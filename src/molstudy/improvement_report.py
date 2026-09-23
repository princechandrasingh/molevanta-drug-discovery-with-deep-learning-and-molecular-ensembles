"""Verify and report exploratory descriptor-augmentation results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256, write_json
from .improved import SPECS
from .improvement import read_json, verify_lock
from .metrics import evaluate
from .report import markdown_table
from .splits import validate_split

MODELS = [
    "prior",
    "compact",
    "descriptor_mlp",
    "fusion_30",
    "fusion_60",
    "selected_blend",
    "forest",
    "chemprop",
]
LABELS = {
    "prior": "Class prior",
    "compact": "Original compact",
    "descriptor_mlp": "Descriptors only",
    "fusion_30": "Compact + descriptors (30 epochs)",
    "fusion_60": "Compact + descriptors (60 epochs)",
    "selected_blend": "Validation-selected combination",
    "forest": "Fixed random forest",
    "chemprop": "Chemprop reference",
}
METRICS = ["average_precision", "roc_auc", "precision_at_20", "brier_score"]
PAIRS = [
    ("fusion_30", "compact"),
    ("fusion_60", "compact"),
    ("selected_blend", "compact"),
    ("fusion_30", "chemprop"),
    ("fusion_60", "chemprop"),
    ("selected_blend", "chemprop"),
    ("selected_blend", "forest"),
    ("fusion_30", "descriptor_mlp"),
]


def verify(output, config, metrics, predictions):
    if config["status"] != "completed":
        raise ValueError("A completed evaluation is required")
    lock = verify_lock(output, config)
    if config["evaluation_started_at"] <= lock["locked_at"]:
        raise ValueError("Test evaluation must begin after all selection decisions are locked")
    if (
        sha256(output / "metrics.csv") != config["metrics_sha256"]
        or sha256(output / "predictions.csv") != config["predictions_sha256"]
    ):
        raise ValueError("Evaluation artifacts changed")
    expected = {
        (kind, seed, name)
        for kind in ("random", "scaffold")
        for seed in config["seeds"]
        for name in MODELS
    }
    if set(
        metrics[["split", "seed", "model"]].itertuples(index=False, name=None)
    ) != expected or len(metrics) != len(expected):
        raise ValueError("Incomplete or duplicate evaluation matrix")
    expected_rows, counts = 0, {}
    for key in lock["decisions"]:
        kind, seed = key.rsplit("_", 1)
        seed = int(seed)
        split = pd.read_csv(output / key / "split.csv", keep_default_na=False)
        validate_split(split, split.partition.to_numpy(), kind)
        test = split[split.partition == "test"].set_index("compound_id").label.sort_index()
        counts[key] = read_json(output / key / "split_summary.json")
        for name in MODELS:
            pred = predictions[
                (predictions.split == kind)
                & (predictions.seed == seed)
                & (predictions.model == name)
            ]
            if pred.compound_id.duplicated().any() or set(pred.compound_id) != set(test.index):
                raise ValueError("Predictions must match exact held-out compounds")
            np.testing.assert_array_equal(pred.set_index("compound_id").label.sort_index(), test)
            # Match the original score dtype so metric arithmetic survives CSV round trips.
            dtype = np.float32 if name in ("compact", "chemprop", *SPECS) else np.float64
            calculated = evaluate(pred.label, pred.score.to_numpy(dtype=dtype))
            row = metrics[
                (metrics.split == kind) & (metrics.seed == seed) & (metrics.model == name)
            ].iloc[0]
            for metric, value in calculated.items():
                np.testing.assert_allclose(value, row[metric], rtol=1e-10, atol=1e-10)
            expected_rows += len(test)
        for name in SPECS:
            if not read_json(output / key / f"{name}_selection.json")["checkpoint_reload_verified"]:
                raise ValueError("Candidate checkpoint reload not verified")
    if len(predictions) != expected_rows:
        raise ValueError("Unexpected predictions")
    return lock, counts


def make_report(run, destination):
    config = read_json(run / "run.json")
    metrics = pd.read_csv(run / "metrics.csv")
    predictions = pd.read_csv(run / "predictions.csv")
    lock, counts = verify(run, config, metrics, predictions)
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "metrics.csv", index=False)
    summary = metrics.groupby(["split", "model"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(destination / "summary.csv", index=False)
    comparison = []
    for kind in ("random", "scaffold"):
        for seed in config["seeds"]:
            rows = metrics[(metrics.split == kind) & (metrics.seed == seed)].set_index("model")
            for model, baseline in PAIRS:
                comparison.append(
                    {
                        "split": kind,
                        "seed": seed,
                        "model": model,
                        "baseline": baseline,
                        **{
                            m + "_delta": float(rows.loc[model, m] - rows.loc[baseline, m])
                            for m in METRICS
                        },
                    }
                )
    paired = pd.DataFrame(comparison)
    paired.to_csv(destination / "paired_differences.csv", index=False)
    table, differences, choices = [], [], []
    for kind in ("random", "scaffold"):
        for name in MODELS:
            row = {"Split": kind, "Model": LABELS[name]}
            for metric, label in [
                ("average_precision", "AP"),
                ("roc_auc", "ROC-AUC"),
                ("precision_at_20", "P@20"),
            ]:
                s = summary.loc[(kind, name)]
                row[label] = f"{s[metric + '_mean']:.3f} ± {s[metric + '_std']:.3f}"
            table.append(row)
        for model, baseline in PAIRS:
            values = paired[
                (paired.split == kind) & (paired.model == model) & (paired.baseline == baseline)
            ].average_precision_delta
            differences.append(
                {
                    "Split": kind,
                    "Candidate": LABELS[model],
                    "Baseline": LABELS[baseline],
                    "Mean ΔAP": f"{values.mean():+.3f}",
                    "SD": f"{values.std():.3f}",
                    "Positive differences": f"{int((values > 0).sum())}/{len(values)}",
                }
            )
    for key, choice in lock["decisions"].items():
        choices.append(
            {
                "Partition": key,
                "Neural candidate": choice["candidate"],
                "Neural weight": choice["neural_weight"],
                "Validation AP": f"{choice['validation_ap']:.3f}",
            }
        )
    os.environ.setdefault("MPLCONFIGDIR", str(run / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(run / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ["compact", "fusion_30", "fusion_60", "selected_blend", "forest", "chemprop"]
    labels = [
        "Original\ncompact",
        "Graph + desc.\n30 epochs",
        "Graph + desc.\n60 epochs",
        "Selected\ncombination",
        "Random\nforest",
        "Chemprop\nreference",
    ]
    colors = ["#989a98", "#209080", "#427b88", "#2258a2", "#b88442", "#8174a5"]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafaf7",
            "axes.facecolor": "#fafaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, kind in zip(axes, ("random", "scaffold")):
        for x, (name, color) in enumerate(zip(shown, colors)):
            values = metrics[
                (metrics.split == kind) & (metrics.model == name)
            ].average_precision.to_numpy()
            ax.bar(x, values.mean(), width=0.65, color=color, alpha=0.73)
            ax.scatter(
                x + np.linspace(-0.13, 0.13, len(values)),
                values,
                color=color,
                s=23,
                edgecolors="white",
                zorder=3,
            )
            ax.text(
                x, min(1.06, values.max() + 0.045), f"{values.mean():.3f}", ha="center", fontsize=9
            )
        ax.set_title(kind.title() + " holdout", loc="left", pad=15)
        ax.set_xticks(range(len(shown)), labels, fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Average precision · higher is better")
    fig.suptitle(
        "Descriptor augmentation: exploratory improvement on the existing benchmark", fontsize=14
    )
    fig.supxlabel(
        "Bars and numeric labels: means · dots: five seeds · reused test sets, not independent confirmation",
        fontsize=10,
    )
    fig.savefig(destination / "model_comparison.png", dpi=180)
    plt.close(fig)
    primary = {
        kind: float(
            paired[
                (paired.split == kind)
                & (paired.model == "fusion_30")
                & (paired.baseline == "compact")
            ].average_precision_delta.mean()
        )
        for kind in ("random", "scaffold")
    }
    pure_forest = sum(c["neural_weight"] == 0 for c in lock["decisions"].values())
    text = f"""# Compact model improvement: exploratory results

Adding molecular descriptors to the compact graph model under its existing 30-epoch budget changed mean average precision by **{primary['random']:+.3f} on random splits** and **{primary['scaffold']:+.3f} on scaffold splits** relative to the original compact model.

These are development results on an already inspected benchmark, not independent confirmation or an improvement claim over the original paper. Every candidate and combination rule was written down before this experiment, and all ten validation-based choices were locked before any new test prediction.

## What changed

- **fusion_30:** retain the original width-64 graph backbone and training schedule; add 200 normalized molecular descriptors to its prediction head.
- **fusion_60:** the same augmented architecture with 60 epochs, constant learning rate 0.001 and weight decay 0.00001. This changes both duration and optimizer settings.
- **descriptor_mlp:** descriptor-only comparison, allowing inspection of whether the graph adds value.
- **selected_blend:** choose the neural candidate and its combination weight with the frozen random forest using validation AP only. Weights tested: 0, 0.25, 0.5, 0.75, 1. This has a larger selection budget than fixed baselines; {pure_forest}/10 choices used the forest alone. It must not be described as a uniformly improved standalone neural model.

Same 2,292 curated compounds, ten original partitions, and seeds {config['seeds']}. Each test set has 229 compounds and 12 active labels. The descriptor normalization/missing-value treatment, chemistry, labels and splits are unchanged. See [the fixed experiment protocol](../../docs/IMPROVEMENT_PROTOCOL.md).

## All held-out results

{markdown_table(pd.DataFrame(table))}

Mean ± sample SD across five seeds; SD is not a confidence interval. AP is scikit-learn average precision. P@20 handles boundary ties. Scores are uncalibrated. Repeated tests reuse compounds; random and scaffold test cohorts differ. The original compact, forest, Chemprop and class-prior values are frozen benchmark results.

![Improvement comparison](model_comparison.png)

## Paired differences on identical test compounds

Positive ΔAP favors the candidate. Negative results and all three standalone candidate settings are retained. The primary comparison is fusion_30 versus original compact. The descriptors-only comparison helps avoid assuming the graph caused all of the gain.

{markdown_table(pd.DataFrame(differences))}

`paired_differences.csv` includes individual seeds and all four metrics; lower Brier is better. These are descriptive differences without a statistical superiority claim.

## Validation decisions fixed before test evaluation

{markdown_table(pd.DataFrame(choices))}

There are only 12 validation positives per split. Candidate/checkpoint/weight selection can overfit this small validation sample. It can therefore underperform a fixed baseline on test data despite a better validation score.

## Verification and provenance

All 30 new models were saved/reloaded and their validation predictions reproduced exactly. The report rechecks selection decisions against saved validation predictions, model/protocol/source/environment hashes, split integrity, all 80 metric rows and all 18,320 saved prediction rows. Neural predictions are restored to their original float32 dtype for metric arithmetic; combinations and forests use float64.

The new head and training orchestration are project-specific AI-assisted code. D-MPNNs, descriptor augmentation and score averaging are established methods. RDKit, descriptastorus, scikit-learn, PyTorch and the Chemprop feature-handling/reference APIs remain attributed third-party software. See [code provenance](../../docs/CODE_PROVENANCE.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).

## Scope of the result

This experiment improves or tests an internal model configuration, not clinical efficacy or discovery of a new drug. It does not reproduce the paper's original optimized setup. An untouched, compatible external assay and sensitivity checks are still needed before generalization or publication-level superiority claims. Dataset redistribution rights and the project's outbound licensing decision remain unresolved.

Only aggregate outputs are included here. Compound data, descriptors, validation/test predictions and checkpoints stay local and are excluded from the source archive.
"""
    (destination / "REPORT.md").write_text(text)
    write_json(destination / "run.json", config)
    write_json(destination / "split_counts.json", counts)
    write_json(
        destination / "verification.json",
        {
            "evaluations_verified": len(metrics),
            "prediction_rows_verified": len(predictions),
            "new_checkpoint_reloads_verified": 30,
            "locked_decisions_verified": len(lock["decisions"]),
            "selection_lock_sha256": config["selection_lock_sha256"],
            "evaluation_started_after_lock": config["evaluation_started_at"] > lock["locked_at"],
            "metrics_sha256": config["metrics_sha256"],
            "predictions_sha256": config["predictions_sha256"],
            "report_generator_sha256": sha256(Path(__file__)),
        },
    )
    print(f"Verified report: {destination / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_report(args.run, args.output)
