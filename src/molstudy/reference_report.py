"""Audit and summarize completed shared-split reference experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256, write_json
from .metrics import evaluate
from .report import markdown_table
from .splits import validate_split

METRICS = ["average_precision", "roc_auc", "precision_at_20", "brier_score"]
LABELS = {
    "prior": "Class prior",
    "forest": "Random forest (500 trees)",
    "compact": "Project compact D-MPNN",
    "chemprop": "Chemprop + descriptors",
}


def paired_differences(scores):
    if scores.duplicated(["split", "seed", "model"]).any():
        raise ValueError("Duplicate model/seed results")
    reference = scores[scores.model == "chemprop"].set_index(["split", "seed"])
    rows = []
    for name, group in scores[scores.model != "chemprop"].groupby("model"):
        candidate = group.set_index(["split", "seed"])
        if set(candidate.index) != set(reference.index):
            raise ValueError("Unpaired candidate and reference results")
        for index, row in candidate.iterrows():
            base = reference.loc[index]
            if row["n"] != base["n"] or row.positives != base.positives:
                raise ValueError("Paired test counts differ")
            rows.append(
                {
                    "split": index[0],
                    "seed": index[1],
                    "model": name,
                    **{metric + "_delta": float(row[metric] - base[metric]) for metric in METRICS},
                }
            )
    return pd.DataFrame(rows)


def verify_run(run_dir, config, scores, predictions):
    if config["status"] != "completed":
        raise ValueError("Report requires a completed run")
    expected = {
        (kind, seed, name)
        for kind in ("random", "scaffold")
        for seed in config["seeds"]
        for name in config["models"]
    }
    actual = set(scores[["split", "seed", "model"]].itertuples(index=False, name=None))
    if actual != expected or len(scores) != len(expected):
        raise ValueError("Missing, duplicate, or unexpected evaluation results")
    for filename, digest in config["source_hashes"].items():
        if sha256(run_dir / "source" / filename) != digest:
            raise ValueError("Model source snapshot changed")
    if sha256(run_dir / "REFERENCE_PROTOCOL.md") != config["protocol_sha256"]:
        raise ValueError("Protocol snapshot changed")
    if sha256(run_dir / "requirements-reference-lock.txt") != config["environment_lock_sha256"]:
        raise ValueError("Environment snapshot changed")
    compound_sets, summaries = {}, {}
    expected_prediction_rows = 0
    for kind in ("random", "scaffold"):
        for seed in config["seeds"]:
            key = f"{kind}_{seed}"
            split_path = run_dir / key / "split.csv"
            if sha256(split_path) != config["split_hashes"][key]:
                raise ValueError("Split membership changed")
            split = pd.read_csv(split_path, keep_default_na=False)
            validate_split(split, split.partition.to_numpy(), kind)
            test = split[split.partition == "test"].set_index("compound_id").label.sort_index()
            compound_sets[kind, seed] = set(test.index)
            summaries[key] = json.loads((run_dir / key / "split_summary.json").read_text())
            for name in config["models"]:
                pred = predictions[
                    (predictions.split == kind)
                    & (predictions.seed == seed)
                    & (predictions.model == name)
                ]
                expected_prediction_rows += len(test)
                if pred.compound_id.duplicated().any() or set(pred.compound_id) != set(test.index):
                    raise ValueError("Models must predict exactly the same held-out compounds")
                np.testing.assert_array_equal(
                    pred.set_index("compound_id").label.sort_index(), test
                )
                result = scores[
                    (scores.split == kind) & (scores.seed == seed) & (scores.model == name)
                ].iloc[0]
                # CSV readers infer float64, while Torch produced float32 scores.
                # Restore their native dtype to reproduce Brier arithmetic exactly.
                score = pred.score.to_numpy(
                    dtype=np.float32 if name in ("compact", "chemprop") else np.float64
                )
                for metric, value in evaluate(pred.label, score).items():
                    np.testing.assert_allclose(value, result[metric], rtol=1e-10, atol=1e-10)
                if name != "prior" and not result.checkpoint_reload_verified:
                    raise ValueError("Saved model reload was not verified")
    if len(predictions) != expected_prediction_rows:
        raise ValueError("Unexpected prediction rows")
    coverage = {
        kind: len(set.union(*(compound_sets[kind, seed] for seed in config["seeds"])))
        for kind in ("random", "scaffold")
    }
    return summaries, coverage


def make_reference_report(run_dir: Path, output: Path):
    config = json.loads((run_dir / "run.json").read_text())
    scores = pd.read_csv(run_dir / "metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    counts, coverage = verify_run(run_dir, config, scores, predictions)
    unique_test_actives = {
        kind: int(
            predictions[(predictions.split == kind) & (predictions.model == "prior")]
            .drop_duplicates("compound_id")
            .label.sum()
        )
        for kind in ("random", "scaffold")
    }
    deltas = paired_differences(scores)
    output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output / "metrics.csv", index=False)
    deltas.to_csv(output / "paired_differences.csv", index=False)
    summary = scores.groupby(["split", "model"])[METRICS + ["seconds"]].agg(["mean", "std"])
    summary.columns = ["_".join(item) for item in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    write_json(output / "split_counts.json", counts)
    write_json(output / "run.json", config)
    rows, comparisons = [], []
    for kind in ("random", "scaffold"):
        for name in config["models"]:
            values = summary.loc[kind, name]
            row = {"Split": kind, "Model": LABELS[name]}
            for metric, label in [
                ("average_precision", "AP"),
                ("roc_auc", "ROC-AUC"),
                ("precision_at_20", "P@20"),
            ]:
                row[label] = f"{values[metric + '_mean']:.3f} ± {values[metric + '_std']:.3f}"
            rows.append(row)
        for name in ("compact", "forest"):
            paired = deltas[(deltas.split == kind) & (deltas.model == name)].average_precision_delta
            comparisons.append(
                {
                    "Split": kind,
                    "Model minus Chemprop": LABELS[name],
                    "Mean ΔAP": f"{paired.mean():+.3f}",
                    "SD of ΔAP": f"{paired.std():.3f}",
                    "Positive differences": f"{int((paired > 0).sum())}/{len(paired)}",
                    "Individual ΔAP": ", ".join(f"{v:+.3f}" for v in paired),
                }
            )
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(run_dir / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafaf7",
            "axes.facecolor": "#fafaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = {"prior": "#9a9d9c", "forest": "#196e67", "compact": "#d88444", "chemprop": "#4664ad"}
    for ax, kind in zip(axes, ("random", "scaffold")):
        for x, name in enumerate(config["models"]):
            values = scores[
                (scores.split == kind) & (scores.model == name)
            ].average_precision.to_numpy()
            ax.bar(x, values.mean(), width=0.65, color=colors[name], alpha=0.7)
            ax.scatter(
                x + np.linspace(-0.14, 0.14, len(values)),
                values,
                s=34,
                c=colors[name],
                edgecolors="white",
                zorder=3,
            )
            ax.text(
                x,
                min(1.035, max(values.max(), values.mean()) + 0.035),
                f"{values.mean():.3f}",
                ha="center",
                fontsize=10,
            )
        ax.set_title(kind.title() + " holdout", loc="left", fontsize=13, pad=14)
        ax.set_ylim(0, 1.11)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.set_xticks(
            range(4),
            ["Class\nprior", "Random\nforest", "Compact\nD-MPNN", "Chemprop +\ndescriptors"],
        )
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Average precision · higher is better")
    fig.suptitle("Shared molecules, shared splits: a reference-method benchmark", fontsize=15)
    fig.supxlabel(
        "Bars: means · dots: five fixed seeds · one configuration per model family", fontsize=10
    )
    fig.savefig(output / "reference_comparison.png", dpi=180)
    plt.close(fig)
    missing = json.loads((run_dir / "descriptor_missingness.json").read_text())["replaced_cells"]
    descriptor_note = f"{len(missing)} NaN descriptor cells across {len({r['row_index'] for r in missing})} molecules were replaced by zero using Chemprop's native policy; no fitted imputation or label-dependent curation."
    compact_deltas = {
        kind: deltas[
            (deltas.split == kind) & (deltas.model == "compact")
        ].average_precision_delta.mean()
        for kind in ("random", "scaffold")
    }
    direction = "; ".join(f"{kind} mean ΔAP {delta:+.3f}" for kind, delta in compact_deltas.items())
    text = f"""# Shared-split reference comparison

The compact project model versus the licensed Chemprop reference: **{direction}**. These are internal benchmark differences, not evidence of an improvement over the paper's published result or a new drug discovery.

## Fixed experiment

- Run: `{run_dir.name}`; protocol `{config['protocol']}`; five seeds {config['seeds']}.
- Same curated Table S1B as the pilot: 2,292 molecules, 119 active. Each model receives the same train/validation/test membership within each split and seed.
- One configuration per model family. Both neural networks train for 30 complete epochs under the same optimizer schedule; validation AP selects their checkpoints. Models differ in size, features and regularization. Equal configuration counts do not mean equal computational cost or optimization quality.
- Reference: Chemprop 1.6.1, hidden size 300, directed bond messages, depth 3, mean pooling and 200 normalized RDKit descriptors. The project compact model uses hidden size 64 without descriptors. The fixed 500-tree forest differs from the class-weighted, tuned pilot forest.
- {descriptor_note}
- Exact membership is retained locally. Aggregate partition counts are in `split_counts.json`. Unique test compounds across the five seeds: random **{coverage['random']}** (including {unique_test_actives['random']} active), scaffold **{coverage['scaffold']}** (including {unique_test_actives['scaffold']} active). The seeds reuse compounds and are not independent external cohorts.

## Held-out results

{markdown_table(pd.DataFrame(rows))}

Values are mean ± sample SD across seeds, not confidence intervals. AP is average precision, not trapezoidal PR-AUC. Test sets are small and imbalanced; class prior provides the prevalence-dependent ranking baseline. P@20 uses tie averaging.

![Reference comparison](reference_comparison.png)

## Paired comparison on identical test compounds

Positive ΔAP favors the named model; negative ΔAP favors Chemprop. Individual differences follow the seed order above. These are descriptive values, with no independence assumption, statistical significance claim or external-validation claim.

{markdown_table(pd.DataFrame(comparisons))}

`paired_differences.csv` also includes ROC-AUC, P@20 and Brier differences. For Brier, lower is better. `metrics.csv` retains runtimes and neural parameter counts.

## What this says about the paper

Stokes et al. report random-split ROC-AUC around 0.896 under their own sample and protocol. Here, standardization reduces the dataset, partitions differ, the reference software postdates the paper, architecture search/ensembling are omitted, and the trainer selects validation AP. Our new comparison is internally matched; it is **not a replication of the published optimized model**. A score above 0.896 cannot be called an improvement over that paper.

The study still lacks an untouched, compatible external assay cohort. The label-aware allocation and treatment of acyclic molecules also limit generalization. Standardization sensitivity and external evaluation remain necessary before broader claims.

Random and scaffold experiments hold out different test cohorts. A higher scaffold mean in these particular runs does not show that unfamiliar chemistry is inherently easier, just as a lower mean would not isolate a universal distribution-shift penalty. The within-split, paired model differences are the more controlled comparisons.

The compact model's learning-rate schedule, batch size and checkpoint budget changed from the pilot so the two neural implementations could share a training recipe. Its new score should not be compared with the old pilot as an isolated model improvement or regression: the seeds and training recipe both changed. This fixed-configuration experiment also cannot identify whether differences arise from model size, descriptors, molecular features or optimization. A subsequent controlled ablation would need to vary those factors separately and keep newly observed test scores out of its selection decisions.

## Verification and provenance

The report generator revalidated all ten splits, matched every model's held-out compound IDs/labels, recomputed all metrics from saved predictions, and verified frozen protocol/source/environment/split hashes. Each of the 30 fitted models was saved, reloaded and checked against its original predictions during the run. The failed descriptor-only preflight is preserved separately; it produced no fitted models or test scores.

Only aggregate results are included in this report snapshot. Raw data, molecular predictions, descriptor arrays and checkpoints remain local. For implementation authorship and license scope, see [code provenance](../../docs/CODE_PROVENANCE.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).

Sources: [Stokes et al. (2020)](https://doi.org/10.1016/j.cell.2020.01.021), [Chemprop](https://github.com/chemprop/chemprop), [Yang et al. (2019)](https://doi.org/10.1021/acs.jcim.9b00237).
"""
    (output / "REPORT.md").write_text(text)
    write_json(
        output / "verification.json",
        {
            "evaluations_verified": len(scores),
            "prediction_rows_verified": len(predictions),
            "fitted_model_reloads_verified": int(scores.checkpoint_reload_verified.sum()),
            "test_compound_coverage": coverage,
            "unique_test_actives": unique_test_actives,
            "descriptor_nan_cells": len(missing),
            "metrics_sha256": sha256(run_dir / "metrics.csv"),
            "predictions_sha256": sha256(run_dir / "predictions.csv"),
            "prediction_dtype_audit": "Torch scores restored to float32 from round-trip CSV; forest/prior float64. All metrics matched at rtol=1e-10, atol=1e-10.",
            "report_generator_sha256": sha256(Path(__file__)),
        },
    )
    print(f"Verified report: {output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    make_reference_report(arguments.run_dir, arguments.output)
