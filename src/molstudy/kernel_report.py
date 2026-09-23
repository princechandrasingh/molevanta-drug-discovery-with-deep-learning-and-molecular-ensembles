"""Verify every saved prediction and report the bounded kernel experiment."""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .data import sha256, write_json
from .improvement import read_json
from .improvement_report import LABELS as OLD_LABELS, METRICS, MODELS as OLD_MODELS
from .kernel import ENSEMBLES, FAMILIES, ensemble_scores
from .kernel_study import NEW_MODELS, verify_lock
from .metrics import evaluate
from .report import markdown_table

MODELS = [*OLD_MODELS, *NEW_MODELS]
LABELS = {
    **OLD_LABELS,
    "kernel_r2": "Tanimoto SVM (radius 2)",
    "kernel_multiscale": "Tanimoto SVM (radii 2 + 3)",
    "kernel_fusion_forest": "Kernel + custom graph + forest",
    "kernel_chemprop_forest": "Kernel + Chemprop + forest",
    "chemprop_forest": "Chemprop + forest",
}
PAIRS = [(name, baseline) for name in NEW_MODELS for baseline in ("forest", "chemprop")] + [
    ("kernel_multiscale", "kernel_r2"),
    ("kernel_fusion_forest", "selected_blend"),
    ("kernel_chemprop_forest", "chemprop_forest"),
]
FLOAT32 = {"compact", "chemprop", "descriptor_mlp", "fusion_30", "fusion_60"}


def verify(run, config, metrics, predictions):
    if config["status"] != "completed":
        raise ValueError("A completed evaluation is required")
    lock = verify_lock(run, config)
    if config["evaluation_started_at"] <= lock["locked_at"]:
        raise ValueError("Evaluation must start after selection is locked")
    for name in ("metrics", "predictions"):
        if sha256(run / f"{name}.csv") != config[f"{name}_sha256"]:
            raise ValueError("Evaluation artifacts changed")
    expected = {
        (kind, seed, name)
        for kind in ("random", "scaffold")
        for seed in config["seeds"]
        for name in MODELS
    }
    if (
        len(metrics) != len(expected)
        or set(metrics[["split", "seed", "model"]].itertuples(index=False, name=None)) != expected
    ):
        raise ValueError("Incomplete or duplicated metric matrix")
    expected_rows, counts = 0, {}
    for key in lock["decisions"]:
        kind, seed = key.rsplit("_", 1)
        seed = int(seed)
        split = pd.read_csv(run / key / "split.csv", keep_default_na=False)
        test = split[split.partition == "test"].set_index("compound_id").label.sort_index()
        counts[key] = read_json(run / key / "split_summary.json")
        score_arrays = {}
        for name in MODELS:
            rows = (
                predictions[
                    (predictions.split == kind)
                    & (predictions.seed == seed)
                    & (predictions.model == name)
                ]
                .set_index("compound_id")
                .sort_index()
            )
            if rows.index.duplicated().any() or set(rows.index) != set(test.index):
                raise ValueError("Predictions differ from exact held-out membership")
            np.testing.assert_array_equal(rows.label, test)
            # CSV erases dtypes; restore neural float32 before recomputing Brier scores.
            score = rows.score.to_numpy(dtype=np.float32 if name in FLOAT32 else np.float64)
            score_arrays[name] = score
            calculated = evaluate(test.to_numpy(), score)
            recorded = metrics[
                (metrics.split == kind) & (metrics.seed == seed) & (metrics.model == name)
            ].iloc[0]
            for metric, value in calculated.items():
                np.testing.assert_allclose(recorded[metric], value, rtol=1e-10, atol=1e-10)
            expected_rows += len(test)
        # Metric agreement alone would not prove that the declared weights were used.
        for name, expected_score in ensemble_scores(score_arrays).items():
            np.testing.assert_allclose(score_arrays[name], expected_score, rtol=1e-14, atol=1e-14)
    if len(predictions) != expected_rows:
        raise ValueError("Unexpected prediction rows")
    return lock, counts


def make_report(run, destination):
    config = read_json(run / "run.json")
    metrics = pd.read_csv(run / "metrics.csv", float_precision="round_trip")
    predictions = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    lock, counts = verify(run, config, metrics, predictions)
    destination.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(destination / "metrics.csv", index=False)
    summary = metrics.groupby(["split", "model"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(destination / "summary.csv", index=False)
    pairs = []
    for kind in ("random", "scaffold"):
        for seed in config["seeds"]:
            block = metrics[(metrics.split == kind) & (metrics.seed == seed)].set_index("model")
            for name, baseline in PAIRS:
                pairs.append(
                    {
                        "split": kind,
                        "seed": seed,
                        "model": name,
                        "baseline": baseline,
                        **{
                            m + "_delta": float(block.loc[name, m] - block.loc[baseline, m])
                            for m in METRICS
                        },
                    }
                )
    paired = pd.DataFrame(pairs)
    paired.to_csv(destination / "paired_differences.csv", index=False)
    table, differences, choices = [], [], []
    for kind in ("random", "scaffold"):
        for name in MODELS:
            s = summary.loc[(kind, name)]
            table.append(
                {
                    "Split": kind,
                    "Model": LABELS[name],
                    **{
                        label: f"{s[m + '_mean']:.3f} ± {s[m + '_std']:.3f}"
                        for m, label in [
                            ("average_precision", "AP"),
                            ("roc_auc", "ROC-AUC"),
                            ("precision_at_20", "P@20"),
                        ]
                    },
                }
            )
        for name, baseline in PAIRS:
            delta = paired[
                (paired.split == kind) & (paired.model == name) & (paired.baseline == baseline)
            ].average_precision_delta
            differences.append(
                {
                    "Split": kind,
                    "Candidate": LABELS[name],
                    "Baseline": LABELS[baseline],
                    "Mean ΔAP": f"{delta.mean():+.3f}",
                    "SD": f"{delta.std():.3f}",
                    "Positive differences": f"{int((delta > 0).sum())}/{len(delta)}",
                }
            )
    for key, decision in lock["decisions"].items():
        for name, selection in decision["selections"].items():
            choices.append(
                {
                    "Partition": key,
                    "Model": name,
                    "C": selection["setting"]["C"],
                    "Class weight": str(selection["setting"]["class_weight"]),
                    "Training CV AP": f"{selection['cv_ap']:.3f}",
                }
            )
    primary = paired[
        (paired.split == "scaffold")
        & (paired.model == "kernel_multiscale")
        & (paired.baseline == "forest")
    ].average_precision_delta
    plot(metrics, destination)
    report = f"""# Fingerprint kernels and fixed ensembles: exploratory results

The predeclared primary comparison, **multiscale Tanimoto SVM versus the fixed forest on scaffold AP**, changed by **{primary.mean():+.3f}** on average, with positive differences in **{int((primary > 0).sum())}/5** seeds. All five new candidates and all eight prior model families are retained below, regardless of outcome.

This is development on the same previously inspected benchmark. It is not independent confirmation, statistical proof of superiority, or a comparison with Stokes' original optimized model. The full [predeclared protocol](../../docs/KERNEL_PROTOCOL.md) records the research sources and selection budget.

## Research translated into the experiment

- [Yang et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6727618/) motivates molecular features and model averaging. We retain the prior graph models and investigate heterogeneous equal-weight ensembles; these combinations are our experimental choices.
- [Tripp et al. (2023)](https://arxiv.org/abs/2306.14809) studies Tanimoto kernels for molecular learning. We implement the established exact binary kernel, which fits this dataset's size, rather than reproducing their random-feature approximation.
- [Cawley and Talbot (2010)](https://www.jmlr.org/papers/v11/cawley10a.html) explains overfitting during model selection. We restrict kernel tuning to three grouped inner folds and lock all choices before a separate test-scoring command.

## What changed

Two SVMs use radius-2 Morgan similarity or the fixed mean of radius-2/radius-3 similarities. Each uses an eight-setting C/class-weight grid, selected by mean training-only inner-fold AP. Connectivity/scaffold groups stay together as appropriate. The three inner folds contain 23–45 positives in this run; the former outer validation sets have just 12. Group sizes still differ, especially for scaffold folds.

Each selected SVM is refitted on all original training rows. A regularized logistic calibration is fitted to its out-of-fold training decisions. Calibration and hyperparameter selection reuse those folds, so CV results are selection diagnostics, not independent estimates. Neither outer validation nor test labels tune the new kernel. Existing neural checkpoints retain their previous validation-based selection.

Three fixed, equal-weight arithmetic means combine: (a) kernel/custom graph/forest, (b) kernel/Chemprop/forest, and (c) Chemprop/forest. The third is a control for whether the kernel adds to a simpler reference ensemble. No ensemble weights were fitted. These ensembles have greater combined model and optimization budgets than any single fixed baseline.

## Complete results

{markdown_table(pd.DataFrame(table))}

Mean ± sample SD over five seeds; SD is not a confidence interval. AP is average precision, not trapezoidal PR-AUC. Higher AP, ROC-AUC and P@20 are better; lower Brier score is better. All Brier values are retained in the CSVs. Each test has 229 compounds and 12 positives. Test sets overlap across seeds, and random/scaffold cohorts differ.

![Kernel and ensemble comparison](model_comparison.png)

## Paired comparisons on identical test compounds

{markdown_table(pd.DataFrame(differences))}

Every seed and all four metrics appear in `paired_differences.csv`. Gains in one metric need not imply gains in every metric or partition. The radius-2 ablation and two-component ensemble control are retained even if they outperform the more complex alternatives. Do not select a new default based solely on these reused test results.

## Training-only choices

{markdown_table(pd.DataFrame(choices))}

All 20 kernel fits were serialized and reloaded with exactly reproduced validation predictions. Training-fold membership, all eight candidates' out-of-fold scores, selected settings, calibration parameters and component checkpoints remain in the local run. A single lock covers all ten partitions before evaluation.

## Verification and interpretation

The report verifies all {len(metrics)} metric rows, {len(predictions):,} prediction rows, 30 ensemble outputs, 20 selection decisions, training group isolation, original test membership/labels, and source/protocol/environment/artifact hashes. The evaluator additionally checks executing source against its frozen snapshot and reproduces the three old component models' test predictions before combining them. Prior evaluation results are carried forward unchanged.

No new packages, pretrained weights, paper figures or third-party implementation files were added. The kernel formula, wrappers, experiments and tests were generated here with AI assistance; scikit-learn supplies SVM/logistic estimators, RDKit supplies fingerprints, and Chemprop remains an explicitly attributed reference implementation. These are established methods, not a claim of a newly invented algorithm. See [code provenance](../../docs/CODE_PROVENANCE.md).

The data, benchmark reuse, overlapping cohorts, greedy scaffold allocation and external descriptor normalization remain limitations. Training-only sigmoid calibration does not establish calibrated probabilities outside this assay. A compatible untouched external assay and prospective laboratory validation remain outstanding. No new drug or clinical benefit is established. Only aggregate results are included in this report; molecular data, predictions and trained models remain excluded from the source bundle.
"""
    (destination / "REPORT.md").write_text(report)
    write_json(destination / "run.json", config)
    write_json(destination / "split_counts.json", counts)
    write_json(
        destination / "verification.json",
        {
            "metric_rows_verified": len(metrics),
            "prediction_rows_verified": len(predictions),
            "ensemble_outputs_verified": 30,
            "kernel_checkpoint_reloads_verified": 20,
            "kernel_selections_verified": 20,
            "inner_folds_verified": 30,
            "selection_lock_sha256": config["selection_lock_sha256"],
            "evaluation_started_after_lock": True,
            "metrics_sha256": config["metrics_sha256"],
            "predictions_sha256": config["predictions_sha256"],
            "report_generator_sha256": sha256(Path(__file__)),
        },
    )
    print(f"Verified report: {destination / 'REPORT.md'}", flush=True)


def plot(metrics, destination):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/molstudy-mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = [
        "fusion_60",
        "selected_blend",
        "forest",
        "chemprop",
        "kernel_multiscale",
        "kernel_fusion_forest",
        "kernel_chemprop_forest",
        "chemprop_forest",
    ]
    labels = [
        "Custom\ngraph",
        "Previous\nblend",
        "Random\nforest",
        "Chemprop",
        "Kernel\nSVM",
        "Kernel + graph\n+ forest",
        "Kernel + CP\n+ forest",
        "CP +\nforest",
    ]
    colors = [
        "#989a98",
        "#aaa19a",
        "#b88442",
        "#8174a5",
        "#43877b",
        "#2258a2",
        "#594494",
        "#796c8b",
    ]
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafaf7",
            "axes.facecolor": "#fafaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), constrained_layout=True)
    for ax, kind in zip(axes, ("random", "scaffold")):
        for position, (name, color) in enumerate(zip(shown, colors)):
            values = metrics[
                (metrics.split == kind) & (metrics.model == name)
            ].average_precision.to_numpy()
            ax.bar(position, values.mean(), width=0.65, color=color, alpha=0.72)
            ax.scatter(
                position + np.linspace(-0.13, 0.13, len(values)),
                values,
                color=color,
                edgecolors="white",
                zorder=3,
                s=24,
            )
            ax.text(position, values.max() + 0.04, f"{values.mean():.3f}", ha="center", fontsize=9)
        ax.set_title(kind.title() + " holdout", loc="left", pad=15)
        ax.set_xticks(range(len(shown)), labels, fontsize=8.5)
        ax.set_ylim(0, 1.1)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Average precision · higher is better")
    fig.suptitle("Fingerprint kernels and fixed ensembles · exploratory benchmark", fontsize=14)
    fig.supxlabel(
        "Bars and numbers: means · dots: five seeds · reused test sets, not independent validation",
        fontsize=10,
    )
    fig.savefig(destination / "model_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    make_report(args.run, args.output)
