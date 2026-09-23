"""Generate a static research report from saved, independently inspectable predictions."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def markdown_table(frame):
    headers = list(frame.columns)
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *[
                "| " + " | ".join(map(str, row)) + " |"
                for row in frame.itertuples(index=False, name=None)
            ],
        ]
    )


def make_report(run_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str(run_dir / ".mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(run_dir / ".cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score

    config = json.loads((run_dir / "run.json").read_text())
    if config["status"] != "completed":
        raise ValueError("Only completed runs can be reported")
    scores = pd.read_csv(run_dir / "metrics.csv")
    predictions = pd.read_csv(run_dir / "predictions.csv")
    metric_names = ["average_precision", "roc_auc", "precision_at_20", "brier_score"]
    summary = scores.groupby(["split", "model"])[metric_names].agg(["mean", "std"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(run_dir / "summary.csv", index=False)
    labels = {
        "prior": "Class prior",
        "logistic": "Logistic regression",
        "forest": "Random forest",
        "dmpnn": "Compact D-MPNN",
    }
    models = config["models"]
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "#fafaf7",
            "axes.facecolor": "#fafaf7",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    x = np.arange(len(models))
    for ax, metric, title in zip(
        axes,
        ["average_precision", "roc_auc"],
        ["Average precision · higher is better", "ROC-AUC · higher is better"],
    ):
        for offset, split, color in [(-0.19, "random", "#177e89"), (0.19, "scaffold", "#df844a")]:
            means = [summary.loc[(split, m), metric + "_mean"] for m in models]
            ax.bar(x + offset, means, width=0.35, color=color, label=split.title(), alpha=0.85)
            for index, model in enumerate(models):
                values = scores.loc[
                    (scores.split == split) & (scores.model == model), metric
                ].to_numpy()
                jitter = (
                    np.linspace(-0.065, 0.065, len(values)) if len(values) > 1 else np.array([0.0])
                )
                ax.scatter(
                    index + offset + jitter,
                    values,
                    s=24,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=4,
                )
        ax.set_ylim(0, 1.04)
        ax.set_xticks(x, [labels[m].replace(" ", "\n", 1) for m in models])
        ax.set_title(title, loc="left", pad=16)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Antibiotic activity prediction: random vs. scaffold holdout", fontsize=15)
    fig.supxlabel(
        f"Bars: mean · dots: individual seeded runs (n={len(config['seeds'])})", fontsize=10
    )
    fig.savefig(run_dir / "model_comparison.png", dpi=180)
    plt.close(fig)

    bins = [0, 0.3, 0.5, 0.7, 1.00001]
    predictions["similarity_bin"] = pd.cut(
        predictions.nearest_train_tanimoto,
        bins=bins,
        labels=["[0,0.3)", "[0.3,0.5)", "[0.5,0.7)", "[0.7,1]"],
        right=False,
    )
    error_rows = []
    for (split, model, band), group in predictions.groupby(
        ["split", "model", "similarity_bin"], observed=True
    ):
        error_rows.append(
            {
                "split": split,
                "model": model,
                "similarity_bin": str(band),
                "prediction_count": len(group),
                "unique_compounds": group.compound_id.nunique(),
                "positives": int(group.label.sum()),
                "mean_absolute_error": float(abs(group.label - group.score).mean()),
                "average_precision": (
                    float(average_precision_score(group.label, group.score))
                    if group.label.nunique() == 2
                    else None
                ),
            }
        )
    pd.DataFrame(error_rows).to_csv(run_dir / "similarity_error_analysis.csv", index=False)
    table_rows = []
    for split in ["random", "scaffold"]:
        for model in models:
            row = {"Split": split, "Model": labels[model]}
            for metric, label in [
                ("average_precision", "AP"),
                ("roc_auc", "ROC-AUC"),
                ("precision_at_20", "P@20"),
            ]:
                mean, sd = summary.loc[(split, model), [metric + "_mean", metric + "_std"]]
                row[label] = (
                    f"{mean:.3f} ± {sd:.3f}" if np.isfinite(sd) else f"{mean:.3f} (single run)"
                )
            table_rows.append(row)
    data = config["data"]
    text = f"""# Pilot results: molecular generalization

Generated from completed run `{run_dir.name}`. This is a computational extension of Stokes et al. (2020), not an exact reproduction or a validated drug discovery.

## Data and evaluation

- Source: [publisher Table S1]({data['source_url']}), sheet S1B; SHA-256 `{data['source_sha256']}`.
- Original: {data['raw_rows']} molecules / {data['raw_actives']} active. Curated: {data['curated_rows']} molecules / {data['curated_actives']} active.
- Seeds fixed in advance: {config['seeds']}. Each model shares the same partitions within a split and seed.
- Split protocol: {config['split_protocol']} Inspect each `split_summary.json` for actual counts.
- Random split groups connectivity-equivalent molecules. Scaffold split groups Bemis–Murcko frameworks, including one shared acyclic group. This prevents exact group overlap, not every possible form of chemical similarity.
- Hyperparameters and D-MPNN checkpoint selection use validation AP only. Models are not refitted on the test set.

## Held-out performance

{markdown_table(pd.DataFrame(table_rows))}

![Model comparison](model_comparison.png)

Values show mean ± sample standard deviation across {len(config['seeds'])} seeds, not confidence intervals. Test compounds overlap between seeds, so these are not independent replications. AP means scikit-learn average precision, not trapezoidal PR-AUC. P@20 averages ties at the ranking boundary; the class-prior baseline therefore has P@20 equal to test prevalence.

## Interpretation limits

This pilot has small, imbalanced test sets. Inspect positive counts before comparing scores. Random and scaffold splits may differ in size and prevalence, affecting AP. A performance difference here does not alone prove an architecture is superior or quantify a universal distribution-shift penalty.

The graph model is a compact, independently implemented D-MPNN with mean pooling, no RDKit descriptor augmentation, no ensembling, and a small training budget. It is not the original Chemprop model. Logistic regression tries three C values and random forest two leaf sizes; the graph model uses one fixed architecture with validation checkpoint selection. Tuning budgets are not matched. Scores are uncalibrated.

`similarity_error_analysis.csv` groups held-out predictions by maximum Morgan-fingerprint Tanimoto similarity to training data. Rows combine seeds, so repeated compounds are counted more than once; both prediction and unique-compound counts are included. Use this as descriptive error analysis, not independent statistical evidence.

## Audit trail

- `run.json`: data checksum, package versions, source-code hashes, configuration, completion status.
- `metrics.csv`: each seed/model result, validation AP and runtime.
- `predictions.csv`: every held-out label, score and nearest-training similarity.
- Per-split folders: exact membership, class counts, model artifacts, validation search/training history.
- Dataset preparation creates `data/processed/curation_audit.csv` with every rejected or merged source row.

## Next experiment

Freeze this pilot. Before claiming a model improvement, predeclare a larger repeated-split study, matched tuning budgets, chemical-standardization sensitivity analysis, and an untouched external evaluation set with compatible assay labels. An additional bootstrap over these reused test molecules would not substitute for independent validation.
"""
    (run_dir / "REPORT.md").write_text(text)
    from .data import sha256, write_json

    write_json(
        run_dir / "report_manifest.json",
        {
            "generator_sha256": sha256(Path(__file__)),
            "metrics_sha256": sha256(run_dir / "metrics.csv"),
            "predictions_sha256": sha256(run_dir / "predictions.csv"),
        },
    )
    print(f"Report: {run_dir / 'REPORT.md'}", flush=True)
