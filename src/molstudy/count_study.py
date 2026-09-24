"""Controlled count-fingerprint extension with locked training and separate evaluation."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .data import sha256, write_json
from .features import count_fingerprints, fingerprints
from .improvement import check_environment, read_json
from .kernel import GRID, fit, select_setting, validate_folds
from .kernel_report import LABELS as BASE_LABELS, METRICS, verify as verify_base_results
from .kernel_study import now, verify_lock as verify_base_lock
from .metrics import evaluate
from .report import markdown_table
from .splits import validate_split

FAMILIES = {"count_r2": (2,), "count_multiscale": (2, 3)}
ENSEMBLES = {
    "count_fusion_forest": ("count_multiscale", "fusion_60", "forest"),
    "count_chemprop_forest": ("count_multiscale", "chemprop", "forest"),
}
PAIRS = {
    "count_r2": "kernel_r2",
    "count_multiscale": "kernel_multiscale",
    "count_fusion_forest": "kernel_fusion_forest",
    "count_chemprop_forest": "kernel_chemprop_forest",
}
LABELS = {
    **BASE_LABELS,
    "count_r2": "Count Tanimoto SVM (radius 2)",
    "count_multiscale": "Count Tanimoto SVM (radii 2 + 3)",
    "count_fusion_forest": "Count kernel + custom graph + forest",
    "count_chemprop_forest": "Count kernel + Chemprop + forest",
}


def specification():
    return {
        "families": {k: list(v) for k, v in FAMILIES.items()},
        "ensembles": {k: list(v) for k, v in ENSEMBLES.items()},
        "grid": GRID,
        "kernel_kind": "count",
    }


def combine(scores):
    """Reuse frozen neural/forest scores with a fixed one-third weight per component."""
    result = {}
    for name, members in ENSEMBLES.items():
        values = np.stack([np.asarray(scores[m], dtype=np.float64) for m in members])
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid ensemble scores")
        result[name] = values.mean(axis=0)
    return result


def feature_matrix(strings):
    return {radius: count_fingerprints(strings, radius) for radius in (2, 3)}


def load_frame(config, key):
    """Align chemistry to the frozen split before selecting train/validation/test rows."""
    path = Path(config["data_file"])
    if sha256(path) != config["data_sha256"]:
        raise ValueError("Curated data changed")
    frame = pd.read_csv(path, keep_default_na=False)
    split = pd.read_csv(Path(config["base_run"]) / key / "split.csv", keep_default_na=False)
    np.testing.assert_array_equal(frame.compound_id, split.compound_id)
    np.testing.assert_array_equal(frame.label, split.label)
    validate_split(split, split.partition.to_numpy(), key.rsplit("_", 1)[0])
    frame["partition"] = split.partition.to_numpy()
    return frame


def check_base(base):
    config = read_json(base / "run.json")
    if config["status"] != "completed":
        raise ValueError("A completed binary-kernel study is required")
    lock = verify_base_lock(base, config)
    for name in ("metrics", "predictions"):
        if sha256(base / f"{name}.csv") != config[f"{name}_sha256"]:
            raise ValueError("Base evaluation changed")
    return config, lock


def train(data_root, base, output):
    versions = check_environment()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    base_config, base_lock = check_base(base)
    project = Path(__file__).resolve().parents[2]
    config = {
        "status": "training",
        "started_at": now(),
        "specification": specification(),
        "base_run": str(base.resolve()),
        "environment": versions,
        "data_file": str((data_root / "data/processed/molecules.csv").resolve()),
        "data_sha256": base_config["data"]["processed_sha256"],
        "base_hashes": {
            name: sha256(base / name) for name in ("run.json", "metrics.csv", "predictions.csv")
        },
    }
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, output / "source" / source.name)
    shutil.copy2(project / "docs/COUNT_PROTOCOL.md", output / "PROTOCOL.md")
    shutil.copy2(
        project / "requirements-reference-lock.txt", output / "requirements-reference-lock.txt"
    )
    config["source_hashes"] = {p.name: sha256(p) for p in sorted((output / "source").glob("*.py"))}
    config["protocol_sha256"] = sha256(output / "PROTOCOL.md")
    config["environment_lock_sha256"] = sha256(output / "requirements-reference-lock.txt")
    write_json(output / "run.json", config)
    decisions = {}
    try:
        for key in base_lock["decisions"]:
            frame = load_frame(config, key)
            training = frame[frame.partition == "train"]
            validation = frame[frame.partition == "validation"]
            folder = output / key
            folder.mkdir()
            for name in ("split.csv", "inner_folds.csv"):
                shutil.copy2(base / key / name, folder / name)
            folds = pd.read_csv(folder / "inner_folds.csv", keep_default_na=False)
            np.testing.assert_array_equal(folds.compound_id, training.compound_id)
            train_x, val_x = feature_matrix(training.smiles), feature_matrix(validation.smiles)
            with np.load(base / key / "validation.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["label"], validation.label)
                scores = {name: saved[name].copy() for name in ("forest", "fusion_60", "chemprop")}
                binary_x = {r: fingerprints(validation.smiles, radius=r)[0] for r in (2, 3)}
                for name in ("kernel_r2", "kernel_multiscale"):
                    # Prove that adding count support did not change old pickle behavior.
                    np.testing.assert_array_equal(
                        joblib.load(base / key / f"{name}.joblib").predict(binary_x), saved[name]
                    )
            selections = {}
            for name, radii in FAMILIES.items():
                model, selection, oof = fit(
                    train_x,
                    training.label.to_numpy(),
                    folds.group.to_numpy(),
                    folds.fold.to_numpy(),
                    radii,
                    kernel_kind="count",
                )
                joblib.dump(model, folder / f"{name}.joblib")
                np.save(folder / f"{name}_oof.npy", oof, allow_pickle=False)
                scores[name] = model.predict(val_x)
                np.testing.assert_array_equal(
                    joblib.load(folder / f"{name}.joblib").predict(val_x), scores[name]
                )
                selection["checkpoint_reload_verified"] = True
                selections[name] = selection
                print(f"{key} {name}: training CV AP={selection['cv_ap']:.3f}", flush=True)
            scores.update(combine(scores))
            np.savez(folder / "validation.npz", label=validation.label.to_numpy(), **scores)
            decisions[key] = {
                "selections": selections,
                "binary_reload_verified": True,
                "artifact_hashes": {p.name: sha256(p) for p in sorted(folder.iterdir())},
            }
        write_json(output / "selection_lock.json", {"locked_at": now(), "decisions": decisions})
        config.update(
            status="selection_locked", selection_lock_sha256=sha256(output / "selection_lock.json")
        )
    except Exception as exc:
        config.update(status="training_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "run.json", config)
    print("All 20 count models locked. No count-model test scores calculated.", flush=True)


def verify_lock(run, config, *, current_source=True):
    """Verify provenance and recompute selections before loading executable checkpoints."""
    if config["status"] not in {"selection_locked", "evaluating", "completed"}:
        raise ValueError("A selection-locked run is required")
    if config["specification"] != specification():
        raise ValueError("Candidate specification changed")
    for name, key in (
        ("selection_lock.json", "selection_lock_sha256"),
        ("PROTOCOL.md", "protocol_sha256"),
        ("requirements-reference-lock.txt", "environment_lock_sha256"),
    ):
        if sha256(run / name) != config[key]:
            raise ValueError(f"Locked artifact changed: {name}")
    for name, digest in config["source_hashes"].items():
        if sha256(run / "source" / name) != digest:
            raise ValueError(f"Source snapshot changed: {name}")
        if current_source and sha256(Path(__file__).parent / name) != digest:
            raise ValueError(f"Executing source differs from snapshot: {name}")
    base = Path(config["base_run"])
    for name, digest in config["base_hashes"].items():
        if sha256(base / name) != digest:
            raise ValueError(f"Base run changed: {name}")
    _, base_lock = check_base(base)
    lock = read_json(run / "selection_lock.json")
    if set(lock["decisions"]) != set(base_lock["decisions"]):
        raise ValueError("Incomplete partition coverage")
    for key, decision in lock["decisions"].items():
        folder = run / key
        for name, digest in decision["artifact_hashes"].items():
            if sha256(folder / name) != digest:
                raise ValueError(f"Training artifact changed: {key}/{name}")
        for name in ("split.csv", "inner_folds.csv"):
            if sha256(folder / name) != sha256(base / key / name):
                raise ValueError("Partitions or training folds differ from baseline")
        folds = pd.read_csv(folder / "inner_folds.csv", keep_default_na=False)
        validate_folds(folds.label, folds.group, folds.fold)
        for name in FAMILIES:
            selection = select_setting(
                folds.label.to_numpy(),
                folds.fold.to_numpy(),
                np.load(folder / f"{name}_oof.npy", allow_pickle=False),
            )
            recorded = decision["selections"][name]
            if any(recorded[k] != v for k, v in selection.items()):
                raise ValueError("Selection differs from training CV evidence")
            if not recorded["checkpoint_reload_verified"] or not decision["binary_reload_verified"]:
                raise ValueError("Missing checkpoint reload verification")
    return lock


def old_scores(predictions, key, test):
    kind, seed = key.rsplit("_", 1)
    block = predictions[(predictions.split == kind) & (predictions.seed == int(seed))]
    scores = {}
    for name in BASE_LABELS:
        rows = block[block.model == name].set_index("compound_id")
        if rows.index.duplicated().any() or set(rows.index) != set(test.compound_id):
            raise ValueError("Base test membership changed")
        rows = rows.loc[test.compound_id]
        np.testing.assert_array_equal(rows.label, test.label)
        scores[name] = rows.score.to_numpy()
    return scores


def evaluate_locked(run):
    check_environment()
    config = read_json(run / "run.json")
    if config["status"] != "selection_locked":
        raise ValueError("Evaluation requires a locked, not yet evaluated run")
    lock = verify_lock(run, config)
    base = Path(config["base_run"])
    old = pd.read_csv(base / "predictions.csv", float_precision="round_trip")
    rows, predictions = [], []
    config.update(status="evaluating", evaluation_started_at=now())
    write_json(run / "run.json", config)
    for key in lock["decisions"]:
        frame = load_frame(config, key)
        test = frame[frame.partition == "test"]
        x = feature_matrix(test.smiles)
        scores = old_scores(old, key, test)
        scores.update(
            {name: joblib.load(run / key / f"{name}.joblib").predict(x) for name in FAMILIES}
        )
        scores.update(combine(scores))
        kind, seed = key.rsplit("_", 1)
        with np.load(run / key / "validation.npz", allow_pickle=False) as validation:
            for name in PAIRS:
                rows.append(
                    {
                        "split": kind,
                        "seed": int(seed),
                        "model": name,
                        **evaluate(test.label.to_numpy(), scores[name]),
                        "validation_ap": evaluate(validation["label"], validation[name])[
                            "average_precision"
                        ],
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "compound_id": test.compound_id,
                            "label": test.label,
                            "split": kind,
                            "seed": int(seed),
                            "model": name,
                            "score": scores[name],
                        }
                    )
                )
        print(f"{key}: count-model and ensemble test predictions saved", flush=True)
    pd.DataFrame(rows).to_csv(run / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(run / "predictions.csv", index=False)
    config.update(
        status="completed",
        completed_at=now(),
        metrics_sha256=sha256(run / "metrics.csv"),
        predictions_sha256=sha256(run / "predictions.csv"),
    )
    write_json(run / "run.json", config)


def verified_results(run):
    config = read_json(run / "run.json")
    if config["status"] != "completed":
        raise ValueError("A completed count evaluation is required")
    lock = verify_lock(run, config, current_source=False)
    if config["evaluation_started_at"] <= lock["locked_at"]:
        raise ValueError("Evaluation preceded the selection lock")
    for name in ("metrics", "predictions"):
        if sha256(run / f"{name}.csv") != config[f"{name}_sha256"]:
            raise ValueError("Evaluation artifact changed")
    base = Path(config["base_run"])
    metrics = pd.read_csv(run / "metrics.csv", float_precision="round_trip")
    predictions = pd.read_csv(run / "predictions.csv", float_precision="round_trip")
    base_metrics = pd.read_csv(base / "metrics.csv", float_precision="round_trip")
    old = pd.read_csv(base / "predictions.csv", float_precision="round_trip")
    verify_base_results(base, read_json(base / "run.json"), base_metrics, old)
    expected_rows = 0
    expected_metrics = set()
    for key in lock["decisions"]:
        kind, seed = key.rsplit("_", 1)
        frame = load_frame(config, key)
        test = frame[frame.partition == "test"]
        scores = old_scores(old, key, test)
        for name in PAIRS:
            expected_metrics.add((kind, int(seed), name))
            part = predictions[
                (predictions.split == kind)
                & (predictions.seed == int(seed))
                & (predictions.model == name)
            ].set_index("compound_id")
            if part.index.duplicated().any() or set(part.index) != set(test.compound_id):
                raise ValueError("Count test membership changed")
            part = part.loc[test.compound_id]
            np.testing.assert_array_equal(part.label, test.label)
            scores[name] = part.score.to_numpy()
            calculated = evaluate(part.label.to_numpy(), scores[name])
            recorded = metrics[
                (metrics.split == kind) & (metrics.seed == int(seed)) & (metrics.model == name)
            ]
            if len(recorded) != 1:
                raise ValueError("Missing or duplicated count metrics")
            for metric, value in calculated.items():
                np.testing.assert_allclose(recorded.iloc[0][metric], value, rtol=1e-12, atol=1e-12)
            expected_rows += len(test)
        for name, expected in combine(scores).items():
            np.testing.assert_allclose(scores[name], expected, rtol=1e-14, atol=1e-14)
    if len(predictions) != expected_rows or len(metrics) != len(expected_metrics):
        raise ValueError("Unexpected evaluation rows")
    if (
        set(metrics[["split", "seed", "model"]].itertuples(index=False, name=None))
        != expected_metrics
    ):
        raise ValueError("Incomplete metric coverage")
    return pd.concat([base_metrics, metrics], ignore_index=True), config, expected_rows


def make_report(run, destination):
    metrics, config, verified_rows = verified_results(run)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.mkdir(parents=True)
    metrics.to_csv(destination / "metrics.csv", index=False)
    summary = metrics.groupby(["split", "model"])[METRICS].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(destination / "summary.csv", index=False)
    paired = []
    for (kind, seed), block in metrics.groupby(["split", "seed"]):
        block = block.set_index("model")
        for candidate, baseline in PAIRS.items():
            paired.append(
                {
                    "split": kind,
                    "seed": seed,
                    "model": candidate,
                    "baseline": baseline,
                    **{
                        m + "_delta": float(block.loc[candidate, m] - block.loc[baseline, m])
                        for m in METRICS
                    },
                }
            )
    paired = pd.DataFrame(paired)
    paired.to_csv(destination / "paired_differences.csv", index=False)
    primary = paired[
        (paired.split == "scaffold") & (paired.model == "count_multiscale")
    ].average_precision_delta
    table, comparisons = [], []
    for kind in ("random", "scaffold"):
        for name, label in LABELS.items():
            row = summary.loc[(kind, name)]
            table.append(
                {
                    "Split": kind,
                    "Model": label,
                    **{
                        label: f"{row[m + '_mean']:.3f} ± {row[m + '_std']:.3f}"
                        for m, label in [
                            ("average_precision", "AP"),
                            ("roc_auc", "ROC-AUC"),
                            ("precision_at_20", "P@20"),
                            ("brier_score", "Brier"),
                        ]
                    },
                }
            )
        for candidate, baseline in PAIRS.items():
            delta = paired[
                (paired.split == kind) & (paired.model == candidate)
            ].average_precision_delta
            comparisons.append(
                {
                    "Split": kind,
                    "Candidate": candidate,
                    "Baseline": baseline,
                    "Mean ΔAP": f"{delta.mean():+.4f}",
                    "SD": f"{delta.std():.4f}",
                    "Positive differences": f"{int((delta > 0).sum())}/5",
                }
            )
    report = f"""# Count fingerprints: controlled exploratory results

The predeclared primary comparison, **count versus binary multiscale Tanimoto SVM on scaffold AP**, changed by **{primary.mean():+.4f}** on average; **{int((primary > 0).sum())}/5** seeds improved. This reused benchmark cannot establish independent generalization or superiority to the original drug-discovery paper.

Counts preserve repeated molecular environments. The exact dot-product Tanimoto kernel follows the established formulation discussed by [Tripp et al. (2023)](https://arxiv.org/html/2306.14809v2); the experimental choice and code are project-specific. We do not use their implementation or reproduce their approximation experiments. Read the [protocol](../../docs/COUNT_PROTOCOL.md).

The count and binary families use the same eight settings, exact grouped inner folds, training-only sigmoid calibration, and outer splits. No neural model was retrained; ensemble members still have equal weights. Every result is retained, including regressions. No new default is selected from these test results.

## Complete results

{markdown_table(pd.DataFrame(table))}

Mean ± sample SD across five overlapping seed cohorts, not confidence intervals. Each test has 229 compounds and 12 positives. AP means average precision; P@20 uses expected precision under tied scores. Higher AP, ROC-AUC, P@20 are better; lower Brier is better. Scores are not externally calibrated probabilities.

![Matched count and binary comparisons](model_comparison.png)

## Paired comparisons

{markdown_table(pd.DataFrame(comparisons))}

All per-seed values and all four metric differences are included in the CSVs. Different split types contain different compounds; comparisons are paired only within the same split and seed.

## Verification and limitations

Verified 20 count checkpoint reloads and 20 historical binary checkpoint reloads against validation predictions, all saved CV selections and group boundaries, {verified_rows:,} new test predictions, 40 new metric rows, and all ensemble arithmetic. The inherited 130 metrics and 29,770 predictions were separately reverified. Source, data, protocol, environment, baseline, and fitted-artifact hashes are recorded in the local run.

This is another exploratory experiment on already inspected tests. Hyperparameter selection and calibration reuse inner folds. Count fingerprints still have hash collisions and ignore chirality. Ensembles have greater compute budgets than individual models. External assay validation remains outstanding. No discovered drug, validated efficacy, clinical safety, or paper-beating result is claimed. Raw molecular data and fitted checkpoints remain excluded from the source distribution.
"""
    (destination / "REPORT.md").write_text(report)
    plot(summary, destination)
    write_json(
        destination / "verification.json",
        {
            "run_manifest_sha256": sha256(run / "run.json"),
            "new_prediction_rows": verified_rows,
            "new_metric_rows": 40,
            "total_metric_rows": len(metrics),
            "count_reload_checks": 20,
            "binary_reload_checks": 20,
            "primary_scaffold_ap_delta": float(primary.mean()),
            "report_hashes": {p.name: sha256(p) for p in sorted(destination.iterdir())},
        },
    )
    print(
        f"Primary scaffold AP change: {primary.mean():+.4f}; improvements {int((primary > 0).sum())}/5",
        flush=True,
    )


def plot(summary, destination):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    names = [name for pair in PAIRS.items() for name in (pair[1], pair[0])]
    short = [
        "Binary radius 2",
        "Count radius 2",
        "Binary multiscale",
        "Count multiscale",
        "Binary + custom + forest",
        "Count + custom + forest",
        "Binary + Chemprop + forest",
        "Count + Chemprop + forest",
    ]
    for axis, kind in zip(axes, ("random", "scaffold")):
        block = summary.loc[kind].loc[names]
        values = block.average_precision_mean.to_numpy()
        axis.barh(
            range(len(names)),
            values,
            xerr=block.average_precision_std,
            color=["#758A99", "#247D70"] * 4,
            alpha=0.9,
            capsize=3,
        )
        axis.set(
            yticks=range(len(names)),
            yticklabels=short,
            xlim=(0, 1),
            xlabel="Average precision (mean ± sample SD)",
            title=kind.capitalize(),
        )
        axis.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            axis.text(0.025, i, f"{value:.3f}", va="center", color="white", fontsize=9)
    # Shared axes need one inversion; inverting both would undo the first change.
    axes[0].invert_yaxis()
    fig.suptitle("Count versus binary fingerprints · exploratory reused benchmark")
    fig.savefig(destination / "model_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--data-root", type=Path, default=Path.cwd())
    training.add_argument("--base", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("run", type=Path)
    report = sub.add_parser("report")
    report.add_argument("run", type=Path)
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.data_root.resolve(), args.base.resolve(), args.output.resolve())
    elif args.command == "evaluate":
        evaluate_locked(args.run.resolve())
    else:
        make_report(args.run.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
