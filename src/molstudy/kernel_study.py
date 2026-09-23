"""Locked two-phase fingerprint-kernel study and reusable ensemble inference."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .data import sha256, standardize, write_json
from .features import fingerprints
from .graph import molecular_graph
from .improved import load_model, predict
from .improvement import check_environment, inputs, read_json, verify_lock as verify_previous
from .kernel import (
    ENSEMBLES,
    FAMILIES,
    GRID,
    ensemble_scores,
    fit,
    make_folds,
    select_setting,
    validate_folds,
)
from .metrics import evaluate
from .reference import SEEDS, make_chemprop, neural_predict, normalized_descriptors
from .splits import validate_split

NEW_MODELS = [*FAMILIES, *ENSEMBLES]
RUNTIME_SOURCES = [
    "kernel.py",
    "kernel_study.py",
    "features.py",
    "data.py",
    "graph.py",
    "improved.py",
    "improvement.py",
    "metrics.py",
    "reference.py",
    "splits.py",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def component_scores(folder, smiles, x, descriptors):
    """Predict only the explicitly supplied rows using the frozen components."""
    from chemprop.features import MolGraph

    cp = make_chemprop(folder / "train.csv", folder / ".chemprop")
    cp.load_state_dict(
        torch.load(folder / "chemprop.pt", map_location="cpu", weights_only=True)["state_dict"]
    )
    return {
        "forest": joblib.load(folder / "forest.joblib").predict_proba(x[2])[:, 1],
        "fusion_60": predict(
            load_model(folder / "fusion_60.pt"), [molecular_graph(s) for s in smiles], descriptors
        ),
        "chemprop": neural_predict(cp, "chemprop", [MolGraph(s) for s in smiles], descriptors),
    }


def train(root, output, reference, previous):
    versions = check_environment()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    frame, descriptors, splits, baseline, _ = inputs(root, reference)
    previous_config = read_json(previous / "run.json")
    if previous_config["status"] != "completed" or previous_config[
        "reference_manifest_sha256"
    ] != sha256(reference / "run.json"):
        raise ValueError("Completed prior experiment with the same reference required")
    # Reuse the prior checkpoints only after verifying their selection evidence.
    verify_previous(previous, previous_config)
    for file in ("metrics", "predictions"):
        if sha256(previous / f"{file}.csv") != previous_config[f"{file}_sha256"]:
            raise ValueError("Previous evaluation changed")
    output.mkdir(parents=True)
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    shutil.copy2(root / "docs/KERNEL_PROTOCOL.md", output / "PROTOCOL.md")
    shutil.copy2(
        root / "requirements-reference-lock.txt", output / "requirements-reference-lock.txt"
    )
    baseline_files = [
        p / name
        for p in (reference, previous)
        for name in ("run.json", "metrics.csv", "predictions.csv")
    ]
    config = {
        "status": "training",
        "started_at": now(),
        "data": baseline["data"],
        "environment": versions,
        "reference_run": str(reference.relative_to(root)),
        "previous_run": str(previous.relative_to(root)),
        "seeds": list(SEEDS),
        "grid": GRID,
        "families": FAMILIES,
        "ensembles": ENSEMBLES,
        "new_models": NEW_MODELS,
        "split_hashes": baseline["split_hashes"],
        "baseline_hashes": {str(p.relative_to(root)): sha256(p) for p in baseline_files},
        "protocol_sha256": sha256(output / "PROTOCOL.md"),
        "environment_lock_sha256": sha256(output / "requirements-reference-lock.txt"),
        "source_hashes": {p.name: sha256(p) for p in sorted((output / "source").glob("*.py"))},
        "interpretation": "Exploratory reused benchmark; no independent confirmation or paper superiority claim",
    }
    write_json(output / "run.json", config)
    x = {r: fingerprints(frame.smiles, radius=r)[0] for r in (2, 3)}
    y = frame.label.to_numpy()
    decisions = {}
    try:
        for key, partition in splits.items():
            kind, seed = key.rsplit("_", 1)
            folder = output / key
            folder.mkdir()
            for name in (
                "split.csv",
                "split_summary.json",
                "forest.joblib",
                "chemprop.pt",
                "train.csv",
            ):
                original = reference / key / name
                shutil.copy2(original, folder / name)
                config["baseline_hashes"][str(original.relative_to(root))] = sha256(original)
            original = previous / key / "fusion_60.pt"
            shutil.copy2(original, folder / "fusion_60.pt")
            config["baseline_hashes"][str(original.relative_to(root))] = sha256(original)
            # Training deliberately creates no test-row indices or test predictions.
            train_ix, val_ix = (
                np.flatnonzero(partition == name) for name in ("train", "validation")
            )
            training = frame.iloc[train_ix].copy()
            groups = training["scaffold" if kind == "scaffold" else "connectivity"].to_numpy()
            fold = make_folds(y[train_ix], groups, int(seed))
            pd.DataFrame(
                {
                    "compound_id": training.compound_id,
                    "label": y[train_ix],
                    "group": groups,
                    "fold": fold,
                }
            ).to_csv(folder / "inner_folds.csv", index=False)
            train_x = {r: values[train_ix] for r, values in x.items()}
            val_x = {r: values[val_ix] for r, values in x.items()}
            selections, validation = {}, {}
            for name, radii in FAMILIES.items():
                started = time.monotonic()
                model, selection, oof = fit(train_x, y[train_ix], groups, fold, radii)
                joblib.dump(model, folder / f"{name}.joblib")
                np.save(folder / f"{name}_oof.npy", oof, allow_pickle=False)
                validation[name] = model.predict(val_x)
                np.testing.assert_array_equal(
                    validation[name], joblib.load(folder / f"{name}.joblib").predict(val_x)
                )
                selection.update(
                    seconds=time.monotonic() - started, checkpoint_reload_verified=True
                )
                write_json(folder / f"{name}_selection.json", selection)
                selections[name] = selection
                print(
                    f"{key} {name}: training CV AP={selection['cv_ap']:.3f}, C={selection['setting']['C']}, class_weight={selection['setting']['class_weight']}, {selection['seconds']:.1f}s",
                    flush=True,
                )
            # Outer-validation scores are diagnostics, not inputs to kernel selection.
            validation.update(
                component_scores(
                    folder, frame.iloc[val_ix].smiles.to_list(), val_x, descriptors[val_ix]
                )
            )
            validation.update(ensemble_scores(validation))
            np.savez(folder / "validation.npz", label=y[val_ix], **validation)
            decisions[key] = {
                "selections": selections,
                "inner_fold_counts": [
                    {
                        "fold": f,
                        "n": int(sum(fold == f)),
                        "positives": int(y[train_ix][fold == f].sum()),
                    }
                    for f in range(3)
                ],
                "artifact_hashes": {
                    p.name: sha256(p) for p in sorted(folder.iterdir()) if p.is_file()
                },
            }
        # One lock covers every partition, trained artifact, and fixed ensemble rule.
        lock = {"locked_at": now(), "ensembles": ENSEMBLES, "decisions": decisions}
        write_json(output / "selection_lock.json", lock)
        config.update(
            status="selection_locked", selection_lock_sha256=sha256(output / "selection_lock.json")
        )
    except Exception as exc:
        config.update(status="training_failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "run.json", config)
    print(
        "All 20 kernel selections and fixed ensembles locked; no new test predictions calculated.",
        flush=True,
    )


def verify_lock(output, config, current_source=False):
    for file, field in [
        ("selection_lock.json", "selection_lock_sha256"),
        ("PROTOCOL.md", "protocol_sha256"),
        ("requirements-reference-lock.txt", "environment_lock_sha256"),
    ]:
        if sha256(output / file) != config[field]:
            raise ValueError(f"Locked artifact changed: {file}")
    for name, digest in config["source_hashes"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError(f"Source snapshot changed: {name}")
    if current_source:
        # Keep this strict byte check; historical commands can use the frozen-run launcher.
        for name in RUNTIME_SOURCES:
            if sha256(Path(__file__).parent / name) != config["source_hashes"][name]:
                raise ValueError(f"Executing source differs from locked source: {name}")
    lock = read_json(output / "selection_lock.json")
    expected = {f"{kind}_{seed}" for kind in ("random", "scaffold") for seed in SEEDS}
    if set(lock["decisions"]) != expected or lock["ensembles"] != {
        k: list(v) for k, v in ENSEMBLES.items()
    }:
        raise ValueError("Incomplete selection lock or changed ensemble definitions")
    if config["grid"] != GRID or config["families"] != {k: list(v) for k, v in FAMILIES.items()}:
        raise ValueError("Changed candidate specification")
    for key, decision in lock["decisions"].items():
        folder = output / key
        for name, digest in decision["artifact_hashes"].items():
            if sha256(folder / name) != digest:
                raise ValueError(f"Training artifact changed: {key}/{name}")
        kind = key.rsplit("_", 1)[0]
        split = pd.read_csv(folder / "split.csv", keep_default_na=False)
        if sha256(folder / "split.csv") != config["split_hashes"][key]:
            raise ValueError("Split differs from frozen reference")
        validate_split(split, split.partition.to_numpy(), kind)
        training = split[split.partition == "train"]
        folds = pd.read_csv(folder / "inner_folds.csv", keep_default_na=False)
        np.testing.assert_array_equal(folds.compound_id, training.compound_id)
        np.testing.assert_array_equal(folds.label, training.label)
        np.testing.assert_array_equal(
            folds.group, training["scaffold" if kind == "scaffold" else "connectivity"]
        )
        validate_folds(folds.label, folds.group, folds.fold)
        for name in FAMILIES:
            # Recompute the winning setting from saved out-of-fold evidence.
            calculated = select_setting(
                folds.label.to_numpy(),
                folds.fold.to_numpy(),
                np.load(folder / f"{name}_oof.npy", allow_pickle=False),
            )
            recorded = decision["selections"][name]
            if (
                any(recorded[k] != value for k, value in calculated.items())
                or not recorded["checkpoint_reload_verified"]
            ):
                raise ValueError("Selection does not match training-only CV evidence")
    return lock


def evaluate_locked(root, output):
    check_environment()
    config = read_json(output / "run.json")
    if config["status"] != "selection_locked":
        raise ValueError("Evaluation requires a locked, not yet evaluated run")
    verify_lock(output, config, current_source=True)
    for file, digest in config["baseline_hashes"].items():
        if sha256(root / file) != digest:
            raise ValueError(f"Frozen baseline changed: {file}")
    frame, descriptors, splits, _, _ = inputs(root, root / config["reference_run"])
    x = {r: fingerprints(frame.smiles, radius=r)[0] for r in (2, 3)}
    y = frame.label.to_numpy()
    previous = root / config["previous_run"]
    old_metrics = pd.read_csv(previous / "metrics.csv", float_precision="round_trip")
    old_pred = pd.read_csv(previous / "predictions.csv", float_precision="round_trip")
    rows, predictions = [], []
    config.update(status="evaluating", evaluation_started_at=now())
    write_json(output / "run.json", config)
    for key, partition in splits.items():
        kind, seed = key.rsplit("_", 1)
        seed = int(seed)
        test = np.flatnonzero(partition == "test")
        test_x = {r: values[test] for r, values in x.items()}
        folder = output / key
        scores = component_scores(
            folder, frame.iloc[test].smiles.to_list(), test_x, descriptors[test]
        )
        ids = frame.iloc[test].compound_id.to_numpy()
        for name, score in scores.items():
            # Confirm the old components reproduce their recorded scores before blending.
            old = (
                old_pred[
                    (old_pred.split == kind) & (old_pred.seed == seed) & (old_pred.model == name)
                ]
                .set_index("compound_id")
                .loc[ids]
            )
            np.testing.assert_array_equal(old.label, y[test])
            np.testing.assert_allclose(old.score, score, rtol=1e-10, atol=1e-10)
        scores.update(
            {name: joblib.load(folder / f"{name}.joblib").predict(test_x) for name in FAMILIES}
        )
        scores.update(ensemble_scores(scores))
        with np.load(folder / "validation.npz", allow_pickle=False) as validation:
            for name in NEW_MODELS:
                rows.append(
                    {
                        "split": kind,
                        "seed": seed,
                        "model": name,
                        **evaluate(y[test], scores[name]),
                        "validation_ap": evaluate(validation["label"], validation[name])[
                            "average_precision"
                        ],
                    }
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "compound_id": ids,
                            "label": y[test],
                            "split": kind,
                            "seed": seed,
                            "model": name,
                            "score": scores[name],
                        }
                    )
                )
        print(
            f"{key}: test AP kernel={rows[-4]['average_precision']:.3f}, custom ensemble={rows[-3]['average_precision']:.3f}, reference ensemble={rows[-2]['average_precision']:.3f}",
            flush=True,
        )
    pd.concat([old_metrics, pd.DataFrame(rows)], ignore_index=True).to_csv(
        output / "metrics.csv", index=False
    )
    pd.concat([old_pred, *predictions], ignore_index=True).to_csv(
        output / "predictions.csv", index=False
    )
    config.update(
        status="completed",
        completed_at=now(),
        metrics_sha256=sha256(output / "metrics.csv"),
        predictions_sha256=sha256(output / "predictions.csv"),
    )
    write_json(output / "run.json", config)


def predict_smiles(output, key, smiles):
    check_environment()
    config = read_json(output / "run.json")
    if config["status"] != "completed":
        raise ValueError("A completed experiment is required")
    verify_lock(output, config, current_source=True)
    record = standardize(smiles)
    strings = [record["smiles"]]
    x = {r: fingerprints(strings, radius=r)[0] for r in (2, 3)}
    descriptors, _ = normalized_descriptors(strings)
    folder = output / key
    scores = component_scores(folder, strings, x, descriptors)
    scores.update({name: joblib.load(folder / f"{name}.joblib").predict(x) for name in FAMILIES})
    scores.update(ensemble_scores(scores))
    return {
        "standardized_smiles": record["smiles"],
        "scores": {k: float(v[0]) for k, v in scores.items()},
        "ensemble_members": ENSEMBLES,
        "interpretation": "Experimental assay scores. Kernel calibration uses training data only; no validated efficacy, safety or external probability calibration.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train")
    training.add_argument("--output", type=Path, default=Path("runs/kernel-v1"))
    training.add_argument("--reference", type=Path, default=Path("runs/reference-v1-1"))
    training.add_argument("--previous", type=Path, default=Path("runs/improvement-v1"))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("run", type=Path)
    prediction = sub.add_parser("predict")
    prediction.add_argument("run", type=Path)
    prediction.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    prediction.add_argument("--seed", choices=list(SEEDS), type=int, default=101)
    prediction.add_argument("--smiles", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "train":
        train(root, root / args.output, root / args.reference, root / args.previous)
    elif args.command == "evaluate":
        evaluate_locked(root, root / args.run)
    else:
        print(
            json.dumps(
                predict_smiles(root / args.run, f"{args.split}_{args.seed}", args.smiles), indent=2
            )
        )


if __name__ == "__main__":
    main()
