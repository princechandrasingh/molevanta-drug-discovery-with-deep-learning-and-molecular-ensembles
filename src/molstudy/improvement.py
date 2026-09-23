"""Two-phase, validation-selected improvement study and saved-pipeline inference."""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import os
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
from .improved import SPECS, fit, load_model, predict, select_pipeline
from .metrics import evaluate
from .reference import PINNED, SEEDS, normalized_descriptors
from .splits import validate_split


def read_json(path):
    return json.loads(path.read_text())


def check_environment():
    actual = {name: metadata.version(name) for name in PINNED}
    if actual != PINNED:
        raise ValueError("Use the pinned reference environment")
    torch.set_num_threads(2)
    return actual


def inputs(root, reference):
    manifest = read_json(reference / "run.json")
    if manifest["status"] != "completed":
        raise ValueError("Reference run must be complete")
    data = root / "data/processed/molecules.csv"
    if sha256(data) != manifest["data"]["processed_sha256"]:
        raise ValueError("Dataset checksum changed")
    frame = pd.read_csv(data, keep_default_na=False)
    descriptor_manifest = read_json(reference / "descriptor_manifest.json")
    if sha256(reference / "normalized_descriptors.npy") != descriptor_manifest["sha256"]:
        raise ValueError("Descriptor cache checksum changed")
    descriptors = np.load(reference / "normalized_descriptors.npy", allow_pickle=False)
    if descriptors.shape != (len(frame), 200) or not np.isfinite(descriptors).all():
        raise ValueError("Invalid descriptor cache")
    splits = {}
    for kind in ("random", "scaffold"):
        for seed in SEEDS:
            key = f"{kind}_{seed}"
            path = reference / key / "split.csv"
            if sha256(path) != manifest["split_hashes"][key]:
                raise ValueError("Split checksum changed")
            split = pd.read_csv(path, keep_default_na=False)
            np.testing.assert_array_equal(frame.compound_id, split.compound_id)
            np.testing.assert_array_equal(frame.label, split.label)
            validate_split(split, split.partition.to_numpy(), kind)
            splits[key] = split.partition.to_numpy()
    return frame, descriptors, splits, manifest, descriptor_manifest


def train(root: Path, output: Path, reference: Path):
    versions = check_environment()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    # Check the frozen dataset/splits before creating an experiment directory.
    frame, descriptors, splits, baseline, descriptor_manifest = inputs(root, reference)
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    (output / "source").mkdir()
    for file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(file, output / "source" / file.name)
    for src, name in [
        (root / "docs/IMPROVEMENT_PROTOCOL.md", "PROTOCOL.md"),
        (root / "requirements-reference-lock.txt", "requirements-reference-lock.txt"),
    ]:
        shutil.copy2(src, output / name)
    config = {
        "status": "training",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "reference_run": str(reference.relative_to(root)),
        "reference_manifest_sha256": sha256(reference / "run.json"),
        "reference_metrics_sha256": sha256(reference / "metrics.csv"),
        "reference_predictions_sha256": sha256(reference / "predictions.csv"),
        "data": baseline["data"],
        "descriptor_cache_sha256": descriptor_manifest["sha256"],
        "environment": versions,
        "specs": SPECS,
        "seeds": list(SEEDS),
        "split_hashes": baseline["split_hashes"],
        "protocol_sha256": sha256(output / "PROTOCOL.md"),
        "environment_lock_sha256": sha256(output / "requirements-reference-lock.txt"),
        "source_hashes": {p.name: sha256(p) for p in sorted((output / "source").glob("*.py"))},
        "interpretation": "Exploratory development on previously inspected benchmark; not independent confirmation",
    }
    write_json(output / "run.json", config)
    x, _ = fingerprints(frame.smiles)
    graphs = [molecular_graph(s) for s in frame.smiles]
    y = frame.label.to_numpy()
    decisions = {}
    try:
        for key, partition in splits.items():
            seed = int(key.rsplit("_", 1)[1])
            folder = output / key
            folder.mkdir()
            shutil.copy2(reference / key / "split.csv", folder / "split.csv")
            shutil.copy2(reference / key / "split_summary.json", folder / "split_summary.json")
            shutil.copy2(reference / key / "forest.joblib", folder / "forest.joblib")
            train_ix, val_ix = (
                np.flatnonzero(partition == part) for part in ("train", "validation")
            )
            forest = joblib.load(folder / "forest.joblib")
            forest_val = forest.predict_proba(x[val_ix])[:, 1]
            validation = {}
            for name, spec in SPECS.items():
                started = time.monotonic()
                model, selection = fit(
                    spec,
                    [graphs[i] for i in train_ix],
                    descriptors[train_ix],
                    y[train_ix],
                    [graphs[i] for i in val_ix],
                    descriptors[val_ix],
                    y[val_ix],
                    seed,
                )
                validation[name] = predict(model, [graphs[i] for i in val_ix], descriptors[val_ix])
                torch.save({"spec": spec, "state_dict": model.state_dict()}, folder / f"{name}.pt")
                np.testing.assert_array_equal(
                    validation[name],
                    predict(
                        load_model(folder / f"{name}.pt"),
                        [graphs[i] for i in val_ix],
                        descriptors[val_ix],
                    ),
                )
                selection.update(
                    seconds=time.monotonic() - started, checkpoint_reload_verified=True
                )
                write_json(folder / f"{name}_selection.json", selection)
                print(
                    f"{key} {name}: validation AP={selection['validation_ap']:.3f}, epoch={selection['selected_epoch']}, {selection['seconds']:.1f}s",
                    flush=True,
                )
            # Save selection evidence so the chosen pipeline can be independently checked.
            np.savez(folder / "validation.npz", label=y[val_ix], forest=forest_val, **validation)
            decision = select_pipeline(y[val_ix], validation, forest_val)
            decision["artifact_hashes"] = {
                p.name: sha256(p) for p in folder.iterdir() if p.is_file()
            }
            decisions[key] = decision
            print(
                f"  Selected on validation: {decision['candidate']} with neural weight {decision['neural_weight']}",
                flush=True,
            )
        # Lock all partitions together before the separate evaluation command can run.
        lock = {"locked_at": datetime.now(timezone.utc).isoformat(), "decisions": decisions}
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
        "All ten selection decisions locked. No new test predictions have been calculated.",
        flush=True,
    )


def verify_lock(output, config):
    if sha256(output / "selection_lock.json") != config["selection_lock_sha256"]:
        raise ValueError("Selection lock changed")
    if sha256(output / "PROTOCOL.md") != config["protocol_sha256"]:
        raise ValueError("Protocol snapshot changed")
    if sha256(output / "requirements-reference-lock.txt") != config["environment_lock_sha256"]:
        raise ValueError("Environment snapshot changed")
    for name, digest in config["source_hashes"].items():
        if sha256(output / "source" / name) != digest:
            raise ValueError("Source snapshot changed")
    lock = read_json(output / "selection_lock.json")
    expected = {f"{kind}_{seed}" for kind in ("random", "scaffold") for seed in SEEDS}
    if set(lock["decisions"]) != expected:
        raise ValueError("All ten decisions must be locked before evaluation")
    for key, decision in lock["decisions"].items():
        for name, digest in decision["artifact_hashes"].items():
            if sha256(output / key / name) != digest:
                raise ValueError(f"Training artifact changed: {key}/{name}")
        with np.load(output / key / "validation.npz", allow_pickle=False) as values:
            recomputed = select_pipeline(
                values["label"], {name: values[name] for name in SPECS}, values["forest"]
            )
        if recomputed != {k: v for k, v in decision.items() if k != "artifact_hashes"}:
            raise ValueError("Saved selection disagrees with validation predictions")
    return lock


def evaluate_locked(root: Path, output: Path):
    check_environment()
    config = read_json(output / "run.json")
    if config["status"] != "selection_locked":
        raise ValueError("Evaluation requires a locked, not yet evaluated run")
    lock = verify_lock(output, config)
    reference = root / config["reference_run"]
    for name in ("manifest", "metrics", "predictions"):
        file = reference / ("run.json" if name == "manifest" else f"{name}.csv")
        if sha256(file) != config[f"reference_{name}_sha256"]:
            raise ValueError("Frozen reference changed")
    frame, descriptors, splits, _, _ = inputs(root, reference)
    graphs = [molecular_graph(s) for s in frame.smiles]
    x, _ = fingerprints(frame.smiles)
    y = frame.label.to_numpy()
    old_scores = pd.read_csv(reference / "metrics.csv")
    old_pred = pd.read_csv(reference / "predictions.csv")
    rows, predictions = [], []
    config.update(status="evaluating", evaluation_started_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "run.json", config)
    for key, partition in splits.items():
        kind, seed_text = key.rsplit("_", 1)
        seed = int(seed_text)
        test = np.flatnonzero(partition == "test")
        decision = lock["decisions"][key]
        forest_score = joblib.load(output / key / "forest.joblib").predict_proba(x[test])[:, 1]
        expected_ids = frame.iloc[test].compound_id.to_numpy()
        old = old_pred[
            (old_pred.split == kind) & (old_pred.seed == seed) & (old_pred.model == "forest")
        ].set_index("compound_id")
        np.testing.assert_array_equal(old.loc[expected_ids].label, y[test])
        np.testing.assert_allclose(
            old.loc[expected_ids].score, forest_score, rtol=1e-10, atol=1e-10
        )
        candidate_scores = {
            name: predict(
                load_model(output / key / f"{name}.pt"),
                [graphs[i] for i in test],
                descriptors[test],
            )
            for name in SPECS
        }
        weight = decision["neural_weight"]
        candidate_scores["selected_blend"] = (
            weight * candidate_scores[decision["candidate"]].astype(np.float64)
            + (1 - weight) * forest_score
        )
        for name, score in candidate_scores.items():
            selection = (
                read_json(output / key / f"{name}_selection.json") if name in SPECS else decision
            )
            rows.append(
                {
                    "split": kind,
                    "seed": seed,
                    "model": name,
                    **evaluate(y[test], score),
                    "validation_ap": selection["validation_ap"],
                    "trainable_parameters": selection.get("trainable_parameters"),
                    "seconds": selection.get("seconds"),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "compound_id": expected_ids,
                        "label": y[test],
                        "split": kind,
                        "seed": seed,
                        "model": name,
                        "score": score,
                    }
                )
            )
        print(
            f"{key}: test AP fusion_30={rows[-3]['average_precision']:.3f}; selected={rows[-1]['average_precision']:.3f}",
            flush=True,
        )
    scores = pd.concat([old_scores, pd.DataFrame(rows)], ignore_index=True)
    pred = pd.concat([old_pred, *predictions], ignore_index=True)
    scores.to_csv(output / "metrics.csv", index=False)
    pred.to_csv(output / "predictions.csv", index=False)
    config.update(
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
        metrics_sha256=sha256(output / "metrics.csv"),
        predictions_sha256=sha256(output / "predictions.csv"),
    )
    write_json(output / "run.json", config)


def predict_selected(output: Path, key: str, smiles: str):
    check_environment()
    config = read_json(output / "run.json")
    if config["status"] != "completed":
        raise ValueError("Prediction demo requires a completed, verified experiment")
    decision = verify_lock(output, config)["decisions"][key]
    record = standardize(smiles)
    forest = joblib.load(output / key / "forest.joblib")
    forest_score = float(forest.predict_proba(fingerprints([record["smiles"]])[0])[0, 1])
    neural_score = None
    weight = decision["neural_weight"]
    if weight:
        descriptors, _ = normalized_descriptors([record["smiles"]])
        model = load_model(output / key / f"{decision['candidate']}.pt")
        neural_score = float(predict(model, [molecular_graph(record["smiles"])], descriptors)[0])
    score = weight * (neural_score or 0.0) + (1 - weight) * forest_score
    return {
        "standardized_smiles": record["smiles"],
        "activity_score": score,
        "selected_neural_candidate": decision["candidate"],
        "neural_weight": weight,
        "forest_score": forest_score,
        "neural_score": neural_score,
        "interpretation": "Uncalibrated assay score from an exploratory model; not validated efficacy, safety, or drug discovery.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--output", type=Path, default=Path("runs/improvement-v1"))
    train_parser.add_argument("--reference", type=Path, default=Path("runs/reference-v1-1"))
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("run", type=Path)
    prediction = sub.add_parser("predict")
    prediction.add_argument("run", type=Path)
    prediction.add_argument("--split", choices=["random", "scaffold"], default="scaffold")
    prediction.add_argument("--seed", type=int, choices=list(SEEDS), default=101)
    prediction.add_argument("--smiles", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "train":
        train(root, root / args.output, root / args.reference)
    elif args.command == "evaluate":
        evaluate_locked(root, root / args.run)
    else:
        print(
            json.dumps(
                predict_selected(root / args.run, f"{args.split}_{args.seed}", args.smiles),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
