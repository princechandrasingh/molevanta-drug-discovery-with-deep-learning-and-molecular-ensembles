"""Shared-split experiment using an attributed, installed Chemprop reference.

This module is project-specific orchestration. Chemprop's graph representation
and MoleculeModel are third-party MIT-licensed implementations, not our code.
Run in the separate pinned reference environment; see REFERENCE_PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata as metadata
import json
import math
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

from .data import sha256, write_json
from .features import fingerprints, nearest_train_similarity
from .graph import DirectedMPNN, batch_graphs, molecular_graph
from .metrics import evaluate
from .splits import describe_split, make_split, validate_split

SEEDS = (101, 202, 303, 404, 505)
MODELS = ("prior", "forest", "compact", "chemprop")
EPOCHS, BATCH_SIZE = 30, 50
PINNED = {
    "chemprop": "1.6.1",
    "descriptastorus": "2.8.0",
    "torch": "2.2.2",
    "numpy": "1.26.4",
    "scipy": "1.12.0",
    "rdkit": "2025.9.6",
    "scikit-learn": "1.7.2",
    "pandas": "2.3.3",
}


def learning_rate(step, steps_per_epoch, epochs=EPOCHS):
    """Step zero starts at 1e-4; warm up for 2 epochs, then decay to 1e-4."""
    warmup, total = 2 * steps_per_epoch, epochs * steps_per_epoch
    if epochs <= 2 or step < 0 or step > total:
        raise ValueError("Schedule needs >2 epochs and a step within its duration")
    if step <= warmup:
        return 1e-4 + (1e-3 - 1e-4) * step / warmup
    return 1e-3 * (1e-4 / 1e-3) ** ((step - warmup) / (total - warmup))


def sanitize_descriptors(values):
    """Use the reference library's own NaN-to-zero policy; reject infinities."""
    from chemprop.data import MoleculeDatapoint

    values = np.asarray(values, dtype=np.float32)
    if np.isinf(values).any():
        raise ValueError("Infinite descriptor values are not supported")
    return MoleculeDatapoint(smiles=["C"], features=values).features


def normalized_descriptors(smiles, audit_path=None):
    from descriptastorus.descriptors.rdNormalizedDescriptors import RDKit2DNormalized

    generator = RDKit2DNormalized()
    rows, missing = [], []
    names = [name for name, _ in generator.GetColumns()[1:]]
    for index, value in enumerate(smiles):
        result = generator.process(value)
        # Descriptastorus prepends a success flag to its 200 descriptor values.
        if result is None or len(result) != 201 or not result[0]:
            raise ValueError(f"Descriptor calculation failed at row {index}")
        row = np.asarray(result[1:], dtype=np.float32)
        for column in np.flatnonzero(np.isnan(row)):
            # Record every replacement; missing descriptors do not remove molecules.
            missing.append({"row_index": index, "descriptor": names[column], "replacement": 0})
        rows.append(sanitize_descriptors(row))
    array = np.asarray(rows, dtype=np.float32)
    if array.shape != (len(smiles), 200) or not np.isfinite(array).all():
        raise ValueError("Expected 200 finite descriptors for every molecule")
    if audit_path is not None:
        write_json(
            audit_path,
            {
                "policy": "Chemprop 1.6.1 MoleculeDatapoint NaN-to-zero; no fitted imputation; infinities rejected",
                "replaced_cells": missing,
            },
        )
        print(
            f"Descriptor checks passed; {len(missing)} NaN cells replaced by the reference policy",
            flush=True,
        )
    return array, names


def make_chemprop(data_path: Path, save_dir: Path):
    from chemprop.args import TrainArgs
    from chemprop.models import MoleculeModel

    args = TrainArgs().parse_args(
        [
            "--data_path",
            str(data_path),
            "--save_dir",
            str(save_dir),
            "--dataset_type",
            "classification",
            "--smiles_columns",
            "smiles",
            "--target_columns",
            "label",
            "--features_generator",
            "rdkit_2d_normalized",
            "--no_features_scaling",
            "--hidden_size",
            "300",
            "--depth",
            "3",
            "--ffn_num_layers",
            "2",
            "--ffn_hidden_size",
            "300",
            "--dropout",
            "0",
            "--aggregation",
            "mean",
            "--no_cuda",
            "--quiet",
        ]
    )
    # The model constructor needs metadata normally populated by Chemprop's data loader.
    args.task_names = ["label"]
    args.features_size = 200
    return MoleculeModel(args)


def forward(model, name, graphs, descriptors, indices):
    chosen = [graphs[i] for i in indices]
    if name == "compact":
        return model(batch_graphs(chosen))
    from chemprop.features import BatchMolGraph

    return model([BatchMolGraph(chosen)], [descriptors[i] for i in indices]).reshape(-1)


def neural_predict(model, name, graphs, descriptors):
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(graphs), BATCH_SIZE):
            output = forward(
                model, name, graphs, descriptors, range(start, min(start + BATCH_SIZE, len(graphs)))
            )
            # Chemprop applies sigmoid itself in evaluation mode, but not training mode.
            if name == "compact":
                output = torch.sigmoid(output)
            parts.append(output.numpy())
    return np.concatenate(parts)


def fit_neural(
    name,
    train_graphs,
    train_descriptors,
    train_y,
    val_graphs,
    val_descriptors,
    val_y,
    seed,
    data_path,
    save_dir,
    epochs=EPOCHS,
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = DirectedMPNN() if name == "compact" else make_chemprop(data_path, save_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    labels = torch.tensor(train_y, dtype=torch.float32)
    steps_per_epoch = math.ceil(len(train_graphs) / BATCH_SIZE)
    rng = np.random.default_rng(seed)
    best_ap, best_state, best_epoch, step = -np.inf, None, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_graphs))
        loss_sum = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            step += 1
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(step, steps_per_epoch, epochs)
            optimizer.zero_grad()
            logits = forward(model, name, train_graphs, train_descriptors, indices)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[indices])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite neural training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        val_score = neural_predict(model, name, val_graphs, val_descriptors)
        ap = float(average_precision_score(val_y, val_score))
        history.append(
            {
                "epoch": epoch,
                "train_bce": loss_sum / len(train_y),
                "validation_ap": ap,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if ap > best_ap:
            best_ap, best_epoch, best_state = ap, epoch, copy.deepcopy(model.state_dict())
        if epoch % 10 == 0:
            print(f"    {name} epoch {epoch}/{epochs}; validation AP={ap:.3f}", flush=True)
    model.load_state_dict(best_state)
    return model, {
        "validation_ap": best_ap,
        "selected_epoch": best_epoch,
        "history": history,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def run_reference(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    versions = {name: metadata.version(name) for name in PINNED}
    if versions != PINNED:
        raise ValueError(f"Reference environment differs from pins: {versions}")
    data_path = root / "data/processed/molecules.csv"
    manifest = json.loads((root / "data/processed/data_manifest.json").read_text())
    if sha256(data_path) != manifest["processed_sha256"]:
        raise ValueError("Curated input checksum mismatch")
    frame = pd.read_csv(data_path, keep_default_na=False)
    partitions = {
        (kind, seed): make_split(frame, kind, seed)
        for kind in ("random", "scaffold")
        for seed in SEEDS
    }
    for (kind, seed), part in partitions.items():
        validate_split(frame, part, kind)
        for subset in ("validation", "test"):
            counts = describe_split(frame, part)[subset]
            if counts["n"] < 150 or counts["positives"] < 5:
                raise ValueError(f"Insufficient {kind}/{seed}/{subset}: {counts}; do not reroll")
    output.mkdir(parents=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    (output / "source").mkdir()
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, output / "source" / path.name)
    protocol = root / "docs/REFERENCE_PROTOCOL.md"
    shutil.copy2(protocol, output / "REFERENCE_PROTOCOL.md")
    lock = root / "requirements-reference-lock.txt"
    shutil.copy2(lock, output / lock.name)
    config = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "shared-split-reference-v1.1",
        "protocol_sha256": sha256(protocol),
        "data": manifest,
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "environment": versions,
        "max_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "environment_lock_sha256": sha256(lock),
        "configuration_trials_per_family": 1,
        "selection": "validation AP only",
        "source_hashes": {p.name: sha256(p) for p in sorted((output / "source").glob("*.py"))},
        "split_hashes": {},
        "reference": "Installed Chemprop 1.6.1 plus descriptastorus 2.8.0; local common trainer; not exact paper replication",
    }
    for (kind, seed), part in partitions.items():
        split_dir = output / f"{kind}_{seed}"
        split_dir.mkdir()
        split_frame = frame[["compound_id", "smiles", "connectivity", "scaffold", "label"]].copy()
        split_frame["partition"] = part
        split_frame.to_csv(split_dir / "split.csv", index=False)
        frame.loc[part == "train", ["smiles", "label"]].to_csv(split_dir / "train.csv", index=False)
        write_json(split_dir / "split_summary.json", describe_split(frame, part))
        config["split_hashes"][f"{kind}_{seed}"] = sha256(split_dir / "split.csv")
    write_json(output / "run.json", config)
    try:
        print(
            "All ten partitions preflighted; calculating fixed molecular representations",
            flush=True,
        )
        x, bits = fingerprints(frame.smiles)
        descriptors, descriptor_names = normalized_descriptors(
            frame.smiles.tolist(), output / "descriptor_missingness.json"
        )
        np.save(output / "normalized_descriptors.npy", descriptors)
        write_json(
            output / "descriptor_manifest.json",
            {
                "names": descriptor_names,
                "sha256": sha256(output / "normalized_descriptors.npy"),
                "normalization": "Fixed descriptastorus CDFs; no dataset-fitted scaling",
            },
        )
        from chemprop.features import MolGraph

        graphs = {
            "compact": [molecular_graph(s) for s in frame.smiles],
            "chemprop": [MolGraph(s) for s in frame.smiles],
        }
        y = frame.label.to_numpy()
        rows, predictions = [], []
        for (kind, seed), part in partitions.items():
            split_dir = output / f"{kind}_{seed}"
            train, val, test = (
                np.flatnonzero(part == key) for key in ("train", "validation", "test")
            )
            similarity = nearest_train_similarity([bits[i] for i in test], [bits[i] for i in train])
            print(f"{kind} seed={seed}: {describe_split(frame, part)}", flush=True)
            for name in MODELS:
                started = time.monotonic()
                if name == "prior":
                    score = np.full(len(test), y[train].mean())
                    selection = {"validation_ap": float(y[val].mean()), "trainable_parameters": 0}
                elif name == "forest":
                    model = RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=2)
                    model.fit(x[train], y[train])
                    selection = {
                        "params": model.get_params(),
                        "validation_ap": float(
                            average_precision_score(y[val], model.predict_proba(x[val])[:, 1])
                        ),
                    }
                    score = model.predict_proba(x[test])[:, 1]
                    joblib.dump(model, split_dir / "forest.joblib")
                    np.testing.assert_allclose(
                        score,
                        joblib.load(split_dir / "forest.joblib").predict_proba(x[test])[:, 1],
                        rtol=1e-6,
                        atol=1e-7,
                    )
                else:
                    graph = graphs[name]
                    model, selection = fit_neural(
                        name,
                        [graph[i] for i in train],
                        descriptors[train],
                        y[train],
                        [graph[i] for i in val],
                        descriptors[val],
                        y[val],
                        seed,
                        split_dir / "train.csv",
                        split_dir / "chemprop_config",
                    )
                    score = neural_predict(model, name, [graph[i] for i in test], descriptors[test])
                    torch.save(
                        {"state_dict": model.state_dict(), "model": name}, split_dir / f"{name}.pt"
                    )
                    restored = (
                        DirectedMPNN()
                        if name == "compact"
                        else make_chemprop(split_dir / "train.csv", split_dir / "chemprop_config")
                    )
                    restored.load_state_dict(
                        torch.load(split_dir / f"{name}.pt", map_location="cpu", weights_only=True)[
                            "state_dict"
                        ]
                    )
                    np.testing.assert_allclose(
                        score,
                        neural_predict(restored, name, [graph[i] for i in test], descriptors[test]),
                        rtol=1e-6,
                        atol=1e-7,
                    )
                write_json(split_dir / f"{name}_selection.json", selection)
                result = {
                    "split": kind,
                    "seed": seed,
                    "model": name,
                    **evaluate(y[test], score),
                    "validation_ap": selection["validation_ap"],
                    "seconds": time.monotonic() - started,
                    "trainable_parameters": selection.get("trainable_parameters"),
                    "checkpoint_reload_verified": name != "prior",
                }
                rows.append(result)
                pred = (
                    frame.iloc[test][["compound_id", "label"]]
                    .copy()
                    .assign(
                        split=kind,
                        seed=seed,
                        model=name,
                        score=score,
                        nearest_train_tanimoto=similarity,
                    )
                )
                pred.to_csv(split_dir / f"{name}_test_predictions.csv", index=False)
                predictions.append(pred)
                pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                print(
                    f"  {name}: AP={result['average_precision']:.3f}, ROC-AUC={result['roc_auc']:.3f}, {result['seconds']:.1f}s",
                    flush=True,
                )
        pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
        config.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
    except Exception as exc:
        config.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output / "run.json", config)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/reference-v1-1"))
    args = parser.parse_args()
    root = args.root.resolve()
    run_reference(root, args.output if args.output.is_absolute() else root / args.output)


if __name__ == "__main__":
    main()
