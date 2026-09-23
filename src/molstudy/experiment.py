from __future__ import annotations

import importlib.metadata
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .data import sha256, write_json
from .features import fingerprints, nearest_train_similarity
from .metrics import evaluate
from .splits import make_split, describe_split


def fit_baseline(name, x, y, validation_x, validation_y, seed):
    if name == "prior":
        candidates = [DummyClassifier(strategy="prior")]
    elif name == "logistic":
        candidates = [
            LogisticRegression(
                C=c, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed
            )
            for c in (0.1, 1.0, 10.0)
        ]
    elif name == "forest":
        candidates = [
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=leaf,
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=seed,
            )
            for leaf in (1, 3)
        ]
    else:
        raise ValueError(name)
    best_model, best_score, search = None, -np.inf, []
    for model in candidates:
        model.fit(x, y)
        score = float(
            average_precision_score(validation_y, model.predict_proba(validation_x)[:, 1])
        )
        search.append({"params": model.get_params(), "validation_ap": score})
        # Fit uses training data; selection uses validation AP, never test scores.
        if score > best_score:
            best_model, best_score = model, score
    return best_model, {
        "validation_ap": best_score,
        "params": best_model.get_params(),
        "search": search,
    }


def run(
    root: Path,
    output: Path,
    seeds=(11, 22, 33),
    models=("prior", "logistic", "forest", "dmpnn"),
    epochs=30,
):
    if output.exists():
        raise FileExistsError(
            f"Run directory already exists: {output}. Choose a new name to preserve earlier results."
        )
    data_path = root / "data/processed/molecules.csv"
    manifest = json.loads((root / "data/processed/data_manifest.json").read_text())
    if sha256(data_path) != manifest["processed_sha256"]:
        raise ValueError("Curated dataset changed since preparation")
    frame = pd.read_csv(data_path, keep_default_na=False)
    seeds = list(seeds)
    if len(set(seeds)) != len(seeds) or epochs < 1:
        raise ValueError("Use unique seeds and at least one epoch")
    # Preflight every split before doing any training or writing scores.
    partitions = {
        (kind, seed): make_split(frame, kind, seed)
        for kind in ("random", "scaffold")
        for seed in seeds
    }
    for (kind, seed), partition in partitions.items():
        for part in ("validation", "test"):
            counts = describe_split(frame, partition)[part]
            if counts["n"] < 150 or counts["positives"] < 5:
                raise ValueError(
                    f"Uninformative {kind}/{seed}/{part}: {counts}; pilot requires >=150 rows and >=5 positives. Revise and document the protocol; do not reroll based on model scores."
                )
    x, bits = fingerprints(frame.smiles)
    y = frame.label.to_numpy()
    graphs = None
    if "dmpnn" in models:
        from .graph import molecular_graph

        graphs = [molecular_graph(s) for s in frame.smiles]
    output.mkdir(parents=True)
    source_root = Path(__file__).parent
    code_hashes = {p.name: sha256(p) for p in sorted(source_root.glob("*.py"))}
    config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "models": list(models),
        "max_epochs": epochs,
        "data": manifest,
        "source_hashes": code_hashes,
        "environment": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "pandas", "rdkit", "scikit-learn", "torch")
        },
        "selection_metric": "validation average precision; no test-based tuning",
        "status": "running",
        "fingerprint": {"radius": 2, "bits": 2048, "chirality": False},
        "protocol_version": "0.2",
        "split_protocol": "One greedy group allocation balancing 80/10/10 row and positive counts; seed perturbs group priority by Uniform(0.8,1.2). No score-based split selection. Preflight: >=150 rows and >=5 positives per validation/test partition.",
    }
    write_json(output / "run.json", config)
    rows, all_predictions = [], []
    for (kind, seed), partition in partitions.items():
        split_dir = output / f"{kind}_{seed}"
        split_dir.mkdir()
        split_frame = frame[["compound_id", "smiles", "connectivity", "scaffold", "label"]].copy()
        split_frame["partition"] = partition
        split_frame.to_csv(split_dir / "split.csv", index=False)
        write_json(split_dir / "split_summary.json", describe_split(frame, partition))
        train, val, test = (
            np.flatnonzero(partition == part) for part in ("train", "validation", "test")
        )
        similarity = nearest_train_similarity([bits[i] for i in test], [bits[i] for i in train])
        print(f"{kind} seed={seed}: {describe_split(frame, partition)}", flush=True)
        for name in models:
            started = time.monotonic()
            if name == "dmpnn":
                import torch
                from .graph import fit_graph, predict_graph

                model, selection = fit_graph(
                    [graphs[i] for i in train],
                    y[train],
                    [graphs[i] for i in val],
                    y[val],
                    seed,
                    epochs,
                )
                score = predict_graph(model, [graphs[i] for i in test])
                torch.save(
                    {"state_dict": model.state_dict(), "hidden": selection["hidden"]},
                    split_dir / "dmpnn.pt",
                )
            else:
                model, selection = fit_baseline(name, x[train], y[train], x[val], y[val], seed)
                score = model.predict_proba(x[test])[:, 1]
                joblib.dump(model, split_dir / f"{name}.joblib")
            write_json(split_dir / f"{name}_selection.json", selection)
            result = {
                "split": kind,
                "seed": seed,
                "model": name,
                **evaluate(y[test], score),
                "validation_ap": selection["validation_ap"],
                "seconds": time.monotonic() - started,
            }
            rows.append(result)
            pred = frame.iloc[test][["compound_id", "smiles", "name", "label", "scaffold"]].copy()
            pred = pred.assign(
                split=kind, seed=seed, model=name, score=score, nearest_train_tanimoto=similarity
            )
            pred.to_csv(split_dir / f"{name}_test_predictions.csv", index=False)
            all_predictions.append(pred)
            pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
            print(
                f"  {name}: AP={result['average_precision']:.3f}, ROC-AUC={result['roc_auc']:.3f}, P@20={result['precision_at_20']:.3f} ({result['seconds']:.1f}s)",
                flush=True,
            )
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    config["status"] = "completed"
    config["completed_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "run.json", config)
    from .report import make_report

    make_report(output)
    return pd.DataFrame(rows)
