"""Score a SMILES string using a trusted, locally trained reference-run checkpoint."""

import argparse
import importlib.metadata as metadata
import json
import os
import tempfile
from pathlib import Path

import torch

from .data import standardize
from .graph import DirectedMPNN, molecular_graph
from .reference import PINNED, make_chemprop, neural_predict, normalized_descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    if any(metadata.version(name) != version for name, version in PINNED.items()):
        raise ValueError("Use the pinned reference environment")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    name = checkpoint["model"]
    if name not in ("compact", "chemprop"):
        raise ValueError("Unsupported checkpoint family")
    record = standardize(args.smiles)
    with tempfile.TemporaryDirectory(prefix="molstudy-predict-") as directory:
        cache = Path(directory)
        os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
        model = (
            DirectedMPNN()
            if name == "compact"
            else make_chemprop(args.model.parent / "train.csv", cache)
        )
        model.load_state_dict(checkpoint["state_dict"])
        descriptors, _ = normalized_descriptors([record["smiles"]])
        if name == "compact":
            graph = molecular_graph(record["smiles"])
        else:
            from chemprop.features import MolGraph

            graph = MolGraph(record["smiles"])
        score = neural_predict(model, name, [graph], descriptors)[0]
    print(
        json.dumps(
            {
                "model": name,
                "standardized_smiles": record["smiles"],
                "activity_score": float(score),
                "interpretation": "Uncalibrated assay score; not validated efficacy, safety, or drug discovery.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
