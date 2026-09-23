"""Predict with a trusted local standalone augmented-model checkpoint."""

import argparse
import json
import os
import tempfile
from pathlib import Path

from .data import standardize
from .graph import molecular_graph
from .improved import load_model, predict
from .improvement import check_environment
from .reference import normalized_descriptors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    check_environment()
    record = standardize(args.smiles)
    with tempfile.TemporaryDirectory(prefix="molstudy-augmented-") as directory:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(directory) / "matplotlib"))
        descriptors, _ = normalized_descriptors([record["smiles"]])
        model = load_model(args.model)
        score = float(predict(model, [molecular_graph(record["smiles"])], descriptors)[0])
    print(
        json.dumps(
            {
                "model": args.model.stem,
                "standardized_smiles": record["smiles"],
                "activity_score": score,
                "interpretation": "Uncalibrated assay score from an exploratory model; not validated efficacy, safety, or drug discovery.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
