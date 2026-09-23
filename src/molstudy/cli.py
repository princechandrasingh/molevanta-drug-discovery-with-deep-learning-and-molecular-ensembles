from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Molevanta: Drug Discovery with Deep Learning and Molecular Ensembles"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project directory (default: current directory)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download and checksum-verify publisher Table S1")
    commands.add_parser("prepare", help="Curate molecules and record exclusions")
    run = commands.add_parser(
        "run", help="Train and evaluate all models on predeclared shared splits"
    )
    run.add_argument("--output", type=Path, default=Path("runs/pilot"))
    run.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    run.add_argument(
        "--models",
        nargs="+",
        choices=["prior", "logistic", "forest", "dmpnn"],
        default=["prior", "logistic", "forest", "dmpnn"],
    )
    run.add_argument("--epochs", type=int, default=30)
    report = commands.add_parser("report", help="Rebuild report from saved results")
    report.add_argument("run_dir", type=Path)
    predict = commands.add_parser("predict", help="Score a molecule using a saved model")
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--smiles", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "download":
        from .data import download_data

        print(download_data(root))
    elif args.command == "prepare":
        from .data import prepare

        print(json.dumps(prepare(root), indent=2))
    elif args.command == "run":
        from .experiment import run

        output = args.output if args.output.is_absolute() else root / args.output
        run(root, output, args.seeds, args.models, args.epochs)
    elif args.command == "report":
        from .report import make_report

        make_report(args.run_dir if args.run_dir.is_absolute() else root / args.run_dir)
    else:
        from .data import standardize

        record = standardize(args.smiles)
        path = args.model if args.model.is_absolute() else root / args.model
        if path.suffix == ".pt":
            import torch
            from .graph import DirectedMPNN, molecular_graph, predict_graph

            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = DirectedMPNN(hidden=checkpoint["hidden"])
            model.load_state_dict(checkpoint["state_dict"])
            score = predict_graph(model, [molecular_graph(record["smiles"])])[0]
        else:
            import joblib
            from .features import fingerprints

            # Only load local artifacts produced by this project; pickle is not an untrusted interchange format.
            model = joblib.load(path)
            score = model.predict_proba(fingerprints([record["smiles"]])[0])[:, 1][0]
        print(
            json.dumps(
                {
                    "standardized_smiles": record["smiles"],
                    "activity_score": float(score),
                    "interpretation": "Uncalibrated model score for this assay; not efficacy, safety, or a validated discovery.",
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
