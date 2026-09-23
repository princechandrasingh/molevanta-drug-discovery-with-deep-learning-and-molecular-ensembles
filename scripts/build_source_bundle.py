"""Build a reviewable source-only archive using an explicit allowlist.

No data, models, installed dependency code or virtual environments are included.
This prepares a local artifact; it does not publish or select an outbound license.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

PROJECT_TITLE = "Molevanta: Drug Discovery with Deep Learning and Molecular Ensembles"
ARCHIVE_ROOT = "molevanta"
TOP_LEVEL = {
    "README.md",
    "pyproject.toml",
    ".gitignore",
    "requirements-lock.txt",
    "requirements-reference-lock.txt",
    "THIRD_PARTY_NOTICES.md",
}
REPORT_FILES = {
    "REPORT.md",
    "metrics.csv",
    "summary.csv",
    "paired_differences.csv",
    "similarity_error_analysis.csv",
    "split_counts.json",
    "run.json",
    "report_manifest.json",
    "verification.json",
    "model_comparison.png",
    "reference_comparison.png",
}
MOLECULAR_COLUMNS = {
    "smiles",
    "raw_smiles",
    "compound_id",
    "label",
    "score",
    "connectivity",
    "scaffold",
    "source_rows",
}


def selected_files(root):
    # Explicit inclusion keeps new data/model files from accidentally entering a release.
    selected = {root / name for name in TOP_LEVEL}
    for folder in ("src", "tests", "scripts"):
        selected.update((root / folder).rglob("*.py"))
    selected.update((root / "docs").rglob("*.md"))
    selected.update((root / "docs/license-audit").glob("*.json"))
    audit = root / "docs/license-audit"
    for inventory in audit.glob("*-inventory.json"):
        for package in json.loads(inventory.read_text())["packages"]:
            selected.update(audit / item["path"] for item in package["notices"])
    for package in json.loads((audit / "supplemental-notices.json").read_text()):
        selected.update(audit / item["path"] for item in package["notices"])
    for report in (root / "reports").iterdir():
        if report.is_dir():
            selected.update(path for path in report.iterdir() if path.name in REPORT_FILES)
    return sorted(selected)


def build(root, output):
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    records, content = [], []
    for path in selected_files(root):
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Invalid source bundle entry: {path}")
        relative = path.relative_to(root).as_posix()
        payload = path.read_bytes()
        if path.suffix == ".csv":
            header = next(csv.reader(io.StringIO(payload.decode())))
            if MOLECULAR_COLUMNS.intersection(column.lower() for column in header):
                raise ValueError(f"Molecule-level table forbidden in source bundle: {relative}")
        records.append(
            {"path": relative, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        )
        content.append((relative, payload))
    manifest = {
        "project_title": PROJECT_TITLE,
        "scope": "Source-only archive; no outbound project-code license granted",
        "excluded": [
            "raw/curated data",
            "molecular predictions",
            "descriptors",
            "model checkpoints",
            "virtual environments",
            "third-party library implementations",
            "paper PDFs/figures",
        ],
        "files": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, payload in content:
            entry = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", date_time=(2026, 9, 23, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, payload)
        archive.writestr(
            f"{ARCHIVE_ROOT}/BUNDLE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n"
        )
    with zipfile.ZipFile(output) as archive:
        # Verify the bytes in the archive itself, not only the original filesystem files.
        assert archive.testzip() is None
        for record in records:
            payload = archive.read(f"{ARCHIVE_ROOT}/" + record["path"])
            assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".sha256").write_text(f"{digest}  {output.name}\n")
    print(
        json.dumps(
            {
                "archive": str(output),
                "files": len(records),
                "bytes": output.stat().st_size,
                "sha256": digest,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/molevanta-source.zip"))
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
