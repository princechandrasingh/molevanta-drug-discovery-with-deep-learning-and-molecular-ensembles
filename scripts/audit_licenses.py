"""Inventory installed distributions and preserve their shipped legal notices.

Run once with each experiment interpreter. Metadata collection is not legal
clearance or a source-code similarity audit. No dependency code is vendored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


def audit(destination: Path, environment: str):
    records = []
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        name, version = dist.metadata["Name"], dist.version
        if name == "molecular-generalization":
            continue
        identifier = re.sub(r"[^a-zA-Z0-9_.-]", "_", f"{name}-{version}")
        notices = []
        for item in dist.files or []:
            if not re.fullmatch(
                r"(?:licen[cs]e|copying|notice|copyright)(?:[._-].*)?", item.name, re.I
            ):
                continue
            if item.suffix.lower() in {".py", ".pyc", ".so", ".dll"}:
                continue
            source = Path(dist.locate_file(item))
            if not source.is_file():
                continue
            # Preserve path context to avoid collisions between bundled licenses.
            relative = (
                Path("licenses")
                / identifier
                / Path(*[p for p in item.parts if p not in (".", "..", "/")])
            )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            notices.append(
                {
                    "path": relative.as_posix(),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        raw_license = dist.metadata.get("License", "")
        expression = dist.metadata.get("License-Expression", "")
        classifiers = [
            c for c in dist.metadata.get_all("Classifier", []) if c.startswith("License ::")
        ]
        records.append(
            {
                "name": name,
                "version": version,
                "license_expression": expression,
                "license_metadata": raw_license,
                "license_classifiers": classifiers,
                "project_urls": dist.metadata.get_all("Project-URL", []),
                "homepage": dist.metadata.get("Home-page", ""),
                "notices": notices,
                "status": "shipped_notices_collected" if notices else "metadata_only_review_needed",
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "environment": environment,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Installed Python distributions, including transitive/build dependencies; package-shipped notices only; not comprehensive binary/source legal clearance",
        "packages": records,
    }
    output = destination / f"{environment}-inventory.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "environment": environment,
                "packages": len(records),
                "notice_files": sum(len(r["notices"]) for r in records),
                "metadata_only": [r["name"] for r in records if not r["notices"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/license-audit"))
    parser.add_argument("--environment", required=True)
    args = parser.parse_args()
    audit(args.output, args.environment)
