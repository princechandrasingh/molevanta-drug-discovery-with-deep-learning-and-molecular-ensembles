"""Run a historical kernel experiment with its checksum-verified source snapshot.

Formatting and comments change file hashes even when model behavior is unchanged.
This launcher preserves the original strict source checks by staging the recorded
Python files in a temporary package and using the current Python interpreter.
Use the pinned reference environment, just as for the original experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def verified_sources(run: Path) -> dict[str, bytes]:
    """Read all recorded Python files and reject modified or unsafe snapshot entries."""
    config = json.loads((run / "run.json").read_text())
    source_hashes = config.get("source_hashes", {})
    required = {"__init__.py", "kernel.py", "kernel_study.py"}
    if not isinstance(source_hashes, dict) or not required.issubset(source_hashes):
        raise ValueError("A kernel experiment with a complete source manifest is required")

    sources = {}
    for name, expected_hash in source_hashes.items():
        # Each manifest entry must name one Python file inside the snapshot directory.
        if Path(name).name != name or not name.endswith(".py"):
            raise ValueError(f"Unsafe source snapshot entry: {name}")
        source = run / "source" / name
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"Missing or linked source snapshot entry: {name}")
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"Source snapshot checksum changed: {name}")
        sources[name] = payload
    return sources


def run_frozen(run: Path, root: Path, command: str, arguments: list[str]) -> int:
    """Execute prediction/evaluation without modifying the historical source files."""
    if command not in {"predict", "evaluate"}:
        raise ValueError("Frozen runs support predict or evaluate")
    run, root = run.resolve(), root.resolve()
    sources = verified_sources(run)

    with tempfile.TemporaryDirectory(prefix="molevanta-frozen-") as directory:
        package = Path(directory) / "molstudy"
        package.mkdir()
        for name, payload in sources.items():
            # Stage the already-verified bytes, avoiding a second read of the snapshot.
            (package / name).write_bytes(payload)

        environment = os.environ.copy()
        environment["PYTHONPATH"] = directory
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        invocation = [
            sys.executable,
            "-m",
            "molstudy.kernel_study",
            "--root",
            str(root),
            command,
            str(run),
            *arguments,
        ]
        # A separate working directory prevents today's editable package from winning
        # import resolution. The child still performs all original run/model checks.
        result = subprocess.run(invocation, cwd=directory, env=environment, check=False)
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("run", type=Path, help="Historical kernel experiment directory")
    parser.add_argument("command", choices=["predict", "evaluate"])
    parser.add_argument(
        "arguments", nargs=argparse.REMAINDER, help="Arguments passed to the saved command"
    )
    args = parser.parse_args()
    raise SystemExit(run_frozen(args.run, args.root, args.command, args.arguments))


if __name__ == "__main__":
    main()
