"""Keep saved experiment execution isolated and reject altered source snapshots."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def launcher():
    path = Path(__file__).resolve().parents[1] / "scripts/run_frozen_kernel.py"
    spec = importlib.util.spec_from_file_location("frozen_kernel_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen_run(tmp_path):
    run = tmp_path / "historical-run"
    source = run / "source"
    source.mkdir(parents=True)
    payloads = {
        "__init__.py": b"",
        "kernel.py": b"MARKER = 'historical implementation'\n",
        "kernel_study.py": b"""import hashlib
import json
from pathlib import Path
import sys
from .kernel import MARKER
print(json.dumps({
    "marker": MARKER,
    "source_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "arguments": sys.argv[1:],
}))
""",
    }
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    (run / "run.json").write_text(json.dumps({"source_hashes": hashes}))
    return run


def test_frozen_launcher_executes_verified_package_and_forwards_arguments(
    launcher, frozen_run, capfd
):
    root = frozen_run.parent
    arguments = ["--split", "scaffold", "--seed", "202", "--smiles", "CCO"]
    assert launcher.run_frozen(frozen_run, root, "predict", arguments) == 0
    result = json.loads(capfd.readouterr().out)
    manifest = json.loads((frozen_run / "run.json").read_text())
    assert result["marker"] == "historical implementation"
    assert result["source_hash"] == manifest["source_hashes"]["kernel_study.py"]
    assert result["arguments"] == ["--root", str(root), "predict", str(frozen_run), *arguments]


def test_frozen_launcher_rejects_tampering_before_execution(launcher, frozen_run, monkeypatch):
    (frozen_run / "source/kernel.py").write_text("raise RuntimeError('Must not execute')\n")
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *a, **kw: pytest.fail("Executed altered source")
    )
    with pytest.raises(ValueError, match="checksum changed"):
        launcher.run_frozen(frozen_run, frozen_run.parent, "predict", [])


def test_frozen_launcher_rejects_manifest_path_traversal(launcher, frozen_run):
    manifest = json.loads((frozen_run / "run.json").read_text())
    manifest["source_hashes"]["../outside.py"] = "0" * 64
    (frozen_run / "run.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsafe source snapshot"):
        launcher.verified_sources(frozen_run)
