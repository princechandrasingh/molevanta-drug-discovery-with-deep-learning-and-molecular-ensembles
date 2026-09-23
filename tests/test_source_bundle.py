"""Verify the release boundary on actual project files, without building an archive."""

import importlib.util
from pathlib import Path


def test_source_allowlist_excludes_data_models_environments():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("bundle", root / "scripts/build_source_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = [path.relative_to(root) for path in module.selected_files(root)]
    assert paths
    assert all(
        path.parts[0] not in {"data", "runs", ".venv", ".reference-venv", "dist"} for path in paths
    )
    assert all(path.suffix not in {".pt", ".joblib", ".npy", ".xlsx", ".pyc"} for path in paths)
    assert any("chemprop-1.6.1.dist-info/LICENSE.txt" in path.as_posix() for path in paths)
    assert not any(path.name.endswith("predictions.csv") for path in paths)
