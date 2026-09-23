import json

import joblib
import numpy as np
import pytest
from rdkit import DataStructs

from molstudy.features import fingerprints
from molstudy.kernel import (
    combined_kernel,
    ensemble_scores,
    fit,
    make_folds,
    select_setting,
    tanimoto,
    validate_folds,
)


def test_tanimoto_matches_rdkit_and_avoids_uint8_overflow():
    x, bits = fingerprints(["CCO", "CCN", "c1ccccc1", "CC(=O)O"])
    expected = np.array([[DataStructs.TanimotoSimilarity(a, b) for b in bits] for a in bits])
    np.testing.assert_allclose(tanimoto(x, x), expected, atol=1e-15)
    dense = np.ones((2, 2048), dtype=np.uint8)
    dense[1, 300:] = 0
    np.testing.assert_allclose(tanimoto(dense, dense), [[1, 300 / 2048], [300 / 2048, 1]])
    zero = np.zeros((1, 2048), dtype=np.uint8)
    assert tanimoto(zero, zero)[0, 0] == 1
    assert tanimoto(zero, dense)[0, 0] == 0
    with pytest.raises(ValueError, match="binary"):
        tanimoto(np.array([[2, 1]]), np.array([[0, 1]]))


def test_multiscale_kernel_is_symmetric_positive_semidefinite():
    strings = ["CCO", "CCN", "CCCC", "c1ccccc1", "c1ccncc1", "CC(=O)O"]
    x = {r: fingerprints(strings, radius=r)[0] for r in (2, 3)}
    matrix = combined_kernel(x, x, (2, 3))
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(np.diag(matrix), 1)
    assert np.linalg.eigvalsh(matrix).min() >= -1e-12


def test_inner_folds_keep_related_compounds_together_and_reject_leakage():
    y = np.tile([0, 0, 1, 1], 15)
    groups = np.repeat(np.arange(30), 2)
    fold = make_folds(y, groups, 17)
    validate_folds(y, groups, fold)
    np.testing.assert_array_equal(fold, make_folds(y, groups, 17))
    broken = fold.copy()
    broken[0] = (broken[1] + 1) % 3
    with pytest.raises(ValueError, match="leakage"):
        validate_folds(y, groups, broken)
    with pytest.raises(ValueError, match="one class"):
        validate_folds(np.zeros(60), groups, fold)


def test_selection_recomputed_from_all_fold_scores_with_stable_ties():
    labels = np.tile([0, 1], 6)
    folds = np.repeat(np.arange(3), 4)
    good = labels * 2.0 - 1.0
    decisions = np.column_stack([-good, good, good])
    grid = [{"C": c, "class_weight": None} for c in (1.0, 10.0, 100.0)]
    selection = select_setting(labels, folds, decisions, grid)
    assert selection["selected_index"] == 1
    assert selection["cv_ap"] == 1


def test_training_calibration_reload_and_batch_invariance(tmp_path):
    rng = np.random.default_rng(12)
    labels = np.tile([0, 0, 1, 1], 15)
    groups = np.repeat(np.arange(30), 2)
    x = rng.integers(0, 2, size=(60, 32), dtype=np.uint8)
    x[:, :16] = labels[:, None]
    features = {2: x, 3: x[:, ::-1]}
    fold = make_folds(labels, groups, 17)
    grid = [{"C": 1.0, "class_weight": None}, {"C": 10.0, "class_weight": "balanced"}]
    model, selection, oof = fit(features, labels, groups, fold, (2, 3), grid)
    assert selection["calibration_slope"] > 0
    assert select_setting(labels, fold, oof, grid)["selected_index"] == selection["selected_index"]
    expected = model.predict(features)
    assert np.isfinite(expected).all() and ((expected >= 0) & (expected <= 1)).all()
    joblib.dump(model, tmp_path / "model.joblib")
    np.testing.assert_array_equal(
        joblib.load(tmp_path / "model.joblib").predict(features), expected
    )
    for ix in ([0], [14, 2, 6]):
        np.testing.assert_allclose(
            model.predict({r: value[ix] for r, value in features.items()}), expected[ix], atol=1e-14
        )


def test_fixed_ensemble_weights_and_float64_arithmetic():
    scores = {
        "kernel_multiscale": np.array([0.7, 0.2]),
        "fusion_60": np.array([0.2, 0.8], dtype=np.float32),
        "chemprop": np.array([0.8, 0.3], dtype=np.float32),
        "forest": np.array([0.4, 0.1]),
    }
    result = ensemble_scores(scores)
    assert result["kernel_fusion_forest"].dtype == np.float64
    np.testing.assert_array_equal(
        result["kernel_fusion_forest"],
        (scores["kernel_multiscale"] + scores["fusion_60"].astype(float) + scores["forest"]) / 3,
    )
    np.testing.assert_array_equal(
        result["chemprop_forest"], (scores["chemprop"].astype(float) + scores["forest"]) / 2
    )
    scores["forest"][0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        ensemble_scores(scores)


def test_test_scoring_rejects_unlocked_runs_and_changed_lock(tmp_path, monkeypatch):
    from molstudy import kernel_study

    monkeypatch.setattr(kernel_study, "check_environment", lambda: {})
    (tmp_path / "run.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(ValueError, match="locked"):
        kernel_study.evaluate_locked(tmp_path, tmp_path)
    (tmp_path / "selection_lock.json").write_text("{}")
    with pytest.raises(ValueError, match="Locked artifact changed"):
        kernel_study.verify_lock(tmp_path, {"selection_lock_sha256": "changed"})
