"""Check count mathematics, historical compatibility, and evaluation boundaries."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from molstudy.count_study import BASE_LABELS, combine, evaluate_locked, old_scores
from molstudy.features import count_fingerprints, fingerprints
from molstudy.kernel import combined_kernel, count_tanimoto, fit, make_folds, tanimoto


def test_counts_preserve_binary_support_and_repeated_environments():
    strings = ["CCCCCC", "c1ccccc1", "CCO"]
    for radius in (2, 3):
        counts = count_fingerprints(strings, radius)
        binary, _ = fingerprints(strings, radius)
        np.testing.assert_array_equal(counts > 0, binary.astype(bool))
        assert counts.dtype == np.int32 and counts.max() > 1
    with pytest.raises(ValueError, match="At least one"):
        count_fingerprints([])
    with pytest.raises(ValueError, match="invalid"):
        count_fingerprints(["not a molecule"])


def test_dot_product_count_kernel_differs_from_min_max():
    # Dot-product: 5/(5+10-5)=1/2; min/max would be 2/5.
    np.testing.assert_array_equal(count_tanimoto([[2, 1]], [[1, 3]]), [[0.5]])
    zero = np.zeros((1, 2))
    assert count_tanimoto(zero, zero)[0, 0] == 1
    assert count_tanimoto(zero, [[1, 0]])[0, 0] == 0
    # Squaring in int32 would overflow: promotion must precede the product.
    large = np.array([[100_000, 0], [50_000, 0]], dtype=np.int32)
    np.testing.assert_allclose(count_tanimoto(large, large), [[1, 2 / 3], [2 / 3, 1]])


@pytest.mark.parametrize("bad", [[[-1, 0]], [[np.nan, 0]], [[np.inf, 0]]])
def test_count_kernel_rejects_invalid_values(bad):
    with pytest.raises(ValueError, match="Finite nonnegative"):
        count_tanimoto(bad, [[1, 0]])


def test_count_kernel_reduces_to_binary_and_has_positive_semidefinite_gram():
    rng = np.random.default_rng(32)
    binary = rng.integers(0, 2, size=(10, 32))
    np.testing.assert_array_equal(count_tanimoto(binary, binary), tanimoto(binary, binary))
    count = rng.integers(0, 9, size=(10, 32))
    gram = combined_kernel(
        {2: count, 3: count[:, ::-1]}, {2: count, 3: count[:, ::-1]}, (2, 3), "count"
    )
    np.testing.assert_array_equal(gram, gram.T)
    np.testing.assert_allclose(np.diag(gram), 1)
    assert np.linalg.eigvalsh(gram).min() >= -1e-12
    with pytest.raises(ValueError, match="Unknown kernel"):
        combined_kernel({2: binary}, {2: binary}, (2,), "typo")
    with pytest.raises(ValueError, match="Aligned"):
        count_tanimoto([[1, 2]], [[1]])


def test_count_checkpoint_retains_representation_and_batch_invariance(tmp_path):
    rng = np.random.default_rng(12)
    labels = np.tile([0, 0, 1, 1], 15)
    groups = np.repeat(np.arange(30), 2)
    counts = rng.integers(0, 5, size=(60, 32), dtype=np.int32)
    counts[:, :16] = 1 + labels[:, None] * 5
    features = {2: counts, 3: counts[:, ::-1]}
    fold = make_folds(labels, groups, 17)
    model, selection, _ = fit(
        features,
        labels,
        groups,
        fold,
        (2, 3),
        [{"C": 1.0, "class_weight": None}],
        kernel_kind="count",
    )
    assert model.kernel_kind == "count" and selection["calibration_slope"] > 0
    scores = model.predict(features)
    joblib.dump(model, tmp_path / "counts.joblib")
    restored = joblib.load(tmp_path / "counts.joblib")
    np.testing.assert_array_equal(restored.predict(features), scores)
    ix = [7, 1, 18]
    np.testing.assert_allclose(
        restored.predict({r: x[ix] for r, x in features.items()}), scores[ix], atol=1e-14
    )


def test_historical_pickle_without_kind_keeps_binary_behavior(tmp_path):
    labels = np.tile([0, 0, 1, 1], 15)
    groups = np.repeat(np.arange(30), 2)
    features = {2: np.column_stack([labels, 1 - labels])}
    model, _, _ = fit(
        features,
        labels,
        groups,
        make_folds(labels, groups, 17),
        (2,),
        [{"C": 1.0, "class_weight": None}],
    )
    expected = model.predict(features)
    del model.kernel_kind  # Historical checkpoints predate this dataclass field.
    joblib.dump(model, tmp_path / "historical.joblib")
    restored = joblib.load(tmp_path / "historical.joblib")
    assert restored.kernel_kind == "binary"
    np.testing.assert_array_equal(restored.predict(features), expected)


def test_count_ensembles_promote_float32_and_keep_equal_weights():
    scores = {
        "count_multiscale": np.array([0.2, 0.7]),
        "fusion_60": np.array([0.4, 0.6], dtype=np.float32),
        "forest": np.array([0.1, 0.9]),
        "chemprop": np.array([0.8, 0.2], dtype=np.float32),
    }
    result = combine(scores)
    np.testing.assert_array_equal(
        result["count_fusion_forest"],
        (scores["count_multiscale"] + scores["fusion_60"].astype(float) + scores["forest"]) / 3,
    )
    scores["forest"][0] = np.nan
    with pytest.raises(ValueError, match="Invalid"):
        combine(scores)


def test_base_score_alignment_rejects_duplicates_and_label_mismatches():
    test = pd.DataFrame({"compound_id": ["a", "b"], "label": [0, 1]})
    predictions = pd.DataFrame(
        [
            {
                "compound_id": cid,
                "label": label,
                "score": label * 0.8,
                "split": "scaffold",
                "seed": 101,
                "model": model,
            }
            for model in BASE_LABELS
            for cid, label in [("b", 1), ("a", 0)]
        ]
    )
    scores = old_scores(predictions, "scaffold_101", test)
    np.testing.assert_array_equal(scores["forest"], [0, 0.8])
    with pytest.raises(ValueError, match="membership"):
        old_scores(pd.concat([predictions, predictions.iloc[:1]]), "scaffold_101", test)
    predictions.loc[0, "label"] = 0
    with pytest.raises(AssertionError):
        old_scores(predictions, "scaffold_101", test)


@pytest.mark.parametrize("status", ["training", "training_failed", "evaluating", "completed"])
def test_evaluation_rejects_unlocked_or_already_scored_run(tmp_path, monkeypatch, status):
    monkeypatch.setattr("molstudy.count_study.check_environment", lambda: {})
    (tmp_path / "run.json").write_text(json.dumps({"status": status}))
    with pytest.raises(ValueError, match="locked, not yet evaluated"):
        evaluate_locked(tmp_path)


def test_count_report_renders_complete_aggregate_tables(tmp_path, monkeypatch):
    from molstudy import count_study

    metrics = pd.DataFrame(
        [
            {
                "split": kind,
                "seed": seed,
                "model": name,
                "average_precision": 0.6,
                "roc_auc": 0.8,
                "precision_at_20": 0.4,
                "brier_score": 0.1,
            }
            for kind in ("random", "scaffold")
            for seed in (101, 202, 303, 404, 505)
            for name in count_study.LABELS
        ]
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text("{}")
    monkeypatch.setattr(count_study, "verified_results", lambda _: (metrics, {}, 9160))
    output = tmp_path / "report"
    count_study.make_report(run, output)
    report = (output / "REPORT.md").read_text()
    assert "Count Tanimoto SVM (radii 2 + 3)" in report
    assert "0.600 ± 0.000" in report
    assert len(pd.read_csv(output / "summary.csv")) == 34
    assert (output / "model_comparison.png").stat().st_size > 0
    verification = json.loads((output / "verification.json").read_text())
    assert verification["total_metric_rows"] == 170
    with pytest.raises(FileExistsError):
        count_study.make_report(run, output)
