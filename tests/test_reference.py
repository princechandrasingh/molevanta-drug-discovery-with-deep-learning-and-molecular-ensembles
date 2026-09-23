"""Tests for the optional licensed reference integration."""

from pathlib import Path

import numpy as np
import pytest
import torch

from molstudy.reference import learning_rate


def test_schedule_endpoints_and_monotonicity():
    rates = np.array([learning_rate(step, 4) for step in range(121)])
    assert rates[0] == pytest.approx(1e-4)
    assert rates[8] == pytest.approx(1e-3)
    assert rates[-1] == pytest.approx(1e-4)
    assert np.all(np.diff(rates[:9]) > 0)
    assert np.all(np.diff(rates[8:]) < 0)
    with pytest.raises(ValueError):
        learning_rate(1, 4, epochs=2)


def test_reference_logits_probabilities_order_and_reload(tmp_path):
    pytest.importorskip("chemprop")
    from chemprop.features import MolGraph
    from molstudy.reference import forward, make_chemprop, neural_predict, normalized_descriptors

    torch.set_num_threads(2)
    path = tmp_path / "train.csv"
    path.write_text("smiles,label\nCCO,0\nCCN,1\n")
    smiles = ["CCO", "CCN", "c1ccccc1", "CC(=O)O"]
    descriptors, names = normalized_descriptors(smiles)
    assert descriptors.shape == (4, 200) and len(set(names)) == 200
    # Descriptor transforms are molecule-local and do not depend on the cohort.
    np.testing.assert_array_equal(descriptors[0], normalized_descriptors(smiles[:1])[0][0])
    model = make_chemprop(path, tmp_path / "config")
    graphs = [MolGraph(s) for s in smiles]
    model.train()
    logits = forward(model, "chemprop", graphs, descriptors, np.arange(4))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.tensor([0.0, 1.0, 0.0, 1.0])
    )
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    scores = neural_predict(model, "chemprop", graphs, descriptors)
    np.testing.assert_allclose(scores, torch.sigmoid(logits.detach()).numpy(), rtol=1e-6)
    order = [2, 0, 3, 1]
    np.testing.assert_allclose(
        scores[order],
        neural_predict(model, "chemprop", [graphs[i] for i in order], descriptors[order]),
        rtol=1e-6,
    )
    torch.save({"state_dict": model.state_dict()}, tmp_path / "model.pt")
    restored = make_chemprop(path, tmp_path / "config")
    restored.load_state_dict(torch.load(tmp_path / "model.pt", weights_only=True)["state_dict"])
    np.testing.assert_allclose(
        scores, neural_predict(restored, "chemprop", graphs, descriptors), rtol=1e-6
    )


def test_reference_missing_descriptor_policy():
    pytest.importorskip("chemprop")
    from molstudy.reference import sanitize_descriptors

    np.testing.assert_array_equal(
        sanitize_descriptors([0.25, np.nan, 0.8]), [0.25, 0, np.float32(0.8)]
    )
    with pytest.raises(ValueError, match="Infinite"):
        sanitize_descriptors([np.inf])


@pytest.mark.parametrize("name", ["compact", "chemprop"])
def test_common_trainer_checkpoint_matches_validation_history(tmp_path, name):
    pytest.importorskip("chemprop")
    from chemprop.features import MolGraph
    from sklearn.metrics import average_precision_score
    from molstudy.graph import molecular_graph
    from molstudy.reference import fit_neural, neural_predict, normalized_descriptors

    path = tmp_path / "train.csv"
    path.write_text("smiles,label\nCCO,0\nCCN,1\n")
    train_s = ["CCO", "CCN", "CCCC", "c1ccccc1"]
    val_s = ["CC(=O)O", "CCCN"]
    create = molecular_graph if name == "compact" else MolGraph
    train_g, val_g = [create(s) for s in train_s], [create(s) for s in val_s]
    train_d, _ = normalized_descriptors(train_s)
    val_d, _ = normalized_descriptors(val_s)
    val_y = np.array([0, 1])
    model, selection = fit_neural(
        name,
        train_g,
        train_d,
        np.array([0, 1, 0, 1]),
        val_g,
        val_d,
        val_y,
        19,
        path,
        tmp_path / "config",
        epochs=3,
    )
    history = selection["history"]
    assert len(history) == 3
    best = max(history, key=lambda row: row["validation_ap"])
    assert selection["selected_epoch"] == best["epoch"]
    assert average_precision_score(
        val_y, neural_predict(model, name, val_g, val_d)
    ) == pytest.approx(best["validation_ap"])
