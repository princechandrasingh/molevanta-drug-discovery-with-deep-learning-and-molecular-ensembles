import json

import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from molstudy.graph import DirectedMPNN, batch_graphs, molecular_graph
from molstudy.improved import DescriptorModel, SPECS, fit, load_model, predict, select_pipeline


def test_augmented_model_preserves_backbone_initialization():
    torch.manual_seed(19)
    original = DirectedMPNN()
    torch.manual_seed(19)
    improved = DescriptorModel(True)
    for name, value in original.state_dict().items():
        if not name.startswith("readout."):
            assert torch.equal(value, improved.graph.state_dict()[name])


@pytest.mark.parametrize("use_graph", [False, True])
def test_descriptor_model_gradients_batch_order_and_reload(tmp_path, use_graph):
    torch.set_num_threads(2)
    graphs = [molecular_graph(s) for s in ["CCO", "CCN", "c1ccccc1"]]
    descriptors = np.random.default_rng(7).uniform(size=(3, 200)).astype(np.float32)
    model = DescriptorModel(use_graph)
    x = torch.tensor(descriptors, requires_grad=True)
    logits = model(batch_graphs(graphs) if use_graph else None, x)
    assert logits.shape == (3,)
    logits.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    if use_graph:
        assert model.graph.edge_in.weight.grad.abs().sum() > 0
    score = predict(model, graphs, descriptors)
    order = [2, 0, 1]
    np.testing.assert_allclose(
        score[order], predict(model, [graphs[i] for i in order], descriptors[order]), atol=1e-7
    )
    torch.save(
        {"spec": {"graph": use_graph}, "state_dict": model.state_dict()}, tmp_path / "model.pt"
    )
    np.testing.assert_array_equal(
        score, predict(load_model(tmp_path / "model.pt"), graphs, descriptors)
    )


def test_validation_selection_ties_prefer_first_candidate_and_forest():
    labels = np.array([0, 1, 0, 1])
    perfect = np.array([0.1, 0.9, 0.2, 0.8])
    decision = select_pipeline(labels, {name: perfect for name in SPECS}, perfect)
    assert decision["candidate"] == "descriptor_mlp"
    assert decision["neural_weight"] == 0
    with pytest.raises(ValueError, match="finite"):
        select_pipeline(labels, {name: perfect for name in SPECS}, np.full(4, np.nan))


@pytest.mark.parametrize("use_graph", [False, True])
def test_augmented_trainer_restores_best_validation_epoch(use_graph):
    graphs = [molecular_graph(s) for s in ["CCO", "CCN", "CCCC", "c1ccccc1"]]
    descriptors = np.random.default_rng(3).uniform(size=(4, 200)).astype(np.float32)
    labels = np.array([0, 1, 0, 1])
    spec = {"graph": use_graph, "epochs": 3, "schedule": "warmup_decay", "weight_decay": 0.0}
    model, info = fit(
        spec, graphs, descriptors, labels, graphs[:2], descriptors[:2], labels[:2], 11
    )
    best = max(info["history"], key=lambda r: r["validation_ap"])
    assert info["selected_epoch"] == best["epoch"]
    assert average_precision_score(
        labels[:2], predict(model, graphs[:2], descriptors[:2])
    ) == pytest.approx(best["validation_ap"])


def test_test_evaluation_requires_completed_selection_lock(tmp_path, monkeypatch):
    from molstudy import improvement

    monkeypatch.setattr(improvement, "check_environment", lambda: {})
    (tmp_path / "run.json").write_text(json.dumps({"status": "training"}))
    with pytest.raises(ValueError, match="locked"):
        improvement.evaluate_locked(tmp_path, tmp_path)


def test_selection_lock_rejects_tampering(tmp_path):
    from molstudy.improvement import verify_lock

    (tmp_path / "selection_lock.json").write_text("{}")
    with pytest.raises(ValueError, match="Selection lock changed"):
        verify_lock(tmp_path, {"selection_lock_sha256": "different"})
