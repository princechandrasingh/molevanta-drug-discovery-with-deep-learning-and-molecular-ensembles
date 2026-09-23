import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from molstudy.data import curate, standardize
from molstudy.metrics import evaluate, precision_at_k
from molstudy.splits import make_split, validate_split


def test_equivalent_smiles_and_salt_parent_have_same_identity():
    assert standardize("CCO")["compound_id"] == standardize("OCC")["compound_id"]
    assert standardize("CC[NH3+].[Cl-]")["compound_id"] == standardize("CCN")["compound_id"]


@pytest.mark.parametrize("smiles", ["", " ", "not-a-molecule"])
def test_invalid_input_is_not_silently_scored(smiles):
    with pytest.raises(ValueError):
        standardize(smiles)


def test_curation_quarantines_conflicts_and_audits_merges():
    frame = pd.DataFrame(
        {
            "SMILES": ["CCO", "OCC", "CCN", "NCC", "not-real", "CCC"],
            "Activity": ["Active", "Inactive", "Active", "Active", "Inactive", "Inactive"],
            "Name": list("abcdef"),
        }
    )
    clean, audit, stats = curate(frame)
    assert len(clean) == 2
    assert clean.source_count.sum() == 3
    assert stats["conflicting_structures"] == 1
    assert len(audit) == 4
    assert audit.reason.str.startswith("structure_error").sum() == 1


def test_unknown_labels_fail_closed():
    with pytest.raises(ValueError, match="Unexpected activity"):
        curate(pd.DataFrame({"SMILES": ["CCO"], "Name": ["x"], "Activity": ["Unknown"]}))


def test_acyclic_and_stereo_group_policy():
    a, b = standardize("C[C@H](O)C(=O)O"), standardize("C[C@@H](O)C(=O)O")
    assert a["compound_id"] != b["compound_id"]
    assert a["connectivity"] == b["connectivity"]
    assert a["scaffold"] == standardize("CCCC")["scaffold"] == "__ACYCLIC__"


def grouped_fixture():
    return pd.DataFrame(
        {
            "compound_id": [f"id{i}" for i in range(200)],
            "connectivity": [f"c{i // 2}" for i in range(200)],
            "scaffold": [f"s{i // 4}" for i in range(200)],
            "label": [i % 2 for i in range(200)],
        }
    )


@pytest.mark.parametrize("kind", ["random", "scaffold"])
def test_splits_are_complete_reproducible_and_group_disjoint(kind):
    frame = grouped_fixture()
    a = make_split(frame, kind, 11)
    b = make_split(frame, kind, 11)
    assert np.array_equal(a, b)
    validate_split(frame, a, kind)
    col = "scaffold" if kind == "scaffold" else "connectivity"
    groups = [set(frame.loc[a == part, col]) for part in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for part, fraction in [("train", 0.8), ("validation", 0.1), ("test", 0.1)]:
        assert abs(np.mean(a == part) - fraction) <= 0.03


def test_large_scaffold_does_not_starve_test_partition():
    frame = pd.DataFrame(
        {
            "compound_id": [f"id{i}" for i in range(1000)],
            "connectivity": [f"c{i}" for i in range(1000)],
            "scaffold": ["large"] * 200 + [f"s{i // 2}" for i in range(800)],
            "label": [int(i % 20 == 0) for i in range(1000)],
        }
    )
    p = make_split(frame, "scaffold", 11)
    assert sum(p == "test") >= 90
    assert frame.loc[p == "test", "label"].sum() >= 4
    assert len(set(p[:200])) == 1


def test_leakage_guard_rejects_cross_partition_stereoisomer():
    frame = grouped_fixture()
    partition = make_split(frame, "random", 11)
    train = np.flatnonzero(partition == "train")[0]
    test = np.flatnonzero(partition == "test")[0]
    frame.loc[test, "connectivity"] = frame.loc[train, "connectivity"]
    with pytest.raises(ValueError, match="leakage"):
        validate_split(frame, partition, "random")


def test_constant_score_ties_do_not_exploit_row_order():
    y = np.array([1] * 10 + [0] * 90)
    score = np.full(100, 0.1)
    assert precision_at_k(y, score, 20) == pytest.approx(0.1)
    assert precision_at_k(y[::-1], score, 20) == pytest.approx(0.1)
    assert evaluate(y, score)["average_precision"] == pytest.approx(0.1)


def test_evaluation_uses_continuous_rankings_and_rejects_nan():
    assert evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])["roc_auc"] == 1
    with pytest.raises(ValueError):
        evaluate([0, 1], [0.2, np.nan])


def test_graph_batching_permutation_and_bondless_molecules():
    import torch
    from molstudy.graph import DirectedMPNN, batch_graphs, molecular_graph, predict_graph

    torch.manual_seed(2)
    model = DirectedMPNN(hidden=16)
    graphs = [molecular_graph(s) for s in ["CCO", "OCC", "[Na+]", "c1ccccc1"]]
    scores = predict_graph(model, graphs)
    assert np.isfinite(scores).all()
    assert scores[0] == pytest.approx(scores[1], abs=1e-7)
    assert scores == pytest.approx(
        np.concatenate([predict_graph(model, [g]) for g in graphs]), abs=1e-7
    )
    for graph in graphs:
        _, _, source, dest, reverse = graph
        assert torch.equal(source[reverse], dest)
        assert torch.equal(reverse[reverse], torch.arange(len(reverse)))
    model.train()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        model(batch_graphs(graphs)), torch.tensor([0.0, 0.0, 1.0, 1.0])
    )
    loss.backward()
    assert model.edge_in.weight.grad is not None
    assert torch.isfinite(model.edge_in.weight.grad).all()
