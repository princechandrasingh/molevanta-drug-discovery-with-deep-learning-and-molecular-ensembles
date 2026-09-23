"""Compact directed-bond MPNN, inspired by Yang et al.; not exact Chemprop reproduction."""

from __future__ import annotations

import copy
import random
import numpy as np
import torch
from rdkit import Chem
from sklearn.metrics import average_precision_score
from torch import nn

ELEMENTS = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53]


def onehot(value, choices):
    # Reserve the last position for elements/categories outside the known vocabulary.
    return [float(value == choice) for choice in choices] + [float(value not in choices)]


def atom_features(atom):
    return (
        onehot(atom.GetAtomicNum(), ELEMENTS)
        + onehot(atom.GetTotalDegree(), list(range(6)))
        + onehot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])
        + onehot(int(atom.GetHybridization()), [2, 3, 4, 5, 6])
        + onehot(int(atom.GetChiralTag()), [0, 1, 2])
        + [float(atom.GetIsAromatic()), atom.GetMass() / 100.0, atom.GetTotalNumHs() / 4.0]
    )


def bond_features(bond):
    return onehot(str(bond.GetBondType()), ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


ATOM_DIM = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
BOND_DIM = 7


def molecular_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Invalid graph input")
    atoms = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    sources, destinations, bonds, reverse = [], [], [], []
    for bond in mol.GetBonds():
        # One chemical bond becomes two directed edges with paired reverse indices.
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        i = len(sources)
        sources.extend([u, v])
        destinations.extend([v, u])
        reverse.extend([i + 1, i])
        bonds.extend([bond_features(bond), bond_features(bond)])
    return (
        atoms,
        torch.tensor(bonds, dtype=torch.float32).reshape(-1, BOND_DIM),
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(destinations, dtype=torch.long),
        torch.tensor(reverse, dtype=torch.long),
    )


def batch_graphs(graphs):
    if not graphs:
        raise ValueError("Cannot batch an empty graph list")
    atoms, bonds, sources, destinations, reverse, membership = [], [], [], [], [], []
    atom_offset = edge_offset = 0
    for graph_id, (a, b, s, d, r) in enumerate(graphs):
        # Offset local indices so messages cannot cross between batched molecules.
        atoms.append(a)
        bonds.append(b)
        sources.append(s + atom_offset)
        destinations.append(d + atom_offset)
        reverse.append(r + edge_offset)
        membership.append(torch.full((len(a),), graph_id, dtype=torch.long))
        atom_offset += len(a)
        edge_offset += len(b)
    return tuple(
        torch.cat(parts) for parts in (atoms, bonds, sources, destinations, reverse, membership)
    ) + (len(graphs),)


class DirectedMPNN(nn.Module):
    def __init__(self, hidden=64, depth=3, dropout=0.1):
        super().__init__()
        self.hidden, self.depth = hidden, depth
        self.edge_in = nn.Linear(ATOM_DIM + BOND_DIM, hidden, bias=False)
        self.message = nn.Linear(hidden, hidden, bias=False)
        self.atom_out = nn.Linear(ATOM_DIM + hidden, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, batch):
        atoms, bonds, source, dest, reverse, membership, n_graphs = batch
        initial = torch.relu(self.edge_in(torch.cat([atoms[source], bonds], dim=1)))
        hidden = initial
        for _ in range(self.depth - 1):
            incoming = atoms.new_zeros((len(atoms), self.hidden)).index_add(0, dest, hidden)
            # Incoming bonds at source u, excluding v -> u when updating u -> v.
            message = incoming[source] - hidden[reverse]
            hidden = torch.relu(initial + self.message(message))
        # Convert the final directed-edge states into atom states before pooling.
        incoming = atoms.new_zeros((len(atoms), self.hidden)).index_add(0, dest, hidden)
        atom_state = torch.relu(self.atom_out(torch.cat([atoms, incoming], dim=1)))
        pooled = atoms.new_zeros((n_graphs, self.hidden)).index_add(0, membership, atom_state)
        # Mean aggregation moderates effects of large molecular size in this compact pilot.
        counts = torch.bincount(membership, minlength=n_graphs).to(pooled.dtype).unsqueeze(1)
        return self.readout(pooled / counts).squeeze(1)


def predict_graph(model, graphs, batch_size=64):
    # Disable dropout/gradients and convert training logits into assay scores.
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            logits = model(batch_graphs(graphs[start : start + batch_size]))
            predictions.extend(torch.sigmoid(logits).tolist())
    return np.asarray(predictions)


def fit_graph(
    train_graphs, train_y, validation_graphs, validation_y, seed, max_epochs=30, hidden=64
):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = DirectedMPNN(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    labels = torch.tensor(train_y, dtype=torch.float32)
    generator = np.random.default_rng(seed)
    best_score, best_state, best_epoch, stale = -np.inf, None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = generator.permutation(len(train_graphs))
        total_loss = 0.0
        for start in range(0, len(order), 64):
            ix = order[start : start + 64]
            optimizer.zero_grad()
            loss = loss_fn(model(batch_graphs([train_graphs[i] for i in ix])), labels[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
        score = float(
            average_precision_score(validation_y, predict_graph(model, validation_graphs))
        )
        history.append(
            {"epoch": epoch, "train_bce": total_loss / len(train_graphs), "validation_ap": score}
        )
        if score > best_score:
            # Clone tensors: a shallow reference would change during later epochs.
            best_score, best_state, best_epoch, stale = (
                score,
                copy.deepcopy(model.state_dict()),
                epoch,
                0,
            )
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    return model, {
        "hidden": hidden,
        "depth": 3,
        "dropout": 0.1,
        "max_epochs": max_epochs,
        "selected_epoch": best_epoch,
        "validation_ap": best_score,
        "history": history,
    }
