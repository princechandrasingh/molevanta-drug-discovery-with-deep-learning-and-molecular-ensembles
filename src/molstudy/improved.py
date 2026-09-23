"""Project-specific descriptor augmentation of the existing compact graph model."""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .graph import DirectedMPNN, batch_graphs
from .reference import learning_rate

SPECS = {
    "descriptor_mlp": {
        "graph": False,
        "epochs": 30,
        "schedule": "warmup_decay",
        "weight_decay": 0.0,
    },
    "fusion_30": {"graph": True, "epochs": 30, "schedule": "warmup_decay", "weight_decay": 0.0},
    "fusion_60": {"graph": True, "epochs": 60, "schedule": "constant", "weight_decay": 1e-5},
}
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


class DescriptorModel(nn.Module):
    def __init__(self, use_graph=True):
        super().__init__()
        self.use_graph = use_graph
        if use_graph:
            self.graph = DirectedMPNN(hidden=64, depth=3, dropout=0.1)
            # Keep the original backbone and initialization; expose its embedding.
            self.graph.readout = nn.Identity()
        # Fusion input: 64 learned graph features plus 200 fixed molecular descriptors.
        self.head = nn.Sequential(
            nn.Linear(264 if use_graph else 200, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, graph_batch, descriptors):
        if descriptors.ndim != 2 or descriptors.shape[1] != 200:
            raise ValueError("Expected a batch of 200 molecular descriptors")
        features = descriptors
        if self.use_graph:
            features = torch.cat([self.graph(graph_batch), descriptors], dim=1)
        return self.head(features).squeeze(1)


def predict(model, graphs, descriptors):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(descriptors), 50):
            stop = min(start + 50, len(descriptors))
            graph_batch = batch_graphs(graphs[start:stop]) if model.use_graph else None
            features = torch.as_tensor(descriptors[start:stop], dtype=torch.float32)
            scores.append(torch.sigmoid(model(graph_batch, features)).numpy())
    return np.concatenate(scores)


def fit(spec, train_graphs, train_x, train_y, val_graphs, val_x, val_y, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = DescriptorModel(spec["graph"])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=spec["weight_decay"])
    labels = torch.as_tensor(train_y, dtype=torch.float32)
    features = torch.as_tensor(train_x, dtype=torch.float32)
    rng = np.random.default_rng(seed)
    history, best_ap, best_state, best_epoch, step = [], -np.inf, None, None, 0
    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        order, total_loss = rng.permutation(len(train_y)), 0.0
        for start in range(0, len(order), 50):
            ix = order[start : start + 50]
            step += 1
            lr = (
                learning_rate(step, math.ceil(len(train_y) / 50), spec["epochs"])
                if spec["schedule"] == "warmup_decay"
                else 1e-3
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            graph_batch = batch_graphs([train_graphs[i] for i in ix]) if spec["graph"] else None
            optimizer.zero_grad()
            # The loss consumes raw logits; sigmoid is applied only when scoring.
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(graph_batch, features[ix]), labels[ix]
            )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total_loss += float(loss.detach()) * len(ix)
        scores = predict(model, val_graphs, val_x)
        ap = float(average_precision_score(val_y, scores))
        history.append(
            {"epoch": epoch, "train_bce": total_loss / len(train_y), "validation_ap": ap}
        )
        if ap > best_ap:
            # Strict improvement preserves the first checkpoint when AP is tied.
            best_ap, best_state, best_epoch = ap, copy.deepcopy(model.state_dict()), epoch
    model.load_state_dict(best_state)
    return model, {
        "spec": spec,
        "selected_epoch": best_epoch,
        "validation_ap": best_ap,
        "history": history,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
    }


def select_pipeline(val_y, neural_scores, forest_score):
    """Only validation labels/scores enter this function; test data is not an input."""
    if tuple(neural_scores) != tuple(SPECS):
        raise ValueError("Candidate order must match the declared protocol")
    for score in [*neural_scores.values(), forest_score]:
        if len(score) != len(val_y) or not np.isfinite(score).all():
            raise ValueError("Validation scores must be finite and aligned")
    candidate_ap = {
        name: float(average_precision_score(val_y, score)) for name, score in neural_scores.items()
    }
    chosen = max(candidate_ap, key=candidate_ap.get)
    # Select a neural model first, then its forest mixture; neither step sees test labels.
    search = [
        {
            "neural_weight": weight,
            "validation_ap": float(
                average_precision_score(
                    val_y,
                    weight * neural_scores[chosen].astype(np.float64) + (1 - weight) * forest_score,
                )
            ),
        }
        for weight in WEIGHTS
    ]
    best = max(search, key=lambda row: row["validation_ap"])
    return {
        "candidate": chosen,
        **best,
        "candidate_validation_ap": candidate_ap,
        "mixture_search": search,
    }


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = DescriptorModel(checkpoint["spec"]["graph"])
    model.load_state_dict(checkpoint["state_dict"])
    return model
