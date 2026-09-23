"""Fixed, structure-aware partitions; no split selection based on model scores."""

import numpy as np
import pandas as pd


def balanced_group_partition(labels, groups, seed):
    """One deterministic greedy allocation; uses counts only, never model scores.

    Place difficult (large or positive-rich) groups first, then minimize the
    normalized squared deviation from 80/10/10 row and positive-count targets.
    Seeded perturbation of group priority allows repeated partition experiments.
    """
    labels, groups = np.asarray(labels), np.asarray(groups)
    rng = np.random.default_rng(seed)
    names = np.array(["train", "validation", "test"], dtype=object)
    fractions = np.array([0.8, 0.1, 0.1])
    target = np.column_stack([fractions * len(labels), fractions * labels.sum()])
    current = np.zeros_like(target)
    group_table = []
    for group in sorted(set(groups)):
        indices = np.flatnonzero(groups == group)
        size = np.array([len(indices), labels[indices].sum()], dtype=float)
        priority = max(size[0] / len(labels), size[1] / max(labels.sum(), 1))
        group_table.append((priority * rng.uniform(0.8, 1.2), group, indices, size))
    # Place difficult groups first; the seeded priority is independent of model scores.
    group_table.sort(key=lambda g: (-g[0], g[1]))
    partition = np.empty(len(labels), dtype=object)
    for _, _, indices, size in group_table:
        costs = []
        for destination in range(3):
            proposed = current.copy()
            proposed[destination] += size
            residual = (proposed - target) ** 2 / np.maximum(target, 1)
            cost = residual[:, 0].sum() / len(labels) + residual[:, 1].sum() / max(labels.sum(), 1)
            costs.append(cost)
        chosen = int(np.argmin(costs))
        current[chosen] += size
        partition[indices] = names[chosen]
    return partition


def make_split(frame: pd.DataFrame, kind: str, seed: int) -> np.ndarray:
    if kind not in ("random", "scaffold"):
        raise ValueError(f"Unknown split: {kind}")
    # Even the random baseline keeps stereoisomers out of separate partitions.
    group_col = "connectivity" if kind == "random" else "scaffold"
    groups = frame[group_col].to_numpy()
    if len(np.unique(groups)) < 10:
        raise ValueError("At least 10 chemical groups are required for the 80/10/10 protocol")
    partition = balanced_group_partition(frame.label.to_numpy(), groups, seed)
    validate_split(frame, partition, kind)
    return partition


def validate_split(frame: pd.DataFrame, partition: np.ndarray, kind: str) -> None:
    if len(partition) != len(frame) or set(partition) != {"train", "validation", "test"}:
        raise ValueError("Invalid partition membership")
    for col in ["compound_id", "connectivity"] + (["scaffold"] if kind == "scaffold" else []):
        check = pd.DataFrame({"group": frame[col].to_numpy(), "partition": partition})
        if check.groupby("group").partition.nunique().max() != 1:
            raise ValueError(f"Chemical leakage across partitions: {col}")
    for split in ("train", "validation", "test"):
        # Reject an unusable split instead of silently searching for an easier seed.
        if frame.loc[partition == split, "label"].nunique() != 2:
            raise ValueError(
                f"{split} has only one class; this split cannot be scored. Do not silently reroll seeds."
            )


def describe_split(frame: pd.DataFrame, partition: np.ndarray) -> dict:
    return {
        split: {
            "n": int(sum(partition == split)),
            "positives": int(frame.loc[partition == split, "label"].sum()),
            "scaffolds": int(frame.loc[partition == split, "scaffold"].nunique()),
        }
        for split in ("train", "validation", "test")
    }
