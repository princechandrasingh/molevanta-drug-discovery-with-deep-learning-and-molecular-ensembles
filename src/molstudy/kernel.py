"""Exact fingerprint kernels with group-aware tuning and training-only calibration.

Project-specific orchestration of established methods; estimators are sklearn.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

# Iteration order is part of the protocol: exact score ties favor the first setting.
GRID = [
    {"C": c, "class_weight": weight}
    for c in (0.1, 1.0, 10.0, 100.0)
    for weight in (None, "balanced")
]
FAMILIES = {"kernel_r2": (2,), "kernel_multiscale": (2, 3)}
ENSEMBLES = {
    "kernel_fusion_forest": ("kernel_multiscale", "fusion_60", "forest"),
    "kernel_chemprop_forest": ("kernel_multiscale", "chemprop", "forest"),
    "chemprop_forest": ("chemprop", "forest"),
}


def tanimoto(left, right):
    """Binary Tanimoto, with K(empty, empty)=1; cast BEFORE multiplying uint8."""
    left, right = np.asarray(left), np.asarray(right)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Aligned two-dimensional fingerprints required")
    if not (np.isin(left, [0, 1]).all() and np.isin(right, [0, 1]).all()):
        raise ValueError("Fingerprints must be binary")
    with threadpool_limits(limits=2):
        # A uint8 dot product would overflow once more than 255 bits overlap.
        common = (left.astype(np.float32) @ right.astype(np.float32).T).astype(np.float64)
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - common
    return np.divide(common, union, out=np.ones_like(common), where=union != 0)


def count_tanimoto(left, right):
    """Dot-product Tanimoto on counts; distinct from the min/max count kernel."""
    left, right = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1]:
        raise ValueError("Aligned two-dimensional fingerprints required")
    if any(not np.isfinite(x).all() or (x < 0).any() for x in (left, right)):
        raise ValueError("Finite nonnegative fingerprints required")
    with threadpool_limits(limits=2):
        common = left @ right.T
    # Count vectors need squared norms, unlike binary vectors whose squares equal bits.
    denominator = (left * left).sum(axis=1)[:, None] + (right * right).sum(axis=1)[None, :] - common
    if not np.isfinite(denominator).all() or not np.isfinite(common).all():
        raise ValueError("Fingerprint magnitudes overflow the kernel computation")
    return np.divide(common, denominator, out=np.ones_like(common), where=denominator != 0)


def combined_kernel(left, right, radii, kind="binary"):
    # Equal weights combine molecular neighborhoods without fitting another parameter.
    if kind not in {"binary", "count"}:
        raise ValueError(f"Unknown kernel kind: {kind}")
    similarity = tanimoto if kind == "binary" else count_tanimoto
    return np.mean([similarity(left[r], right[r]) for r in radii], axis=0)


def make_folds(labels, groups, seed):
    labels, groups = np.asarray(labels), np.asarray(groups)
    fold = np.full(len(labels), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for number, (_, held) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups)):
        # Every training molecule receives exactly one held-out prediction per setting.
        if np.any(fold[held] != -1):
            raise ValueError("Duplicate fold assignment")
        fold[held] = number
    validate_folds(labels, groups, fold)
    return fold


def validate_folds(labels, groups, fold):
    labels, groups, fold = np.asarray(labels), np.asarray(groups), np.asarray(fold)
    if not len(labels) == len(groups) == len(fold) or set(fold) != {0, 1, 2}:
        raise ValueError("Exactly three complete inner folds required")
    for number in range(3):
        held = fold == number
        if set(groups[held]) & set(groups[~held]):
            raise ValueError("Chemical group leakage in inner folds")
        if len(np.unique(labels[held])) != 2 or len(np.unique(labels[~held])) != 2:
            raise ValueError("Inner fold has only one class; do not reroll")


def select_setting(labels, fold, decisions, grid=GRID):
    decisions = np.asarray(decisions)
    if decisions.shape != (len(labels), len(grid)) or not np.isfinite(decisions).all():
        raise ValueError("Complete finite out-of-fold decisions required")
    history = []
    for i, setting in enumerate(grid):
        scores = [
            float(average_precision_score(labels[fold == f], decisions[fold == f, i]))
            for f in range(3)
        ]
        history.append({**setting, "fold_ap": scores, "mean_ap": float(np.mean(scores))})
    selected = max(range(len(grid)), key=lambda i: history[i]["mean_ap"])
    return {
        "selected_index": selected,
        "setting": grid[selected],
        "cv_ap": history[selected]["mean_ap"],
        "history": history,
    }


@dataclass
class KernelClassifier:
    radii: tuple
    training_fingerprints: dict
    estimator: SVC
    calibrator: LogisticRegression
    # A class-level default also keeps historical pickles without this field readable.
    kernel_kind: str = "binary"

    def predict(self, fingerprints):
        # Kernel columns must follow the exact training-molecule order used by SVC.
        kernel = combined_kernel(
            fingerprints, self.training_fingerprints, self.radii, self.kernel_kind
        )
        decision = self.estimator.decision_function(kernel)
        return self.calibrator.predict_proba(decision[:, None])[:, 1]


def fit(fingerprints, labels, groups, fold, radii, grid=GRID, *, kernel_kind="binary"):
    labels, fold = np.asarray(labels), np.asarray(fold)
    validate_folds(labels, groups, fold)
    # This matrix contains outer-training rows only; validation/test data never enter.
    kernel = combined_kernel(fingerprints, fingerprints, radii, kernel_kind)
    decisions = np.full((len(labels), len(grid)), np.nan, dtype=np.float64)
    for i, setting in enumerate(grid):
        for number in range(3):
            train, held = np.flatnonzero(fold != number), np.flatnonzero(fold == number)
            estimator = SVC(kernel="precomputed", probability=False, **setting)
            # Fit on training-by-training similarities, then score held-by-training.
            estimator.fit(kernel[np.ix_(train, train)], labels[train])
            decisions[held, i] = estimator.decision_function(kernel[np.ix_(held, train)])
    selection = select_setting(labels, fold, decisions, grid)
    # Calibrate held-out training decisions, not the overconfident in-sample margins.
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(decisions[:, selection["selected_index"], None], labels)
    if calibrator.coef_[0, 0] <= 0:
        raise ValueError("Non-positive calibration slope; insufficient training-only signal")
    # Refit the selected SVM on all outer-training molecules for later inference.
    estimator = SVC(kernel="precomputed", probability=False, **selection["setting"])
    estimator.fit(kernel, labels)
    model = KernelClassifier(
        tuple(radii), {r: fingerprints[r].copy() for r in radii}, estimator, calibrator, kernel_kind
    )
    selection.update(
        calibration_slope=float(calibrator.coef_[0, 0]),
        calibration_intercept=float(calibrator.intercept_[0]),
        support_vectors=int(estimator.n_support_.sum()),
    )
    return model, selection, decisions


def ensemble_scores(scores):
    result = {}
    for name, members in ENSEMBLES.items():
        # Promote neural float32 scores before arithmetic to make blends reproducible.
        values = np.stack([np.asarray(scores[member], dtype=np.float64) for member in members])
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid component scores")
        result[name] = values.mean(axis=0)
    return result
