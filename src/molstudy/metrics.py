import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def precision_at_k(y, score, k=20):
    """Expected precision under uniform tie breaking; constant models cannot exploit row ordering."""
    y, score = np.asarray(y), np.asarray(score)
    if k <= 0 or len(y) == 0:
        raise ValueError("k and sample count must be positive")
    k = min(k, len(y))
    boundary = np.sort(score)[-k]
    above, tied = score > boundary, score == boundary
    # Share the remaining top-k slots across ties instead of favoring CSV row order.
    return float((y[above].sum() + (k - above.sum()) * y[tied].mean()) / k)


def evaluate(y, score):
    y, score = np.asarray(y), np.asarray(score)
    if len(y) != len(score) or len(np.unique(y)) != 2:
        raise ValueError("Evaluation requires aligned observations and both classes")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Scores must be finite values between 0 and 1")
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        # Average precision is the primary ranking metric for this imbalanced assay.
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision_at_20": precision_at_k(y, score),
        "brier_score": float(brier_score_loss(y, score)),
    }
