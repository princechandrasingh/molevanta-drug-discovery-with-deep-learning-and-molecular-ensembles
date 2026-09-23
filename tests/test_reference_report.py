import numpy as np
import pandas as pd
import pytest

from molstudy.reference_report import paired_differences


def results():
    rows = []
    for seed, base, candidate in [(101, 0.6, 0.7), (202, 0.8, 0.5)]:
        for model, ap in [("chemprop", base), ("compact", candidate)]:
            rows.append(
                {
                    "split": "scaffold",
                    "seed": seed,
                    "model": model,
                    "n": 100,
                    "positives": 10,
                    "average_precision": ap,
                    "roc_auc": 0.8,
                    "precision_at_20": 0.4,
                    "brier_score": 0.1,
                }
            )
    return pd.DataFrame(rows)


def test_differences_pair_by_seed_not_row_order():
    paired = paired_differences(results().sample(frac=1, random_state=9)).sort_values("seed")
    np.testing.assert_allclose(paired.average_precision_delta, [0.1, -0.3])


def test_differences_reject_unpaired_or_duplicate_or_different_counts():
    frame = results()
    with pytest.raises(ValueError, match="Unpaired"):
        paired_differences(frame.iloc[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_differences(pd.concat([frame, frame.iloc[:1]]))
    frame.loc[0, "n"] = 101
    with pytest.raises(ValueError, match="counts"):
        paired_differences(frame)
