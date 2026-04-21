from __future__ import annotations

import numpy as np
import pandas as pd

from mlops_aml_transactions.modeling.train import tune_threshold_recall_target


def test_recall_target_threshold_reachable() -> None:
    y = pd.Series([1, 1, 0, 0, 0])
    proba = np.array([0.9, 0.7, 0.6, 0.2, 0.1], dtype=float)

    threshold, meta = tune_threshold_recall_target(
        y,
        proba,
        target_recall=1.0,
        points=100,
        min_threshold=0.5,
        max_threshold=0.99,
    )

    assert threshold <= 0.7
    assert meta["recall_val"] >= 1.0
    assert meta["target_recall_met_val"] == 1.0


def test_recall_target_threshold_fallback_when_unreachable() -> None:
    y = pd.Series([1, 1, 0, 0, 0])
    proba = np.array([0.7, 0.6, 0.55, 0.2, 0.1], dtype=float)

    threshold, meta = tune_threshold_recall_target(
        y,
        proba,
        target_recall=0.9,
        points=80,
        min_threshold=0.8,
        max_threshold=0.99,
    )

    assert 0.8 <= threshold <= 0.99
    assert meta["target_recall_met_val"] == 0.0
