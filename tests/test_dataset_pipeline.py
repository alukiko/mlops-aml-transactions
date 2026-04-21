from __future__ import annotations

import pandas as pd

from mlops_aml_transactions.dataset import ensure_min_positive_rows, stratified_subsample


def _build_df(n_neg: int, n_pos: int) -> pd.DataFrame:
    vals = [0] * n_neg + [1] * n_pos
    return pd.DataFrame({"Is Laundering": vals, "Amount Paid": range(len(vals))})


def test_ensure_min_positive_rows_meets_target() -> None:
    full_df = _build_df(n_neg=980, n_pos=20)
    sampled = stratified_subsample(full_df, sample_size=200, random_state=42)

    out = ensure_min_positive_rows(full_df, sampled, min_positive_rows=30, random_state=42)

    assert len(out) == len(sampled)
    assert int((out["Is Laundering"] == 1).sum()) >= 30


def test_ensure_min_positive_rows_zero_keeps_data() -> None:
    full_df = _build_df(n_neg=900, n_pos=100)
    sampled = stratified_subsample(full_df, sample_size=200, random_state=42)

    out = ensure_min_positive_rows(full_df, sampled, min_positive_rows=0, random_state=42)

    assert len(out) == len(sampled)
    assert int((out["Is Laundering"] == 1).sum()) == int((sampled["Is Laundering"] == 1).sum())
