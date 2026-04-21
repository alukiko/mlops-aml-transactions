from __future__ import annotations

import pandas as pd

from mlops_aml_transactions.features import engineer_features


def _base_df() -> pd.DataFrame:
    # Intentionally unsorted by time to verify internal sort + no leakage behavior.
    return pd.DataFrame(
        {
            "Timestamp": [
                "2020-01-01 10:00",
                "2020-01-01 09:00",
                "2020-01-01 11:00",
            ],
            "From Bank": ["1", "1", "1"],
            "from_account": ["A", "A", "A"],
            "To Bank": ["2", "2", "2"],
            "to_account": ["X", "Y", "X"],
            "Amount Received": [100.0, 200.0, 300.0],
            "Receiving Currency": ["USD", "USD", "USD"],
            "Amount Paid": [100.0, 200.0, 300.0],
            "Payment Currency": ["USD", "USD", "USD"],
            "Payment Format": ["Wire", "Wire", "Wire"],
            "Is Laundering": [0, 1, 0],
        }
    )


def test_history_features_no_future_leakage() -> None:
    out = engineer_features(_base_df())

    # Row index 1 is earliest event (09:00), so history must be zero.
    assert out.loc[1, "sender_tx_count_prev"] == 0
    assert out.loc[1, "sender_mean_paid_prev"] == 0
    assert out.loc[1, "time_since_prev_sender_tx"] == 0


def test_history_features_values_for_unsorted_input() -> None:
    out = engineer_features(_base_df())

    # Row index 0 is 10:00, one previous sender transaction at 09:00 (amount 200).
    assert out.loc[0, "sender_tx_count_prev"] == 1
    assert out.loc[0, "sender_mean_paid_prev"] == 200.0
    assert out.loc[0, "time_since_prev_sender_tx"] == 3600
    assert out.loc[0, "sender_unique_receivers_prev"] == 1

    # Row index 2 is 11:00, previous amounts are [200, 100] -> mean=150.
    assert out.loc[2, "sender_tx_count_prev"] == 2
    assert out.loc[2, "sender_mean_paid_prev"] == 150.0
    assert out.loc[2, "time_since_prev_sender_tx"] == 3600
    assert out.loc[2, "sender_unique_receivers_prev"] == 2
