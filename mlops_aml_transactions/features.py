"""Признаки для строк AML-транзакций."""

from __future__ import annotations

import numpy as np
import pandas as pd

# Имена колонок после переименования в dataset (сырой CSV → RAW_COLUMNS)
RAW_COLUMNS = [
    "Timestamp",
    "From Bank",
    "from_account",
    "To Bank",
    "to_account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

NUMERIC_FEATURES = [
    "hour",
    "dayofweek",
    "day",
    "is_weekend",
    "Amount Received",
    "Amount Paid",
    "log_amt_received",
    "log_amt_paid",
    "amount_abs_diff",
    "same_currency",
    "in_known_pattern",
    "sender_tx_count_prev",
    "sender_mean_paid_prev",
    "sender_std_paid_prev",
    "receiver_tx_count_prev",
    "receiver_mean_received_prev",
    "time_since_prev_sender_tx",
    "time_since_prev_receiver_tx",
    "sender_unique_receivers_prev",
    "receiver_unique_senders_prev",
]

CATEGORICAL_FEATURES = [
    "From Bank",
    "To Bank",
    "from_account",
    "to_account",
    "Receiving Currency",
    "Payment Currency",
    "Payment Format",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Парсинг времени, преобразования сумм, категориальные поля в строки."""
    out = df.copy()
    ts = pd.to_datetime(out["Timestamp"], format="mixed", errors="coerce")

    out["hour"] = ts.dt.hour.fillna(0).astype(np.int32)
    out["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(np.int32)
    out["day"] = ts.dt.day.fillna(1).astype(np.int32)
    out["is_weekend"] = (ts.dt.dayofweek >= 5).fillna(False).astype(np.int32)

    ar = pd.to_numeric(out["Amount Received"], errors="coerce").fillna(0.0)
    ap = pd.to_numeric(out["Amount Paid"], errors="coerce").fillna(0.0)
    out["Amount Received"] = ar
    out["Amount Paid"] = ap
    out["log_amt_received"] = np.log1p(np.maximum(ar.to_numpy(dtype=float), 0.0))
    out["log_amt_paid"] = np.log1p(np.maximum(ap.astype(float).to_numpy(), 0.0))
    out["amount_abs_diff"] = (ap - ar).abs()

    rc = out["Receiving Currency"].astype(str)
    pc = out["Payment Currency"].astype(str)
    out["same_currency"] = (rc == pc).astype(np.int32)
    if "in_known_pattern" not in out.columns:
        out["in_known_pattern"] = 0
    out["in_known_pattern"] = pd.to_numeric(out["in_known_pattern"], errors="coerce").fillna(0).astype(
        np.int32
    )

    # Time-safe history features: compute on time-sorted copy, then restore original row order.
    out["__orig_idx"] = np.arange(len(out), dtype=np.int64)
    out["__ts"] = ts
    sorted_out = out.sort_values(["__ts", "__orig_idx"]).copy()
    sorted_out["__ts_int"] = sorted_out["__ts"].astype("int64", copy=False)

    sender_group = sorted_out.groupby("from_account", sort=False)
    receiver_group = sorted_out.groupby("to_account", sort=False)

    sorted_out["sender_tx_count_prev"] = sender_group.cumcount().astype(np.int32)
    sorted_out["receiver_tx_count_prev"] = receiver_group.cumcount().astype(np.int32)

    sender_paid_exp = sender_group["Amount Paid"].expanding()
    sorted_out["sender_mean_paid_prev"] = (
        sender_paid_exp.mean().shift(1).reset_index(level=0, drop=True).fillna(0.0).astype(float)
    )
    sorted_out["sender_std_paid_prev"] = (
        sender_paid_exp.std().shift(1).reset_index(level=0, drop=True).fillna(0.0).astype(float)
    )

    receiver_recv_exp = receiver_group["Amount Received"].expanding()
    sorted_out["receiver_mean_received_prev"] = (
        receiver_recv_exp.mean().shift(1).reset_index(level=0, drop=True).fillna(0.0).astype(float)
    )

    prev_sender_ts = sender_group["__ts_int"].shift(1)
    prev_receiver_ts = receiver_group["__ts_int"].shift(1)
    sec = 1_000_000_000
    sorted_out["time_since_prev_sender_tx"] = (
        (sorted_out["__ts_int"] - prev_sender_ts).clip(lower=0).fillna(0) // sec
    ).astype(np.float64)
    sorted_out["time_since_prev_receiver_tx"] = (
        (sorted_out["__ts_int"] - prev_receiver_ts).clip(lower=0).fillna(0) // sec
    ).astype(np.float64)

    sender_is_new_pair = (~sorted_out.duplicated(subset=["from_account", "to_account"])).astype(np.int32)
    receiver_is_new_pair = (~sorted_out.duplicated(subset=["to_account", "from_account"])).astype(np.int32)
    sorted_out["sender_unique_receivers_prev"] = (
        sender_is_new_pair.groupby(sorted_out["from_account"]).cumsum() - sender_is_new_pair
    ).astype(np.float64)
    sorted_out["receiver_unique_senders_prev"] = (
        receiver_is_new_pair.groupby(sorted_out["to_account"]).cumsum() - receiver_is_new_pair
    ).astype(np.float64)

    out = sorted_out.sort_values("__orig_idx").drop(columns=["__orig_idx", "__ts", "__ts_int"])
    out = out.reset_index(drop=True)

    out = out.drop(columns=["Timestamp"])
    for col in CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype(str)
    return out


def X_y_from_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Разделение на матрицу признаков и цель; колонка цели убирается из X."""
    engineered = engineer_features(df)
    y = engineered["Is Laundering"].astype(int)
    X = engineered.drop(columns=["Is Laundering"])
    return X, y
