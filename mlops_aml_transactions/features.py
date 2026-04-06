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
