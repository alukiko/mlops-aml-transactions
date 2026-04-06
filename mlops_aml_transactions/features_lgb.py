"""Признаки и порог для LightGBM-пайплайна из ноутбука (отдельно от sklearn pipeline в features.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLS_LGB = [
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_night",
    "amount_ratio",
    "amount_diff",
    "log_amount_paid",
    "log_amount_recv",
    "is_round_amount",
    "currency_mismatch",
    "same_bank",
    "self_transfer",
    "sender_tx_count",
    "sender_total_paid",
    "sender_mean_paid",
    "sender_std_paid",
    "sender_unique_recv",
    "fanout_ratio",
    "recv_tx_count",
    "recv_total_received",
    "recv_mean_received",
    "recv_unique_sender",
    "fanin_ratio",
    "Payment Format_enc",
    "Receiving Currency_enc",
    "Payment Currency_enc",
    "From Bank_enc",
    "To Bank_enc",
]

TARGET_LGB = "Is Laundering"


def engineer_features_lgb(df: pd.DataFrame) -> pd.DataFrame:
    """Временные признаки, суммы, агрегаты по счетам, label encoding категорий."""
    logger.info("Engineering features (LGB pipeline)...")
    data = df.copy()

    data["hour"] = data["Timestamp"].dt.hour.astype(np.int8)
    data["day_of_week"] = data["Timestamp"].dt.dayofweek.astype(np.int8)
    data["day_of_month"] = data["Timestamp"].dt.day.astype(np.int8)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(np.int8)
    data["is_night"] = ((data["hour"] < 6) | (data["hour"] >= 22)).astype(np.int8)

    data["Amount Received"] = pd.to_numeric(data["Amount Received"], errors="coerce").fillna(0)
    data["Amount Paid"] = pd.to_numeric(data["Amount Paid"], errors="coerce").fillna(0)
    data["amount_ratio"] = np.where(
        data["Amount Paid"] > 0,
        data["Amount Received"] / (data["Amount Paid"] + 1e-9),
        1.0,
    )
    data["amount_diff"] = data["Amount Received"] - data["Amount Paid"]
    data["log_amount_paid"] = np.log1p(data["Amount Paid"])
    data["log_amount_recv"] = np.log1p(data["Amount Received"])
    data["is_round_amount"] = (data["Amount Paid"] % 1000 == 0).astype(np.int8)

    data["currency_mismatch"] = (data["Receiving Currency"] != data["Payment Currency"]).astype(
        np.int8
    )
    data["same_bank"] = (data["From Bank"] == data["To Bank"]).astype(np.int8)
    data["self_transfer"] = (data["From Account"] == data["To Account"]).astype(np.int8)

    logger.info("  Computing sender aggregates...")
    sender_stats = (
        data.groupby("From Account")
        .agg(
            sender_tx_count=("Amount Paid", "count"),
            sender_total_paid=("Amount Paid", "sum"),
            sender_mean_paid=("Amount Paid", "mean"),
            sender_std_paid=("Amount Paid", "std"),
            sender_unique_recv=("To Account", "nunique"),
        )
        .reset_index()
        .rename(columns={"From Account": "account_key"})
    )

    logger.info("  Computing receiver aggregates...")
    receiver_stats = (
        data.groupby("To Account")
        .agg(
            recv_tx_count=("Amount Received", "count"),
            recv_total_received=("Amount Received", "sum"),
            recv_mean_received=("Amount Received", "mean"),
            recv_unique_sender=("From Account", "nunique"),
        )
        .reset_index()
        .rename(columns={"To Account": "account_key"})
    )

    data = data.merge(sender_stats, left_on="From Account", right_on="account_key", how="left").drop(
        columns="account_key"
    )
    data = data.merge(receiver_stats, left_on="To Account", right_on="account_key", how="left").drop(
        columns="account_key"
    )

    data["fanout_ratio"] = data["sender_unique_recv"] / (data["sender_tx_count"] + 1)
    data["fanin_ratio"] = data["recv_unique_sender"] / (data["recv_tx_count"] + 1)
    data["sender_std_paid"] = data["sender_std_paid"].fillna(0)

    categorical_cols = [
        "Payment Format",
        "Receiving Currency",
        "Payment Currency",
        "From Bank",
        "To Bank",
    ]
    for col in categorical_cols:
        encoder = LabelEncoder()
        data[f"{col}_enc"] = encoder.fit_transform(data[col].fillna("UNK").astype(str))

    logger.info("  Features ready. Shape: {}", data.shape)
    return data


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray, beta: float = 1.0) -> float:
    """Порог, максимизирующий F_beta по OOF-вероятностям."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_threshold = 0.5
    best_score = 0.0

    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)

        if precision + recall == 0:
            continue

        fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        if fb > best_score:
            best_score = fb
            best_threshold = threshold

    return float(best_threshold)
