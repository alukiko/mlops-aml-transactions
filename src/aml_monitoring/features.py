import numpy as np
import pandas as pd

from .config import FEATURE_COLS
from .data import normalize_transactions

CATEGORICAL_COLS = ["Payment Format", "Receiving Currency", "Payment Currency", "From Bank", "To Bank"]
SENDER_AGG_COLS = [
    "sender_tx_count",
    "sender_total_paid",
    "sender_mean_paid",
    "sender_std_paid",
    "sender_unique_recv",
]
RECEIVER_AGG_COLS = [
    "recv_tx_count",
    "recv_total_received",
    "recv_mean_received",
    "recv_unique_sender",
]


def build_preprocessor(df: pd.DataFrame) -> dict:
    data = normalize_transactions(df)
    category_mappings = {}
    for col in CATEGORICAL_COLS:
        values = sorted(data[col].fillna("UNK").astype(str).unique().tolist())
        values = ["UNK"] + [value for value in values if value != "UNK"]
        category_mappings[col] = {value: idx for idx, value in enumerate(values)}

    sender_stats = compute_sender_stats(data)
    receiver_stats = compute_receiver_stats(data)
    return {
        "category_mappings": category_mappings,
        "sender_stats": sender_stats,
        "receiver_stats": receiver_stats,
        "sender_defaults": sender_stats[SENDER_AGG_COLS].mean(numeric_only=True).fillna(0).to_dict(),
        "receiver_defaults": receiver_stats[RECEIVER_AGG_COLS].mean(numeric_only=True).fillna(0).to_dict(),
        "feature_cols": FEATURE_COLS,
    }


def compute_sender_stats(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("From Account", dropna=False)
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


def compute_receiver_stats(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby("To Account", dropna=False)
        .agg(
            recv_tx_count=("Amount Received", "count"),
            recv_total_received=("Amount Received", "sum"),
            recv_mean_received=("Amount Received", "mean"),
            recv_unique_sender=("From Account", "nunique"),
        )
        .reset_index()
        .rename(columns={"To Account": "account_key"})
    )


def engineer_features(df: pd.DataFrame, preprocessor: dict | None = None) -> pd.DataFrame:
    data = normalize_transactions(df)
    timestamps = pd.to_datetime(data["Timestamp"], errors="coerce")
    data["hour"] = timestamps.dt.hour.fillna(0).astype(np.int8)
    data["day_of_week"] = timestamps.dt.dayofweek.fillna(0).astype(np.int8)
    data["day_of_month"] = timestamps.dt.day.fillna(1).astype(np.int8)
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(np.int8)
    data["is_night"] = ((data["hour"] < 6) | (data["hour"] >= 22)).astype(np.int8)

    data["amount_ratio"] = np.where(
        data["Amount Paid"] > 0,
        data["Amount Received"] / (data["Amount Paid"] + 1e-9),
        1.0,
    )
    data["amount_diff"] = data["Amount Received"] - data["Amount Paid"]
    data["log_amount_paid"] = np.log1p(data["Amount Paid"].clip(lower=0))
    data["log_amount_recv"] = np.log1p(data["Amount Received"].clip(lower=0))
    data["is_round_amount"] = (data["Amount Paid"] % 1000 == 0).astype(np.int8)

    data["currency_mismatch"] = (data["Receiving Currency"] != data["Payment Currency"]).astype(np.int8)
    data["same_bank"] = (data["From Bank"] == data["To Bank"]).astype(np.int8)
    data["self_transfer"] = (data["From Account"] == data["To Account"]).astype(np.int8)

    if preprocessor:
        sender_stats = preprocessor["sender_stats"]
        receiver_stats = preprocessor["receiver_stats"]
        sender_defaults = preprocessor["sender_defaults"]
        receiver_defaults = preprocessor["receiver_defaults"]
    else:
        sender_stats = compute_sender_stats(data)
        receiver_stats = compute_receiver_stats(data)
        sender_defaults = sender_stats[SENDER_AGG_COLS].mean(numeric_only=True).fillna(0).to_dict()
        receiver_defaults = receiver_stats[RECEIVER_AGG_COLS].mean(numeric_only=True).fillna(0).to_dict()

    data = data.merge(sender_stats, left_on="From Account", right_on="account_key", how="left").drop(columns="account_key")
    data = data.merge(receiver_stats, left_on="To Account", right_on="account_key", how="left").drop(columns="account_key")
    for col in SENDER_AGG_COLS:
        data[col] = data[col].fillna(sender_defaults.get(col, 0))
    for col in RECEIVER_AGG_COLS:
        data[col] = data[col].fillna(receiver_defaults.get(col, 0))

    data["fanout_ratio"] = data["sender_unique_recv"] / (data["sender_tx_count"] + 1)
    data["fanin_ratio"] = data["recv_unique_sender"] / (data["recv_tx_count"] + 1)
    data["sender_std_paid"] = data["sender_std_paid"].fillna(0)

    for col in CATEGORICAL_COLS:
        values = data[col].fillna("UNK").astype(str)
        if preprocessor:
            mapping = preprocessor["category_mappings"][col]
        else:
            categories = ["UNK"] + sorted(value for value in values.unique().tolist() if value != "UNK")
            mapping = {value: idx for idx, value in enumerate(categories)}
        data[f"{col}_enc"] = values.map(lambda value: mapping.get(value, mapping["UNK"])).astype(np.int32)

    return data


def to_feature_matrix(df: pd.DataFrame, preprocessor: dict | None = None) -> pd.DataFrame:
    features = engineer_features(df, preprocessor=preprocessor)
    for column in FEATURE_COLS:
        if column not in features.columns:
            features[column] = 0
    return features[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
