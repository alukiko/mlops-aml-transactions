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

# Columns expected by the trained sklearn pipeline (ColumnTransformer)
_PIPELINE_COLS = [
    "From Bank", "from_account", "To Bank", "to_account",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency", "Payment Format",
    "in_known_pattern", "hour", "dayofweek", "day", "is_weekend",
    "log_amt_received", "log_amt_paid", "amount_abs_diff", "same_currency",
    "sender_tx_count_prev", "receiver_tx_count_prev",
    "sender_mean_paid_prev", "sender_std_paid_prev",
    "receiver_mean_received_prev", "time_since_prev_sender_tx",
    "time_since_prev_receiver_tx", "sender_unique_receivers_prev",
    "receiver_unique_senders_prev",
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


def _compute_prev_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-account rolling stats from previous transactions in the batch."""
    df = df.copy()
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["_ts"] = ts
    df["_order"] = range(len(df))

    prev_cols = [
        "sender_tx_count_prev", "sender_mean_paid_prev", "sender_std_paid_prev",
        "sender_unique_receivers_prev", "time_since_prev_sender_tx",
        "receiver_tx_count_prev", "receiver_mean_received_prev",
        "receiver_unique_senders_prev", "time_since_prev_receiver_tx",
    ]
    for col in prev_cols:
        df[col] = 0.0

    for i in range(1, len(df)):
        sender = df.at[df.index[i], "from_account"]
        recv = df.at[df.index[i], "to_account"]
        cur_ts = df.at[df.index[i], "_ts"]

        prev_s = df.iloc[:i][df.iloc[:i]["from_account"] == sender]
        if len(prev_s) > 0:
            df.at[df.index[i], "sender_tx_count_prev"] = len(prev_s)
            df.at[df.index[i], "sender_mean_paid_prev"] = prev_s["Amount Paid"].mean()
            df.at[df.index[i], "sender_std_paid_prev"] = prev_s["Amount Paid"].std(ddof=0)
            df.at[df.index[i], "sender_unique_receivers_prev"] = prev_s["to_account"].nunique()
            last_ts = prev_s["_ts"].iloc[-1]
            if pd.notna(last_ts) and pd.notna(cur_ts):
                df.at[df.index[i], "time_since_prev_sender_tx"] = max(0.0, (cur_ts - last_ts).total_seconds() / 60)

        prev_r = df.iloc[:i][df.iloc[:i]["to_account"] == recv]
        if len(prev_r) > 0:
            df.at[df.index[i], "receiver_tx_count_prev"] = len(prev_r)
            df.at[df.index[i], "receiver_mean_received_prev"] = prev_r["Amount Received"].mean()
            df.at[df.index[i], "receiver_unique_senders_prev"] = prev_r["from_account"].nunique()
            last_ts = prev_r["_ts"].iloc[-1]
            if pd.notna(last_ts) and pd.notna(cur_ts):
                df.at[df.index[i], "time_since_prev_receiver_tx"] = max(0.0, (cur_ts - last_ts).total_seconds() / 60)

    return df.drop(columns=["_ts", "_order"])


def to_feature_matrix(df: pd.DataFrame, preprocessor: dict | None = None) -> pd.DataFrame:
    """Build the feature matrix expected by the trained sklearn pipeline."""
    data = df.copy()

    # Normalize account and bank column names; pipeline OrdinalEncoder expects strings
    if "From Account" in data.columns:
        data["from_account"] = data["From Account"].fillna("UNK").astype(str)
    if "To Account" in data.columns:
        data["to_account"] = data["To Account"].fillna("UNK").astype(str)
    data["from_account"] = data.get("from_account", pd.Series(["UNK"] * len(data), index=data.index)).fillna("UNK").astype(str)
    data["to_account"] = data.get("to_account", pd.Series(["UNK"] * len(data), index=data.index)).fillna("UNK").astype(str)
    data["From Bank"] = data["From Bank"].fillna("UNK").astype(str)
    data["To Bank"] = data["To Bank"].fillna("UNK").astype(str)

    # Temporal features
    ts = pd.to_datetime(data["Timestamp"], errors="coerce")
    data["hour"] = ts.dt.hour.fillna(0).astype(int)
    data["dayofweek"] = ts.dt.dayofweek.fillna(0).astype(int)
    data["day"] = ts.dt.day.fillna(1).astype(int)
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    # Amount features
    data["Amount Received"] = pd.to_numeric(data["Amount Received"], errors="coerce").fillna(0.0)
    data["Amount Paid"] = pd.to_numeric(data["Amount Paid"], errors="coerce").fillna(0.0)
    data["log_amt_received"] = np.log1p(data["Amount Received"].clip(lower=0))
    data["log_amt_paid"] = np.log1p(data["Amount Paid"].clip(lower=0))
    data["amount_abs_diff"] = (data["Amount Received"] - data["Amount Paid"]).abs()
    data["same_currency"] = (data["Receiving Currency"] == data["Payment Currency"]).astype(int)
    data["in_known_pattern"] = 0

    # Per-account rolling features from batch history
    data = _compute_prev_features(data)

    for col in _PIPELINE_COLS:
        if col not in data.columns:
            data[col] = 0

    return data[_PIPELINE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
