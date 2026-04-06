"""EDA-графики Plotly для сырых AML-транзакций (ноутбук first_steps)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure


def data_overview(df: pd.DataFrame) -> dict:
    """Краткая сводка по датасету."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "laundering_rate": float(df["Is Laundering"].mean()),
        "start_ts": df["Timestamp"].min(),
        "end_ts": df["Timestamp"].max(),
    }


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Типы, пропуски и число уникальных значений по колонкам."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "unique_values": df.nunique(dropna=False),
        }
    )


def plot_class_balance(df: pd.DataFrame) -> Figure:
    """Столбчатый график баланса классов (ось Y — log)."""
    class_df = pd.DataFrame(
        {
            "class": ["Normal", "Laundering"],
            "count": [
                int((df["Is Laundering"] == 0).sum()),
                int((df["Is Laundering"] == 1).sum()),
            ],
        }
    )
    fig = px.bar(class_df, x="class", y="count", color="class", text="count", title="Class Balance")
    fig.update_yaxes(type="log")
    return fig


def plot_daily_laundering_rate(df: pd.DataFrame) -> Figure:
    """Доля подозрительных транзакций по дням."""
    daily_stats = (
        df.assign(date=df["Timestamp"].dt.floor("D"))
        .groupby("date")
        .agg(
            transactions=("Is Laundering", "size"),
            laundering_rate=("Is Laundering", "mean"),
        )
        .reset_index()
    )
    return px.line(daily_stats, x="date", y="laundering_rate", title="Daily Laundering Rate")


def plot_hourly_laundering_rate(df: pd.DataFrame) -> Figure:
    """Доля подозрительных транзакций по часу суток."""
    hourly_stats = (
        df.groupby(df["Timestamp"].dt.hour.rename("hour"))
        .agg(
            transactions=("Is Laundering", "size"),
            laundering_rate=("Is Laundering", "mean"),
        )
        .reset_index()
    )
    return px.line(
        hourly_stats,
        x="hour",
        y="laundering_rate",
        markers=True,
        title="Hourly Laundering Rate",
    )


def plot_log_amount_paid_distribution(df: pd.DataFrame) -> Figure:
    """Гистограмма log(Amount Paid + 1) для Normal vs Laundering."""
    amount_plot = df[["Amount Paid", "Is Laundering"]].dropna().copy()
    amount_plot["log_amount_paid"] = np.log1p(amount_plot["Amount Paid"].clip(lower=0))
    amount_plot["label"] = amount_plot["Is Laundering"].map({0: "Normal", 1: "Laundering"})
    return px.histogram(
        amount_plot,
        x="log_amount_paid",
        color="label",
        barmode="overlay",
        histnorm="probability density",
        nbins=60,
        title="Distribution of log(Amount Paid + 1)",
    )


def payment_format_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по способу платежа."""
    return (
        df.groupby("Payment Format")
        .agg(
            rows=("Is Laundering", "size"),
            laundering_rate=("Is Laundering", "mean"),
        )
        .sort_values(["laundering_rate", "rows"], ascending=[False, False])
        .reset_index()
    )


def payment_currency_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Агрегаты по валюте платежа."""
    return (
        df.groupby("Payment Currency")
        .agg(
            rows=("Is Laundering", "size"),
            laundering_rate=("Is Laundering", "mean"),
        )
        .sort_values(["laundering_rate", "rows"], ascending=[False, False])
        .reset_index()
    )
