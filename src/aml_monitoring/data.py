import os
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_COLUMNS


def load_transactions(files: list[str | Path], nrows: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = nrows
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            continue
        read_nrows = remaining if remaining is not None else None
        df = pd.read_csv(
            path,
            names=RAW_COLUMNS,
            header=0,
            nrows=read_nrows,
            dtype={
                "From Bank": str,
                "From Account": str,
                "To Bank": str,
                "To Account": str,
                "Payment Format": str,
                "Receiving Currency": str,
                "Payment Currency": str,
            },
            low_memory=False,
        )
        frames.append(normalize_transactions(df))
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break
    if not frames:
        raise FileNotFoundError("No transaction CSV files were loaded")
    return pd.concat(frames, ignore_index=True)


def normalize_transactions(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    if list(data.columns).count("Account") == 2:
        data.columns = RAW_COLUMNS
    if "Account" in data.columns and "Account.1" in data.columns:
        data = data.rename(columns={"Account": "From Account", "Account.1": "To Account"})
    for column in RAW_COLUMNS:
        if column not in data.columns and column != "Is Laundering":
            data[column] = np.nan
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    data["Amount Received"] = pd.to_numeric(data["Amount Received"], errors="coerce").fillna(0.0)
    data["Amount Paid"] = pd.to_numeric(data["Amount Paid"], errors="coerce").fillna(0.0)
    if "Is Laundering" in data.columns:
        data["Is Laundering"] = pd.to_numeric(data["Is Laundering"], errors="coerce")
    return data


def sample_reference(files: list[str | Path], output_path: Path, nrows: int = 50000) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_transactions(files, nrows=nrows)
    data.to_csv(output_path, index=False)
    return data


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default
