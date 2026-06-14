"""Загрузка сырых CSV IBM synthetic AML (формат как в ноутбуке first_steps)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from mlops_aml_transactions.storage.s3 import s3_download_if_missing, s3_key_for_local_path

RAW_TRANSACTION_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]


def load_raw_transaction_files(files: Sequence[str | Path]) -> pd.DataFrame:
    """Читает один или несколько CSV, объединяет в один DataFrame, парсит время и суммы."""
    frames: list[pd.DataFrame] = []

    for path in files:
        p = Path(path)
        if not p.is_file():
            # Если файла нет локально — пробуем скачать из S3 (если сконфигурировано).
            try:
                key = s3_key_for_local_path(p, kind="data")
                s3_download_if_missing(p, key)
            except Exception:
                # Если S3 не настроен/недоступен — продолжаем как раньше.
                pass
        if not p.is_file():
            logger.warning("File not found, skip: {}", p)
            continue

        logger.info("Reading {} ...", p)
        started_at = time.time()
        df = pd.read_csv(
            p,
            names=RAW_TRANSACTION_COLUMNS,
            header=0,
            dtype={
                "From Bank": str,
                "From Account": str,
                "To Bank": str,
                "To Account": str,
                "Payment Format": str,
                "Receiving Currency": str,
                "Payment Currency": str,
                "Is Laundering": np.int8,
            },
            low_memory=False,
        )

        elapsed = time.time() - started_at
        logger.info("  -> {:,} rows [{:.1f}s]", len(df), elapsed)
        frames.append(df)

    if not frames:
        raise ValueError("No input files were loaded. Check paths to raw CSV files.")

    data = pd.concat(frames, ignore_index=True)
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
    data["Amount Received"] = pd.to_numeric(data["Amount Received"], errors="coerce")
    data["Amount Paid"] = pd.to_numeric(data["Amount Paid"], errors="coerce")

    laundering_rate = data["Is Laundering"].mean() * 100
    logger.info("Total rows: {:,} | Laundering rate: {:.3f}%", len(data), laundering_rate)
    return data
