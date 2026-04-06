from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import typer

from mlops_aml_transactions.config import DEFAULT_RAW_CSV, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_aml_transactions.features import RAW_COLUMNS

app = typer.Typer()

# Сколько строк читать с диска до стратифицированной выборки (полный файл — миллионы строк)
DEFAULT_READ_CAP = 1_000_000


def stratified_subsample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Стратифицированная случайная подвыборка; при ошибке — обычная случайная."""
    sample_size = min(sample_size, len(df))
    if sample_size == len(df):
        return df
    try:
        out, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df["Is Laundering"],
            random_state=random_state,
        )
        return out
    except ValueError:
        logger.warning("Стратифицированная выборка не удалась; используется случайная.")
        return df.sample(n=sample_size, random_state=random_state)


@app.command()
def main(
    input_path: Path = DEFAULT_RAW_CSV,
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    sample_size: int = typer.Option(200_000, help="Rows to keep after stratified sampling."),
    read_cap: int = typer.Option(
        DEFAULT_READ_CAP,
        help="Max rows read from CSV before sampling (reduces memory/time).",
    ),
    random_state: int = 42,
) -> None:
    """Сборка обработанного dataset.csv из сырого CSV IBM AML."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not input_path.is_file():
        raise typer.BadParameter(f"Input file not found: {input_path}")

    logger.info("Reading up to {} rows from {} ...", read_cap, input_path)
    df = pd.read_csv(input_path, nrows=read_cap)
    if len(df.columns) != len(RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(RAW_COLUMNS)} columns (got {len(df.columns)}); check CSV format."
        )
    df.columns = RAW_COLUMNS
    logger.info("Loaded {} rows; subsampling to {} (stratified) ...", len(df), sample_size)
    df = stratified_subsample(df, sample_size, random_state)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    pos = int((df["Is Laundering"] == 1).sum())
    logger.success(
        "Wrote {} rows (positive class: {}, {:.4%}) to {}",
        len(df),
        pos,
        pos / len(df) if len(df) else 0,
        output_path,
    )


if __name__ == "__main__":
    app()
