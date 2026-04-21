from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
import typer

from mlops_aml_transactions.config import DEFAULT_RAW_CSV, PROCESSED_DATA_DIR, RAW_DATA_DIR
from mlops_aml_transactions.data.s3 import s3_download_if_missing, s3_key_for_local_path
from mlops_aml_transactions.features import RAW_COLUMNS

app = typer.Typer()

DEFAULT_READ_CAP = 1_000_000
DEFAULT_SAMPLE_SIZE = 500_000
DEFAULT_MIN_POSITIVE_ROWS = 500
TARGET_COLUMN = "Is Laundering"
DEFAULT_INPUT_FILES = [
    "HI-Small_Trans.csv",
    "LI-Small_Trans.csv",
    "HI-Medium_Trans.csv",
    "LI-Medium_Trans.csv",
]


def stratified_subsample(df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    """Stratified random sample with fallback to standard random sample."""
    sample_size = min(sample_size, len(df))
    if sample_size == len(df):
        return df
    try:
        out, _ = train_test_split(
            df,
            train_size=sample_size,
            stratify=df[TARGET_COLUMN],
            random_state=random_state,
        )
        return out
    except ValueError:
        logger.warning("Stratified sampling failed; using standard random sample.")
        return df.sample(n=sample_size, random_state=random_state)


def ensure_min_positive_rows(
    full_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    *,
    min_positive_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Ensure at least min_positive_rows positives while keeping output size unchanged."""
    if min_positive_rows <= 0 or sampled_df.empty:
        return sampled_df

    target = min(min_positive_rows, len(sampled_df))
    current_pos = int((sampled_df[TARGET_COLUMN] == 1).sum())
    if current_pos >= target:
        return sampled_df

    need = target - current_pos
    sampled_neg = sampled_df[sampled_df[TARGET_COLUMN] == 0]
    if sampled_neg.empty:
        return sampled_df

    full_pos = full_df[full_df[TARGET_COLUMN] == 1]
    if full_pos.empty:
        return sampled_df

    sampled_idx = set(sampled_df.index)
    extra_pos = full_pos.loc[~full_pos.index.isin(sampled_idx)]
    if extra_pos.empty:
        extra_pos = full_pos

    replace_pos = len(extra_pos) < need
    injected_pos = extra_pos.sample(n=need, random_state=random_state, replace=replace_pos)
    dropped_neg = sampled_neg.sample(n=need, random_state=random_state, replace=False)

    out = sampled_df.drop(index=dropped_neg.index)
    out = pd.concat([out, injected_pos], axis=0, ignore_index=False)
    return out.sample(frac=1.0, random_state=random_state)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if len(df.columns) != len(RAW_COLUMNS):
        raise ValueError(f"Expected {len(RAW_COLUMNS)} columns (got {len(df.columns)}); check CSV format.")
    out = df.copy()
    out.columns = RAW_COLUMNS
    return out


def parse_csv_input_files(input_path: Path, input_files: str) -> list[Path]:
    if not input_files.strip():
        return [input_path]

    paths: list[Path] = []
    for token in [p.strip() for p in input_files.split(",") if p.strip()]:
        p = Path(token)
        if not p.is_absolute():
            p = RAW_DATA_DIR / p
        paths.append(p)
    return paths


def parse_pattern_row_ids(pattern_path: Path) -> set[int]:
    if not pattern_path.is_file():
        return set()
    row_ids: set[int] = set()
    for line in pattern_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        nums = re.findall(r"\d+", line)
        if not nums:
            continue
        # Kaggle pattern files usually contain transaction index references.
        idx = int(nums[0]) - 1
        if idx >= 0:
            row_ids.add(idx)
    return row_ids


def read_with_chunking(
    input_path: Path,
    *,
    read_cap: int,
    sample_size: int,
    min_positive_rows: int,
    chunksize: int,
) -> pd.DataFrame:
    """Read CSV progressively until enough rows and positives are collected."""
    read_rows = 0
    positives = 0
    chunks: list[pd.DataFrame] = []

    for raw_chunk in pd.read_csv(input_path, chunksize=chunksize):
        remaining = read_cap - read_rows
        if remaining <= 0:
            break

        chunk = raw_chunk.head(remaining)
        chunk = _normalize_columns(chunk)
        chunk = chunk.reset_index(drop=True)
        chunk["__source_file"] = input_path.name
        chunk["__source_row_id"] = range(read_rows, read_rows + len(chunk))
        chunks.append(chunk)

        read_rows += len(chunk)
        positives += int((chunk[TARGET_COLUMN] == 1).sum())

        enough_rows = read_rows >= sample_size
        enough_pos = positives >= min_positive_rows
        if enough_rows and enough_pos:
            break

    if not chunks:
        raise ValueError(f"No rows loaded from {input_path}")

    df = pd.concat(chunks, axis=0, ignore_index=True)
    pattern_ids = parse_pattern_row_ids(input_path.with_name(input_path.name.replace("_Trans.csv", "_Patterns.txt")))
    if pattern_ids:
        df["in_known_pattern"] = df["__source_row_id"].isin(pattern_ids).astype(int)
    else:
        df["in_known_pattern"] = 0
    return df


def download_if_missing(path: Path) -> None:
    if path.is_file():
        return
    key = s3_key_for_local_path(path, kind="data")
    s3_download_if_missing(path, key)


@app.command()
def main(
    input_path: Path = DEFAULT_RAW_CSV,
    input_files: str = typer.Option(
        ",".join(DEFAULT_INPUT_FILES),
        help="Comma-separated raw CSV files (relative to data/raw or absolute paths).",
    ),
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    sample_size: int = typer.Option(DEFAULT_SAMPLE_SIZE, help="Rows to keep after stratified sampling."),
    read_cap: int = typer.Option(
        DEFAULT_READ_CAP,
        help="Max rows read from all CSV files before sampling.",
    ),
    min_positive_rows: int = typer.Option(
        DEFAULT_MIN_POSITIVE_ROWS,
        help="Minimum positive rows in output; set 0 for fully backward-compatible behavior.",
    ),
    chunksize: int = typer.Option(200_000, help="Chunk size for progressive CSV reading."),
    random_state: int = 42,
) -> None:
    """Build processed dataset.csv from one or many raw IBM AML CSV files."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_paths = parse_csv_input_files(input_path, input_files)
    for p in all_paths:
        download_if_missing(p)

    existing_paths = [p for p in all_paths if p.is_file()]
    missing_paths = [p for p in all_paths if not p.is_file()]
    for p in missing_paths:
        logger.warning("Input file not found (skipped): {}", p)

    if not existing_paths:
        raise typer.BadParameter("No input files found after local/S3 checks.")

    if min_positive_rows < 0:
        raise typer.BadParameter("min_positive_rows must be >= 0")
    if chunksize <= 0:
        raise typer.BadParameter("chunksize must be > 0")

    logger.info(
        "Reading {} files with cap={} sample_size={} min_positive_rows={} chunksize={} ...",
        len(existing_paths),
        read_cap,
        sample_size,
        min_positive_rows,
        chunksize,
    )

    per_file_cap = max(1, read_cap // len(existing_paths))
    per_file_sample = max(1, sample_size // len(existing_paths))
    per_file_pos = 0 if min_positive_rows == 0 else max(1, min_positive_rows // len(existing_paths))

    frames: list[pd.DataFrame] = []
    for p in existing_paths:
        logger.info("Loading source {} ...", p)
        if min_positive_rows == 0:
            raw_df = pd.read_csv(p, nrows=per_file_cap)
            f = _normalize_columns(raw_df)
            f["__source_file"] = p.name
            f["__source_row_id"] = range(len(f))
            f["in_known_pattern"] = 0
        else:
            f = read_with_chunking(
                p,
                read_cap=per_file_cap,
                sample_size=per_file_sample,
                min_positive_rows=per_file_pos,
                chunksize=chunksize,
            )
        frames.append(f)

    df = pd.concat(frames, axis=0, ignore_index=True)
    logger.info("Loaded {} rows from {} files; subsampling to {} ...", len(df), len(existing_paths), sample_size)

    sampled = stratified_subsample(df, sample_size, random_state)
    sampled = ensure_min_positive_rows(
        df,
        sampled,
        min_positive_rows=min_positive_rows,
        random_state=random_state,
    )

    sampled = sampled.drop(columns=["__source_file", "__source_row_id"], errors="ignore")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)

    pos = int((sampled[TARGET_COLUMN] == 1).sum())
    patt = int(sampled.get("in_known_pattern", pd.Series(dtype=int)).sum())
    logger.success(
        "Wrote {} rows (positive: {}, {:.4%}; known-pattern rows: {}) to {}",
        len(sampled),
        pos,
        pos / len(sampled) if len(sampled) else 0,
        patt,
        output_path,
    )


if __name__ == "__main__":
    app()
