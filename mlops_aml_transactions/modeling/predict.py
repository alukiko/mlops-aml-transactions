from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger
import typer

from mlops_aml_transactions.config import MODELS_DIR, PROCESSED_DATA_DIR
from mlops_aml_transactions.features import engineer_features
from mlops_aml_transactions.modeling.artifacts import load_model
from mlops_aml_transactions.storage.s3 import s3_download_if_missing, s3_key_for_local_path

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    max_rows: int = typer.Option(
        10_000,
        help="Сколько строк скорить (полный файл может быть большим).",
    ),
) -> None:
    """Пакетный скоринг: предсказания в CSV."""
    if not model_path.is_file():
        # If missing locally, try S3 (if configured).
        try:
            key = s3_key_for_local_path(model_path, kind="models")
            s3_download_if_missing(model_path, key)
        except Exception:
            pass
    if not model_path.is_file():
        raise typer.BadParameter(f"Модель не найдена: {model_path}. Сначала обучите модель.")
    if not features_path.is_file():
        raise typer.BadParameter(f"Файл признаков не найден: {features_path}")

    pipeline, threshold = load_model(model_path)
    df = pd.read_csv(features_path, nrows=max_rows)
    if "Is Laundering" in df.columns:
        y_true = df["Is Laundering"].astype(int)
        raw = df.drop(columns=["Is Laundering"])
    else:
        y_true = None
        raw = df

    X = engineer_features(raw)
    proba = pipeline.predict_proba(X)[:, 1]
    if threshold is not None:
        pred = (proba >= threshold).astype(int)
    else:
        pred = pipeline.predict(X)

    out = pd.DataFrame({"proba_laundering": proba, "pred_laundering": pred})
    if threshold is not None:
        out["decision_threshold"] = threshold
    if y_true is not None:
        out["is_laundering_true"] = y_true.values
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(predictions_path, index=False)
    logger.success("Wrote {} rows to {}", len(out), predictions_path)


if __name__ == "__main__":
    app()
