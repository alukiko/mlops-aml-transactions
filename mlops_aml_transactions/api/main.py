from __future__ import annotations

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from mlops_aml_transactions.config import MODELS_DIR
from mlops_aml_transactions.features import engineer_features
from mlops_aml_transactions.modeling.artifacts import load_model
from mlops_aml_transactions.storage.s3 import s3_download_if_missing, s3_key_for_local_path

MODEL_PATH = MODELS_DIR / "model.pkl"

app = FastAPI(title="AML transaction scoring", version="0.1.0")


class TransactionIn(BaseModel):
    """Одна строка как в обработанном датасете; имена полей — в features.RAW_COLUMNS, в JSON — с alias."""

    model_config = ConfigDict(populate_by_name=True)

    Timestamp: str
    From_Bank: str = Field(alias="From Bank")
    To_Bank: str = Field(alias="To Bank")
    from_account: str
    to_account: str
    Amount_Received: float = Field(alias="Amount Received")
    Receiving_Currency: str = Field(alias="Receiving Currency")
    Amount_Paid: float = Field(alias="Amount Paid")
    Payment_Currency: str = Field(alias="Payment Currency")
    Payment_Format: str = Field(alias="Payment Format")


class PredictOut(BaseModel):
    proba_laundering: float
    pred_laundering: int


class BatchIn(BaseModel):
    transactions: list[TransactionIn]


@lru_cache(maxsize=1)
def load_model_cached():
    if not MODEL_PATH.is_file():
        # If missing locally, try S3 (if configured).
        try:
            key = s3_key_for_local_path(MODEL_PATH, kind="models")
            s3_download_if_missing(MODEL_PATH, key)
        except Exception:
            pass
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training first (or configure S3 download)."
        )
    return load_model(MODEL_PATH)


@app.on_event("startup")
def startup_load_model() -> None:
    try:
        load_model_cached()
    except FileNotFoundError:
        pass


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictOut)
def predict_one(tx: TransactionIn) -> PredictOut:
    out = predict_batch(BatchIn(transactions=[tx]))
    return out[0]


@app.post("/predict/batch", response_model=list[PredictOut])
def predict_batch(body: BatchIn) -> list[PredictOut]:
    try:
        pipeline, threshold = load_model_cached()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    df = pd.DataFrame([t.model_dump(by_alias=True) for t in body.transactions])
    X = engineer_features(df)
    proba = pipeline.predict_proba(X)[:, 1]
    if threshold is not None:
        pred = (proba >= threshold).astype(int)
    else:
        pred = pipeline.predict(X)
    return [
        PredictOut(proba_laundering=float(p), pred_laundering=int(lbl)) for p, lbl in zip(proba, pred)
    ]
