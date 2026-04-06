"""Тесты HTTP API (FastAPI)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlops_aml_transactions.api import main as api_main
from mlops_aml_transactions.config import MODELS_DIR


def test_health_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_docs_page_ok(client: TestClient) -> None:
    r = client.get("/docs")
    assert r.status_code == 200
    body = r.text.lower()
    assert "swagger" in body or "openapi" in body


def test_openapi_json_has_endpoints(client: TestClient) -> None:
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()
    paths = spec.get("paths", {})
    assert "/health" in paths
    assert "/predict" in paths
    assert "/predict/batch" in paths


def test_predict_malformed_json_returns_422(client: TestClient) -> None:
    r = client.post(
        "/predict",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


def test_predict_missing_required_field_returns_422(client: TestClient) -> None:
    incomplete = {"Timestamp": "2023-01-15 14:30:00"}
    r = client.post("/predict", json=incomplete)
    assert r.status_code == 422
    detail = r.json().get("detail")
    assert detail is not None


def test_predict_batch_missing_transactions_key_returns_422(client: TestClient) -> None:
    r = client.post("/predict/batch", json={})
    assert r.status_code == 422


@pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").is_file(),
    reason="models/model.pkl отсутствует — пропуск интеграционных тестов предсказания",
)
def test_predict_one_returns_proba_and_pred(client: TestClient, sample_tx: dict) -> None:
    r = client.post("/predict", json=sample_tx)
    assert r.status_code == 200
    data = r.json()
    assert "proba_laundering" in data
    assert "pred_laundering" in data
    assert isinstance(data["proba_laundering"], (int, float))
    assert data["pred_laundering"] in (0, 1)


@pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").is_file(),
    reason="models/model.pkl отсутствует",
)
def test_predict_batch_two_rows(client: TestClient, sample_tx: dict) -> None:
    body = {"transactions": [sample_tx, sample_tx]}
    r = client.post("/predict/batch", json=body)
    assert r.status_code == 200
    out = r.json()
    assert len(out) == 2
    for row in out:
        assert 0.0 <= row["proba_laundering"] <= 1.0
        assert row["pred_laundering"] in (0, 1)


@pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").is_file(),
    reason="models/model.pkl отсутствует",
)
def test_predict_example_class_0(client: TestClient, sample_tx_pred_class_0: dict) -> None:
    r = client.post("/predict", json=sample_tx_pred_class_0)
    assert r.status_code == 200
    data = r.json()
    assert data["pred_laundering"] == 0
    assert 0.0 <= data["proba_laundering"] <= 1.0


@pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").is_file(),
    reason="models/model.pkl отсутствует",
)
def test_predict_example_class_1(client: TestClient, sample_tx_pred_class_1: dict) -> None:
    r = client.post("/predict", json=sample_tx_pred_class_1)
    assert r.status_code == 200
    data = r.json()
    assert data["pred_laundering"] == 1
    assert 0.0 <= data["proba_laundering"] <= 1.0


@pytest.mark.skipif(
    not (MODELS_DIR / "model.pkl").is_file(),
    reason="models/model.pkl отсутствует",
)
def test_predict_batch_mixed_classes(
    client: TestClient, sample_tx_pred_class_0: dict, sample_tx_pred_class_1: dict
) -> None:
    r = client.post(
        "/predict/batch",
        json={"transactions": [sample_tx_pred_class_0, sample_tx_pred_class_1]},
    )
    assert r.status_code == 200
    out = r.json()
    assert out[0]["pred_laundering"] == 0
    assert out[1]["pred_laundering"] == 1


def test_predict_503_when_model_missing(
    client: TestClient, sample_tx: dict, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = tmp_path / "no_model_here.pkl"
    monkeypatch.setattr(api_main, "MODEL_PATH", fake)
    api_main.load_model_cached.cache_clear()
    try:
        r = client.post("/predict", json=sample_tx)
        assert r.status_code == 503
        assert "detail" in r.json()
    finally:
        api_main.load_model_cached.cache_clear()
