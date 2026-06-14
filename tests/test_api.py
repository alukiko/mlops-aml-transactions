"""Тесты HTTP API (FastAPI)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlops_aml_transactions.api import main as api_main


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
