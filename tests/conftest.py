"""Общие фикстуры для тестов."""

from __future__ import annotations

from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from mlops_aml_transactions.api.main import app  # noqa: E402
from mlops_aml_transactions.config import MODELS_DIR  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    """Скачать модель с S3 перед запуском тестов, если её нет локально."""
    model_path = MODELS_DIR / "model.pkl"
    if model_path.is_file():
        return
    try:
        from aml_monitoring.s3_data import _download_files, s3_credentials_configured  # noqa: PLC0415
        from aml_monitoring.config import MODEL_FILES, S3_MODELS_PREFIX  # noqa: PLC0415
        if not s3_credentials_configured():
            return
        print("\n[conftest] model.pkl not found — downloading from S3...")
        _download_files(MODEL_FILES, S3_MODELS_PREFIX, required=False)
    except Exception as exc:
        print(f"\n[conftest] S3 model download skipped: {exc}")


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_tx() -> dict:
    return {
        "Timestamp": "2023-01-15 14:30:00",
        "From Bank": "bank_a",
        "To Bank": "bank_b",
        "from_account": "acc_1",
        "to_account": "acc_2",
        "Amount Received": 1500.0,
        "Receiving Currency": "USD",
        "Amount Paid": 1500.0,
        "Payment Currency": "USD",
        "Payment Format": "wire",
    }
