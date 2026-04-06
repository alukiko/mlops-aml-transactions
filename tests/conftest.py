"""Общие фикстуры для тестов."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlops_aml_transactions.api.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_tx() -> dict:
    """Одна транзакция в формате JSON с alias-полями, как в API."""
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


# Примеры из IBM AML (HI-Small_Trans.csv), подобраны под текущий models/model.pkl:
# при смене артефакта или порога — перепроверить предсказания.
@pytest.fixture
def sample_tx_pred_class_0() -> dict:
    """Ожидается pred_laundering == 0 (типичная операция)."""
    return {
        "Timestamp": "2022/09/01 00:20",
        "From Bank": "10",
        "To Bank": "10",
        "from_account": "8000EBD30",
        "to_account": "8000EBD30",
        "Amount Received": 3697.34,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 3697.34,
        "Payment Currency": "US Dollar",
        "Payment Format": "Reinvestment",
    }


@pytest.fixture
def sample_tx_pred_class_1() -> dict:
    """Ожидается pred_laundering == 1 (модель помечает как подозрительную)."""
    return {
        "Timestamp": "2022/09/01 00:01",
        "From Bank": "70",
        "To Bank": "11157",
        "from_account": "100428660",
        "to_account": "8022AB410",
        "Amount Received": 29219.5,
        "Receiving Currency": "US Dollar",
        "Amount Paid": 29219.5,
        "Payment Currency": "US Dollar",
        "Payment Format": "Credit Card",
    }
