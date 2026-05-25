from typing import Any

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    timestamp: str | None = Field(default=None, alias="Timestamp")
    from_bank: str | None = Field(default=None, alias="From Bank")
    from_account: str | None = Field(default=None, alias="From Account")
    to_bank: str | None = Field(default=None, alias="To Bank")
    to_account: str | None = Field(default=None, alias="To Account")
    amount_received: float | None = Field(default=None, alias="Amount Received")
    receiving_currency: str | None = Field(default=None, alias="Receiving Currency")
    amount_paid: float | None = Field(default=None, alias="Amount Paid")
    payment_currency: str | None = Field(default=None, alias="Payment Currency")
    payment_format: str | None = Field(default=None, alias="Payment Format")
    is_laundering: int | None = Field(default=None, alias="Is Laundering")

    model_config = {"populate_by_name": True, "extra": "allow"}


class PredictRequest(BaseModel):
    transactions: list[dict[str, Any]]


class DriftRunRequest(BaseModel):
    transactions: list[dict[str, Any]] | None = None
    batch_path: str | None = None
    use_recent_predictions: bool = True
    prediction_limit: int | None = None
    min_rows: int | None = None
