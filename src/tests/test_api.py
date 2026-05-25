from fastapi.testclient import TestClient

from aml_monitoring import main


class StubModelService:
    threshold = 0.99

    def predict(self, records):
        return [
            {
                "payload": record,
                "probability": 0.12,
                "predicted_class": 0,
                "threshold": self.threshold,
                "anomaly_flag": False,
                "drift_flag": False,
            }
            for record in records
        ]


def test_predict_endpoint(monkeypatch):
    monkeypatch.setattr(main, "get_model_service", lambda: StubModelService())
    client = TestClient(main.app)
    response = client.post(
        "/predict",
        json={
            "transactions": [
                {
                    "Timestamp": "2022/09/01 00:20",
                    "From Bank": "010",
                    "From Account": "A1",
                    "To Bank": "011",
                    "To Account": "A2",
                    "Amount Received": 100.0,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": 100.0,
                    "Payment Currency": "US Dollar",
                    "Payment Format": "Wire",
                }
            ]
        },
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["predicted_class"] == 0


def test_metrics_endpoint():
    client = TestClient(main.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "aml_api_requests_total" in response.text


def test_drift_run_uses_recent_predictions_by_default(monkeypatch):
    captured = {}

    class StubStore:
        def recent_prediction_payloads(self, limit):
            captured["limit"] = limit
            return [{"Amount Paid": 100.0}]

        def add_drift_run(self, *args):
            captured["stored"] = True
            return 1

    def fake_run_drift(batch=None, batch_path=None, min_rows=1, source="dataset"):
        captured["batch"] = batch
        captured["source"] = source
        captured["min_rows"] = min_rows
        return {
            "status": "not_enough_data",
            "data_drift": {"score": 0.0, "threshold": 0.2},
            "report_json": "report.json",
            "report_html": "report.html",
        }

    monkeypatch.setattr(main, "store", StubStore())
    monkeypatch.setattr(main, "run_drift", fake_run_drift)
    monkeypatch.setattr(main, "record_drift", lambda result: None)

    client = TestClient(main.app)
    response = client.post("/drift/run", json={})

    assert response.status_code == 200
    assert captured["batch"] == [{"Amount Paid": 100.0}]
    assert captured["source"] == "recent_predictions"
    assert captured["stored"] is True
