from fastapi.testclient import TestClient

from aml_monitoring import main
from aml_monitoring.storage import Store


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


def test_predict_returns_503_when_model_is_missing(monkeypatch):
    def missing_model():
        raise FileNotFoundError("models/aml_lgbm.pkl")

    monkeypatch.setattr(main, "get_model_service", missing_model)
    client = TestClient(main.app)
    response = client.post("/predict", json={"transactions": [{"Amount Paid": 100.0}]})

    assert response.status_code == 503
    assert "Model is not available" in response.json()["detail"]


def test_metrics_endpoint():
    client = TestClient(main.app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "aml_api_requests_total" in response.text


def test_prediction_label_endpoint_adds_ground_truth_to_drift_payload(monkeypatch, tmp_path):
    store = Store(tmp_path / "monitoring.db")
    prediction_id = store.add_prediction(
        {"Amount Paid": 100.0},
        probability=0.12,
        predicted_class=0,
        anomaly_flag=False,
        drift_flag=False,
    )
    monkeypatch.setattr(main, "store", store)
    client = TestClient(main.app)

    response = client.patch(f"/predictions/{prediction_id}/label", json={"actual_label": 1})

    assert response.status_code == 200
    assert response.json()["item"]["actual_label"] == 1
    assert response.json()["labeling"]["labelled"] == 1
    assert response.json()["labeling"]["remaining"] == 20
    assert store.recent_prediction_payloads(1)[0]["Is Laundering"] == 1


def test_prediction_label_endpoint_validates_label_and_prediction_id(monkeypatch, tmp_path):
    store = Store(tmp_path / "monitoring.db")
    monkeypatch.setattr(main, "store", store)
    client = TestClient(main.app)

    invalid_label = client.patch("/predictions/1/label", json={"actual_label": 2})
    missing_prediction = client.patch("/predictions/1/label", json={"actual_label": 0})

    assert invalid_label.status_code == 422
    assert missing_prediction.status_code == 404


def test_prediction_labeling_status_is_ready_after_21_labels(tmp_path):
    store = Store(tmp_path / "monitoring.db")
    for index in range(21):
        store.add_prediction(
            {"Amount Paid": float(index), "Is Laundering": index % 2},
            probability=0.12,
            predicted_class=0,
            anomaly_flag=False,
            drift_flag=False,
        )

    status = store.prediction_labeling_status()

    assert status["labelled"] == 21
    assert status["remaining"] == 0
    assert status["ready"] is True


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


def test_drift_run_uses_dataset_when_recent_predictions_are_disabled(monkeypatch):
    captured = {}

    class StubStore:
        def add_drift_run(self, *args):
            return 1

    def fake_run_drift(batch=None, batch_path=None, min_rows=1, source="dataset"):
        captured["batch"] = batch
        captured["source"] = source
        return {
            "status": "ok",
            "data_drift": {"status": "ok", "score": 0.0, "threshold": 0.2},
            "report_json": "report.json",
            "report_html": "report.html",
        }

    monkeypatch.setattr(main, "store", StubStore())
    monkeypatch.setattr(main, "run_drift", fake_run_drift)
    monkeypatch.setattr(main, "record_drift", lambda result: None)

    client = TestClient(main.app)
    response = client.post("/drift/run", json={"use_recent_predictions": False})

    assert response.status_code == 200
    assert captured["batch"] is None
    assert captured["source"] == "dataset"
