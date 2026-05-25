import time
from collections.abc import Callable

from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_COUNT = Counter("aml_api_requests_total", "Total API requests", ["method", "path", "status"])
REQUEST_LATENCY = Histogram("aml_api_request_latency_seconds", "API request latency", ["method", "path"])
PREDICTION_COUNT = Counter("aml_predictions_total", "Total predictions", ["predicted_class"])
ANOMALY_RATE = Gauge("aml_anomaly_rate", "Last batch anomaly rate")
AVERAGE_PROBABILITY = Gauge("aml_average_laundering_probability", "Last batch average laundering probability")
DATA_DRIFT_SCORE = Gauge("aml_data_drift_score", "Latest data drift score")
TARGET_DRIFT_SCORE = Gauge("aml_target_drift_score", "Latest target drift score")
CONCEPT_DRIFT_SCORE = Gauge("aml_concept_drift_score", "Latest concept drift score")
RETRAINING_STATUS = Gauge("aml_retraining_status", "Latest retraining status: queued=1 running=2 completed=3 failed=4")
MODEL_ROC_AUC = Gauge("aml_model_roc_auc", "Current model ROC-AUC")
MODEL_PR_AUC = Gauge("aml_model_pr_auc", "Current model average precision / PR-AUC")
MODEL_PRECISION = Gauge("aml_model_precision", "Current model precision at selected threshold")
MODEL_RECALL = Gauge("aml_model_recall", "Current model recall at selected threshold")
MODEL_F1 = Gauge("aml_model_f1", "Current model F1 at selected threshold")
MODEL_F2 = Gauge("aml_model_f2", "Current model F2 at selected threshold")
MODEL_THRESHOLD = Gauge("aml_model_threshold", "Current model decision threshold")
MODEL_TRAIN_ROWS = Gauge("aml_model_train_rows", "Rows used for latest accepted training run")
MODEL_TRAIN_POSITIVES = Gauge("aml_model_train_positives", "Positive labels used for latest accepted training run")


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        started_at = time.time()
        response = await call_next(request)
        path = request.url.path
        REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(time.time() - started_at)
        return response


def metrics_response() -> Response:
    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


def record_predictions(results: list[dict]) -> None:
    if not results:
        return
    for result in results:
        PREDICTION_COUNT.labels(str(result["predicted_class"])).inc()
    ANOMALY_RATE.set(sum(int(r["anomaly_flag"]) for r in results) / len(results))
    AVERAGE_PROBABILITY.set(sum(float(r["probability"]) for r in results) / len(results))


def record_drift(result: dict) -> None:
    DATA_DRIFT_SCORE.set(float(result["data_drift"]["score"]))
    target_score = result.get("target_drift", {}).get("score")
    concept_score = result.get("concept_drift", {}).get("score")
    if target_score is not None:
        TARGET_DRIFT_SCORE.set(float(target_score))
    if concept_score is not None:
        CONCEPT_DRIFT_SCORE.set(float(concept_score))


def record_retraining_status(status: str) -> None:
    RETRAINING_STATUS.set({"queued": 1, "running": 2, "completed": 3, "failed": 4}.get(status, 0))


def record_model_metrics(metrics: dict, threshold: float | None = None) -> None:
    roc_auc = metrics.get("oof_roc_auc", metrics.get("roc_auc"))
    pr_auc = metrics.get("oof_avg_prec", metrics.get("avg_precision"))
    precision = metrics.get("oof_precision", metrics.get("precision"))
    recall = metrics.get("oof_recall", metrics.get("recall"))
    f1 = metrics.get("oof_f1", metrics.get("f1"))
    f2 = metrics.get("oof_f2", metrics.get("f2"))
    train_rows = metrics.get("train_rows")
    train_positives = metrics.get("train_positives")

    for value, gauge in [
        (roc_auc, MODEL_ROC_AUC),
        (pr_auc, MODEL_PR_AUC),
        (precision, MODEL_PRECISION),
        (recall, MODEL_RECALL),
        (f1, MODEL_F1),
        (f2, MODEL_F2),
        (train_rows, MODEL_TRAIN_ROWS),
        (train_positives, MODEL_TRAIN_POSITIVES),
    ]:
        if value is not None:
            gauge.set(float(value))
    if threshold is not None:
        MODEL_THRESHOLD.set(float(threshold))
