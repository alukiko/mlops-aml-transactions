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

# Aggregate drift scores
DATA_DRIFT_SCORE = Gauge("aml_data_drift_score", "Latest data drift score (max PSI across features)")
TARGET_DRIFT_SCORE = Gauge("aml_target_drift_score", "Latest target drift score (|cur_rate - ref_rate|)")
CONCEPT_DRIFT_SCORE = Gauge("aml_concept_drift_score", "Latest concept drift score (F1 drop vs baseline)")

# Drift status: 1=ok, 2=drift, 0=not_enough_data/labels
DATA_DRIFT_STATUS = Gauge("aml_data_drift_status", "Data drift status: 0=no_data 1=ok 2=drift")
TARGET_DRIFT_STATUS = Gauge("aml_target_drift_status", "Target drift status: 0=no_labels 1=ok 2=drift")
CONCEPT_DRIFT_STATUS = Gauge("aml_concept_drift_status", "Concept drift status: 0=no_labels 1=ok 2=drift")

# Per-feature drift (label = feature name + kind)
FEATURE_DRIFT_PSI = Gauge("aml_feature_drift_psi", "Per-feature PSI score", ["feature", "kind"])
FEATURE_DRIFT_KS = Gauge("aml_feature_drift_ks", "Per-feature KS statistic (numeric only)", ["feature"])

# Target drift detail
TARGET_DRIFT_REF_RATE = Gauge("aml_target_drift_reference_rate", "Reference positive rate (laundering fraction)")
TARGET_DRIFT_CUR_RATE = Gauge("aml_target_drift_current_rate", "Current positive rate (laundering fraction)")

# Concept drift model metrics on current batch
CONCEPT_DRIFT_PRECISION = Gauge("aml_concept_drift_precision", "Precision on last labelled batch")
CONCEPT_DRIFT_RECALL = Gauge("aml_concept_drift_recall", "Recall on last labelled batch")
CONCEPT_DRIFT_F1 = Gauge("aml_concept_drift_f1", "F1 on last labelled batch")
CONCEPT_DRIFT_ROC_AUC = Gauge("aml_concept_drift_roc_auc", "ROC-AUC on last labelled batch")
CONCEPT_DRIFT_BASELINE_F1 = Gauge("aml_concept_drift_baseline_f1", "Training baseline F1 for concept drift comparison")

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


_DRIFT_STATUS_MAP = {"ok": 1, "drift": 2, "not_enough_data": 0, "not_enough_labels": 0}


def record_drift(result: dict) -> None:
    data_drift = result.get("data_drift", {})
    DATA_DRIFT_SCORE.set(float(data_drift.get("score", 0.0)))
    DATA_DRIFT_STATUS.set(_DRIFT_STATUS_MAP.get(result.get("status", ""), 0))

    # Per-feature PSI and KS
    for metric in data_drift.get("metrics", []):
        feature = metric["feature"]
        kind = metric["kind"]
        FEATURE_DRIFT_PSI.labels(feature=feature, kind=kind).set(float(metric.get("psi", 0.0)))
        if metric.get("ks") is not None:
            FEATURE_DRIFT_KS.labels(feature=feature).set(float(metric["ks"]))

    # Target drift
    target_drift = result.get("target_drift", {})
    target_score = target_drift.get("score")
    TARGET_DRIFT_STATUS.set(_DRIFT_STATUS_MAP.get(target_drift.get("status", "not_enough_labels"), 0))
    if target_score is not None:
        TARGET_DRIFT_SCORE.set(float(target_score))
    if target_drift.get("reference_rate") is not None:
        TARGET_DRIFT_REF_RATE.set(float(target_drift["reference_rate"]))
    if target_drift.get("current_rate") is not None:
        TARGET_DRIFT_CUR_RATE.set(float(target_drift["current_rate"]))

    # Concept drift
    concept_drift = result.get("concept_drift", {})
    concept_score = concept_drift.get("score")
    CONCEPT_DRIFT_STATUS.set(_DRIFT_STATUS_MAP.get(concept_drift.get("status", "not_enough_labels"), 0))
    if concept_score is not None:
        CONCEPT_DRIFT_SCORE.set(float(concept_score))
    if concept_drift.get("baseline_f1") is not None:
        CONCEPT_DRIFT_BASELINE_F1.set(float(concept_drift["baseline_f1"]))
    for key, gauge in [
        ("precision", CONCEPT_DRIFT_PRECISION),
        ("recall", CONCEPT_DRIFT_RECALL),
        ("f1", CONCEPT_DRIFT_F1),
        ("roc_auc", CONCEPT_DRIFT_ROC_AUC),
    ]:
        val = concept_drift.get("metrics", {}).get(key)
        if val is not None:
            gauge.set(float(val))


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
