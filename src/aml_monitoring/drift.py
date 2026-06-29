import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .config import DATA_FILES, MODEL_META_PATH, REFERENCE_SAMPLE_PATH, REPORT_DIR, TARGET
from .data import load_transactions, sample_reference
from .features import engineer_features
from .inference import get_model_service


NUMERIC_RAW = ["Amount Received", "Amount Paid"]
CATEGORICAL_RAW = ["Receiving Currency", "Payment Currency", "Payment Format", "From Bank", "To Bank"]
STABLE_ENGINEERED_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_night",
    "amount_ratio",
    "amount_diff",
    "log_amount_paid",
    "log_amount_recv",
    "is_round_amount",
    "currency_mismatch",
    "same_bank",
    "self_transfer",
]


def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").dropna()
    actual = pd.to_numeric(actual, errors="coerce").dropna()
    if len(expected) < 2 or len(actual) < 2:
        return 0.0
    interior_edges = np.unique(np.quantile(expected, np.linspace(0, 1, buckets + 1))[1:-1])
    bins = np.concatenate(([-np.inf], interior_edges, [np.inf]))
    expected_counts = np.histogram(expected, bins=bins)[0] / max(len(expected), 1)
    actual_counts = np.histogram(actual, bins=bins)[0] / max(len(actual), 1)
    expected_counts = np.clip(expected_counts, 1e-6, None)
    actual_counts = np.clip(actual_counts, 1e-6, None)
    return float(np.sum((actual_counts - expected_counts) * np.log(actual_counts / expected_counts)))


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected_counts = expected.fillna("UNK").astype(str).value_counts(normalize=True)
    actual_counts = actual.fillna("UNK").astype(str).value_counts(normalize=True)
    categories = sorted(set(expected_counts.index) | set(actual_counts.index))
    score = 0.0
    for category in categories:
        exp = max(float(expected_counts.get(category, 0.0)), 1e-6)
        act = max(float(actual_counts.get(category, 0.0)), 1e-6)
        score += (act - exp) * np.log(act / exp)
    return float(score)


def ks_statistic(expected: pd.Series, actual: pd.Series) -> float:
    expected_values = np.sort(pd.to_numeric(expected, errors="coerce").dropna().to_numpy())
    actual_values = np.sort(pd.to_numeric(actual, errors="coerce").dropna().to_numpy())
    if len(expected_values) == 0 or len(actual_values) == 0:
        return 0.0
    values = np.sort(np.unique(np.concatenate([expected_values, actual_values])))
    exp_cdf = np.searchsorted(expected_values, values, side="right") / len(expected_values)
    act_cdf = np.searchsorted(actual_values, values, side="right") / len(actual_values)
    return float(np.max(np.abs(exp_cdf - act_cdf)))


def get_reference() -> pd.DataFrame:
    if REFERENCE_SAMPLE_PATH.exists():
        return pd.read_csv(REFERENCE_SAMPLE_PATH)
    return sample_reference(DATA_FILES, REFERENCE_SAMPLE_PATH)


def run_drift(
    batch: list[dict[str, Any]] | None = None,
    batch_path: str | None = None,
    min_rows: int = 1,
    source: str = "dataset",
) -> dict[str, Any]:
    reference = get_reference()
    if batch_path:
        current = load_transactions([batch_path])
        source = f"batch_path:{batch_path}"
    elif batch is not None:
        current = pd.DataFrame(batch)
    else:
        current = load_transactions([DATA_FILES[-1]], nrows=50000)

    if len(current) < min_rows:
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "not_enough_data",
            "source": source,
            "data_drift": {"status": "not_enough_data", "score": 0.0, "threshold": 0.2, "metrics": [], "feature_metrics": []},
            "target_drift": {"status": "not_enough_labels", "score": None, "threshold": 0.02},
            "concept_drift": {"status": "not_enough_labels", "metrics": {}, "threshold": 0.05},
            "rows": {"reference": len(reference), "current": len(current), "min_required": min_rows},
        }
        json_path, html_path = write_report(result)
        result["report_json"] = str(json_path)
        result["report_html"] = str(html_path)
        return result

    reference_features = engineer_features(reference)
    current_features = engineer_features(current)

    data_metrics = []
    for column in NUMERIC_RAW:
        data_metrics.append(
            {
                "feature": column,
                "kind": "numeric",
                "psi": psi(reference_features[column], current_features[column]),
                "ks": ks_statistic(reference_features[column], current_features[column]),
            }
        )
    for column in CATEGORICAL_RAW:
        data_metrics.append(
            {
                "feature": column,
                "kind": "categorical",
                "psi": categorical_psi(reference_features[column], current_features[column]),
                "ks": None,
            }
        )

    feature_metrics = [
        {"feature": column, "kind": "model_feature", "psi": psi(reference_features[column], current_features[column]), "ks": ks_statistic(reference_features[column], current_features[column])}
        for column in STABLE_ENGINEERED_FEATURES
        if column in reference_features.columns and column in current_features.columns
    ]

    all_scores = [metric["psi"] for metric in data_metrics + feature_metrics]
    data_drift_score = float(max(all_scores) if all_scores else 0.0)
    data_drift_threshold = 0.2

    target_drift = {"status": "not_enough_labels", "score": None, "threshold": 0.02}
    concept_drift = {"status": "not_enough_labels", "metrics": {}, "threshold": 0.05}
    if TARGET in current_features.columns and current_features[TARGET].notna().sum() > 20:
        ref_rate = float(reference_features[TARGET].mean()) if TARGET in reference_features else 0.0
        cur_rate = float(current_features[TARGET].mean())
        score = abs(cur_rate - ref_rate)
        target_drift = {"status": "drift" if score >= 0.02 else "ok", "score": score, "reference_rate": ref_rate, "current_rate": cur_rate, "threshold": 0.02}

        label_mask = current_features[TARGET].notna()
        labelled_current = current.loc[label_mask].copy()
        service = get_model_service()
        probabilities = [row["probability"] for row in service.predict(labelled_current.to_dict(orient="records"))]
        y_true = current_features.loc[label_mask, TARGET].astype(int).to_numpy()
        y_pred = (np.array(probabilities) >= service.threshold).astype(int)
        metrics = {
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(set(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
        baseline = service.meta.get("metrics", {})
        if not baseline and MODEL_META_PATH.exists():
            import joblib

            baseline = joblib.load(MODEL_META_PATH).get("metrics", {})
        baseline_f1 = float(baseline.get("oof_f1", baseline.get("f1", metrics["f1"])))
        f1_drop = max(0.0, baseline_f1 - metrics["f1"])
        concept_drift = {"status": "drift" if f1_drop >= 0.05 else "ok", "metrics": metrics, "baseline_f1": baseline_f1, "score": f1_drop, "threshold": 0.05}

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "drift" if data_drift_score >= data_drift_threshold or target_drift.get("status") == "drift" or concept_drift.get("status") == "drift" else "ok",
        "source": source,
        "data_drift": {
            "status": "drift" if data_drift_score >= data_drift_threshold else "ok",
            "score": data_drift_score,
            "threshold": data_drift_threshold,
            "metrics": data_metrics,
            "feature_metrics": feature_metrics,
        },
        "target_drift": target_drift,
        "concept_drift": concept_drift,
        "rows": {"reference": len(reference), "current": len(current)},
    }
    json_path, html_path = write_report(result)
    result["report_json"] = str(json_path)
    result["report_html"] = str(html_path)
    return result


def _status_badge(status: str) -> str:
    colors = {"ok": "#27ae60", "drift": "#e74c3c", "not_enough_data": "#7f8c8d", "not_enough_labels": "#7f8c8d"}
    color = colors.get(status, "#7f8c8d")
    return f'<span style="background:{color};color:#fff;padding:3px 10px;border-radius:4px;font-size:13px">{status}</span>'


def _psi_color(psi: float) -> str:
    if psi >= 0.2:
        return "#e74c3c"
    if psi >= 0.1:
        return "#f39c12"
    return "#27ae60"


def write_report(result: dict[str, Any]) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = REPORT_DIR / f"drift_{stamp}.json"
    html_path = REPORT_DIR / f"drift_{stamp}.html"
    json_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # Data drift table sorted by PSI descending
    all_metrics = result["data_drift"]["metrics"] + result["data_drift"].get("feature_metrics", [])
    all_metrics_sorted = sorted(all_metrics, key=lambda m: m.get("psi", 0.0), reverse=True)
    row_parts = []
    for metric in all_metrics_sorted:
        psi_val = metric.get("psi", 0.0)
        ks_str = "" if metric.get("ks") is None else f"{metric['ks']:.4f}"
        psi_color = _psi_color(psi_val)
        row_parts.append(
            f"<tr><td>{metric['feature']}</td><td>{metric['kind']}</td>"
            f"<td style='color:{psi_color};font-weight:bold'>{psi_val:.4f}</td><td>{ks_str}</td></tr>"
        )
    rows = "".join(row_parts)

    # Target drift section
    td = result.get("target_drift", {})
    target_html = f"""
      <p>Status: {_status_badge(td.get('status','?'))}&nbsp; Score: <b>{td.get('score', 'N/A') if td.get('score') is None else f"{td['score']:.4f}"}</b> / threshold {td.get('threshold', 0.02)}</p>
      {"<p>Reference positive rate: <b>" + f"{td['reference_rate']:.4f}" + "</b> → Current: <b>" + f"{td['current_rate']:.4f}" + "</b></p>" if td.get('reference_rate') is not None else "<p>No labels in current batch — target drift not calculated.</p>"}
    """

    # Concept drift section
    cd = result.get("concept_drift", {})
    cd_metrics = cd.get("metrics", {})
    concept_rows = ""
    if cd_metrics:
        for key in ["precision", "recall", "f1", "roc_auc"]:
            val = cd_metrics.get(key)
            if val is not None:
                concept_rows += f"<tr><td>{key.upper()}</td><td>{val:.4f}</td></tr>"
    concept_html = f"""
      <p>Status: {_status_badge(cd.get('status','?'))}&nbsp; F1 drop: <b>{"N/A" if cd.get('score') is None else f"{cd['score']:.4f}"}</b> / threshold {cd.get('threshold', 0.05)}</p>
      {"<p>Baseline F1: <b>" + f"{cd['baseline_f1']:.4f}" + "</b></p>" if cd.get('baseline_f1') is not None else ""}
      {"<table><thead><tr><th>Metric</th><th>Value on current batch</th></tr></thead><tbody>" + concept_rows + "</tbody></table>" if concept_rows else "<p>No labels in current batch — concept drift not calculated.</p>"}
    """

    overall_color = {"ok": "#27ae60", "drift": "#e74c3c", "not_enough_data": "#7f8c8d"}.get(result["status"], "#7f8c8d")
    rows_info = result.get("rows", {})

    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AML Drift Report — {stamp}</title>
<style>
  body{{font-family:Arial,sans-serif;margin:32px;color:#17202a;max-width:1100px}}
  h1{{border-bottom:3px solid {overall_color};padding-bottom:8px}}
  h2{{margin-top:32px;color:#2c3e50;border-left:4px solid #3498db;padding-left:10px}}
  table{{border-collapse:collapse;width:100%;margin-top:10px}}
  td,th{{border:1px solid #d5d8dc;padding:8px 12px}}
  th{{background:#f4f6f7;font-weight:600}}
  .card{{background:#fafafa;border:1px solid #e5e8ea;border-radius:6px;padding:16px 20px;margin-top:12px}}
  .meta{{color:#7f8c8d;font-size:13px;margin-bottom:4px}}
</style></head>
<body>
<h1>AML Drift Report</h1>
<div class="card">
  <p class="meta">Generated: {result.get('created_at','')} &nbsp;|&nbsp; Source: <b>{result.get('source','?')}</b></p>
  <p class="meta">Rows — Reference: <b>{rows_info.get('reference','?')}</b> &nbsp;|&nbsp; Current: <b>{rows_info.get('current','?')}</b></p>
  <p style="font-size:18px">Overall status: {_status_badge(result['status'])}</p>
</div>

<h2>Data Drift</h2>
<div class="card">
  <p>Score (max PSI): <b style="color:{_psi_color(result['data_drift']['score'])}">{result['data_drift']['score']:.4f}</b> / threshold {result['data_drift']['threshold']}
  &nbsp;→&nbsp; {_status_badge("drift" if result['data_drift']['score'] >= result['data_drift']['threshold'] else "ok")}</p>
  <p style="font-size:12px;color:#7f8c8d">PSI &lt; 0.1 = no drift &nbsp;|&nbsp; 0.1–0.2 = moderate &nbsp;|&nbsp; &gt; 0.2 = significant</p>
  <table><thead><tr><th>Feature</th><th>Kind</th><th>PSI ↓</th><th>KS</th></tr></thead><tbody>{rows}</tbody></table>
</div>

<h2>Target Drift</h2>
<div class="card">{target_html}</div>

<h2>Concept Drift</h2>
<div class="card">{concept_html}</div>
</body></html>"""
    html_path.write_text(html, encoding="utf-8")
    return json_path, html_path
