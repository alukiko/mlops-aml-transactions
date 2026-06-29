import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import (
    DATA_FILES,
    DEFAULT_MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_PATH,
    RAW_COLUMNS,
    REFERENCE_SAMPLE_PATH,
    RUNTIME_DIR,
    TARGET,
)
from .data import get_env_int, normalize_transactions
from .features import build_preprocessor, to_feature_matrix
from .metrics import record_model_metrics, record_retraining_status
from .s3_data import s3_credentials_configured, upload_models
from .storage import Store


def load_stratified_training_sample(
    files,
    max_rows: int = 300000,
    min_positives: int = 500,
    random_state: int = 42,
    chunksize: int = 200000,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    positives = []
    negatives = []
    negative_budget = max(max_rows - min_positives, min_positives)
    target_negative_per_chunk = max(1000, negative_budget // 20)

    for path in files:
        for chunk in pd.read_csv(
            path,
            names=RAW_COLUMNS,
            header=0,
            chunksize=chunksize,
            dtype={
                "From Bank": str,
                "From Account": str,
                "To Bank": str,
                "To Account": str,
                "Payment Format": str,
                "Receiving Currency": str,
                "Payment Currency": str,
            },
            low_memory=False,
        ):
            chunk = normalize_transactions(chunk)
            pos = chunk[chunk[TARGET] == 1]
            neg = chunk[chunk[TARGET] == 0]
            if not pos.empty:
                positives.append(pos)
            if len(negatives) < 20 and not neg.empty:
                sample_size = min(len(neg), target_negative_per_chunk)
                seed = int(rng.integers(0, 2**31 - 1))
                negatives.append(neg.sample(n=sample_size, random_state=seed))
            positive_count = sum(len(frame) for frame in positives)
            negative_count = sum(len(frame) for frame in negatives)
            if positive_count >= min_positives and negative_count >= negative_budget:
                return finalize_training_sample(positives, negatives, max_rows, random_state)

    return finalize_training_sample(positives, negatives, max_rows, random_state)


def finalize_training_sample(positives, negatives, max_rows: int, random_state: int) -> pd.DataFrame:
    if not positives:
        raise ValueError("No positive examples found for retraining")
    pos_df = pd.concat(positives, ignore_index=True)
    neg_df = pd.concat(negatives, ignore_index=True) if negatives else pd.DataFrame(columns=pos_df.columns)
    neg_limit = max(max_rows - len(pos_df), 0)
    if len(neg_df) > neg_limit:
        neg_df = neg_df.sample(n=neg_limit, random_state=random_state)
    data = pd.concat([pos_df, neg_df], ignore_index=True)
    return data.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def find_best_threshold(y_true, probabilities, beta: float = 2.0, min_precision: float = 0.001) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "f2": 0.0}
    for threshold in np.linspace(0.001, 0.999, 400):
        y_pred = (probabilities >= threshold).astype(int)
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        if precision < min_precision and recall < 1.0:
            continue
        denom = beta**2 * precision + recall
        f2 = 0.0 if denom == 0 else float((1 + beta**2) * precision * recall / denom)
        if (f2, recall, precision) > (best_metrics["f2"], best_metrics["recall"], best_metrics["precision"]):
            best_threshold = float(threshold)
            best_metrics = {"precision": precision, "recall": recall, "f1": f1, "f2": f2}
    return best_threshold, best_metrics


def load_existing_quality() -> dict[str, float]:
    meta_path = MODEL_DIR / "model_meta.pkl"
    if not meta_path.exists():
        return {}
    try:
        metrics = joblib.load(meta_path).get("metrics", {})
        return {
            "recall": float(metrics.get("oof_recall", metrics.get("recall", 0.0))),
            "f2": float(metrics.get("oof_f2", metrics.get("f2", 0.0))),
            "roc_auc": float(metrics.get("oof_roc_auc", metrics.get("roc_auc", 0.0))),
        }
    except Exception:
        return {}


def passes_quality_gate(candidate: dict[str, float], baseline: dict[str, float]) -> bool:
    if not baseline:
        return True
    baseline_f2 = baseline.get("f2", 0.0)
    baseline_recall = baseline.get("recall", 0.0)
    if baseline_f2 == 0 and baseline_recall == 0:
        return True
    return candidate["f2"] >= baseline_f2 and candidate["recall"] >= max(0.90 * baseline_recall, 0.01)


def run_retraining(job_id: int, store: Store) -> None:
    store.update_retraining_job(job_id, "running")
    record_retraining_status("running")
    try:
        import lightgbm as lgb
        import mlflow
        import mlflow.lightgbm

        max_rows = get_env_int("RETRAIN_MAX_ROWS", 300000)
        min_positives = get_env_int("RETRAIN_MIN_POSITIVES", 500)
        data = load_stratified_training_sample(DATA_FILES, max_rows=max_rows, min_positives=min_positives)
        y = data[TARGET].fillna(0).astype(int).to_numpy()
        if int(y.sum()) < 2:
            raise ValueError("Retraining sample must contain at least two positive examples")
        train_df, val_df, y_train, y_val = train_test_split(
            data,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        preprocessor = build_preprocessor(train_df)
        x_train = to_feature_matrix(train_df, preprocessor=preprocessor)
        x_val = to_feature_matrix(val_df, preprocessor=preprocessor)
        categorical_cols = x_train.select_dtypes(include="object").columns.tolist()
        for col in categorical_cols:
            x_train[col] = x_train[col].astype("category")
            x_val[col] = pd.Categorical(x_val[col], categories=x_train[col].cat.categories)
        pos = max(int(y_train.sum()), 1)
        neg = max(len(y_train) - pos, 1)
        model = lgb.LGBMClassifier(
            objective="binary",
            metric=["auc", "average_precision"],
            n_estimators=get_env_int("RETRAIN_N_ESTIMATORS", 200),
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=neg / pos,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)[:, 1]
        threshold, threshold_metrics = find_best_threshold(y_val, probabilities)
        y_pred = (probabilities >= threshold).astype(int)
        metrics: dict[str, Any] = {
            "roc_auc": float(roc_auc_score(y_val, probabilities)) if len(set(y_val)) > 1 else 0.0,
            "avg_precision": float(average_precision_score(y_val, probabilities)) if len(set(y_val)) > 1 else 0.0,
            "f1": float(f1_score(y_val, y_pred, zero_division=0)),
            "precision": float(precision_score(y_val, y_pred, zero_division=0)),
            "recall": float(recall_score(y_val, y_pred, zero_division=0)),
            "f2": threshold_metrics["f2"],
            "best_threshold": threshold,
            "train_rows": len(data),
            "train_positives": int(y.sum()),
        }
        baseline = load_existing_quality()
        accepted = passes_quality_gate(metrics, baseline)
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI))
        mlflow.set_registry_uri(os.getenv("MLFLOW_REGISTRY_URI", os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI)))
        mlflow.set_experiment("aml-detection")
        registered_model_name = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "aml-laundering-detector")
        with mlflow.start_run(run_name=f"api_retrain_{job_id}") as run:
            mlflow.log_param("train_rows", len(data))
            mlflow.log_param("train_positives", int(y.sum()))
            mlflow.log_param("accepted", accepted)
            mlflow.log_metrics(metrics)
            candidate_model_path = RUNTIME_DIR / f"candidate_aml_lgbm_{job_id}.pkl"
            candidate_meta_path = RUNTIME_DIR / f"candidate_model_meta_{job_id}.pkl"
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            meta = {
                "feature_cols": list(x_train.columns),
                "best_threshold": threshold,
                "preprocessor": preprocessor,
                "metrics": {
                    "oof_roc_auc": metrics["roc_auc"],
                    "oof_avg_prec": metrics["avg_precision"],
                    "oof_f1": metrics["f1"],
                    "oof_f2": metrics["f2"],
                    "oof_precision": metrics["precision"],
                    "oof_recall": metrics["recall"],
                    "best_threshold": threshold,
                    "train_rows": len(data),
                    "train_positives": int(y.sum()),
                },
            }
            joblib.dump(model, candidate_model_path)
            joblib.dump(meta, candidate_meta_path)
            if accepted:
                mlflow.lightgbm.log_model(model.booster_, "model", registered_model_name=registered_model_name)
            else:
                mlflow.lightgbm.log_model(model.booster_, "model")
            mlflow.log_artifact(str(candidate_model_path))
            mlflow.log_artifact(str(candidate_meta_path))
            if accepted:
                model_path = MODEL_PATH
                meta_path = MODEL_DIR / "model_meta.pkl"
                joblib.dump(model, model_path)
                joblib.dump(meta, meta_path)
                reference = data.sample(n=min(50000, len(data)), random_state=42)
                REFERENCE_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
                reference.to_csv(REFERENCE_SAMPLE_PATH, index=False)
                from .inference import get_model_service

                get_model_service.cache_clear()
                record_model_metrics(meta["metrics"], threshold)
                mlflow.log_artifact(str(model_path))
                mlflow.log_artifact(str(meta_path))
                if s3_credentials_configured():
                    upload_models(required=False)
            metrics["registered_model_name"] = registered_model_name
            metrics["accepted"] = accepted
            metrics["baseline_f2"] = baseline.get("f2", 0.0)
            metrics["baseline_recall"] = baseline.get("recall", 0.0)
            store.update_retraining_job(job_id, "completed", mlflow_run_id=run.info.run_id, metrics=metrics)
        record_retraining_status("completed")
    except Exception as exc:
        store.update_retraining_job(job_id, "failed", error=str(exc))
        record_retraining_status("failed")
