from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from loguru import logger
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import typer

from mlops_aml_transactions.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLRUNS_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from mlops_aml_transactions.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, X_y_from_frame
from mlops_aml_transactions.modeling.artifacts import save_model
from mlops_aml_transactions.storage.s3 import s3_key_for_local_path, s3_upload_file

app = typer.Typer()

AVAILABLE_MODELS: dict[str, str] = {
    "hgb": "HistGradientBoostingClassifier",
    "rf": "RandomForestClassifier",
    "et": "ExtraTreesClassifier",
    "lr": "LogisticRegression",
}

SelectBy = Literal["average_precision", "roc_auc", "f1"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def make_estimator(
    name: str,
    *,
    random_state: int,
    hgb_max_depth: int,
    hgb_learning_rate: float,
    hgb_max_iter: int,
):
    if name == "hgb":
        return HistGradientBoostingClassifier(
            max_depth=hgb_max_depth,
            learning_rate=hgb_learning_rate,
            max_iter=hgb_max_iter,
            min_samples_leaf=40,
            l2_regularization=0.05,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=15,
            random_state=random_state,
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "et":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=8,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        )
    if name == "lr":
        return LogisticRegression(
            max_iter=4000,
            C=0.5,
            class_weight="balanced",
            random_state=random_state,
            solver="saga",
            n_jobs=-1,
        )
    raise ValueError(f"Unknown model key: {name}")


def default_params_for_mlflow(name: str, **hgb_kw) -> dict[str, Any]:
    if name == "hgb":
        return {
            "clf_max_depth": hgb_kw["hgb_max_depth"],
            "clf_learning_rate": hgb_kw["hgb_learning_rate"],
            "clf_max_iter": hgb_kw["hgb_max_iter"],
            "clf_min_samples_leaf": 40,
            "clf_l2_regularization": 0.05,
            "clf_early_stopping": True,
            "clf_class_weight": "balanced",
        }
    if name == "rf":
        return {
            "clf_n_estimators": 300,
            "clf_max_depth": 18,
            "clf_min_samples_leaf": 8,
            "clf_max_features": "sqrt",
            "clf_class_weight": "balanced_subsample",
        }
    if name == "et":
        return {
            "clf_n_estimators": 300,
            "clf_max_depth": 18,
            "clf_min_samples_leaf": 8,
            "clf_max_features": "sqrt",
            "clf_class_weight": "balanced_subsample",
        }
    if name == "lr":
        return {
            "clf_max_iter": 4000,
            "clf_C": 0.5,
            "clf_solver": "saga",
            "clf_class_weight": "balanced",
        }
    return {}


def tune_threshold_f1(pipeline: Pipeline, X_val: pd.DataFrame, y_val: pd.Series) -> float:
    """Подбор порога по валидации для максимизации F1 (редкий положительный класс)."""
    proba = pipeline.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.02, 0.98, 97)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        f = f1_score(y_val, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t


def parse_model_list(spec: str) -> list[str]:
    parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
    unknown = set(parts) - set(AVAILABLE_MODELS)
    if unknown:
        raise typer.BadParameter(
            f"Unknown model(s): {sorted(unknown)}. "
            f"Choose from: {', '.join(sorted(AVAILABLE_MODELS))}"
        )
    if not parts:
        raise typer.BadParameter("Provide at least one model name.")
    return parts


def parse_select_by(s: str) -> SelectBy:
    m = {
        "ap": "average_precision",
        "average_precision": "average_precision",
        "roc": "roc_auc",
        "roc_auc": "roc_auc",
        "f1": "f1",
    }
    key = s.strip().lower()
    if key not in m:
        raise typer.BadParameter("select-by must be one of: ap, roc_auc, f1")
    return m[key]  # type: ignore[return-value]


def main_score(select_by: SelectBy, roc: float, pr: float, f1: float) -> float:
    if select_by == "average_precision":
        return pr
    if select_by == "roc_auc":
        return roc
    return f1


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    test_size: float = typer.Option(0.2, help="Holdout fraction for metrics."),
    random_state: int = 42,
    models: str = typer.Option(
        "hgb,rf,et,lr",
        help="Comma-separated models to train: hgb, rf, et, lr.",
    ),
    experiment_name: str = typer.Option(
        MLFLOW_EXPERIMENT_NAME,
        help="MLflow experiment name (or set MLFLOW_EXPERIMENT_NAME).",
    ),
    tracking_uri: str | None = typer.Option(
        None,
        help="MLflow tracking URI; default: local file store in project mlruns/.",
    ),
    run_name: str | None = typer.Option(None, help="Optional parent MLflow run name."),
    hgb_max_depth: int = typer.Option(12, help="[hgb] max_depth."),
    hgb_learning_rate: float = typer.Option(0.05, help="[hgb] learning_rate."),
    hgb_max_iter: int = typer.Option(400, help="[hgb] max_iter."),
    tune_threshold: bool = typer.Option(
        True,
        help="Подобрать порог вероятности на валидации (улучшает F1 при дисбалансе).",
    ),
    val_fraction: float = typer.Option(
        0.15,
        help="Доля train для валидации при подборе порога (если tune-threshold).",
    ),
    select_by: str = typer.Option(
        "ap",
        help="Критерий лучшей модели: ap (average_precision), roc_auc, f1.",
    ),
) -> None:
    """Обучение моделей, MLflow, лучшая по метрике -> model.pkl."""
    if not dataset_path.is_file():
        raise typer.BadParameter(f"Dataset not found: {dataset_path}. Run dataset step first.")

    model_keys = parse_model_list(models)
    select_key = parse_select_by(select_by)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(dataset_path)
    X, y = X_y_from_frame(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    pos_train = int(y_train.sum())
    pos_test = int(y_test.sum())

    shared_params = {
        "test_size": test_size,
        "random_state": random_state,
        "ordinal_unknown_value": -1,
        "n_features": X.shape[1],
        "feature_set": "v2_log_amounts_calendar_currency_match",
        "n_rows_total": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_positive_train": pos_train,
        "n_positive_test": pos_test,
        "dataset_path": str(dataset_path.resolve()),
        "models_trained": ",".join(model_keys),
        "tune_threshold": tune_threshold,
        "val_fraction": val_fraction if tune_threshold else 0.0,
        "select_best_by": select_key,
    }

    best_score = -1.0
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_threshold: float | None = None

    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params(shared_params)
        mlflow.set_tag("run_kind", "multi_model_train")

        for key in model_keys:
            estimator = make_estimator(
                key,
                random_state=random_state,
                hgb_max_depth=hgb_max_depth,
                hgb_learning_rate=hgb_learning_rate,
                hgb_max_iter=hgb_max_iter,
            )
            pipeline = build_pipeline(estimator)

            threshold: float | None = None
            if tune_threshold:
                X_fit, X_val, y_fit, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=val_fraction,
                    stratify=y_train,
                    random_state=random_state,
                )
                pipeline.fit(X_fit, y_fit)
                threshold = tune_threshold_f1(pipeline, X_val, y_val)
                pipeline.fit(X_train, y_train)
            else:
                pipeline.fit(X_train, y_train)

            proba = pipeline.predict_proba(X_test)[:, 1]
            if threshold is not None:
                pred = (proba >= threshold).astype(int)
            else:
                pred = pipeline.predict(X_test)

            roc = roc_auc_score(y_test, proba)
            pr = average_precision_score(y_test, proba)
            f1 = f1_score(y_test, pred, zero_division=0)
            bacc = balanced_accuracy_score(y_test, pred)

            with mlflow.start_run(run_name=key, nested=True):
                mlflow.set_tags(
                    {
                        "model_key": key,
                        "model_class": AVAILABLE_MODELS[key],
                        "preprocessor": "ColumnTransformer_OrdinalEncoder",
                    }
                )
                clf_params = default_params_for_mlflow(
                    key,
                    hgb_max_depth=hgb_max_depth,
                    hgb_learning_rate=hgb_learning_rate,
                    hgb_max_iter=hgb_max_iter,
                )
                mlflow.log_params(clf_params)
                if threshold is not None:
                    mlflow.log_param("decision_threshold", threshold)

                mlflow.log_metrics(
                    {
                        "roc_auc": roc,
                        "average_precision": pr,
                        "f1": f1,
                        "balanced_accuracy": bacc,
                    }
                )
                mlflow.log_text(
                    classification_report(y_test, pred),
                    f"classification_report_{key}.txt",
                )

                logger.info(
                    "[{}] ROC-AUC: {:.4f} | AP: {:.4f} | F1: {:.4f} | bal_acc: {:.4f}{}",
                    key,
                    roc,
                    pr,
                    f1,
                    bacc,
                    f" | thr={threshold:.4f}" if threshold is not None else "",
                )

                signature = infer_signature(X_train, pipeline.predict(X_train))
                mlflow.sklearn.log_model(
                    pipeline,
                    name=f"sklearn_pipeline_{key}",
                    signature=signature,
                )

                out_file = MODELS_DIR / f"{key}.pkl"
                save_model(out_file, pipeline, threshold=threshold)
                mlflow.log_artifact(str(out_file))
                # Upload trained model artifact to S3 (if configured).
                try:
                    s3_key = s3_key_for_local_path(out_file, kind="models")
                    s3_upload_file(out_file, s3_key)
                except Exception:
                    pass

                score = main_score(select_key, roc, pr, f1)
                if score > best_score:
                    best_score = score
                    best_name = key
                    best_pipeline = pipeline
                    best_threshold = threshold

        if best_pipeline is not None and best_name is not None:
            save_model(model_path, best_pipeline, threshold=best_threshold)
            # Upload best model artifact to S3 (if configured).
            try:
                s3_key = s3_key_for_local_path(model_path, kind="models")
                s3_upload_file(model_path, s3_key)
            except Exception:
                pass
            mlflow.set_tag("best_model", best_name)
            mlflow.set_tag("best_select_metric", select_key)
            mlflow.log_metric(f"best_{select_key}", best_score)
            if best_threshold is not None:
                mlflow.log_param("best_decision_threshold", best_threshold)
            logger.success(
                "Best model: {} ({}={:.4f}) -> {}",
                best_name,
                select_key,
                best_score,
                model_path,
            )
            logger.success("Parent MLflow run id: {}", parent.info.run_id)
        else:
            logger.error("No model was trained.")

    logger.info("Open MLflow UI: mlflow ui --backend-store-uri {}", uri)


if __name__ == "__main__":
    app()
