"""Обучение LightGBM с OOF, MLflow и SHAP (логика из ноутбука first_steps)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import shap
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from mlops_aml_transactions.config import MLFLOW_EXPERIMENT_LGB_NOTEBOOK, MLRUNS_DIR, MODELS_DIR
from mlops_aml_transactions.features_lgb import (
    FEATURE_COLS_LGB,
    TARGET_LGB,
    find_best_threshold,
)


def train_lgbm_aml(
    df: pd.DataFrame,
    *,
    model_dir: Path | None = None,
    mlflow_uri: str | None = None,
    experiment_name: str | None = None,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    """Stratified K-Fold OOF, финальная модель на полных данных, артефакты в model_dir и MLflow."""
    X = df[FEATURE_COLS_LGB].copy()
    y = df[TARGET_LGB].values

    pos = y.sum()
    neg = len(y) - pos
    scale_pos_weight = neg / pos
    logger.info(
        "Class balance neg={:,} pos={:,} scale_pos_weight={:.1f}",
        neg,
        pos,
        scale_pos_weight,
    )

    lgb_params = {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "boosting_type": "gbdt",
        "num_leaves": 127,
        "max_depth": -1,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "n_jobs": -1,
        "random_state": random_state,
        "verbose": -1,
    }

    out_dir = model_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    uri = mlflow_uri or MLRUNS_DIR.resolve().as_uri()
    exp = experiment_name or MLFLOW_EXPERIMENT_LGB_NOTEBOOK
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    mlflow.set_experiment(exp)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    oof_proba = np.zeros(len(y), dtype=np.float32)
    feature_importances = np.zeros(len(FEATURE_COLS_LGB))

    with mlflow.start_run(run_name="lgbm_aml"):
        mlflow.log_params(lgb_params)
        mlflow.log_param("n_splits", n_splits)
        mlflow.log_param("train_rows", len(df))
        mlflow.log_param("laundering_rate", float(y.mean()))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
            logger.info("Fold {}/{} ...", fold, n_splits)

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
            )

            val_proba = model.predict_proba(X_val)[:, 1]
            oof_proba[val_idx] = val_proba
            feature_importances += model.feature_importances_

            fold_auc = roc_auc_score(y_val, val_proba)
            fold_ap = average_precision_score(y_val, val_proba)
            logger.info("  Fold {}: ROC-AUC={:.4f} AP={:.4f}", fold, fold_auc, fold_ap)

        best_threshold = find_best_threshold(y, oof_proba)
        y_pred = (oof_proba >= best_threshold).astype(int)

        metrics = {
            "oof_roc_auc": roc_auc_score(y, oof_proba),
            "oof_avg_prec": average_precision_score(y, oof_proba),
            "oof_f1": f1_score(y, y_pred),
            "oof_precision": precision_score(y, y_pred),
            "oof_recall": recall_score(y, y_pred),
            "best_threshold": best_threshold,
        }
        mlflow.log_metrics(metrics)

        logger.info("\n" + "=" * 60)
        logger.info("OOF Results:")
        for key, value in metrics.items():
            logger.info("  {}: {:.4f}", key, value)

        logger.info("\nClassification Report (OOF):")
        logger.info(
            "\n"
            + classification_report(y, y_pred, target_names=["Normal", "Laundering"]),
        )

        final_model = lgb.LGBMClassifier(**lgb_params)
        logger.info("Training final model on full dataset...")
        final_model.fit(X, y, callbacks=[lgb.log_evaluation(100)])

        model_path = os.path.join(out_dir, "aml_lgbm.pkl")
        meta_path = os.path.join(out_dir, "model_meta.pkl")

        joblib.dump(final_model, model_path)
        joblib.dump(
            {
                "feature_cols": FEATURE_COLS_LGB,
                "best_threshold": best_threshold,
                "metrics": metrics,
                "feature_importances": dict(zip(FEATURE_COLS_LGB, feature_importances / n_splits)),
            },
            meta_path,
        )

        mlflow.lightgbm.log_model(final_model.booster_, "model")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(meta_path)

        logger.info("Model saved -> {}", model_path)

        logger.info("Computing SHAP values on sample...")
        sample_idx = np.random.choice(len(X), min(1000, len(X)), replace=False)
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X.iloc[sample_idx])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = (
            pd.DataFrame(np.abs(shap_values).mean(axis=0)[np.newaxis, :], columns=FEATURE_COLS_LGB)
            .T.rename(columns={0: "mean_abs_shap"})
            .sort_values("mean_abs_shap", ascending=False)
        )

        shap_path = os.path.join(out_dir, "shap_importance.csv")
        shap_df.to_csv(shap_path)
        mlflow.log_artifact(shap_path)

        logger.info("\nTop-10 SHAP features:")
        logger.info("\n" + shap_df.head(10).to_string())

        run_id = mlflow.active_run().info.run_id
        logger.info("\nMLflow run_id: {}", run_id)
        logger.info("View UI with: mlflow ui --backend-store-uri {}", uri)

    return final_model, metrics
