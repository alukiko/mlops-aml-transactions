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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import typer

from mlops_aml_transactions.config import MLFLOW_EXPERIMENT_NAME, MLRUNS_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from mlops_aml_transactions.data.s3 import s3_key_for_local_path, s3_upload_file
from mlops_aml_transactions.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, X_y_from_frame
from mlops_aml_transactions.modeling.artifacts import save_model

app = typer.Typer()

AVAILABLE_MODELS: dict[str, str] = {
    "hgb": "HistGradientBoostingClassifier",
    "rf": "RandomForestClassifier",
    "et": "ExtraTreesClassifier",
    "lr": "LogisticRegression",
}

SelectBy = Literal["average_precision", "roc_auc", "f1"]
SplitStrategy = Literal["random", "group", "time"]
ThresholdObjective = Literal["fbeta", "recall"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def make_et_estimator(
    *,
    random_state: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: str | float,
    min_samples_split: int,
):
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        min_samples_split=min_samples_split,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


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
        return make_et_estimator(
            random_state=random_state,
            n_estimators=300,
            max_depth=18,
            min_samples_leaf=8,
            max_features="sqrt",
            min_samples_split=2,
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
            "clf_min_samples_split": 2,
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


def parse_model_list(spec: str) -> list[str]:
    parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
    unknown = set(parts) - set(AVAILABLE_MODELS)
    if unknown:
        raise typer.BadParameter(
            f"Unknown model(s): {sorted(unknown)}. Choose from: {', '.join(sorted(AVAILABLE_MODELS))}"
        )
    if not parts:
        raise typer.BadParameter("Provide at least one model name.")
    return parts


def parse_int_list(spec: str) -> list[int]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise typer.BadParameter("seeds must contain at least one integer")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise typer.BadParameter("seeds must be a comma-separated list of integers") from exc


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


def parse_split_strategy(s: str) -> SplitStrategy:
    key = s.strip().lower()
    allowed = {"random", "group", "time"}
    if key not in allowed:
        raise typer.BadParameter("split-strategy must be one of: random, group, time")
    return key  # type: ignore[return-value]


def parse_threshold_objective(s: str) -> ThresholdObjective:
    key = s.strip().lower()
    allowed = {"fbeta", "recall"}
    if key not in allowed:
        raise typer.BadParameter("threshold-objective must be one of: fbeta, recall")
    return key  # type: ignore[return-value]


def _safe_random_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    raw_df: pd.DataFrame,
    *,
    strategy: SplitStrategy,
    test_size: float,
    random_state: int,
    group_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if strategy == "random":
        return _safe_random_split(X, y, test_size=test_size, random_state=random_state)

    if strategy == "group":
        if group_col not in raw_df.columns:
            raise typer.BadParameter(f"group column not found: {group_col}")
        groups = raw_df[group_col].fillna("UNK").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        if time_col not in raw_df.columns:
            raise typer.BadParameter(f"time column not found: {time_col}")
        ts = pd.to_datetime(raw_df[time_col], errors="coerce")
        ts = ts.fillna(pd.Timestamp.min)
        order = np.argsort(ts.to_numpy())
        cutoff = int((1 - test_size) * len(order))
        cutoff = max(1, min(cutoff, len(order) - 1))
        train_idx = order[:cutoff]
        test_idx = order[cutoff:]

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # If split is degenerate for rare positives, fallback to stratified random split.
    if int(y_train.sum()) == 0 or int(y_test.sum()) == 0:
        logger.warning(
            "Split strategy '{}' produced no positives in one split; fallback to random stratified split.",
            strategy,
        )
        return _safe_random_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    raw_df: pd.DataFrame,
    *,
    strategy: SplitStrategy,
    test_size: float,
    val_size: float,
    random_state: int,
    group_col: str,
    time_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame, pd.Series, pd.Series | None, pd.Series]:
    if strategy != "time":
        X_train, X_test, y_train, y_test = split_train_test(
            X,
            y,
            raw_df,
            strategy=strategy,
            test_size=test_size,
            random_state=random_state,
            group_col=group_col,
            time_col=time_col,
        )
        return X_train, None, X_test, y_train, None, y_test

    if time_col not in raw_df.columns:
        raise typer.BadParameter(f"time column not found: {time_col}")
    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 0.9:
        raise typer.BadParameter("for time split, val_size and test_size must be > 0 and val_size+test_size < 0.9")

    ts = pd.to_datetime(raw_df[time_col], errors="coerce").fillna(pd.Timestamp.min)
    order = np.argsort(ts.to_numpy())
    n = len(order)

    train_end = int((1 - val_size - test_size) * n)
    val_end = int((1 - test_size) * n)
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    y_test = y.iloc[test_idx]

    if int(y_train.sum()) == 0 or int(y_val.sum()) == 0 or int(y_test.sum()) == 0:
        logger.warning(
            "Time split produced zero positives in one partition; fallback to random stratified split."
        )
        X_train, X_test, y_train, y_test = _safe_random_split(
            X, y, test_size=test_size, random_state=random_state
        )
        return X_train, None, X_test, y_train, None, y_test

    return X_train, X_val, X_test, y_train, y_val, y_test


def main_score(select_by: SelectBy, roc: float, pr: float, f1: float) -> float:
    if select_by == "average_precision":
        return pr
    if select_by == "roc_auc":
        return roc
    return f1


def tune_threshold_fbeta(
    y_val: pd.Series,
    proba_val: np.ndarray,
    *,
    beta: float,
    points: int,
    min_threshold: float,
    max_threshold: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(min_threshold, max_threshold, points)
    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        pred = (proba_val >= t).astype(int)
        fb = fbeta_score(y_val, pred, beta=beta, zero_division=0)
        if fb > best_f:
            best_f, best_t = fb, t
    pred_best = (proba_val >= best_t).astype(int)
    return float(best_t), {
        "recall_val": float(recall_score(y_val, pred_best, zero_division=0)),
        "precision_val": float(precision_score(y_val, pred_best, zero_division=0)),
        "f1_val": float(f1_score(y_val, pred_best, zero_division=0)),
        "target_recall_met_val": 0.0,
    }


def tune_threshold_recall_target(
    y_val: pd.Series,
    proba_val: np.ndarray,
    *,
    target_recall: float,
    points: int,
    min_threshold: float,
    max_threshold: float,
) -> tuple[float, dict[str, float]]:
    thresholds = np.linspace(min_threshold, max_threshold, points)
    rows: list[tuple[float, float, float, float]] = []
    for t in thresholds:
        pred = (proba_val >= t).astype(int)
        rec = float(recall_score(y_val, pred, zero_division=0))
        prec = float(precision_score(y_val, pred, zero_division=0))
        f1 = float(f1_score(y_val, pred, zero_division=0))
        rows.append((float(t), rec, prec, f1))

    feasible = [r for r in rows if r[1] >= target_recall]
    if feasible:
        # minimal threshold that reaches recall target; tiebreak by precision then f1
        feasible_sorted = sorted(feasible, key=lambda r: (r[0], -r[2], -r[3]))
        chosen = feasible_sorted[0]
        met = 1.0
    else:
        # fallback: max recall, then max precision, then max f1, then lowest threshold
        chosen = sorted(rows, key=lambda r: (-r[1], -r[2], -r[3], r[0]))[0]
        met = 0.0

    return chosen[0], {
        "recall_val": chosen[1],
        "precision_val": chosen[2],
        "f1_val": chosen[3],
        "target_recall_met_val": met,
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    X_val: pd.DataFrame | None,
    y_val: pd.Series | None,
    split_random_state: int,
    tune_threshold: bool,
    val_fraction: float,
    threshold_objective: ThresholdObjective,
    target_recall: float,
    threshold_beta: float,
    threshold_grid_points: int,
    min_threshold: float,
    max_threshold: float,
) -> tuple[dict[str, float], float | None, dict[str, float]]:
    threshold: float | None = None
    threshold_meta = {
        "recall_val": 0.0,
        "precision_val": 0.0,
        "f1_val": 0.0,
        "target_recall_met_val": 0.0,
    }
    if tune_threshold:
        if X_val is not None and y_val is not None:
            pipeline.fit(X_train, y_train)
        else:
            X_fit, X_val_inner, y_fit, y_val_inner = train_test_split(
                X_train,
                y_train,
                test_size=val_fraction,
                stratify=y_train,
                random_state=split_random_state,
            )
            pipeline.fit(X_fit, y_fit)
            X_val = X_val_inner
            y_val = y_val_inner
        proba_val = pipeline.predict_proba(X_val)[:, 1]
        if threshold_objective == "recall":
            threshold, threshold_meta = tune_threshold_recall_target(
                y_val,
                proba_val,
                target_recall=target_recall,
                points=threshold_grid_points,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )
        else:
            threshold, threshold_meta = tune_threshold_fbeta(
                y_val,
                proba_val,
                beta=threshold_beta,
                points=threshold_grid_points,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )
        pipeline.fit(X_train, y_train)
    else:
        pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    if threshold is None:
        pred = pipeline.predict(X_test)
    else:
        pred = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "average_precision_test": float(average_precision_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "f1_test": float(f1_score(y_test, pred, zero_division=0)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "precision_test": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "recall_test": float(recall_score(y_test, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
    }
    metrics["classification_report"] = classification_report(y_test, pred)
    return metrics, threshold, threshold_meta


def et_candidate_grid(random_state: int, n_configs: int) -> list[dict[str, Any]]:
    if n_configs <= 0:
        return [
            {
                "n_estimators": 300,
                "max_depth": 18,
                "min_samples_leaf": 8,
                "max_features": "sqrt",
                "min_samples_split": 2,
            }
        ]

    rng = np.random.default_rng(random_state)
    n_estimators_choices = np.array([200, 300, 400, 500])
    max_depth_choices = np.array([12, 16, 20, 24, 28])
    min_leaf_choices = np.array([2, 4, 8, 12])
    max_features_choices = np.array(["sqrt", "log2", 0.6], dtype=object)
    min_split_choices = np.array([2, 4, 8])

    configs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    max_trials = max(60, n_configs * 8)

    while len(configs) < n_configs and max_trials > 0:
        max_trials -= 1
        cfg = {
            "n_estimators": int(rng.choice(n_estimators_choices)),
            "max_depth": int(rng.choice(max_depth_choices)),
            "min_samples_leaf": int(rng.choice(min_leaf_choices)),
            "max_features": rng.choice(max_features_choices),
            "min_samples_split": int(rng.choice(min_split_choices)),
        }
        key = (
            cfg["n_estimators"],
            cfg["max_depth"],
            cfg["min_samples_leaf"],
            cfg["max_features"],
            cfg["min_samples_split"],
        )
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    if not configs:
        configs = [
            {
                "n_estimators": 300,
                "max_depth": 18,
                "min_samples_leaf": 8,
                "max_features": "sqrt",
                "min_samples_split": 2,
            }
        ]
    return configs


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    test_size: float = typer.Option(0.2, help="Holdout fraction for metrics."),
    val_size: float = typer.Option(0.2, help="[time split] Validation fraction before holdout test."),
    random_state: int = 42,
    split_strategy: str = typer.Option("random", help="Outer split strategy: random, group, time."),
    group_col: str = typer.Option("from_account", help="[group split] Group column to isolate entities."),
    time_col: str = typer.Option("Timestamp", help="[time split] Timestamp column for chronological holdout."),
    models: str = typer.Option(
        "et,rf",
        help="Comma-separated models to train: hgb, rf, et, lr.",
    ),
    seeds: str = typer.Option("42,43,44", help="Comma-separated seeds used for ET stability runs."),
    et_tune_configs: int = typer.Option(16, help="Number of ET hyperparameter configs to evaluate per seed."),
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
    tune_threshold: bool = typer.Option(True, help="Tune decision threshold on validation split."),
    val_fraction: float = typer.Option(0.15, help="Validation split fraction used for threshold tuning."),
    threshold_objective: str = typer.Option(
        "fbeta",
        help="Threshold objective: fbeta or recall.",
    ),
    target_recall: float = typer.Option(0.6, help="[recall objective] target recall on validation set."),
    min_threshold: float = typer.Option(0.001, help="Minimum threshold for threshold search grid."),
    max_threshold: float = typer.Option(0.999, help="Maximum threshold for threshold search grid."),
    threshold_beta: float = typer.Option(1.0, help="Beta in F-beta for threshold tuning."),
    threshold_grid_points: int = typer.Option(199, help="Threshold grid points in [0.01, 0.99]."),
    select_by: str = typer.Option("f1", help="Best-model metric: ap, roc_auc, f1."),
) -> None:
    """Train AML models with optional ET tuning and multi-seed stability control."""
    if not dataset_path.is_file():
        raise typer.BadParameter(f"Dataset not found: {dataset_path}. Run dataset step first.")
    if threshold_grid_points < 10:
        raise typer.BadParameter("threshold_grid_points must be >= 10")
    if et_tune_configs < 0:
        raise typer.BadParameter("et_tune_configs must be >= 0")
    if not (0.0 < target_recall <= 1.0):
        raise typer.BadParameter("target_recall must be in (0, 1]")
    if not (0.0 <= min_threshold < max_threshold <= 1.0):
        raise typer.BadParameter("threshold bounds must satisfy 0<=min<threshold<max<=1")

    model_keys = parse_model_list(models)
    select_key = parse_select_by(select_by)
    split_key = parse_split_strategy(split_strategy)
    threshold_key = parse_threshold_objective(threshold_objective)
    seed_values = parse_int_list(seeds)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    df = pd.read_csv(dataset_path)
    X, y = X_y_from_frame(df)

    shared_params = {
        "test_size": test_size,
        "val_size": val_size,
        "random_state": random_state,
        "split_strategy": split_key,
        "group_col": group_col,
        "time_col": time_col,
        "ordinal_unknown_value": -1,
        "n_features": X.shape[1],
        "feature_set": "v2_log_amounts_calendar_currency_match",
        "n_rows_total": len(df),
        "dataset_path": str(dataset_path.resolve()),
        "models_trained": ",".join(model_keys),
        "tune_threshold": tune_threshold,
        "val_fraction": val_fraction if tune_threshold else 0.0,
        "threshold_objective": threshold_key,
        "target_recall": target_recall,
        "min_threshold": min_threshold,
        "max_threshold": max_threshold,
        "threshold_beta": threshold_beta,
        "threshold_grid_points": threshold_grid_points,
        "select_best_by": select_key,
        "et_tune_configs": et_tune_configs,
        "et_seeds": ",".join(str(s) for s in seed_values),
    }

    best_selector = (-1.0, -1.0)
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_threshold: float | None = None
    best_score_value = -1.0

    with mlflow.start_run(run_name=run_name) as parent:
        mlflow.log_params(shared_params)
        mlflow.set_tag("run_kind", "multi_model_train")

        for key in model_keys:
            if key == "et":
                seed_runs: list[dict[str, Any]] = []

                for seed in seed_values:
                    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                        X,
                        y,
                        df,
                        strategy=split_key,
                        test_size=test_size,
                        val_size=val_size,
                        random_state=seed,
                        group_col=group_col,
                        time_col=time_col,
                    )

                    candidate_rows: list[dict[str, Any]] = []
                    best_local: dict[str, Any] | None = None
                    best_local_score = -1.0

                    for idx, cfg in enumerate(et_candidate_grid(seed, et_tune_configs), start=1):
                        clf = make_et_estimator(random_state=seed, **cfg)
                        pipeline = build_pipeline(clf)
                        metrics, threshold, threshold_meta = evaluate_pipeline(
                            pipeline,
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                            X_val=X_val,
                            y_val=y_val,
                            split_random_state=seed,
                            tune_threshold=tune_threshold,
                            val_fraction=val_fraction,
                            threshold_objective=threshold_key,
                            target_recall=target_recall,
                            threshold_beta=threshold_beta,
                            threshold_grid_points=threshold_grid_points,
                            min_threshold=min_threshold,
                            max_threshold=max_threshold,
                        )

                        row = {
                            "seed": seed,
                            "candidate_idx": idx,
                            **cfg,
                            "threshold": threshold,
                            "roc_auc": metrics["roc_auc"],
                            "average_precision": metrics["average_precision"],
                            "f1": metrics["f1"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "precision": metrics["precision"],
                            "recall": metrics["recall"],
                            "recall_val": threshold_meta["recall_val"],
                            "precision_val": threshold_meta["precision_val"],
                            "f1_val": threshold_meta["f1_val"],
                            "target_recall_met_val": threshold_meta["target_recall_met_val"],
                        }
                        candidate_rows.append(row)

                        if metrics["f1"] > best_local_score:
                            best_local_score = metrics["f1"]
                            best_local = {
                                "seed": seed,
                                "config": cfg,
                                "pipeline": pipeline,
                                "threshold": threshold,
                                "metrics": metrics,
                                "threshold_meta": threshold_meta,
                            }

                    if best_local is None:
                        raise RuntimeError("ET tuning produced no candidate")

                    with mlflow.start_run(run_name=f"et_seed_{seed}", nested=True):
                        mlflow.set_tags(
                            {
                                "model_key": "et",
                                "model_class": AVAILABLE_MODELS["et"],
                                "seed": seed,
                                "preprocessor": "ColumnTransformer_OrdinalEncoder",
                            }
                        )
                        mlflow.log_params(best_local["config"])
                        mlflow.log_param("et_candidates_evaluated", len(candidate_rows))
                        if best_local["threshold"] is not None:
                            mlflow.log_param("decision_threshold", best_local["threshold"])
                        if threshold_key == "recall" and best_local["threshold_meta"]["target_recall_met_val"] < 0.5:
                            mlflow.set_tag("threshold_fallback", "recall_target_unmet")

                        metrics = best_local["metrics"]
                        mlflow.log_metrics(
                            {
                                "roc_auc": metrics["roc_auc"],
                                "average_precision": metrics["average_precision"],
                                "average_precision_test": metrics["average_precision_test"],
                                "f1": metrics["f1"],
                                "f1_test": metrics["f1_test"],
                                "precision": metrics["precision"],
                                "precision_test": metrics["precision_test"],
                                "recall": metrics["recall"],
                                "recall_test": metrics["recall_test"],
                                "recall_val": threshold_meta["recall_val"],
                                "precision_val": threshold_meta["precision_val"],
                                "f1_val": threshold_meta["f1_val"],
                                "target_recall_met_val": threshold_meta["target_recall_met_val"],
                                "balanced_accuracy": metrics["balanced_accuracy"],
                            }
                        )
                        mlflow.log_text(metrics["classification_report"], f"classification_report_et_seed_{seed}.txt")

                        cand_df = pd.DataFrame(candidate_rows).sort_values("f1", ascending=False)
                        mlflow.log_text(cand_df.to_csv(index=False), f"et_candidates_seed_{seed}.csv")

                        seed_file = MODELS_DIR / f"et_seed_{seed}.pkl"
                        save_model(seed_file, best_local["pipeline"], threshold=best_local["threshold"])
                        mlflow.log_artifact(str(seed_file))
                        try:
                            s3_key = s3_key_for_local_path(seed_file, kind="models")
                            s3_upload_file(seed_file, s3_key)
                        except Exception:
                            pass

                    logger.info(
                        "[et seed={}] ROC-AUC: {:.4f} | AP: {:.4f} | F1: {:.4f} | bal_acc: {:.4f}{}",
                        seed,
                        best_local["metrics"]["roc_auc"],
                        best_local["metrics"]["average_precision"],
                        best_local["metrics"]["f1"],
                        best_local["metrics"]["balanced_accuracy"],
                        f" | thr={best_local['threshold']:.4f}" if best_local["threshold"] is not None else "",
                    )
                    if threshold_key == "recall" and best_local["threshold_meta"]["target_recall_met_val"] < 0.5:
                        logger.warning(
                            "[et seed={}] target recall {:.2f} not reached on val (recall_val={:.4f}).",
                            seed,
                            target_recall,
                            best_local["threshold_meta"]["recall_val"],
                        )
                    seed_runs.append(best_local)

                f1_values = np.array([r["metrics"]["f1"] for r in seed_runs], dtype=float)
                mean_f1 = float(f1_values.mean())
                std_f1 = float(f1_values.std(ddof=0))
                rep = sorted(seed_runs, key=lambda r: r["metrics"]["f1"], reverse=True)[0]

                mlflow.log_metrics(
                    {
                        "et_mean_f1": mean_f1,
                        "et_std_f1": std_f1,
                        "et_mean_roc_auc": float(np.mean([r["metrics"]["roc_auc"] for r in seed_runs])),
                        "et_mean_average_precision": float(
                            np.mean([r["metrics"]["average_precision"] for r in seed_runs])
                        ),
                    }
                )

                signature = infer_signature(X, rep["pipeline"].predict(X.head(min(2000, len(X)))))
                mlflow.sklearn.log_model(rep["pipeline"], name="sklearn_pipeline_et_stable", signature=signature)

                out_file = MODELS_DIR / "et.pkl"
                save_model(out_file, rep["pipeline"], threshold=rep["threshold"])
                mlflow.log_artifact(str(out_file))
                try:
                    s3_key = s3_key_for_local_path(out_file, kind="models")
                    s3_upload_file(out_file, s3_key)
                except Exception:
                    pass

                selector = (mean_f1, -std_f1)
                score_value = mean_f1
                if selector > best_selector:
                    best_selector = selector
                    best_name = "et"
                    best_pipeline = rep["pipeline"]
                    best_threshold = rep["threshold"]
                    best_score_value = score_value
                continue

            # Single-run branch for non-ET models.
            X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
                X,
                y,
                df,
                strategy=split_key,
                test_size=test_size,
                val_size=val_size,
                random_state=random_state,
                group_col=group_col,
                time_col=time_col,
            )
            estimator = make_estimator(
                key,
                random_state=random_state,
                hgb_max_depth=hgb_max_depth,
                hgb_learning_rate=hgb_learning_rate,
                hgb_max_iter=hgb_max_iter,
            )
            pipeline = build_pipeline(estimator)
            metrics, threshold, threshold_meta = evaluate_pipeline(
                pipeline,
                X_train,
                y_train,
                X_test,
                y_test,
                X_val=X_val,
                y_val=y_val,
                split_random_state=random_state,
                tune_threshold=tune_threshold,
                val_fraction=val_fraction,
                threshold_objective=threshold_key,
                target_recall=target_recall,
                threshold_beta=threshold_beta,
                threshold_grid_points=threshold_grid_points,
                min_threshold=min_threshold,
                max_threshold=max_threshold,
            )

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
                if threshold_key == "recall" and threshold_meta["target_recall_met_val"] < 0.5:
                    mlflow.set_tag("threshold_fallback", "recall_target_unmet")

                mlflow.log_metrics(
                    {
                        "roc_auc": metrics["roc_auc"],
                        "average_precision": metrics["average_precision"],
                        "average_precision_test": metrics["average_precision_test"],
                        "f1": metrics["f1"],
                        "f1_test": metrics["f1_test"],
                        "precision": metrics["precision"],
                        "precision_test": metrics["precision_test"],
                        "recall": metrics["recall"],
                        "recall_test": metrics["recall_test"],
                        "recall_val": threshold_meta["recall_val"],
                        "precision_val": threshold_meta["precision_val"],
                        "f1_val": threshold_meta["f1_val"],
                        "target_recall_met_val": threshold_meta["target_recall_met_val"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                    }
                )
                mlflow.log_text(metrics["classification_report"], f"classification_report_{key}.txt")

                logger.info(
                    "[{}] ROC-AUC: {:.4f} | AP: {:.4f} | F1: {:.4f} | bal_acc: {:.4f}{}",
                    key,
                    metrics["roc_auc"],
                    metrics["average_precision"],
                    metrics["f1"],
                    metrics["balanced_accuracy"],
                    f" | thr={threshold:.4f}" if threshold is not None else "",
                )
                if threshold_key == "recall" and threshold_meta["target_recall_met_val"] < 0.5:
                    logger.warning(
                        "[{}] target recall {:.2f} not reached on val (recall_val={:.4f}).",
                        key,
                        target_recall,
                        threshold_meta["recall_val"],
                    )

                signature = infer_signature(X_train, pipeline.predict(X_train))
                mlflow.sklearn.log_model(pipeline, name=f"sklearn_pipeline_{key}", signature=signature)

                out_file = MODELS_DIR / f"{key}.pkl"
                save_model(out_file, pipeline, threshold=threshold)
                mlflow.log_artifact(str(out_file))
                try:
                    s3_key = s3_key_for_local_path(out_file, kind="models")
                    s3_upload_file(out_file, s3_key)
                except Exception:
                    pass

            score = main_score(select_key, metrics["roc_auc"], metrics["average_precision"], metrics["f1"])
            selector = (score, 0.0)
            if selector > best_selector:
                best_selector = selector
                best_name = key
                best_pipeline = pipeline
                best_threshold = threshold
                best_score_value = score

        if best_pipeline is not None and best_name is not None:
            save_model(model_path, best_pipeline, threshold=best_threshold)
            try:
                s3_key = s3_key_for_local_path(model_path, kind="models")
                s3_upload_file(model_path, s3_key)
            except Exception:
                pass

            mlflow.set_tag("best_model", best_name)
            if best_name == "et" and len(seed_values) > 1:
                mlflow.set_tag("best_select_metric", "mean_f1")
                mlflow.log_metric("best_mean_f1", best_score_value)
            else:
                mlflow.set_tag("best_select_metric", select_key)
                mlflow.log_metric(f"best_{select_key}", best_score_value)
            if best_threshold is not None:
                mlflow.log_param("best_decision_threshold", best_threshold)

            logger.success("Best model: {} (score={:.4f}) -> {}", best_name, best_score_value, model_path)
            logger.success("Parent MLflow run id: {}", parent.info.run_id)
        else:
            logger.error("No model was trained.")

    logger.info("Open MLflow UI: mlflow ui --backend-store-uri {}", uri)


if __name__ == "__main__":
    app()
