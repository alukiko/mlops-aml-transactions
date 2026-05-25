import os
from pathlib import Path
from typing import Any

from .config import DEFAULT_MLFLOW_TRACKING_URI, MLRUNS_DIR


def _read_text(path: Path) -> str | None:
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def list_experiments(limit: int = 50) -> list[dict[str, Any]]:
    try:
        return list_experiments_from_mlflow(limit)
    except Exception:
        return list_experiments_from_files(limit)


def list_experiments_from_mlflow(limit: int = 50) -> list[dict[str, Any]]:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI))
    runs: list[dict[str, Any]] = []
    experiments = client.search_experiments()
    for experiment in experiments:
        for run in client.search_runs([experiment.experiment_id], max_results=limit):
            runs.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "experiment_name": experiment.name,
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("mlflow.runName"),
                    "status": run.info.status,
                    "metrics": dict(run.data.metrics),
                    "params": dict(run.data.params),
                }
            )
            if len(runs) >= limit:
                return runs
    return runs


def list_experiments_from_files(limit: int = 50) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    if not MLRUNS_DIR.exists():
        return runs
    for experiment_dir in MLRUNS_DIR.iterdir():
        if not experiment_dir.is_dir() or experiment_dir.name == "0":
            continue
        for run_dir in experiment_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name == "models":
                continue
            metrics = {}
            params = {}
            metrics_dir = run_dir / "metrics"
            params_dir = run_dir / "params"
            if metrics_dir.exists():
                for metric_file in metrics_dir.iterdir():
                    text = _read_text(metric_file)
                    if text:
                        parts = text.split()
                        metrics[metric_file.name] = float(parts[-2] if len(parts) >= 2 else parts[0])
            if params_dir.exists():
                for param_file in params_dir.iterdir():
                    params[param_file.name] = _read_text(param_file)
            runs.append(
                {
                    "experiment_id": experiment_dir.name,
                    "run_id": run_dir.name,
                    "metrics": metrics,
                    "params": params,
                    "run_name": _read_text(run_dir / "tags" / "mlflow.runName"),
                }
            )
    return runs[:limit]
