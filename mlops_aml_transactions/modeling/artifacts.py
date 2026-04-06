"""Сохранение/загрузка модели: sklearn Pipeline или dict с порогом для бинарной классификации."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_model(
    path: Path,
    pipeline: Pipeline,
    *,
    threshold: float | None = None,
) -> None:
    """Если задан threshold — сохраняем dict; иначе только pipeline (обратная совместимость)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if threshold is not None:
        joblib.dump({"pipeline": pipeline, "threshold": float(threshold)}, path)
    else:
        joblib.dump(pipeline, path)


def load_model(path: Path | str) -> tuple[Pipeline, float | None]:
    """Возвращает (pipeline, threshold или None для стандартного predict)."""
    obj: Any = joblib.load(path)
    if isinstance(obj, dict) and "pipeline" in obj:
        return obj["pipeline"], obj.get("threshold")
    if isinstance(obj, Pipeline):
        return obj, None
    raise TypeError(f"Unsupported artifact type: {type(obj)}")
