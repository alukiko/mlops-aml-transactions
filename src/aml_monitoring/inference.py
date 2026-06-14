from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import DEFAULT_THRESHOLD, MODEL_META_PATH, MODEL_PATH
from .features import to_feature_matrix
from .metrics import record_model_metrics


class ModelService:
    def __init__(self) -> None:
        raw = joblib.load(MODEL_PATH)
        if isinstance(raw, dict):
            self.model = raw["pipeline"]
            self.threshold = float(raw.get("threshold", DEFAULT_THRESHOLD))
            self.meta: dict = {}
        else:
            self.model = raw
            self.meta = joblib.load(MODEL_META_PATH) if MODEL_META_PATH.exists() else {}
            self.threshold = float(self.meta.get("best_threshold", DEFAULT_THRESHOLD))
        self.preprocessor = self.meta.get("preprocessor")
        record_model_metrics(self.meta.get("metrics", {}), self.threshold)

    def predict(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        df = pd.DataFrame(records)
        x = to_feature_matrix(df, preprocessor=self.preprocessor)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(x)[:, 1]
        else:
            probabilities = self.model.predict(x)
        results = []
        for payload, probability in zip(records, probabilities):
            prob = float(np.clip(probability, 0, 1))
            predicted_class = int(prob >= self.threshold)
            results.append(
                {
                    "payload": payload,
                    "probability": prob,
                    "predicted_class": predicted_class,
                    "threshold": self.threshold,
                    "anomaly_flag": predicted_class == 1,
                    "drift_flag": False,
                }
            )
        return results


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    return ModelService()
