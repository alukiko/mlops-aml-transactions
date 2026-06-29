# Models

Production model artifacts are versioned by the DVC pipeline and stored in the configured remote.

- `hgb_compat.pkl` — model loaded by the FastAPI service by default.
- `model_meta.pkl` — preprocessing metadata, threshold and baseline metrics.
