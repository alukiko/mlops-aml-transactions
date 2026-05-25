# AML Monitoring Platform

Production-like MLOps project for AML transaction classification. The repository contains a trained LightGBM model, FastAPI inference service, React operations UI, drift reports, Prometheus/Grafana monitoring, MLflow experiment tracking, Docker Compose, Kubernetes manifests and an Argo CD application.

## Components

- `src/aml_monitoring` - backend package: data loading, feature engineering, inference, drift analysis, retraining, SQLite storage and API.
- `frontend` - React/Vite UI for inference, recent predictions, drift alerts, experiments and retraining.
- `models` - trained model artifacts: `aml_lgbm.pkl`, `model_meta.pkl`, SHAP importance.
- `data` - IBM AML synthetic transaction CSV files.
- `runtime/mlflow.db` - MLflow SQLite backend store and local Model Registry metadata.
- `mlartifacts` - MLflow model artifacts.
- `monitoring` - Prometheus scrape config and Grafana dashboard provisioning.
- `k8s` - Kubernetes manifests and Argo CD application.
- `reports/drift` and `runtime` - generated reports and SQLite/reference runtime files.
- `.github/workflows/ci_cd.yaml` - CI/CD pipeline with lint, tests, Docker image build and Argo CD sync.

## Local Python Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=%CD%\src
```

Run the backend:

```bash
uvicorn aml_monitoring.main:app --reload --host 0.0.0.0 --port 8000
```

Useful URLs:

- API docs: `http://localhost:8000/docs`
- Prometheus metrics: `http://localhost:8000/metrics`
- Health: `http://localhost:8000/health`

## Run with Docker Compose

```bash
docker compose up --build
```

Services:

- UI: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001` with `admin/admin`

Docker Compose uses SQLite for MLflow metadata and Model Registry:

- backend store: `sqlite:///runtime/mlflow.db`;
- artifact root: `mlartifacts/`.

## API Examples

Prediction:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"Timestamp\":\"2022/09/01 00:20\",\"From Bank\":\"010\",\"From Account\":\"8000EBD30\",\"To Bank\":\"010\",\"To Account\":\"8000EBD30\",\"Amount Received\":3697.34,\"Receiving Currency\":\"US Dollar\",\"Amount Paid\":3697.34,\"Payment Currency\":\"US Dollar\",\"Payment Format\":\"Reinvestment\"}]}"
```

Run drift report:

```bash
curl -X POST http://localhost:8000/drift/run -H "Content-Type: application/json" -d "{}"
```

By default `/drift/run` uses the latest prediction payloads saved in SQLite, so alerts change as new inference traffic arrives. To run a drift check for an explicit batch:

```bash
curl -X POST http://localhost:8000/drift/run ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"Timestamp\":\"2022/09/01 00:20\",\"From Bank\":\"010\",\"From Account\":\"A1\",\"To Bank\":\"011\",\"To Account\":\"A2\",\"Amount Received\":100,\"Receiving Currency\":\"US Dollar\",\"Amount Paid\":100,\"Payment Currency\":\"US Dollar\",\"Payment Format\":\"Wire\"}]}"
```

Start retraining:

```bash
curl -X POST http://localhost:8000/retrain
```

Retraining logs parameters, metrics and artifacts to MLflow and registers the model under `aml-laundering-detector` by default. Override with:

```bash
set MLFLOW_REGISTERED_MODEL_NAME=my-model-name
```

For local Python runs, point MLflow to the same SQLite store:

```bash
set MLFLOW_TRACKING_URI=sqlite:///%CD%/runtime/mlflow.db
set MLFLOW_REGISTRY_URI=sqlite:///%CD%/runtime/mlflow.db
```

## Drift Reports

Drift jobs compare current data with a reference sample. If `runtime/reference_sample.csv` is missing, it is created from the source CSV files on first run.

In the API/UI path, current data means recent inference traffic stored in SQLite. If there are fewer than `DRIFT_MIN_ROWS` records, the run is saved with `not_enough_data` status. The local Docker Compose stack also starts an in-process scheduler:

- `DRIFT_SCHEDULER_ENABLED=true`;
- `DRIFT_SCHEDULER_INTERVAL_SECONDS=300`;
- `DRIFT_PREDICTION_LIMIT=1000`;
- `DRIFT_MIN_ROWS=30`.

In Kubernetes, `k8s/base/drift-cronjob.yaml` runs the same check every 15 minutes by calling `POST /drift/run`. Change the CronJob `schedule` to tune the interval.

Implemented checks:

- data drift: PSI and KS for numeric features, PSI for categorical features;
- target drift: target-rate change when `Is Laundering` labels are present;
- concept drift: precision, recall, F1 and ROC-AUC drop when labels are present.

Reports are generated in:

- `reports/drift/*.json`
- `reports/drift/*.html`

## UI

The React UI is an operations workspace, not a landing page. It includes:

- inference form with JSON single/batch input;
- table of recent predictions and anomaly flags;
- drift alert panel and report list;
- MLflow experiment table;
- retraining button and job status table.

For local frontend development:

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL=http://localhost:8000` when running Vite directly.

## Monitoring

The backend exposes Prometheus metrics at `/metrics`, including:

- API request count and latency;
- prediction count;
- anomaly rate;
- average laundering probability;
- data, target and concept drift scores;
- retraining status.

Grafana is provisioned with the dashboard `AML Monitoring`.

## Kubernetes and Argo CD

Manifests are in `k8s/base`. The default namespace is `aml-monitoring`.

Before deploying, replace placeholders:

- image names in `k8s/base/backend.yaml`, `frontend.yaml`, `mlflow.yaml`, `retraining-job.yaml`;
- `repoURL` in `k8s/argocd/application.yaml`;
- secret values in `k8s/base/secret.yaml`;
- storage and ingress/load balancer settings as required by your cluster.

Apply directly:

```bash
kubectl apply -k k8s/base
```

Apply through Argo CD:

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Argo CD will sync the manifests from the configured Git repository path `k8s/base`.

## CI/CD

GitHub Actions workflow `.github/workflows/ci_cd.yaml` runs on pull requests to `main` and pushes to `main`.

Pipeline stages:

- lint backend with `ruff`;
- run backend tests with `pytest`;
- build frontend with `npm run build`;
- build backend and frontend Docker images;
- push images to GHCR on `main`;
- sync the Argo CD application on `main`.

Required GitHub secrets for deploy:

- `ARGOCD_SERVER`
- `ARGOCD_AUTH_TOKEN`
- optional `ARGOCD_APP`, default `aml-monitoring`

## Tests

```bash
set PYTHONPATH=%CD%\src
pytest src/tests
```

The test suite covers feature engineering and drift calculators. API and UI smoke tests can be added after the Docker stack is running in CI.
