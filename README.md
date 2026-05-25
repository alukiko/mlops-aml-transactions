# AML Monitoring Platform

Production-like MLOps-проект для классификации AML-транзакций. В проекте есть FastAPI backend для инференса, React UI для оператора, расчет drift, отчеты, Prometheus/Grafana мониторинг, MLflow tracking/model registry, Docker Compose для локальной отладки, Kubernetes manifests и Argo CD application.

## Из чего состоит проект

- `src/aml_monitoring` - backend-пакет: загрузка данных, feature engineering, инференс, drift-анализ, retraining, SQLite-хранилище и API.
- `frontend` - React/Vite интерфейс: инференс, последние предсказания, drift alerts, эксперименты и запуск переобучения.
- `models` - артефакты модели. Тяжелые `.pkl` и `.csv` артефакты не коммитятся, см. `models/README.md`.
- `data` - место для IBM AML synthetic transaction CSV. CSV-файлы не коммитятся, см. `data/README.md`.
- `runtime/mlflow.db` - локальное SQLite-хранилище MLflow tracking и Model Registry.
- `mlartifacts` - MLflow artifacts.
- `monitoring` - конфигурация Prometheus и provisioning dashboard для Grafana.
- `k8s` - Kubernetes manifests и Argo CD application.
- `reports/drift` и `runtime` - сгенерированные drift-отчеты, SQLite runtime DB и reference sample.
- `.github/workflows/ci_cd.yaml` - CI/CD pipeline: линтер, тесты, сборка Docker images и Argo CD sync.

## Локальный запуск backend без Docker

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set PYTHONPATH=%CD%\src
```

Запуск FastAPI:

```bash
uvicorn aml_monitoring.main:app --reload --host 0.0.0.0 --port 8000
```

Полезные URL:

- OpenAPI/Swagger: `http://localhost:8000/docs`
- Prometheus metrics: `http://localhost:8000/metrics`
- Health check: `http://localhost:8000/health`

## Запуск через Docker Compose

```bash
docker compose up --build
```

Сервисы:

- Web UI: `http://localhost:3000`
- Backend API: `http://localhost:8000`
- MLflow UI: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`, логин/пароль `admin/admin`

Остановка:

```bash
docker compose down
```

Docker Compose использует SQLite для MLflow metadata и Model Registry:

- backend store: `sqlite:///runtime/mlflow.db`
- artifact root: `mlartifacts/`

## Примеры API

Инференс:

```bash
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"Timestamp\":\"2022/09/01 00:20\",\"From Bank\":\"010\",\"From Account\":\"8000EBD30\",\"To Bank\":\"010\",\"To Account\":\"8000EBD30\",\"Amount Received\":3697.34,\"Receiving Currency\":\"US Dollar\",\"Amount Paid\":3697.34,\"Payment Currency\":\"US Dollar\",\"Payment Format\":\"Reinvestment\"}]}"
```

Запуск drift report:

```bash
curl -X POST http://localhost:8000/drift/run -H "Content-Type: application/json" -d "{}"
```

По умолчанию `/drift/run` берет последние payload предсказаний из SQLite. Поэтому drift alerts меняются после нового inference-трафика, а не считаются по одному и тому же файлу.

Запуск drift по явному batch:

```bash
curl -X POST http://localhost:8000/drift/run ^
  -H "Content-Type: application/json" ^
  -d "{\"transactions\":[{\"Timestamp\":\"2022/09/01 00:20\",\"From Bank\":\"010\",\"From Account\":\"A1\",\"To Bank\":\"011\",\"To Account\":\"A2\",\"Amount Received\":100,\"Receiving Currency\":\"US Dollar\",\"Amount Paid\":100,\"Payment Currency\":\"US Dollar\",\"Payment Format\":\"Wire\"}]}"
```

Запуск переобучения:

```bash
curl -X POST http://localhost:8000/retrain
```

Retraining логирует параметры, метрики и артефакты в MLflow. Модель регистрируется под именем `aml-laundering-detector` по умолчанию. Имя можно переопределить:

```bash
set MLFLOW_REGISTERED_MODEL_NAME=my-model-name
```

Для локального Python-запуска можно указать тот же SQLite backend:

```bash
set MLFLOW_TRACKING_URI=sqlite:///%CD%/runtime/mlflow.db
set MLFLOW_REGISTRY_URI=sqlite:///%CD%/runtime/mlflow.db
```

## Drift Reports

Drift job сравнивает текущие данные с reference sample. Если `runtime/reference_sample.csv` отсутствует, он создается из исходных CSV при первом запуске.

В API/UI текущие данные - это последние inference-запросы, сохраненные в SQLite. Если записей меньше `DRIFT_MIN_ROWS`, run сохраняется со статусом `not_enough_data`.

В Docker Compose включен in-process scheduler:

- `DRIFT_SCHEDULER_ENABLED=true`
- `DRIFT_SCHEDULER_INTERVAL_SECONDS=300`
- `DRIFT_PREDICTION_LIMIT=1000`
- `DRIFT_MIN_ROWS=30`

В Kubernetes файл `k8s/base/drift-cronjob.yaml` запускает такую же проверку каждые 15 минут через `POST /drift/run`. Интервал меняется через поле `schedule`.

Реализованные проверки:

- `data drift`: PSI и KS для числовых признаков, PSI для категориальных признаков;
- `target drift`: изменение доли `Is Laundering`, если в batch есть labels;
- `concept drift`: падение precision/recall/F1/ROC-AUC, если в batch есть labels.

Отчеты создаются здесь:

- `reports/drift/*.json`
- `reports/drift/*.html`

## Web UI

React UI - это рабочий интерфейс оператора, не landing page. В нем есть:

- страница инференса с JSON single/batch input;
- таблица последних предсказаний;
- anomaly flags;
- Drift Alerts с историей проверок и ссылками на отчеты;
- страница экспериментов MLflow;
- кнопка `Start retraining` и таблица retraining jobs.

Локальный запуск frontend для разработки:

```bash
cd frontend
npm install
npm run dev
```

Если Vite запускается отдельно, задайте:

```bash
set VITE_API_URL=http://localhost:8000
```

## Monitoring

Backend отдает Prometheus metrics на `/metrics`.

Основные метрики:

- количество API-запросов и latency;
- количество prediction-запросов;
- anomaly rate;
- средняя вероятность laundering;
- data/target/concept drift scores;
- retraining status;
- качество модели: ROC-AUC, PR-AUC, precision, recall, F1, F2, threshold.

Grafana автоматически поднимает dashboard `AML Monitoring`.

## Kubernetes и Argo CD

Манифесты находятся в `k8s/base`. Namespace по умолчанию:

```text
aml-monitoring
```

Перед деплоем нужно заменить placeholders:

- image names в `k8s/base/backend.yaml`, `frontend.yaml`, `mlflow.yaml`, `retraining-job.yaml`;
- `repoURL` в `k8s/argocd/application.yaml`;
- secret values в `k8s/base/secret.yaml`;
- storage/ingress/load balancer настройки под ваш cluster или Minikube.

Прямой деплой:

```bash
kubectl apply -k k8s/base
```

Деплой через Argo CD:

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Argo CD будет синхронизировать manifests из пути `k8s/base`.

## CI/CD

GitHub Actions workflow `.github/workflows/ci_cd.yaml` запускается на pull request в `main` и push в `main`.

Этапы pipeline:

- lint backend через `ruff`;
- backend tests через `pytest`;
- frontend build через `npm run build`;
- сборка backend/frontend Docker images;
- push images в GHCR на `main`;
- Argo CD sync на `main`.

Secrets для deploy:

- `ARGOCD_SERVER`
- `ARGOCD_AUTH_TOKEN`
- optional `ARGOCD_APP`, по умолчанию `aml-monitoring`

## Тесты

```bash
set PYTHONPATH=%CD%\src
pytest src/tests
```

Тесты покрывают feature engineering, стабильное кодирование категорий, drift calculators, API endpoints, retraining sampling и подбор threshold.
