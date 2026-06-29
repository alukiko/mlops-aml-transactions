# AML Monitoring Platform

MLOps-проект для детекции транзакций по отмыванию денег (AML). Включает FastAPI-backend для инференса, React UI для оператора, расчёт drift, MLflow tracking и Model Registry, Prometheus/Grafana мониторинг, Docker Compose для локальной разработки, Kubernetes-манифесты и Argo CD GitOps-деплой.

## Структура проекта

```
src/aml_monitoring/          # Backend-пакет: инференс, drift, retraining, API
mlops_aml_transactions/      # Пакет обработки транзакций: feature engineering, scoring API
frontend/                    # React/Vite интерфейс оператора
k8s/base/                    # Kubernetes manifests (kustomize)
k8s/argocd/                  # Argo CD Application
.github/workflows/           # CI/CD pipelines
monitoring/                  # Prometheus config и Grafana dashboard
models/                      # Артефакты модели (не коммитятся, хранятся в S3)
data/                        # IBM AML synthetic CSV (не коммитятся, хранятся в S3)
reports/drift/               # Сгенерированные drift-отчёты
```

## Выбор данных и модели

Проект использует открытый синтетический датасет IBM AML (`HI-Small_Trans.csv` и
`LI-Small_Trans.csv`) с целевой колонкой `Is Laundering`.

В качестве baseline используется Logistic Regression. В MLflow сравниваются
Logistic Regression, Random Forest, Extra Trees и HistGradientBoosting. Лучшая
модель выбирается по Average Precision; FastAPI по умолчанию загружает
`models/hgb_compat.pkl`.

## Git workflow и Conventional Commits

Используется GitHub Flow: рабочая ветка → Pull Request в `main` → CI → merge.
Сообщения коммитов должны иметь вид:

```text
feat(scope): add feature
fix: correct drift calculation
docs: update runbook
```

Формат проверяется локальным `commit-msg` hook и GitHub Actions:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

## Cookiecutter

В репозитории находится рабочий шаблон `{{cookiecutter.repo_name}}`. Создание
нового проекта:

```bash
pip install -r requirements-dev.txt
cookiecutter .
```

## DVC

Данные, модель и drift-отчёты описаны в `dvc.yaml` и зафиксированы в
`dvc.lock`. Remote `yandex-s3` настроен в `.dvc/config`.

```bash
pip install -r requirements-dev.txt
dvc pull
dvc repro
dvc push
```

## Локальный запуск без Docker

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
set PYTHONPATH=%CD%\src       # Windows
# export PYTHONPATH=$PWD/src  # Linux/Mac
```

Запуск backend:

```bash
uvicorn aml_monitoring.main:app --reload --host 0.0.0.0 --port 8000
```

Полезные URL при локальном запуске:

| URL | Описание |
|-----|----------|
| `http://localhost:8000/docs` | OpenAPI / Swagger |
| `http://localhost:8000/health` | Health check |
| `http://localhost:8000/metrics` | Prometheus metrics |

## Запуск через Docker Compose

```bash
docker compose up --build
```

| Сервис | URL |
|--------|-----|
| Web UI | `http://localhost:3000` |
| Backend API | `http://localhost:8000` |
| MLflow UI | `http://localhost:5000` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3001` (admin / admin) |

```bash
docker compose down
```

MLflow использует SQLite: `sqlite:///runtime/mlflow.db`, артефакты в `mlartifacts/`.

## Данные и модели в S3

CSV-файлы и модели не хранятся в Git — только в S3-совместимом хранилище (Yandex Object Storage).

Ожидаемые ключи в bucket:

```
data/raw/HI-Small_Trans.csv
data/raw/LI-Small_Trans.csv
models/hgb_compat.pkl
models/model_meta.pkl
models/reference_sample.csv
```

Переменные окружения для S3:

```bash
set AWS_ACCESS_KEY_ID=...
set AWS_SECRET_ACCESS_KEY=...
set S3_ENDPOINT_URL=https://storage.yandexcloud.net
set S3_BUCKET=mlops-aml-transactions
set S3_DATA_PREFIX=data/raw
set S3_MODELS_PREFIX=models
```

Загрузка данных в S3:

```bash
set PYTHONPATH=%CD%\src
python -m aml_monitoring.s3_data upload --required
```

Скачивание данных из S3:

```bash
set PYTHONPATH=%CD%\src
python -m aml_monitoring.s3_data download --required
python -m aml_monitoring.s3_data models-download --required
```

## Примеры API

Инференс (одна транзакция):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"transactions":[{"Timestamp":"2022/09/01 00:20","From Bank":"010","From Account":"8000EBD30","To Bank":"010","To Account":"8000EBD30","Amount Received":3697.34,"Receiving Currency":"US Dollar","Amount Paid":3697.34,"Payment Currency":"US Dollar","Payment Format":"Reinvestment"}]}'
```

Запуск drift-анализа:

```bash
curl -X POST http://localhost:8000/drift/run -H "Content-Type: application/json" -d "{}"
```

Добавление фактической метки к сохранённому предсказанию:

```bash
curl -X PATCH http://localhost:8000/predictions/123/label \
  -H "Content-Type: application/json" \
  -d '{"actual_label":1}'
```

Запуск переобучения:

```bash
curl -X POST http://localhost:8000/retrain
```

## Drift Monitoring

`/drift/run` сравнивает последние inference-запросы (из SQLite) с reference sample. Если записей меньше `DRIFT_MIN_ROWS` — статус `not_enough_data`.

Реализованные проверки:

- **data drift** — PSI и KS для числовых признаков, PSI для категориальных
- **target drift** — изменение доли `Is Laundering` (если в batch есть labels)
- **concept drift** — падение precision/recall/F1/ROC-AUC (если в batch есть labels)

Для Target Drift и Concept Drift требуется минимум 21 фактическая метка среди последних
`DRIFT_PREDICTION_LIMIT` предсказаний. Метки `0` (обычная транзакция) и `1` (отмывание)
назначаются в таблице Recent Predictions или через `PATCH /predictions/{id}/label`.
Прогресс возвращается полем `labeling` в `GET /predictions/recent`.

В Kubernetes drift-check запускается CronJob каждые 15 минут (`k8s/base/drift-cronjob.yaml`).

В Docker Compose drift работает in-process scheduler (переменные `DRIFT_SCHEDULER_ENABLED`, `DRIFT_SCHEDULER_INTERVAL_SECONDS`).

Отчёты сохраняются в `reports/drift/*.json` и `reports/drift/*.html`.

## Тесты

```bash
set PYTHONPATH=%CD%\src
pytest src/tests tests
```

Если S3 сконфигурирован, `conftest.py` автоматически скачает модель перед запуском тестов.

Тесты покрывают: feature engineering, кодирование категорий, drift calculators, API endpoints (health, predict, 503 при отсутствии модели), retraining sampling, подбор threshold.

## Kubernetes и Argo CD

Все манифесты находятся в `k8s/base` (kustomize). Namespace: `aml-monitoring`.

Основные ресурсы:

| Файл | Что создаёт |
|------|-------------|
| `backend.yaml` | Deployment aml-backend (FastAPI) |
| `frontend.yaml` | Deployment aml-frontend (React) |
| `mlflow.yaml` | Deployment aml-mlflow |
| `storage.yaml` | PVC aml-data-pvc (5Gi) и aml-runtime-pvc (2Gi) |
| `secret.yaml` | S3 credentials → envFrom в backend |
| `configmap.yaml` | Переменные окружения |
| `drift-cronjob.yaml` | CronJob drift-check (каждые 15 мин) |
| `retraining-job.yaml` | Опциональный шаблон Job; Argo CD его не применяет, UI запускает `/retrain` через FastAPI |
| `prometheus-grafana.yaml` | Prometheus + Grafana |

Деплой через Argo CD (уже настроен):

```bash
kubectl apply -f k8s/argocd/application.yaml
```

Argo CD следит за веткой `main`, путь `k8s/base`, с `selfHeal: true` и `prune: true`. После каждого merge в main CI автоматически обновляет image tags в манифестах, Argo CD подхватывает изменения.

Прямой деплой без Argo CD:

```bash
kubectl apply -k k8s/base
```

## CI/CD

Файл `.github/workflows/ci_cd.yaml` запускается на push и PR в `main`.

Этапы:

1. **lint-test** — `ruff check`, `pytest`, `npm run build`
2. **build-images** — скачивает модель из S3, собирает и пушит Docker images в GHCR с тегами `:{sha}` и `:latest`
3. **update-manifests** — заменяет image tags во всех манифестах `k8s/base` на `{sha}`, коммитит `[skip ci]` обратно в main
4. **deploy-argocd** — запускает `argocd app sync` (только если `ENABLE_ARGOCD_DEPLOY=true`)

Secrets для CI/CD:

| Secret | Обязателен | Описание |
|--------|-----------|----------|
| `AWS_ACCESS_KEY_ID` | Да | S3 доступ для скачивания модели |
| `AWS_SECRET_ACCESS_KEY` | Да | S3 доступ |
| `S3_BUCKET` | Нет | По умолчанию `mlops-aml-transactions` |
| `S3_ENDPOINT_URL` | Нет | По умолчанию `https://storage.yandexcloud.net` |
| `ARGOCD_SERVER` | Нет | Нужен если `ENABLE_ARGOCD_DEPLOY=true` |
| `ARGOCD_AUTH_TOKEN` | Нет | Нужен если `ENABLE_ARGOCD_DEPLOY=true` |

## Web UI

React UI — рабочий интерфейс оператора:

- инференс: JSON single/batch input
- таблица последних предсказаний с anomaly flags
- Drift Alerts с историей и ссылками на HTML-отчёты
- страница экспериментов MLflow
- кнопка запуска переобучения и таблица retraining jobs

Локальная разработка frontend:

```bash
cd frontend
npm install
npm run dev
# VITE_API_URL=http://localhost:8000 — если backend на другом порту
```

## Мониторинг

Backend отдаёт Prometheus metrics на `/metrics`. Grafana автоматически поднимает dashboard `AML Monitoring`.

Основные метрики: latency API, количество предсказаний, anomaly rate, средняя вероятность laundering, drift scores, статус retraining, качество модели (ROC-AUC, F1, precision, recall).
