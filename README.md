# mlops-aml-transactions

Проект на Python для **MLOps** вокруг задачи **AML** (Anti-Money Laundering): бинарная классификация банковских транзакций по синтетическому датасету IBM. Пакет `mlops_aml_transactions` включает подготовку данных, обучение модели, батч-скоринг, HTTP API и Docker-образ для предсказаний.

## Требования

- Python **3.10–3.13** (см. [pyproject.toml](pyproject.toml)).

## Установка

Из каталога `mlops-aml-transactions`:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Режим разработки (редактируемый пакет):

```bash
python -m pip install -e .
```

## Данные

Сырой CSV **IBM AML** по умолчанию читается из **`data/raw/HI-Small_Trans.csv`** (внутри этого подпроекта). Скопируйте туда файл с диска или из каталога `mlops/`, где могут лежать копии датасетов — см. [README в корне репозитория](../README.md) (описание колонок и источники).

Другой путь к CSV можно передать аргументом CLI (см. `--help` у `dataset.py` и `train.py`).

## S3 (Yandex Object Storage): автоскачивание данных и моделей

Проект умеет работать с S3-совместимым хранилищем (например, **Yandex Object Storage**):

- если **сырого CSV** нет локально в `data/raw/`, он будет скачан из S3;
- если **модели** нет локально в `models/`, API/скоринг попробуют скачать её из S3;
- при **обучении** модели сохраняются локально и дополнительно загружаются в S3.

Для включения S3 создайте локальный файл `.env` (он уже игнорируется git) по образцу `.env.example`:

```bash
S3_ENDPOINT_URL=https://storage.yandexcloud.net
S3_BUCKET=mlops-aml-transactions
S3_DATA_PREFIX=data/raw
S3_MODELS_PREFIX=models
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Ожидаемые ключи объектов в бакете:

- `data/raw/HI-Small_Trans.csv`
- `data/raw/LI-Small_Trans.csv`
- `models/model.pkl` (и при обучении: `models/{rf,et,lr,hgb}.pkl`)

## Конвейер: датасет, обучение, скоринг

Команды реализованы через **Typer**; полный список опций — у каждого модуля:

```bash
python mlops_aml_transactions/dataset.py --help
python mlops_aml_transactions/modeling/train.py --help
python mlops_aml_transactions/modeling/predict.py --help
```

Типичный порядок:

1. **Подготовка `data/processed/dataset.csv`** из сырого CSV (стратифицированная выборка и т.д.) — `dataset.py`.
2. **Обучение** и сохранение **`models/model.pkl`** — `modeling/train.py` (MLflow по желанию).
3. **Батч-предсказания** в CSV — `modeling/predict.py`.

Удобная цель `make data` в [Makefile](Makefile) ставит зависимости и запускает `dataset.py` (при установленном `make`).

## HTTP API

Запуск сервера (из каталога проекта, с установленным пакетом):

```bash
python -m uvicorn mlops_aml_transactions.api.main:app --host 127.0.0.1 --port 8000
```

- `GET /health` — проверка работоспособности.
- `POST /predict` — одна транзакция.
- `POST /predict/batch` — список транзакций.
- `GET /docs` — интерактивная документация (Swagger UI).
- `GET /redoc` — ReDoc.

Модель читается из `models/model.pkl` относительно корня подпроекта.

## Docker

Минимальные зависимости для образа — [requirements-api.txt](requirements-api.txt). Сборка и запуск:

```bash
docker compose up --build
```

Сервис слушает порт **8000**. При сборке в образ копируется каталог **`models/`** (положите туда `model.pkl` до `docker build`). Чтобы подменить модель без пересборки образа, в [docker-compose.yml](docker-compose.yml) можно раскомментировать том `./models:/app/models:ro`.

## Тесты

Из **корня** репозитория `mlops` (там лежит [pytest.ini](../pytest.ini), указывающий на тесты подпроекта):

```bash
cd ..
python -m pytest
```

Или из каталога `mlops-aml-transactions`:

```bash
python -m pytest tests
```

## Структура каталогов

```
mlops-aml-transactions/
├── Dockerfile              # образ API
├── docker-compose.yml
├── requirements.txt        # полное окружение (разработка, ноутбуки)
├── requirements-api.txt    # только зависимости API для Docker
├── pyproject.toml
├── models/                 # model.pkl и др. артефакты
├── data/
│   ├── raw/
│   ├── processed/          # dataset.csv и пр.
│   └── ...
├── notebooks/
├── tests/                  # pytest (test_api.py и др.)
├── docs/                   # MkDocs (при необходимости)
├── reports/
├── mlops_aml_transactions/
│   ├── config.py
│   ├── features.py
│   ├── dataset.py
│   ├── api/
│   │   └── main.py         # FastAPI
│   └── modeling/
│       ├── train.py
│       ├── predict.py
│       └── artifacts.py
└── README.md
```

## Лицензия

См. файл [LICENSE](LICENSE).
