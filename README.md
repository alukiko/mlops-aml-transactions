# mlops-aml-transactions

Проект на Python для **MLOps** вокруг задачи **AML** (Anti-Money Laundering): бинарная классификация банковских транзакций на синтетическом датасете IBM.

Пакет `mlops_aml_transactions` включает:
- подготовку данных;
- обучение моделей;
- batch scoring;
- HTTP API (FastAPI);
- Docker-образ для продового запуска.

## Требования

- Python **3.10-3.13** (см. [pyproject.toml](pyproject.toml)).

## Установка

Из каталога `mlops-aml-transactions`:

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Режим разработки:

```bash
python -m pip install -e .
```

## Данные

По умолчанию сырой CSV читается из `data/raw/HI-Small_Trans.csv`.

Можно:
- положить файлы вручную в `data/raw/`;
- или дать системе скачать их из S3 (см. раздел ниже).

## S3 (Yandex Object Storage)

Поддерживается S3-совместимое хранилище:
- автоскачивание сырых данных в `data/raw/`, если локально файлов нет;
- автоскачивание `models/model.pkl`, если модель не найдена локально;
- автозагрузка обученных артефактов в S3.

Создайте `.env` по шаблону `.env.example`:

```bash
S3_ENDPOINT_URL=https://storage.yandexcloud.net
S3_BUCKET=mlops-aml-transactions
S3_DATA_PREFIX=data/raw
S3_MODELS_PREFIX=models
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

Ожидаемые ключи в бакете:
- `data/raw/HI-Small_Trans.csv`
- `data/raw/LI-Small_Trans.csv`
- `models/model.pkl`

## Практические сценарии

### Вариант 1: полный цикл с нуля

1. Заполнить `.env`.
2. Собрать датасет:

```bash
python -m mlops_aml_transactions.dataset \
  --input-files HI-Small_Trans.csv,LI-Small_Trans.csv \
  --read-cap 2000000 --sample-size 600000 --min-positive-rows 800
```

3. Запустить обучение:

```bash
python -m mlops_aml_transactions.modeling.train \
  --run-name exp_time_recall_t06 \
  --model-path models/model_exp_time_recall_t06.pkl \
  --models et \
  --split-strategy time --val-size 0.2 --test-size 0.2 \
  --threshold-objective recall --target-recall 0.6 \
  --min-threshold 0.001 --max-threshold 0.999
```

4. Назначить боевую модель как `models/model.pkl` и загрузить в S3.

### Вариант 2: боевой Docker

1. Убедиться, что есть `models/model.pkl`.
2. Собрать и поднять сервис:

```bash
docker compose up --build -d
```

3. Проверить здоровье:

```bash
curl http://127.0.0.1:8000/health
```

Ожидаемо:

```json
{"status":"ok"}
```

## Конвейер

Справка по CLI:

```bash
python mlops_aml_transactions/dataset.py --help
python mlops_aml_transactions/modeling/train.py --help
python mlops_aml_transactions/modeling/predict.py --help
```

Порядок:
1. Подготовка `data/processed/dataset.csv` (`dataset.py`).
2. Обучение и сохранение `models/model.pkl` (`modeling/train.py`).
3. Batch-предсказания (`modeling/predict.py`).

## Быстрый гид по режимам

| Вариант | Split | Цель порога | Плюсы | Минусы |
|---|---|---|---|---|
| `ET + random + fbeta` | `random` | `fbeta (beta=1.0)` | Лучшие цифры на baseline, быстрый sanity-check | Оптимистичная оценка, риск утечки по сущностям/времени |
| `ET + time + fbeta` | `time (60/20/20)` | `fbeta (beta=1.0)` | Реалистичнее для будущих данных, лучше контролируемый `F1` | `F1` обычно ниже, сложнее выбор порога |
| `ET + time + recall` | `time (60/20/20)` | `recall-first` (`target_recall`) | Позволяет держать целевой recall (если достижим) | Может сильно просесть precision, много алертов |

## Метрики экспериментов

| Run name | Run ID | Split | Threshold objective | ROC-AUC | AP (PR-AUC) | F1 | Threshold |
|---|---|---|---|---:|---:|---:|---:|
| `exp_time_recall_t06` | `4f0189fe55f14692be50e28f06bcf3c0` | `time` | `recall (target=0.6)` | 0.8968 | 0.0690 | 0.0268 | 0.0010 |
| `exp_time_fbeta_b1` | `555ff3dff7614aa1b7df7b9587f28130` | `time` | `fbeta (beta=1.0)` | 0.8968 | 0.0690 | 0.1322 | 0.3185 |
| `exp_random_fbeta_b1` | `78e99381641c4d2294532640c110a3a8` | `random` | `fbeta (beta=1.0)` | 0.9738 | 0.5316 | 0.6066 | 0.7067 |

Текущая боевая модель для Docker/API: `exp_random_fbeta_b1`  
Артефакт: `models/model.pkl` (синхронизирован с `s3://mlops-aml-transactions/models/model.pkl`).

## HTTP API

Запуск:

```bash
python -m uvicorn mlops_aml_transactions.api.main:app --host 127.0.0.1 --port 8000
```

Эндпоинты:
- `GET /health`
- `POST /predict`
- `POST /predict/batch`
- `GET /docs`
- `GET /redoc`

## Docker

```bash
docker compose up --build
```

Сервис слушает порт `8000`.

## Тесты

Из `mlops-aml-transactions`:

```bash
python -m pytest tests
```

Из корня `mlops`:

```bash
python -m pytest
```

## Лицензия

См. [LICENSE](LICENSE).
