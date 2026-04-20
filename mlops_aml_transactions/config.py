import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Загрузка переменных окружения из .env, если файл есть
load_dotenv()

# Корень подпроекта mlops-aml-transactions
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Сырой CSV IBM AML по умолчанию: data/raw/HI-Small_Trans.csv (положите файл в эту папку)
DEFAULT_RAW_CSV = RAW_DATA_DIR / "HI-Small_Trans.csv"

# Файлы для ноутбука / LightGBM-пайплайна (IBM synthetic AML)
RAW_AML_DEFAULT_FILES = [
    RAW_DATA_DIR / "HI-Small_Trans.csv",
    RAW_DATA_DIR / "LI-Small_Trans.csv",
]

# Эксперимент MLflow для LGBM-ноутбука
MLFLOW_EXPERIMENT_LGB_NOTEBOOK = "aml-detection"

MODELS_DIR = PROJ_ROOT / "models"

# MLflow: по умолчанию локальное хранилище рядом с корнем проекта
MLRUNS_DIR = PROJ_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "aml-transactions")

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# -----------------------------
# S3 / Yandex Object Storage
# -----------------------------
# Если переменные не заданы — интеграция S3 отключена (код должен работать локально как раньше).
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
S3_BUCKET = os.environ.get("S3_BUCKET")
S3_DATA_PREFIX = os.environ.get("S3_DATA_PREFIX", "data/raw").strip("/")
S3_MODELS_PREFIX = os.environ.get("S3_MODELS_PREFIX", "models").strip("/")

# Стандартные переменные boto3
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")


try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
