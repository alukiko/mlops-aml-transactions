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

MODELS_DIR = PROJ_ROOT / "models"

# MLflow: по умолчанию локальное хранилище рядом с корнем проекта
MLRUNS_DIR = PROJ_ROOT / "mlruns"
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "aml-transactions")

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
