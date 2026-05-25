from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
REPORT_DIR = PROJECT_ROOT / "reports" / "drift"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MLARTIFACTS_DIR = PROJECT_ROOT / "mlartifacts"
MLFLOW_SQLITE_PATH = RUNTIME_DIR / "mlflow.db"
DEFAULT_MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_SQLITE_PATH.as_posix()}"

MODEL_PATH = MODEL_DIR / "aml_lgbm.pkl"
MODEL_META_PATH = MODEL_DIR / "model_meta.pkl"
REFERENCE_SAMPLE_PATH = RUNTIME_DIR / "reference_sample.csv"
DB_PATH = RUNTIME_DIR / "monitoring.db"

DATA_FILES = [DATA_DIR / "HI-Small_Trans.csv", DATA_DIR / "LI-Small_Trans.csv"]

RAW_COLUMNS = [
    "Timestamp",
    "From Bank",
    "From Account",
    "To Bank",
    "To Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

FEATURE_COLS = [
    "hour",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "is_night",
    "amount_ratio",
    "amount_diff",
    "log_amount_paid",
    "log_amount_recv",
    "is_round_amount",
    "currency_mismatch",
    "same_bank",
    "self_transfer",
    "sender_tx_count",
    "sender_total_paid",
    "sender_mean_paid",
    "sender_std_paid",
    "sender_unique_recv",
    "fanout_ratio",
    "recv_tx_count",
    "recv_total_received",
    "recv_mean_received",
    "recv_unique_sender",
    "fanin_ratio",
    "Payment Format_enc",
    "Receiving Currency_enc",
    "Payment Currency_enc",
    "From Bank_enc",
    "To Bank_enc",
]

TARGET = "Is Laundering"
DEFAULT_THRESHOLD = 0.99

DRIFT_PREDICTION_LIMIT = int(os.getenv("DRIFT_PREDICTION_LIMIT", "1000"))
DRIFT_MIN_ROWS = int(os.getenv("DRIFT_MIN_ROWS", "30"))
DRIFT_SCHEDULER_ENABLED = os.getenv("DRIFT_SCHEDULER_ENABLED", "false").lower() in {"1", "true", "yes"}
DRIFT_SCHEDULER_INTERVAL_SECONDS = int(os.getenv("DRIFT_SCHEDULER_INTERVAL_SECONDS", "900"))
