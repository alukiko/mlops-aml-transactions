import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import DB_PATH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists predictions (
                    id integer primary key autoincrement,
                    created_at text not null,
                    payload_json text not null,
                    probability real not null,
                    predicted_class integer not null,
                    anomaly_flag integer not null,
                    drift_flag integer not null
                );
                create table if not exists drift_runs (
                    id integer primary key autoincrement,
                    created_at text not null,
                    drift_type text not null,
                    status text not null,
                    score real,
                    threshold real,
                    report_json text,
                    report_html text,
                    details_json text
                );
                create table if not exists retraining_jobs (
                    id integer primary key autoincrement,
                    created_at text not null,
                    finished_at text,
                    status text not null,
                    mlflow_run_id text,
                    error text,
                    metrics_json text
                );
                """
            )

    def add_prediction(self, payload: dict[str, Any], probability: float, predicted_class: int, anomaly_flag: bool, drift_flag: bool) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into predictions(created_at, payload_json, probability, predicted_class, anomaly_flag, drift_flag)
                values (?, ?, ?, ?, ?, ?)
                """,
                (utc_now(), json.dumps(payload, default=str), probability, predicted_class, int(anomaly_flag), int(drift_flag)),
            )
            return int(cur.lastrowid)

    def recent_predictions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from predictions order by id desc limit ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def recent_prediction_payloads(self, limit: int = 1000) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select payload_json from predictions order by id desc limit ?", (limit,)).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def add_drift_run(self, drift_type: str, status: str, score: float | None, threshold: float | None, report_json: str | None, report_html: str | None, details: dict[str, Any]) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                insert into drift_runs(created_at, drift_type, status, score, threshold, report_json, report_html, details_json)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (utc_now(), drift_type, status, score, threshold, report_json, report_html, json.dumps(details, default=str)),
            )
            return int(cur.lastrowid)

    def drift_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from drift_runs order by id desc limit ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    def create_retraining_job(self) -> int:
        with self.connect() as conn:
            cur = conn.execute("insert into retraining_jobs(created_at, status) values (?, ?)", (utc_now(), "queued"))
            return int(cur.lastrowid)

    def update_retraining_job(self, job_id: int, status: str, mlflow_run_id: str | None = None, error: str | None = None, metrics: dict[str, Any] | None = None) -> None:
        finished_at = utc_now() if status in {"completed", "failed"} else None
        with self.connect() as conn:
            conn.execute(
                """
                update retraining_jobs
                set status = ?, finished_at = coalesce(?, finished_at), mlflow_run_id = coalesce(?, mlflow_run_id),
                    error = ?, metrics_json = coalesce(?, metrics_json)
                where id = ?
                """,
                (status, finished_at, mlflow_run_id, error, json.dumps(metrics, default=str) if metrics else None, job_id),
            )

    def retraining_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("select * from retraining_jobs order by id desc limit ?", (limit,)).fetchall()
        return [self._row(row) for row in rows]

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        for key in ("payload_json", "details_json", "metrics_json"):
            if key in item and item[key]:
                item[key.replace("_json", "")] = json.loads(item[key])
        return item
