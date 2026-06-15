import asyncio
import contextlib

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import DRIFT_MIN_ROWS, DRIFT_PREDICTION_LIMIT, DRIFT_SCHEDULER_ENABLED, DRIFT_SCHEDULER_INTERVAL_SECONDS, REPORT_DIR
from .drift import run_drift
from .experiments import list_experiments
from .inference import get_model_service
from .metrics import PrometheusMiddleware, metrics_response, record_drift, record_predictions, record_retraining_status
from .retraining import run_retraining
from .schemas import DriftRunRequest, PredictRequest
from .storage import Store

store = Store()


def execute_drift_run(request: DriftRunRequest) -> dict:
    prediction_limit = request.prediction_limit or DRIFT_PREDICTION_LIMIT
    min_rows = request.min_rows or DRIFT_MIN_ROWS
    source = "recent_predictions"
    if request.transactions or request.batch_path or not request.use_recent_predictions:
        result = run_drift(batch=request.transactions, batch_path=request.batch_path, min_rows=min_rows, source="request_batch")
    else:
        payloads = store.recent_prediction_payloads(prediction_limit)
        result = run_drift(batch=payloads, min_rows=min_rows, source=source)
    store.add_drift_run(
        "combined",
        result["status"],
        result["data_drift"]["score"],
        result["data_drift"]["threshold"],
        result["report_json"],
        result["report_html"],
        result,
    )
    record_drift(result)
    return result


async def drift_scheduler_loop() -> None:
    while True:
        await asyncio.sleep(DRIFT_SCHEDULER_INTERVAL_SECONDS)
        try:
            execute_drift_run(DriftRunRequest())
        except Exception:
            # The scheduled job must not stop the API process; failures are visible in container logs.
            pass


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    if DRIFT_SCHEDULER_ENABLED:
        app.state.drift_scheduler_task = asyncio.create_task(drift_scheduler_loop())
    yield
    task = getattr(app.state, "drift_scheduler_task", None)
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="AML Monitoring API", version="1.0.0", lifespan=lifespan)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/reports/drift", StaticFiles(directory=str(REPORT_DIR)), name="drift-reports")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictRequest) -> dict:
    if not request.transactions:
        raise HTTPException(status_code=400, detail="transactions must not be empty")
    try:
        svc = get_model_service()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model is not available: {e}") from e
    results = svc.predict(request.transactions)
    for result in results:
        store.add_prediction(
            result["payload"],
            result["probability"],
            result["predicted_class"],
            result["anomaly_flag"],
            result["drift_flag"],
        )
    record_predictions(results)
    return {"count": len(results), "results": results}


@app.get("/predictions/recent")
def recent_predictions(limit: int = 100) -> dict:
    return {"items": store.recent_predictions(limit)}


@app.get("/drift/status")
def drift_status() -> dict:
    return {"items": store.drift_runs(10)}


@app.get("/drift/reports")
def drift_reports(limit: int = 50) -> dict:
    return {"items": store.drift_runs(limit)}


@app.get("/drift/report/{report_id}", response_class=HTMLResponse)
def drift_report_html(report_id: int):
    runs = store.drift_runs(200)
    run = next((r for r in runs if r["id"] == report_id), None)
    if not run:
        raise HTTPException(status_code=404, detail="Report not found")
    html_path = run.get("report_html", "")
    from pathlib import Path
    path = Path(html_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report file not found on disk")
    return FileResponse(path, media_type="text/html")


@app.post("/drift/run")
def drift_run(request: DriftRunRequest) -> dict:
    return execute_drift_run(request)


@app.post("/retrain")
def retrain(background_tasks: BackgroundTasks) -> dict:
    job_id = store.create_retraining_job()
    record_retraining_status("queued")
    background_tasks.add_task(run_retraining, job_id, store)
    return {"job_id": job_id, "status": "queued"}


@app.get("/retrain/jobs")
def retrain_jobs(limit: int = 20) -> dict:
    return {"items": store.retraining_jobs(limit)}


@app.get("/experiments")
def experiments(limit: int = 50) -> dict:
    return {"items": list_experiments(limit)}


@app.get("/metrics")
def metrics():
    return metrics_response()
