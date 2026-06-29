import argparse

from .drift import run_drift
from .retraining import run_retraining
from .storage import Store


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["drift", "retrain"])
    args = parser.parse_args()
    store = Store()
    if args.command == "drift":
        result = run_drift()
        store.add_drift_run("combined", result["status"], result["data_drift"]["score"], result["data_drift"]["threshold"], result["report_json"], result["report_html"], result)
        print(result)
    if args.command == "retrain":
        job_id = store.create_retraining_job()
        run_retraining(job_id, store)
        job = store.retraining_jobs(1)[0]
        print(job)
        if job["status"] == "failed":
            raise SystemExit(1)


if __name__ == "__main__":
    main()
