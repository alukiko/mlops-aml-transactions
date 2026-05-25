import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from .config import DATA_FILES, S3_BUCKET, S3_DATA_PREFIX, S3_ENDPOINT_URL


def _client():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)


def s3_key_for(path: Path) -> str:
    return f"{S3_DATA_PREFIX.rstrip('/')}/{path.name}"


def download_data(required: bool = False) -> list[Path]:
    downloaded = []
    client = _client()
    for path in DATA_FILES:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        key = s3_key_for(path)
        try:
            client.download_file(S3_BUCKET, key, str(path))
            downloaded.append(path)
        except (BotoCoreError, ClientError) as exc:
            if required:
                raise RuntimeError(f"Failed to download s3://{S3_BUCKET}/{key}") from exc
    return downloaded


def upload_data(required: bool = True) -> list[str]:
    uploaded = []
    client = _client()
    for path in DATA_FILES:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Local data file is missing: {path}")
            continue
        key = s3_key_for(path)
        client.upload_file(str(path), S3_BUCKET, key)
        uploaded.append(f"s3://{S3_BUCKET}/{key}")
    return uploaded


def ensure_data_available(required: bool = False) -> None:
    missing = [path for path in DATA_FILES if not path.exists()]
    if missing and os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
        download_data(required=required)
    if required:
        still_missing = [str(path) for path in DATA_FILES if not path.exists()]
        if still_missing:
            raise FileNotFoundError(f"Missing data files: {', '.join(still_missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload/download AML CSV files to an S3-compatible bucket.")
    parser.add_argument("command", choices=["download", "upload"])
    parser.add_argument("--required", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        paths = download_data(required=args.required)
        print("\n".join(str(path) for path in paths) or "No files downloaded")
    else:
        keys = upload_data(required=args.required)
        print("\n".join(keys) or "No files uploaded")


if __name__ == "__main__":
    main()
