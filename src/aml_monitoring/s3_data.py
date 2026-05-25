import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

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
        except NoCredentialsError as exc:
            if required:
                raise RuntimeError("AWS credentials are not configured for S3 download") from exc
        except (BotoCoreError, ClientError) as exc:
            if required:
                raise RuntimeError(f"Failed to download s3://{S3_BUCKET}/{key}: {describe_s3_error(exc)}") from exc
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


def describe_s3_error(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = error.get("Code", "Unknown")
        message = error.get("Message", "No message")
        return f"{code} - {message}"
    return exc.__class__.__name__


def check_data() -> list[str]:
    client = _client()
    results = []
    for path in DATA_FILES:
        key = s3_key_for(path)
        try:
            response = client.head_object(Bucket=S3_BUCKET, Key=key)
            size = response.get("ContentLength", 0)
            results.append(f"OK s3://{S3_BUCKET}/{key} size={size}")
        except NoCredentialsError as exc:
            raise RuntimeError("AWS credentials are not configured for S3 check") from exc
        except (BotoCoreError, ClientError) as exc:
            raise RuntimeError(f"FAILED s3://{S3_BUCKET}/{key}: {describe_s3_error(exc)}") from exc
    return results


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
    parser.add_argument("command", choices=["download", "upload", "check"])
    parser.add_argument("--required", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        paths = download_data(required=args.required)
        print("\n".join(str(path) for path in paths) or "No files downloaded")
    elif args.command == "upload":
        keys = upload_data(required=args.required)
        print("\n".join(keys) or "No files uploaded")
    else:
        print("\n".join(check_data()))


if __name__ == "__main__":
    main()
