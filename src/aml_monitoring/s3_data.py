import argparse
import os
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from .config import DATA_FILES, MODEL_FILES, S3_BUCKET, S3_DATA_PREFIX, S3_ENDPOINT_URL, S3_MODELS_PREFIX


def _client():
    return boto3.client("s3", endpoint_url=S3_ENDPOINT_URL)


def s3_credentials_configured() -> bool:
    return bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))


def s3_key_for(path: Path, prefix: str) -> str:
    return f"{prefix.rstrip('/')}/{path.name}"


# Keep old signature for backward compatibility
def s3_key_for_local_path(path: Path, kind: str = "data") -> str:
    prefix = S3_DATA_PREFIX if kind == "data" else S3_MODELS_PREFIX
    return s3_key_for(path, prefix)


def s3_download_if_missing(local_path: Path, key: str) -> bool:
    if local_path.exists():
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(S3_BUCKET, key, str(local_path))
    return True


def _download_files(files: list[Path], prefix: str, required: bool = False) -> list[Path]:
    downloaded = []
    client = _client()
    for path in files:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        key = s3_key_for(path, prefix)
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


def download_data(required: bool = False) -> list[Path]:
    return _download_files(DATA_FILES, S3_DATA_PREFIX, required=required)


def download_models(required: bool = False) -> list[Path]:
    return _download_files(MODEL_FILES, S3_MODELS_PREFIX, required=required)


def _upload_files(files: list[Path], prefix: str, required: bool = True) -> list[str]:
    uploaded = []
    client = _client()
    for path in files:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Local data file is missing: {path}")
            continue
        key = s3_key_for(path, prefix)
        try:
            client.upload_file(str(path), S3_BUCKET, key)
        except (BotoCoreError, ClientError, NoCredentialsError):
            if required:
                raise
            continue
        uploaded.append(f"s3://{S3_BUCKET}/{key}")
    return uploaded


def upload_data(required: bool = True) -> list[str]:
    return _upload_files(DATA_FILES, S3_DATA_PREFIX, required=required)


def upload_models(required: bool = True) -> list[str]:
    return _upload_files(MODEL_FILES, S3_MODELS_PREFIX, required=required)


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
        key = s3_key_for(path, S3_DATA_PREFIX)
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
    if missing and s3_credentials_configured():
        download_data(required=required)
    if required:
        still_missing = [str(path) for path in DATA_FILES if not path.exists()]
        if still_missing:
            raise FileNotFoundError(f"Missing data files: {', '.join(still_missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload/download AML files to an S3-compatible bucket.")
    parser.add_argument("command", choices=["download", "upload", "check", "models-download", "models-upload"])
    parser.add_argument("--required", action="store_true")
    args = parser.parse_args()

    if args.command == "download":
        paths = download_data(required=args.required)
        print("\n".join(str(p) for p in paths) or "No files downloaded")
    elif args.command == "models-download":
        paths = download_models(required=args.required)
        print("\n".join(str(p) for p in paths) or "No files downloaded")
    elif args.command == "models-upload":
        keys = upload_models(required=args.required)
        print("\n".join(keys) or "No files uploaded")
    elif args.command == "upload":
        keys = upload_data(required=args.required)
        print("\n".join(keys) or "No files uploaded")
    else:
        print("\n".join(check_data()))


if __name__ == "__main__":
    main()
