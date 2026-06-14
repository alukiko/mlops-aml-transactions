from __future__ import annotations

from pathlib import Path

from mlops_aml_transactions.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    S3_BUCKET,
    S3_DATA_PREFIX,
    S3_ENDPOINT_URL,
    S3_MODELS_PREFIX,
)


def _client():
    if not S3_BUCKET:
        raise RuntimeError("S3_BUCKET is not configured")

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def s3_key_for_local_path(path: str | Path, kind: str) -> str:
    local_path = Path(path)
    if kind == "data":
        prefix = S3_DATA_PREFIX
    elif kind == "models":
        prefix = S3_MODELS_PREFIX
    else:
        raise ValueError(f"Unsupported S3 artifact kind: {kind}")
    return f"{prefix.rstrip('/')}/{local_path.name}"


def s3_download_if_missing(path: str | Path, key: str) -> bool:
    local_path = Path(path)
    if local_path.is_file():
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    _client().download_file(S3_BUCKET, key, str(local_path))
    return True


def s3_upload_file(path: str | Path, key: str) -> str:
    local_path = Path(path)
    if not local_path.is_file():
        raise FileNotFoundError(f"Cannot upload missing file: {local_path}")
    _client().upload_file(str(local_path), S3_BUCKET, key)
    return f"s3://{S3_BUCKET}/{key}"
