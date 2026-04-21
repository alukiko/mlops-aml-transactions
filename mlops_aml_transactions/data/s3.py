from __future__ import annotations

from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger

from mlops_aml_transactions.config import (
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    PROJ_ROOT,
    S3_BUCKET,
    S3_DATA_PREFIX,
    S3_ENDPOINT_URL,
    S3_MODELS_PREFIX,
)


def _s3_is_configured() -> bool:
    return bool(S3_BUCKET and AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY)


def _build_client():
    if not _s3_is_configured():
        return None
    return boto3.client(
        service_name="s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )


def s3_key_for_local_path(local_path: str | Path, *, kind: str) -> str:
    p = Path(local_path)
    try:
        rel = p.resolve().relative_to(PROJ_ROOT.resolve())
    except ValueError:
        rel = Path(p.name)

    rel_posix = rel.as_posix().lstrip("/")

    if kind == "data":
        if rel_posix.startswith("data/raw/"):
            suffix = rel_posix.removeprefix("data/raw/")
        else:
            suffix = p.name
        return f"{S3_DATA_PREFIX}/{suffix}".strip("/")

    if kind == "models":
        if rel_posix.startswith("models/"):
            suffix = rel_posix.removeprefix("models/")
        else:
            suffix = p.name
        return f"{S3_MODELS_PREFIX}/{suffix}".strip("/")

    return rel_posix


def s3_download_if_missing(local_path: str | Path, key: str) -> bool:
    p = Path(local_path)
    if p.is_file():
        return False

    client = _build_client()
    if client is None:
        logger.info("S3 is not configured, skip download: {}", p)
        return False

    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading s3://{}/{} -> {}", S3_BUCKET, key, p)
        client.download_file(S3_BUCKET, key, str(p))
        return True
    except (ClientError, BotoCoreError) as exc:
        logger.warning("S3 download failed for s3://{}/{}: {}", S3_BUCKET, key, exc)
        return False


def s3_upload_file(local_path: str | Path, key: str) -> bool:
    p = Path(local_path)
    if not p.is_file():
        logger.warning("Skip S3 upload, file missing: {}", p)
        return False

    client = _build_client()
    if client is None:
        logger.info("S3 is not configured, skip upload: {}", p)
        return False

    try:
        logger.info("Uploading {} -> s3://{}/{}", p, S3_BUCKET, key)
        client.upload_file(str(p), S3_BUCKET, key)
        return True
    except (ClientError, BotoCoreError) as exc:
        logger.warning("S3 upload failed for {}: {}", p, exc)
        return False
