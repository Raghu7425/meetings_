import logging
import asyncio
import boto3
from botocore.exceptions import ClientError
from app.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET

log = logging.getLogger("storage")


def _get_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        region_name="us-east-1",  # MinIO ignores this but boto3 requires a value
    )


def _ensure_bucket(bucket: str = MINIO_BUCKET) -> None:
    client = _get_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            client.create_bucket(Bucket=bucket)
            log.info(f"[storage] Created bucket: {bucket}")
        else:
            raise


def _upload(local_path: str, bucket: str, object_name: str) -> str:
    client = _get_client()
    client.upload_file(local_path, bucket, object_name)
    url = f"{MINIO_ENDPOINT}/{bucket}/{object_name}"
    log.info(f"[storage] Uploaded {local_path} → {url}")
    return url


def _download(bucket: str, object_name: str, local_path: str) -> None:
    client = _get_client()
    client.download_file(bucket, object_name, local_path)
    log.info(f"[storage] Downloaded {bucket}/{object_name} → {local_path}")


async def ensure_bucket(bucket: str = MINIO_BUCKET) -> None:
    await asyncio.to_thread(_ensure_bucket, bucket)


async def upload_file(local_path: str, object_name: str, bucket: str = MINIO_BUCKET) -> str:
    return await asyncio.to_thread(_upload, local_path, bucket, object_name)


async def download_file(object_name: str, local_path: str, bucket: str = MINIO_BUCKET) -> None:
    await asyncio.to_thread(_download, bucket, object_name, local_path)


async def check_minio_health() -> bool:
    try:
        await asyncio.to_thread(_ensure_bucket)
        return True
    except Exception as e:
        log.error(f"[storage] Health check failed: {e}")
        return False
