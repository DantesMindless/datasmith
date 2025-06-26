import boto3
import os


def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT"),
        aws_access_key_id=os.getenv("MINIO_ACCESS_KEY"),
        aws_secret_access_key=os.getenv("MINIO_SECRET_KEY"),
    )


def upload_to_minio(
    file_bytes: bytes, key: str, content_type: str = "application/octet-stream"
):
    client = get_minio_client()
    client.put_object(
        Bucket=os.getenv("MINIO_BUCKET_NAME"),
        Key=key,
        Body=file_bytes,
        ContentType=content_type,
    )
    return key


def download_from_minio(key: str) -> bytes:
    client = get_minio_client()
    response = client.get_object(Bucket=os.getenv("MINIO_BUCKET_NAME"), Key=key)
    return response["Body"].read()
