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


def delete_from_minio(key: str) -> bool:
    """
    Delete an object from MinIO.

    Args:
        key: The MinIO object key to delete

    Returns:
        bool: True if deletion was successful
    """
    client = get_minio_client()
    client.delete_object(Bucket=os.getenv("MINIO_BUCKET_NAME"), Key=key)
    return True


def delete_minio_prefix(prefix: str) -> int:
    """
    Delete all objects with a given prefix from MinIO.
    
    Args:
        prefix: The prefix of objects to delete (e.g., "datasets/uuid/")
        
    Returns:
        int: Number of objects deleted
    """
    client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_NAME")
    
    # List all objects with this prefix
    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    objects_to_delete = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                objects_to_delete.append({'Key': obj['Key']})
    
    if not objects_to_delete:
        return 0
    
    # Delete objects in batches of 1000 (AWS/MinIO limit)
    deleted_count = 0
    for i in range(0, len(objects_to_delete), 1000):
        batch = objects_to_delete[i:i+1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={'Objects': batch}
        )
        deleted_count += len(batch)
    
    return deleted_count


def upload_file_to_minio(file_obj, key: str, content_type: str = "application/octet-stream"):
    """
    Upload Django UploadedFile to MinIO

    Args:
        file_obj: Django UploadedFile (from request.FILES)
        key: MinIO object key
        content_type: MIME type

    Returns:
        str: The MinIO key
    """
    client = get_minio_client()
    file_obj.seek(0)

    client.upload_fileobj(
        Fileobj=file_obj,
        Bucket=os.getenv("MINIO_BUCKET_NAME"),
        Key=key,
        ExtraArgs={'ContentType': content_type}
    )
    return key


def upload_directory_to_minio(local_dir: str, minio_prefix: str):
    """Upload entire directory to MinIO with prefix"""
    from pathlib import Path

    client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_NAME")
    uploaded_count = 0

    for file_path in Path(local_dir).rglob('*'):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_dir)
            minio_key = f"{minio_prefix}{relative_path}".replace('\\', '/')

            # Determine content type
            content_type = 'application/octet-stream'
            if file_path.suffix.lower() in ['.jpg', '.jpeg']:
                content_type = 'image/jpeg'
            elif file_path.suffix.lower() == '.png':
                content_type = 'image/png'
            elif file_path.suffix.lower() == '.gif':
                content_type = 'image/gif'
            elif file_path.suffix.lower() == '.bmp':
                content_type = 'image/bmp'
            elif file_path.suffix.lower() in ['.tiff', '.tif']:
                content_type = 'image/tiff'
            elif file_path.suffix.lower() == '.webp':
                content_type = 'image/webp'

            client.upload_file(
                str(file_path),
                bucket,
                minio_key,
                ExtraArgs={'ContentType': content_type}
            )
            uploaded_count += 1

    return uploaded_count


def download_minio_directory(minio_prefix: str, local_dir: str):
    """Download all objects with prefix to local directory"""
    from pathlib import Path

    client = get_minio_client()
    bucket = os.getenv("MINIO_BUCKET_NAME")

    paginator = client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=minio_prefix)

    downloaded_count = 0
    for page in pages:
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            key = obj['Key']
            relative_path = key[len(minio_prefix):]
            local_path = Path(local_dir) / relative_path

            local_path.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(local_path))
            downloaded_count += 1

    return downloaded_count
