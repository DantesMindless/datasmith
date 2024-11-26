import torch
import boto3

def save_model_to_minio(model, bucket_name, key):
    torch.save(model.state_dict(), "model.pt")
    s3 = boto3.client(
        's3',
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    s3.upload_file("model.pt", bucket_name, key)

def load_model_from_minio(model, bucket_name, key):
    s3 = boto3.client(
        's3',
        endpoint_url="http://localhost:9000",
        aws_access_key_id="minioadmin",
        aws_secret_access_key="minioadmin",
    )
    s3.download_file(bucket_name, key, "model.pt")
    model.load_state_dict(torch.load("model.pt"))
    return model
