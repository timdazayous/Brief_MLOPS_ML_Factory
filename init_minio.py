"""
Init Minio Bucket for MLflow.
This script creates the 'mlflow' bucket in MinIO if it does not already exist.
Ensure that the MinIO service is running before executing this script.
"""

import boto3
from botocore.exceptions import ClientError
import os

def init_bucket(bucket_name="mlflow", endpoint_url="http://localhost:9000", access_key="minioadmin", secret_key="minioadmin"):
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=boto3.session.Config(signature_version='s3v4')
    )

    try:
        # Check if bucket exists
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code')
        if error_code == '404':
            print(f"Bucket '{bucket_name}' does not exist. Creating it...")
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")
        else:
            print(f"An error occurred: {e}")
            raise

if __name__ == "__main__":
    init_bucket()
