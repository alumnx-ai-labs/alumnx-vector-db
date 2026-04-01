import logging
import boto3
from botocore.exceptions import ClientError
from app.config import get_config

logger = logging.getLogger("nexvec.s3_store")

class S3Store:
    def __init__(self):
        self.config = get_config()
        if self.config.s3_bucket_name:
            self.s3_client = boto3.client(
                "s3",
                region_name=self.config.s3_region or "ap-south-1",
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key,
            )
        else:
            self.s3_client = None
        self.bucket = self.config.s3_bucket_name

    def upload_file(self, file_path: str, object_name: str) -> bool:
        if not self.bucket or not self.s3_client:
            logger.warning("S3 bucket not configured. Skipping upload.")
            return False
        try:
            self.s3_client.upload_file(file_path, self.bucket, object_name)
            logger.info("Uploaded %s to S3 bucket %s", object_name, self.bucket)
            return True
        except Exception as e:
            logger.error("Failed to upload %s to S3: %s", object_name, str(e))
            return False

    def delete_file(self, object_name: str) -> bool:
        if not self.bucket or not self.s3_client:
            logger.warning("S3 bucket not configured. Skipping delete.")
            return False
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=object_name)
            logger.info("Deleted %s from S3 bucket %s", object_name, self.bucket)
            return True
        except Exception as e:
            logger.error("Failed to delete %s from S3: %s", object_name, str(e))
            return False
