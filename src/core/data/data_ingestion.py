# data import

import pandas as pd
import boto3
from io import BytesIO
import zipfile
from dotenv import load_dotenv
load_dotenv()
import os
import logging


aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
verbose=True
bucket_name="disease-risk-app"
zip_key="disease-risk-app_zip.zip"



logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_csv_from_s3_zip(bucket_name, zip_key, aws_access_key_id, aws_secret_access_key, verbose=False):

    try:
        logger.info(f"Connecting to S3 bucket: {bucket_name}")
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )

        logger.info(f"Fetching ZIP file from S3: {zip_key}")
        response = s3.get_object(Bucket=bucket_name, Key=zip_key)
        zip_bytes = BytesIO(response['Body'].read())

        with zipfile.ZipFile(zip_bytes) as z:
            file_list = z.namelist()
            logger.info(f"ZIP file contains {len(file_list)} files")
            if verbose:
                print("Files in ZIP:", file_list)

            first_file = file_list[0]
            logger.info(f"Reading first file in ZIP: {first_file}")
            with z.open(first_file) as f:
                df = pd.read_csv(f)
                logger.info(f"Loaded DataFrame with shape: {df.shape}")
                return df

    except Exception as e:
        logger.error(f"Error loading CSV from S3 ZIP: {e}")
        raise

def save_data(s3_data: pd.DataFrame, data_path: str) -> None:
    """Save raw dataset."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
       
        s3_data.to_csv(os.path.join(raw_data_path, "s3_data.csv"), index=False)

        logger.info('raw data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:

        df=load_csv_from_s3_zip(bucket_name,zip_key,aws_access_key_id,aws_secret_access_key)

        # drop use less columns
        df= df.drop(columns="id")
        logger.info("dropping useless column")
        
        save_data(df, data_path='./output')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()