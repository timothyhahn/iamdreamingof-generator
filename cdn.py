import os

import botocore
import boto3
import requests

ENDPOINT_URL = "https://nyc3.digitaloceanspaces.com"
CONFIG = botocore.config.Config(s3={"addressing_style": "virtual"})
REGION = "nyc3"
CDN_ACCESS_KEY_ID = os.environ["CDN_ACCESS_KEY_ID"]
CDN_SECRET_ACCESS_KEY = os.environ["CDN_SECRET_ACCESS_KEY"]
BUCKET = "iamdreamingof"
CDN_BASE_URL = "https://cdn.iamdreamingof.com"


def get_client():
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        config=CONFIG,
        region_name=REGION,
        aws_access_key_id=CDN_ACCESS_KEY_ID,
        aws_secret_access_key=CDN_SECRET_ACCESS_KEY,
    )


def upload_file(path: str, key: str) -> str:
    client = get_client()
    client.upload_file(path, BUCKET, key, ExtraArgs={"ACL": "public-read"})
    return f"{CDN_BASE_URL}/{key}"


# TODO: This is easier, but this is hitting the CDN's edge cache, which means it's not always up to date. Switch to hit the origin direectly.
def read_public_json(path: str) -> str:
    return requests.get(f"{CDN_BASE_URL}/{path}").json()
