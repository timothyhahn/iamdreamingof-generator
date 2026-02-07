use super::CdnService;
use crate::{Error, Result};
use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::{config::Region, types::ObjectCannedAcl, Client as S3Client};

pub struct CdnClient {
    client: S3Client,
    bucket: String,
    base_url: String,
}

impl CdnClient {
    pub async fn new(
        access_key_id: String,
        secret_access_key: String,
        endpoint: String,
        bucket: String,
        base_url: String,
    ) -> Result<Self> {
        let credentials = aws_sdk_s3::config::Credentials::new(
            access_key_id,
            secret_access_key,
            None,
            None,
            "digital-ocean-spaces",
        );

        // Create custom config for DigitalOcean Spaces
        let config = aws_config::defaults(BehaviorVersion::latest())
            .credentials_provider(credentials)
            .region(Region::new("us-east-1")) // DigitalOcean Spaces doesn't really use regions
            .endpoint_url(endpoint)
            .load()
            .await;

        let client = S3Client::new(&config);

        Ok(Self {
            client,
            bucket,
            base_url,
        })
    }

    fn get_public_url(&self, key: &str) -> String {
        format!("{}/{}", self.base_url, key)
    }
}

#[async_trait]
impl CdnService for CdnClient {
    async fn upload_file(&self, key: &str, data: &[u8], content_type: &str) -> Result<String> {
        let body = ByteStream::from(data.to_vec());

        self.client
            .put_object()
            .bucket(&self.bucket)
            .key(key)
            .body(body)
            .content_type(content_type)
            .acl(ObjectCannedAcl::PublicRead)
            .send()
            .await
            .map_err(|e| Error::S3(format!("Failed to upload file: {}", e)))?;

        Ok(self.get_public_url(key))
    }

    async fn read_json(&self, key: &str) -> Result<String> {
        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .map_err(|e| Error::S3(format!("Failed to read file: {}", e)))?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| Error::S3(format!("Failed to read body: {}", e)))?;

        String::from_utf8(bytes.to_vec()).map_err(|e| Error::S3(format!("Invalid UTF-8: {}", e)))
    }
}
