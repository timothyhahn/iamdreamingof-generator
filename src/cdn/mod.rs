//! CDN integration for uploading generated content
//!
//! Handles uploading images and JSON files to S3-compatible storage
//! (DigitalOcean Spaces) for web distribution.

pub mod client;
pub mod mock;

pub use client::CdnClient;
pub use mock::MockCdnClient;

use crate::Result;
use async_trait::async_trait;

#[async_trait]
pub trait CdnService: Send + Sync {
    async fn upload_file(&self, key: &str, data: &[u8], content_type: &str) -> Result<String>;
    async fn read_json(&self, key: &str) -> Result<String>;
    async fn file_exists(&self, key: &str) -> Result<bool>;
}
