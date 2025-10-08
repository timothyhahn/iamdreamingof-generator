use super::CdnService;
use crate::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct MockCdnClient {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    base_url: String,
    upload_count: Arc<Mutex<usize>>,
    read_count: Arc<Mutex<usize>>,
}

impl MockCdnClient {
    pub fn new() -> Self {
        Self {
            files: Arc::new(Mutex::new(HashMap::new())),
            base_url: "https://mock-cdn.example.com".to_string(),
            upload_count: Arc::new(Mutex::new(0)),
            read_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_file(self, key: String, content: Vec<u8>) -> Self {
        self.files.lock().unwrap().insert(key, content);
        self
    }

    pub fn get_upload_count(&self) -> usize {
        *self.upload_count.lock().unwrap()
    }

    pub fn get_read_count(&self) -> usize {
        *self.read_count.lock().unwrap()
    }

    pub fn get_files(&self) -> HashMap<String, Vec<u8>> {
        self.files.lock().unwrap().clone()
    }
}

impl Default for MockCdnClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl CdnService for MockCdnClient {
    async fn upload_file(&self, key: &str, data: &[u8], _content_type: &str) -> Result<String> {
        let mut count = self.upload_count.lock().unwrap();
        *count += 1;

        self.files
            .lock()
            .unwrap()
            .insert(key.to_string(), data.to_vec());
        Ok(format!("{}/{}", self.base_url, key))
    }

    async fn read_json(&self, key: &str) -> Result<String> {
        let mut count = self.read_count.lock().unwrap();
        *count += 1;

        let files = self.files.lock().unwrap();
        match files.get(key) {
            Some(data) => String::from_utf8(data.clone())
                .map_err(|e| crate::Error::S3(format!("Invalid UTF-8: {}", e))),
            None => Err(crate::Error::S3(format!("File not found: {}", key))),
        }
    }

    async fn file_exists(&self, key: &str) -> Result<bool> {
        Ok(self.files.lock().unwrap().contains_key(key))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_cdn_upload_and_read() {
        let client = MockCdnClient::new();

        // Upload a file
        let url = client
            .upload_file("test.json", b"{\"test\": true}", "application/json")
            .await
            .unwrap();

        assert_eq!(url, "https://mock-cdn.example.com/test.json");
        assert_eq!(client.get_upload_count(), 1);

        // Read the file back
        let content = client.read_json("test.json").await.unwrap();
        assert_eq!(content, "{\"test\": true}");
        assert_eq!(client.get_read_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_cdn_file_exists() {
        let client =
            MockCdnClient::new().with_file("existing.json".to_string(), b"content".to_vec());

        assert!(client.file_exists("existing.json").await.unwrap());
        assert!(!client.file_exists("missing.json").await.unwrap());
    }

    #[tokio::test]
    async fn test_mock_cdn_with_custom_base_url() {
        let client = MockCdnClient::new().with_base_url("https://custom-cdn.com".to_string());

        let url = client
            .upload_file("file.txt", b"data", "text/plain")
            .await
            .unwrap();

        assert_eq!(url, "https://custom-cdn.com/file.txt");
    }

    #[tokio::test]
    async fn test_mock_cdn_read_missing_file() {
        let client = MockCdnClient::new();
        let result = client.read_json("missing.json").await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }
}
