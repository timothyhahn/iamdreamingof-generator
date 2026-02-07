use super::{ImageService, ProcessedImages};
use crate::Result;
use async_trait::async_trait;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub struct MockImageProcessor {
    process_count: Arc<Mutex<usize>>,
    base_path: PathBuf,
    should_fail: Arc<Mutex<bool>>,
}

impl MockImageProcessor {
    pub fn new() -> Self {
        Self {
            process_count: Arc::new(Mutex::new(0)),
            base_path: PathBuf::from("/tmp"),
            should_fail: Arc::new(Mutex::new(false)),
        }
    }

    pub fn with_base_path(mut self, path: String) -> Self {
        self.base_path = PathBuf::from(path);
        self
    }

    pub fn with_failure(self, should_fail: bool) -> Self {
        *self.should_fail.lock().unwrap() = should_fail;
        self
    }

    pub fn get_process_count(&self) -> usize {
        *self.process_count.lock().unwrap()
    }
}

impl Default for MockImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageService for MockImageProcessor {
    async fn process_image(&self, _image_data: &[u8], base_name: &str) -> Result<ProcessedImages> {
        if *self.should_fail.lock().unwrap() {
            return Err(crate::Error::Image(image::ImageError::IoError(
                std::io::Error::other("Mock failure"),
            )));
        }

        let mut count = self.process_count.lock().unwrap();
        *count += 1;

        let uuid = Uuid::new_v4();
        let jpeg_path = self.base_path.join(format!("{}_{}.jpg", base_name, uuid));
        let webp_path = self.base_path.join(format!("{}_{}.webp", base_name, uuid));

        fs::create_dir_all(&self.base_path)?;
        // Write lightweight fixtures so callers that read from returned paths
        // exercise the same contract as production ImageProcessor.
        fs::write(&jpeg_path, b"mock-jpeg-bytes")?;
        fs::write(&webp_path, b"mock-webp-bytes")?;

        Ok(ProcessedImages {
            jpeg_path,
            webp_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_image_processor() {
        let processor = MockImageProcessor::new();

        let result = processor
            .process_image(b"fake image data", "test")
            .await
            .unwrap();

        assert!(result.jpeg_path.to_string_lossy().contains("test"));
        assert!(result.jpeg_path.to_string_lossy().ends_with(".jpg"));
        assert!(result.webp_path.to_string_lossy().contains("test"));
        assert!(result.webp_path.to_string_lossy().ends_with(".webp"));
        assert!(result.jpeg_path.exists());
        assert!(result.webp_path.exists());

        assert_eq!(processor.get_process_count(), 1);
    }

    #[tokio::test]
    async fn test_mock_with_custom_path() {
        let dir = tempfile::tempdir().unwrap();
        let processor =
            MockImageProcessor::new().with_base_path(dir.path().to_string_lossy().to_string());

        let result = processor.process_image(b"data", "image").await.unwrap();

        assert!(result.jpeg_path.starts_with(dir.path()));
        assert!(result.webp_path.starts_with(dir.path()));
    }

    #[tokio::test]
    async fn test_mock_with_failure() {
        let processor = MockImageProcessor::new().with_failure(true);

        let result = processor.process_image(b"data", "test").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_mock_unique_paths() {
        let processor = MockImageProcessor::new();

        let result1 = processor.process_image(b"data", "test").await.unwrap();

        let result2 = processor.process_image(b"data", "test").await.unwrap();

        // Paths should be unique due to UUID
        assert_ne!(result1.jpeg_path, result2.jpeg_path);
        assert_ne!(result1.webp_path, result2.webp_path);
    }
}
