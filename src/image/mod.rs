//! Image processing and format conversion
//!
//! Handles resizing and converting generated images to web-optimized
//! formats (JPEG and WebP) for efficient delivery.

pub mod mock;
pub mod processor;

pub use mock::MockImageProcessor;
pub use processor::ImageProcessor;

use crate::Result;
use async_trait::async_trait;

pub struct ProcessedImages {
    pub jpeg_path: String,
    pub webp_path: String,
}

#[async_trait]
pub trait ImageService: Send + Sync {
    async fn process_image(&self, image_data: &[u8], base_name: &str) -> Result<ProcessedImages>;
}
