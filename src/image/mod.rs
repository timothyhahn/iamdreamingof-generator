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
use std::path::PathBuf;

pub struct ProcessedImages {
    /// Absolute path to the generated JPEG file on disk.
    pub jpeg_path: PathBuf,
    /// Absolute path to the generated WebP file on disk.
    pub webp_path: PathBuf,
}

#[async_trait]
pub trait ImageService: Send + Sync {
    /// Process and persist image outputs, returning filesystem paths to the
    /// created JPEG and WebP files.
    async fn process_image(&self, image_data: &[u8], base_name: &str) -> Result<ProcessedImages>;
}
