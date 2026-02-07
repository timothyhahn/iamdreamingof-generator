use super::{ImageService, ProcessedImages};
use crate::{Error, Result};
use async_trait::async_trait;
use image::{DynamicImage, ImageFormat};
use little_exif::metadata::Metadata;
use std::path::{Path, PathBuf};
use uuid::Uuid;

pub struct ImageProcessor {
    output_dir: PathBuf,
}

impl ImageProcessor {
    pub fn new(output_dir: &Path) -> Result<Self> {
        Ok(Self {
            output_dir: output_dir.to_path_buf(),
        })
    }

    fn save_variants_sync(
        image: DynamicImage,
        jpeg_path: PathBuf,
        webp_path: PathBuf,
    ) -> Result<()> {
        let resized = image.resize_exact(800, 800, image::imageops::FilterType::Lanczos3);
        resized.save_with_format(jpeg_path, ImageFormat::Jpeg)?;
        resized.save_with_format(webp_path, ImageFormat::WebP)?;
        Ok(())
    }

    async fn save_variants(
        &self,
        image: DynamicImage,
        jpeg_path: &Path,
        webp_path: &Path,
    ) -> Result<()> {
        tokio::task::spawn_blocking({
            let jpeg_path = jpeg_path.to_path_buf();
            let webp_path = webp_path.to_path_buf();
            move || Self::save_variants_sync(image, jpeg_path, webp_path)
        })
        .await
        .map_err(|e| Error::Invariant(format!("Image processing task join error: {}", e)))?
    }
}

#[async_trait]
impl ImageService for ImageProcessor {
    async fn process_image(&self, image_data: &[u8], base_name: &str) -> Result<ProcessedImages> {
        let img = image::load_from_memory(image_data)?;

        // Generate unique filename
        let uuid = Uuid::new_v4();
        let jpeg_filename = format!("{}_{}.jpg", base_name, uuid);
        let webp_filename = format!("{}_{}.webp", base_name, uuid);

        let jpeg_path = self.output_dir.join(&jpeg_filename);
        let webp_path = self.output_dir.join(&webp_filename);

        self.save_variants(img, &jpeg_path, &webp_path).await?;
        // Strip EXIF to prevent stale orientation tags from confusing viewers
        if let Err(e) = Metadata::file_clear_metadata(&jpeg_path) {
            tracing::warn!("Failed to strip EXIF from JPEG: {}", e);
        }
        if let Err(e) = Metadata::file_clear_metadata(&webp_path) {
            tracing::warn!("Failed to strip EXIF from WebP: {}", e);
        }

        Ok(ProcessedImages {
            jpeg_path,
            webp_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_image() -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(10, 10, image::Rgba([255, 0, 0, 255]));
        let mut bytes = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut bytes), ImageFormat::Png)
            .unwrap();
        bytes
    }

    struct TestImageProcessor {
        processor: ImageProcessor,
        _temp_dir: TempDir,
    }

    impl TestImageProcessor {
        fn new() -> Result<Self> {
            let temp_dir = TempDir::new()?;

            let processor = ImageProcessor {
                output_dir: temp_dir.path().to_path_buf(),
            };

            Ok(Self {
                processor,
                _temp_dir: temp_dir,
            })
        }
    }

    #[tokio::test]
    async fn test_image_processing() {
        let test_processor = TestImageProcessor::new().unwrap();
        let processor = &test_processor.processor;
        let test_image = create_test_image();

        let result = processor.process_image(&test_image, "test").await.unwrap();

        assert!(result.jpeg_path.exists());
        assert!(result.webp_path.exists());

        assert!(result.jpeg_path.to_string_lossy().ends_with(".jpg"));
        assert!(result.webp_path.to_string_lossy().ends_with(".webp"));

        let jpeg_img = image::open(&result.jpeg_path).unwrap();
        assert_eq!(jpeg_img.width(), 800);
        assert_eq!(jpeg_img.height(), 800);

        let webp_img = image::open(&result.webp_path).unwrap();
        assert_eq!(webp_img.width(), 800);
        assert_eq!(webp_img.height(), 800);
    }

    #[tokio::test]
    async fn test_unique_filenames() {
        let test_processor = TestImageProcessor::new().unwrap();
        let processor = &test_processor.processor;
        let test_image = create_test_image();

        let result1 = processor.process_image(&test_image, "test").await.unwrap();

        let result2 = processor.process_image(&test_image, "test").await.unwrap();

        assert_ne!(result1.jpeg_path, result2.jpeg_path);
        assert_ne!(result1.webp_path, result2.webp_path);
    }
}
