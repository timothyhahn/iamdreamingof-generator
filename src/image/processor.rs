use super::{ImageService, ProcessedImages};
use crate::Result;
use async_trait::async_trait;
use image::{DynamicImage, ImageFormat};
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

    async fn resize_and_save(
        &self,
        image: DynamicImage,
        output_path: &Path,
        format: ImageFormat,
    ) -> Result<()> {
        let resized = image.resize_exact(800, 800, image::imageops::FilterType::Lanczos3);

        resized.save_with_format(output_path, format)?;

        Ok(())
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

        self.resize_and_save(img.clone(), &jpeg_path, ImageFormat::Jpeg)
            .await?;
        self.resize_and_save(img, &webp_path, ImageFormat::WebP)
            .await?;

        Ok(ProcessedImages {
            jpeg_path: jpeg_path.to_string_lossy().to_string(),
            webp_path: webp_path.to_string_lossy().to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
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
            let temp_dir = TempDir::new()
                .map_err(|e| Error::Generic(format!("Failed to create temp dir: {}", e)))?;

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

        assert!(Path::new(&result.jpeg_path).exists());
        assert!(Path::new(&result.webp_path).exists());

        assert!(result.jpeg_path.ends_with(".jpg"));
        assert!(result.webp_path.ends_with(".webp"));

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
