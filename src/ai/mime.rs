pub fn detect_image_mime(bytes: &[u8]) -> &'static str {
    match bytes {
        [0xFF, 0xD8, 0xFF, ..] => "image/jpeg",
        [0x89, 0x50, 0x4E, 0x47, ..] => "image/png",
        [0x52, 0x49, 0x46, 0x46, _, _, _, _, 0x57, 0x45, 0x42, 0x50, ..] => "image/webp",
        _ => {
            tracing::warn!(
                "Unrecognized image format (first 4 bytes: {:02X?}), falling back to image/png",
                &bytes[..bytes.len().min(4)]
            );
            "image/png"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_png() {
        assert_eq!(
            detect_image_mime(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A]),
            "image/png"
        );
    }

    #[test]
    fn test_detect_jpeg() {
        assert_eq!(detect_image_mime(&[0xFF, 0xD8, 0xFF, 0xE0]), "image/jpeg");
    }

    #[test]
    fn test_detect_webp() {
        assert_eq!(
            detect_image_mime(&[
                0x52, 0x49, 0x46, 0x46, 0x00, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50
            ]),
            "image/webp"
        );
    }

    #[test]
    fn test_unknown_falls_back_to_png() {
        assert_eq!(detect_image_mime(&[0x00, 0x01, 0x02, 0x03]), "image/png");
    }

    #[test]
    fn test_empty_falls_back_to_png() {
        assert_eq!(detect_image_mime(&[]), "image/png");
    }
}
