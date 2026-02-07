use super::client::GeminiHttpClient;
use super::types::{Content, GenerateContentResponse, Part};
use crate::ai::{words_to_csv, ImageGenerationService};
use crate::models::Word;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Serialize)]
struct ImageRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: ImageGenerationConfig,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageGenerationConfig {
    response_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_config: Option<ImageConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageConfig {
    aspect_ratio: String,
}

pub struct GeminiImageClient {
    http: GeminiHttpClient,
}

impl GeminiImageClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_client(api_key, model, reqwest::Client::new())
    }

    pub fn new_with_client(api_key: String, model: String, client: reqwest::Client) -> Self {
        Self {
            http: GeminiHttpClient::new_with_client(
                api_key,
                model,
                Duration::from_secs(120),
                client,
            ),
        }
    }
}

#[cfg(test)]
super::impl_with_gemini_base_url!(GeminiImageClient);

#[async_trait]
impl ImageGenerationService for GeminiImageClient {
    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>> {
        let words_str = words_to_csv(words);

        let enhanced_prompt = prompts::render(
            prompts::IMAGE_ENHANCEMENT,
            &[("prompt", prompt), ("words", &words_str)],
        );

        let request = ImageRequest {
            contents: vec![Content {
                role: None,
                parts: vec![Part::Text {
                    text: enhanced_prompt,
                }],
            }],
            generation_config: ImageGenerationConfig {
                response_modalities: vec!["IMAGE".to_string()],
                image_config: Some(ImageConfig {
                    aspect_ratio: "1:1".to_string(),
                }),
            },
        };

        let gemini_response: GenerateContentResponse = self.http.generate_content(&request).await?;

        let image_data = gemini_response
            .candidates
            .first()
            .and_then(|c| {
                c.content.parts.iter().find_map(|p| match p {
                    Part::InlineData { inline_data } => Some(inline_data),
                    _ => None,
                })
            })
            .ok_or_else(|| Error::AiProvider("No image data in Gemini response".to_string()))?;

        tracing::debug!(
            "Gemini returned image with mime_type: {}",
            image_data.mime_type
        );

        use base64::Engine as _;
        base64::engine::general_purpose::STANDARD
            .decode(&image_data.data)
            .map_err(|e| Error::AiProvider(format!("Failed to decode Gemini base64 image: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::gemini::test_support;
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gemini-2.5-flash-image";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> GeminiImageClient {
        GeminiImageClient::new(api_key.to_string(), model.to_string()).with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_generate_image_parses_inline_data() {
        let server = MockServer::start().await;

        use base64::Engine as _;
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&fake_image);

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": b64
                            }
                        }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        let result = client.generate_image("a dream", &[]).await.unwrap();
        assert_eq!(result, fake_image);
    }

    #[tokio::test]
    async fn test_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(429).set_body_string("quota exceeded"))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_request_uses_square_aspect_ratio() {
        let server = MockServer::start().await;

        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode([0x00]);

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .and(wiremock::matchers::body_string_contains(
                "\"aspectRatio\":\"1:1\"",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": { "mimeType": "image/png", "data": b64 }
                        }]
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        client.generate_image("test", &[]).await.unwrap();
    }

    #[tokio::test]
    async fn test_generate_image_rejects_missing_inline_data() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": { "parts": [{ "text": "no image here" }] }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);
        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_generate_image_rejects_invalid_base64() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": "!!!invalid-base64!!!"
                            }
                        }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);
        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
