use super::client::GeminiHttpClient;
use crate::ai::ImageGenerationService;
use crate::models::Word;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize)]
struct GenerateContentRequest {
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: GenerationConfig,
}

#[derive(Debug, Serialize)]
struct Content {
    parts: Vec<TextPart>,
}

#[derive(Debug, Serialize)]
struct TextPart {
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    response_modalities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_config: Option<ImageConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ImageConfig {
    aspect_ratio: String,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: ResponseContent,
}

#[derive(Debug, Deserialize)]
struct ResponseContent {
    parts: Vec<ResponsePart>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponsePart {
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
    #[allow(dead_code)]
    Text { text: String },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct InlineData {
    mime_type: String,
    data: String,
}

pub struct GeminiImageClient {
    http: GeminiHttpClient,
}

impl GeminiImageClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: GeminiHttpClient::new(api_key, model, Duration::from_secs(120)),
        }
    }

    #[cfg(test)]
    fn with_base_url(mut self, base_url: String) -> Self {
        self.http = self.http.with_base_url(base_url);
        self
    }
}

#[async_trait]
impl ImageGenerationService for GeminiImageClient {
    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>> {
        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        let words_str = word_list.join(", ");

        let enhanced_prompt = prompts::render(
            prompts::IMAGE_ENHANCEMENT,
            &[("prompt", prompt), ("words", &words_str)],
        );

        let request = GenerateContentRequest {
            contents: vec![Content {
                parts: vec![TextPart {
                    text: enhanced_prompt,
                }],
            }],
            generation_config: GenerationConfig {
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
                    ResponsePart::InlineData { inline_data } => Some(inline_data),
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
            .map_err(|e| Error::Generic(format!("Failed to decode Gemini base64 image: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_generate_image_parses_inline_data() {
        let server = MockServer::start().await;

        use base64::Engine as _;
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&fake_image);

        Mock::given(method("POST"))
            .and(path_regex(r"/v1beta/models/.+:generateContent"))
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

        let client =
            GeminiImageClient::new("key".to_string(), "gemini-2.5-flash-image".to_string())
                .with_base_url(server.uri());

        let result = client.generate_image("a dream", &[]).await.unwrap();
        assert_eq!(result, fake_image);
    }

    #[tokio::test]
    async fn test_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path_regex(r"/v1beta/models/.+:generateContent"))
            .respond_with(ResponseTemplate::new(429).set_body_string("quota exceeded"))
            .mount(&server)
            .await;

        let client =
            GeminiImageClient::new("key".to_string(), "gemini-2.5-flash-image".to_string())
                .with_base_url(server.uri());

        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_request_uses_square_aspect_ratio() {
        let server = MockServer::start().await;

        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode([0x00]);

        Mock::given(method("POST"))
            .and(path_regex(r"/v1beta/models/.+:generateContent"))
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

        let client =
            GeminiImageClient::new("key".to_string(), "gemini-2.5-flash-image".to_string())
                .with_base_url(server.uri());

        client.generate_image("test", &[]).await.unwrap();
    }
}
