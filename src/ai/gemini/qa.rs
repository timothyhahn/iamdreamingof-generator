use super::client::GeminiHttpClient;
use super::types::{Content, GenerateContentResponse, InlineData, Part};
use crate::ai::ImageQaService;
use crate::models::TextDetectionResponse;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Serialize)]
struct QaRequest {
    system_instruction: Option<Content>,
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: Option<QaGenerationConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct QaGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
}

pub struct GeminiImageQaClient {
    http: GeminiHttpClient,
}

impl GeminiImageQaClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_client(api_key, model, reqwest::Client::new())
    }

    pub fn new_with_client(api_key: String, model: String, client: reqwest::Client) -> Self {
        Self {
            http: GeminiHttpClient::new_with_client(
                api_key,
                model,
                Duration::from_secs(30),
                client,
            ),
        }
    }
}

#[cfg(test)]
super::impl_with_gemini_base_url!(GeminiImageQaClient);

#[async_trait]
impl ImageQaService for GeminiImageQaClient {
    async fn detect_text(&self, image_bytes: &[u8]) -> Result<bool> {
        tracing::debug!(
            "Detecting text in image ({} bytes) via Gemini",
            image_bytes.len()
        );

        use base64::Engine as _;
        let base64_image = base64::engine::general_purpose::STANDARD.encode(image_bytes);

        let request = QaRequest {
            system_instruction: Some(Content {
                role: None,
                parts: vec![Part::Text {
                    text: prompts::QA_SYSTEM.to_string(),
                }],
            }),
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![
                    Part::InlineData {
                        inline_data: InlineData {
                            mime_type: crate::ai::mime::detect_image_mime(image_bytes).to_string(),
                            data: base64_image,
                        },
                    },
                    Part::Text {
                        text: prompts::QA_USER.to_string(),
                    },
                ],
            }],
            generation_config: Some(QaGenerationConfig {
                max_output_tokens: Some(100),
                response_mime_type: Some("application/json".to_string()),
            }),
        };

        let response: GenerateContentResponse = self.http.generate_content(&request).await?;

        let text = response
            .candidates
            .first()
            .and_then(|c| {
                c.content.parts.iter().find_map(|p| match p {
                    Part::Text { text } => Some(text.clone()),
                    _ => None,
                })
            })
            .ok_or_else(|| {
                Error::AiProvider("No response from Gemini text detection".to_string())
            })?;

        let detection: TextDetectionResponse = serde_json::from_str(&text).map_err(|e| {
            Error::AiProvider(format!(
                "Failed to parse Gemini text detection response: {}",
                e
            ))
        })?;

        tracing::info!(
            "Gemini text detection result: includes_text={}",
            detection.includes_text
        );

        Ok(detection.includes_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::gemini::test_support;
    use wiremock::matchers::body_string_contains;
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gemini-3-flash-preview";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> GeminiImageQaClient {
        GeminiImageQaClient::new(api_key.to_string(), model.to_string()).with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_detect_text_returns_false_when_no_text() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .and(body_string_contains("\"inlineData\""))
            .and(body_string_contains("\"mimeType\""))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "{\"includes_text\": false}" }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let has_text = client.detect_text(&[0x89, 0x50]).await.unwrap();
        assert!(!has_text);
    }

    #[tokio::test]
    async fn test_detect_text_returns_true_when_text_found() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "{\"includes_text\": true}" }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let has_text = client.detect_text(&[0x89, 0x50]).await.unwrap();
        assert!(has_text);
    }

    #[tokio::test]
    async fn test_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        let err = client.detect_text(&[0x89, 0x50]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_detect_text_rejects_empty_candidates() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": []
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);
        let err = client.detect_text(&[0x89, 0x50]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_detect_text_rejects_invalid_json_payload() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "{\"includes_text\":\"sometimes\"}" }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);
        let err = client.detect_text(&[0x89, 0x50]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
