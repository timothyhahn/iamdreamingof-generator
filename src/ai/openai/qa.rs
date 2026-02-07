use super::client::OpenAiHttpClient;
use super::types::{
    ChatCompletionRequest, ChatMessage, ChatMessageContent, ImageUrl, JsonSchema, MessagePart,
    ResponseFormat,
};
use crate::ai::ImageQaService;
use crate::models::TextDetectionResponse;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use std::time::Duration;

pub struct OpenAiImageQaClient {
    http: OpenAiHttpClient,
    model: String,
}

impl OpenAiImageQaClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_client(api_key, model, reqwest::Client::new())
    }

    pub fn new_with_client(api_key: String, model: String, client: reqwest::Client) -> Self {
        Self {
            http: OpenAiHttpClient::new_with_client(api_key, Duration::from_secs(30), client),
            model,
        }
    }
}

#[cfg(test)]
super::impl_with_openai_base_url!(OpenAiImageQaClient);

#[async_trait]
impl ImageQaService for OpenAiImageQaClient {
    async fn detect_text(&self, image_bytes: &[u8]) -> Result<bool> {
        tracing::debug!("Detecting text in image ({} bytes)", image_bytes.len());

        use base64::Engine as _;
        let base64_image = base64::engine::general_purpose::STANDARD.encode(image_bytes);
        let mime = crate::ai::mime::detect_image_mime(image_bytes);
        let data_url = format!("data:{};base64,{}", mime, base64_image);

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "includes_text": {
                    "type": "boolean",
                    "description": "True if the image contains any text, letters, words, or writing"
                }
            },
            "required": ["includes_text"],
            "additionalProperties": false
        });

        let response_format = ResponseFormat {
            format_type: "json_schema".to_string(),
            json_schema: JsonSchema {
                name: "text_detection".to_string(),
                schema,
                strict: true,
            },
        };

        let system_message = ChatMessage {
            role: "system".to_string(),
            content: Some(ChatMessageContent::Text(prompts::QA_SYSTEM.to_string())),
        };

        let user_message = ChatMessage {
            role: "user".to_string(),
            content: Some(ChatMessageContent::ImageContent(vec![
                MessagePart {
                    part_type: "text".to_string(),
                    text: Some(prompts::QA_USER.to_string()),
                    image_url: None,
                },
                MessagePart {
                    part_type: "image_url".to_string(),
                    text: None,
                    image_url: Some(ImageUrl { url: data_url }),
                },
            ])),
        };

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![system_message, user_message],
            max_completion_tokens: 100,
            response_format: Some(response_format),
        };

        let response = self.http.chat_completion(&request).await?;

        let json_str = response
            .choices
            .first()
            .and_then(|choice| match &choice.message.content {
                Some(ChatMessageContent::Text(text)) => Some(text.clone()),
                _ => None,
            })
            .ok_or_else(|| Error::AiProvider("No response from text detection".to_string()))?;

        let detection_result: TextDetectionResponse =
            serde_json::from_str(&json_str).map_err(|e| {
                Error::AiProvider(format!("Failed to parse text detection response: {}", e))
            })?;

        tracing::info!(
            "Text detection result: includes_text={}",
            detection_result.includes_text
        );

        Ok(detection_result.includes_text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::openai::test_support;
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gpt-4o-mini";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> OpenAiImageQaClient {
        OpenAiImageQaClient::new(api_key.to_string(), model.to_string()).with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_detect_text_returns_false_when_no_text() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "{\"includes_text\": false}"
                    },
                    "finish_reason": "stop"
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

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "{\"includes_text\": true}"
                    },
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let has_text = client.detect_text(&[0x89, 0x50]).await.unwrap();
        assert!(has_text);
    }

    #[tokio::test]
    async fn test_detect_text_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(500).set_body_string("internal error"))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);
        let err = client.detect_text(&[0x89, 0x50]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_detect_text_rejects_empty_choices() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": []
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

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "{\"includes_text\":\"maybe\"}"
                    },
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);
        let err = client.detect_text(&[0x89, 0x50]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
