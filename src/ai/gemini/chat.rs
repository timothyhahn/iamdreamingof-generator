use super::client::GeminiHttpClient;
use super::types::{Content, GenerateContentResponse, Part};
use crate::ai::{words_to_csv, ChatService};
use crate::models::Word;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use serde::Serialize;
use std::time::Duration;

#[derive(Debug, Serialize)]
struct ChatRequest {
    system_instruction: Option<Content>,
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: Option<ChatGenerationConfig>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ChatGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

pub struct GeminiChatClient {
    http: GeminiHttpClient,
}

impl GeminiChatClient {
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

    fn extract_text(response: &GenerateContentResponse) -> Option<String> {
        response.candidates.first().and_then(|c| {
            c.content.parts.iter().find_map(|p| match p {
                Part::Text { text } => Some(text.clone()),
                Part::InlineData { .. } => None,
            })
        })
    }
}

#[cfg(test)]
super::impl_with_gemini_base_url!(GeminiChatClient);

#[async_trait]
impl ChatService for GeminiChatClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let words_str = words_to_csv(words);

        let request = ChatRequest {
            system_instruction: Some(Content {
                role: None,
                parts: vec![Part::Text {
                    text: prompts::CHAT_SYSTEM.to_string(),
                }],
            }),
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::Text {
                    text: prompts::render(prompts::CHAT_USER, &[("words", &words_str)]),
                }],
            }],
            generation_config: Some(ChatGenerationConfig {
                max_output_tokens: Some(3000),
            }),
        };

        let response: GenerateContentResponse = self.http.generate_content(&request).await?;

        Self::extract_text(&response)
            .ok_or_else(|| Error::AiProvider("No text in Gemini chat response".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::gemini::test_support;
    use crate::models::WordType;
    use wiremock::matchers::{method, path};
    use wiremock::Mock;
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gemini-3-flash-preview";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> GeminiChatClient {
        GeminiChatClient::new(api_key.to_string(), model.to_string()).with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_generate_prompt_parses_response() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Clouds of honey drift past a floating cat" }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let words = vec![
            Word {
                word: "honey".to_string(),
                word_type: WordType::Object,
            },
            Word {
                word: "floating".to_string(),
                word_type: WordType::Gerund,
            },
            Word {
                word: "cat".to_string(),
                word_type: WordType::Object,
            },
        ];

        let prompt = client.generate_prompt(&words).await.unwrap();
        assert_eq!(prompt, "Clouds of honey drift past a floating cat");
    }

    #[tokio::test]
    async fn test_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
            .mount(&server)
            .await;

        let client = make_client(&server, "bad-key", DEFAULT_MODEL);

        let err = client.generate_prompt(&[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_generate_prompt_rejects_empty_candidates() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::GENERATE_CONTENT_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": []
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);
        let err = client.generate_prompt(&[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_generate_prompt_strips_models_prefix_from_model_id() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path(
                "/v1beta/models/gemini-3-flash-preview:generateContent",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "dream scene" }]
                    }
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", "models/gemini-3-flash-preview");

        client.generate_prompt(&[]).await.unwrap();
    }
}
