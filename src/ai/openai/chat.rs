use super::client::OpenAiHttpClient;
use super::types::{ChatCompletionRequest, ChatMessage, ChatMessageContent};
use crate::ai::{words_to_csv, ChatService};
use crate::models::Word;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use std::time::Duration;

pub struct OpenAiChatClient {
    http: OpenAiHttpClient,
    model: String,
}

impl OpenAiChatClient {
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
super::impl_with_openai_base_url!(OpenAiChatClient);

#[async_trait]
impl ChatService for OpenAiChatClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let words_str = words_to_csv(words);

        let system_message = ChatMessage {
            role: "system".to_string(),
            content: Some(ChatMessageContent::Text(prompts::CHAT_SYSTEM.to_string())),
        };

        let user_message = ChatMessage {
            role: "user".to_string(),
            content: Some(ChatMessageContent::Text(prompts::render(
                prompts::CHAT_USER,
                &[("words", &words_str)],
            ))),
        };

        let request = ChatCompletionRequest {
            model: self.model.clone(),
            messages: vec![system_message, user_message],
            max_completion_tokens: 3000,
            response_format: None,
        };

        let response = self.http.chat_completion(&request).await?;

        response
            .choices
            .first()
            .and_then(|choice| match &choice.message.content {
                Some(ChatMessageContent::Text(text)) => Some(text.clone()),
                _ => None,
            })
            .ok_or_else(|| Error::AiProvider("No response from OpenAI chat API".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::openai::test_support;
    use crate::models::WordType;
    use wiremock::matchers::header;
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gpt-5";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> OpenAiChatClient {
        OpenAiChatClient::new(api_key.to_string(), model.to_string()).with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_generate_prompt_parses_response() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "A floating apple drifts through clouds"
                    },
                    "finish_reason": "stop"
                }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let words = vec![
            Word {
                word: "apple".to_string(),
                word_type: WordType::Object,
            },
            Word {
                word: "running".to_string(),
                word_type: WordType::Gerund,
            },
        ];

        let prompt = client.generate_prompt(&words).await.unwrap();
        assert_eq!(prompt, "A floating apple drifts through clouds");
    }

    #[tokio::test]
    async fn test_generate_prompt_sends_configured_model() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .and(wiremock::matchers::body_string_contains(
                "\"model\":\"custom-model\"",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": [{
                    "message": { "role": "assistant", "content": "dream scene" },
                    "finish_reason": "stop"
                }]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "key", "custom-model");

        client.generate_prompt(&[]).await.unwrap();
    }

    #[tokio::test]
    async fn test_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        let err = client.generate_prompt(&[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_generate_prompt_rejects_empty_choices() {
        let server = MockServer::start().await;

        test_support::post(test_support::CHAT_COMPLETIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "choices": []
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);
        let err = client.generate_prompt(&[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
