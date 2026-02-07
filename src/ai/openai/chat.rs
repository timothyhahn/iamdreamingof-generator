use super::client::OpenAiHttpClient;
use crate::ai::ChatService;
use crate::models::{ChatCompletionRequest, ChatMessage, ChatMessageContent, Word};
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use std::time::Duration;

pub struct OpenAiChatClient {
    http: OpenAiHttpClient,
    model: String,
}

impl OpenAiChatClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: OpenAiHttpClient::new(api_key, Duration::from_secs(30)),
            model,
        }
    }

    #[cfg(test)]
    fn with_base_url(mut self, base_url: String) -> Self {
        self.http = self.http.with_base_url(base_url);
        self
    }
}

#[async_trait]
impl ChatService for OpenAiChatClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        let words_str = word_list.join(", ");

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

        let response = self.http.chat_completion(request).await?;

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
    use crate::models::WordType;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_generate_prompt_parses_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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

        let client = OpenAiChatClient::new("test-key".to_string(), "gpt-5".to_string())
            .with_base_url(server.uri());

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

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
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

        let client = OpenAiChatClient::new("key".to_string(), "custom-model".to_string())
            .with_base_url(server.uri());

        client.generate_prompt(&[]).await.unwrap();
    }

    #[tokio::test]
    async fn test_api_error_returns_ai_provider_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limited"))
            .mount(&server)
            .await;

        let client = OpenAiChatClient::new("key".to_string(), "gpt-5".to_string())
            .with_base_url(server.uri());

        let err = client.generate_prompt(&[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
