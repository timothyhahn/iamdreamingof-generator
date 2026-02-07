use super::client::GeminiHttpClient;
use crate::ai::ChatService;
use crate::models::Word;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize)]
struct GenerateContentRequest {
    system_instruction: Option<Content>,
    contents: Vec<Content>,
    #[serde(rename = "generationConfig")]
    generation_config: Option<GenerationConfig>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<Part>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum Part {
    Text { text: String },
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GenerateContentResponse {
    candidates: Vec<Candidate>,
}

#[derive(Debug, Deserialize)]
struct Candidate {
    content: Content,
}

pub struct GeminiChatClient {
    http: GeminiHttpClient,
}

impl GeminiChatClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: GeminiHttpClient::new(api_key, model, Duration::from_secs(30)),
        }
    }

    #[cfg(test)]
    fn with_base_url(mut self, base_url: String) -> Self {
        self.http = self.http.with_base_url(base_url);
        self
    }

    fn extract_text(response: &GenerateContentResponse) -> Option<String> {
        response.candidates.first().and_then(|c| {
            c.content
                .parts
                .iter()
                .map(|p| match p {
                    Part::Text { text } => text.clone(),
                })
                .next()
        })
    }
}

#[async_trait]
impl ChatService for GeminiChatClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        let words_str = word_list.join(", ");

        let request = GenerateContentRequest {
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
            generation_config: Some(GenerationConfig {
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
    use crate::models::WordType;
    use wiremock::matchers::{method, path_regex};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_generate_prompt_parses_response() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path_regex(r"/v1beta/models/.+:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "candidates": [{
                    "content": {
                        "parts": [{ "text": "Clouds of honey drift past a floating cat" }]
                    }
                }]
            })))
            .mount(&server)
            .await;

        let client =
            GeminiChatClient::new("test-key".to_string(), "gemini-3-flash-preview".to_string())
                .with_base_url(server.uri());

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

        Mock::given(method("POST"))
            .and(path_regex(r"/v1beta/models/.+:generateContent"))
            .respond_with(ResponseTemplate::new(403).set_body_string("forbidden"))
            .mount(&server)
            .await;

        let client =
            GeminiChatClient::new("bad-key".to_string(), "gemini-3-flash-preview".to_string())
                .with_base_url(server.uri());

        let err = client.generate_prompt(&[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
