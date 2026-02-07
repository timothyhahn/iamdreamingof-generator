//! Gemini embedding client implementation.
//!
//! Uses `batchEmbedContents` to embed text inputs in a single provider call.

use super::client::GeminiHttpClient;
use super::types::{Content, Part};
use crate::ai::EmbeddingService;
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Maximum number of inputs supported by Gemini `batchEmbedContents`.
pub const GEMINI_MAX_BATCH_EMBED_ITEMS: usize = 100;

#[derive(Debug, Serialize)]
struct BatchEmbedContentsRequest {
    requests: Vec<EmbedContentRequest>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbedContentRequest {
    model: String,
    content: Content,
}

#[derive(Debug, Deserialize)]
struct BatchEmbedContentsResponse {
    embeddings: Vec<ContentEmbedding>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ContentEmbedding {
    values: Vec<f32>,
}

/// Gemini implementation of [`EmbeddingService`].
pub struct GeminiEmbeddingClient {
    http: GeminiHttpClient,
}

impl GeminiEmbeddingClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_client(api_key, model, reqwest::Client::new())
    }

    pub fn new_with_client(api_key: String, model: String, client: reqwest::Client) -> Self {
        Self {
            http: GeminiHttpClient::new_with_client(
                api_key,
                model,
                Duration::from_secs(60),
                client,
            ),
        }
    }
}

#[cfg(test)]
super::impl_with_gemini_base_url!(GeminiEmbeddingClient);

#[async_trait]
impl EmbeddingService for GeminiEmbeddingClient {
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        if texts.len() > GEMINI_MAX_BATCH_EMBED_ITEMS {
            return Err(Error::Config(format!(
                "Gemini batchEmbedContents allows at most {} texts per request (got {}).",
                GEMINI_MAX_BATCH_EMBED_ITEMS,
                texts.len()
            )));
        }

        // batchEmbedContents expects each request item to carry a fully-qualified model name.
        let model_name = format!("models/{}", self.http.model());
        let request = BatchEmbedContentsRequest {
            requests: texts
                .iter()
                .map(|text| EmbedContentRequest {
                    model: model_name.clone(),
                    content: Content {
                        role: None,
                        parts: vec![Part::Text {
                            text: (*text).to_string(),
                        }],
                    },
                })
                .collect(),
        };

        let response: BatchEmbedContentsResponse = self.http.batch_embed_contents(&request).await?;

        if response.embeddings.len() != texts.len() {
            return Err(Error::AiProvider(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                response.embeddings.len()
            )));
        }

        Ok(response
            .embeddings
            .into_iter()
            .map(|embedding| embedding.values)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::gemini::test_support;
    use wiremock::matchers::{body_string_contains, header};
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gemini-embedding-001";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> GeminiEmbeddingClient {
        GeminiEmbeddingClient::new(api_key.to_string(), model.to_string())
            .with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_embed_texts_parses_response() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::BATCH_EMBED_CONTENTS_PATH_REGEX)
            .and(header("x-goog-api-key", "test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [
                    {"values": [0.5, 0.6]},
                    {"values": [0.7, 0.8]}
                ]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let result = client.embed_texts(&["alpha", "beta"]).await.unwrap();

        assert_eq!(result, vec![vec![0.5, 0.6], vec![0.7, 0.8]]);
    }

    #[tokio::test]
    async fn test_embed_texts_validates_response_count() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::BATCH_EMBED_CONTENTS_PATH_REGEX)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [
                    {"values": [0.5, 0.6]}
                ]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha", "beta"]).await.unwrap_err();

        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_texts_sends_fully_qualified_model_in_requests() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::BATCH_EMBED_CONTENTS_PATH_REGEX)
            .and(body_string_contains(
                "\"model\":\"models/gemini-embedding-001\"",
            ))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "embeddings": [
                    {"values": [0.5, 0.6]}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        client.embed_texts(&["alpha"]).await.unwrap();
    }

    #[tokio::test]
    async fn test_embed_texts_empty_input() {
        let client = GeminiEmbeddingClient::new("test-key".to_string(), DEFAULT_MODEL.to_string());

        let result = client.embed_texts(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_embed_texts_propagates_http_error_status() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::BATCH_EMBED_CONTENTS_PATH_REGEX)
            .respond_with(ResponseTemplate::new(429).set_body_string("quota exceeded"))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha"]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_texts_propagates_malformed_json_error() {
        let server = MockServer::start().await;

        test_support::post_path_regex(test_support::BATCH_EMBED_CONTENTS_PATH_REGEX)
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("{\"embeddings\":[{\"values\":[0.1,0.2]}"),
            )
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha"]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_texts_rejects_batch_sizes_above_provider_limit() {
        let client = GeminiEmbeddingClient::new("test-key".to_string(), DEFAULT_MODEL.to_string());
        let words: Vec<String> = (0..101).map(|idx| format!("word-{idx}")).collect();
        let word_refs: Vec<&str> = words.iter().map(String::as_str).collect();

        let err = client.embed_texts(&word_refs).await.unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }
}
