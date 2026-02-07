//! OpenAI embedding client implementation.
//!
//! Uses `/v1/embeddings` to batch-embed text inputs for semantic similarity.

use super::client::OpenAiHttpClient;
use crate::ai::EmbeddingService;
use crate::{Error, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Serialize)]
struct EmbeddingsRequest {
    model: String,
    input: Vec<String>,
    encoding_format: String,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

/// OpenAI implementation of [`EmbeddingService`].
pub struct OpenAiEmbeddingClient {
    http: OpenAiHttpClient,
    model: String,
}

impl OpenAiEmbeddingClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_client(api_key, model, reqwest::Client::new())
    }

    pub fn new_with_client(api_key: String, model: String, client: reqwest::Client) -> Self {
        Self {
            http: OpenAiHttpClient::new_with_client(api_key, Duration::from_secs(60), client),
            model,
        }
    }
}

#[cfg(test)]
super::impl_with_openai_base_url!(OpenAiEmbeddingClient);

#[async_trait]
impl EmbeddingService for OpenAiEmbeddingClient {
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let request = EmbeddingsRequest {
            model: self.model.clone(),
            input: texts.iter().map(|text| (*text).to_string()).collect(),
            // Request raw floats instead of base64 payloads so the response
            // can be deserialized directly into `Vec<f32>`.
            encoding_format: "float".to_string(),
        };

        let response: EmbeddingsResponse = self.http.embeddings(&request).await?;

        let mut indexed = response.data;
        // OpenAI returns an explicit `index` per embedding item. We sort defensively so
        // callers always get vectors aligned to the original input order.
        indexed.sort_by_key(|item| item.index);

        if indexed.len() != texts.len() {
            return Err(Error::AiProvider(format!(
                "Expected {} embeddings, got {}",
                texts.len(),
                indexed.len()
            )));
        }

        if indexed
            .iter()
            .enumerate()
            .any(|(expected_idx, item)| item.index != expected_idx)
        {
            return Err(Error::AiProvider(
                "Embedding indices were non-contiguous or out of range".to_string(),
            ));
        }

        Ok(indexed.into_iter().map(|item| item.embedding).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::openai::test_support;
    use wiremock::matchers::{body_string_contains, header};
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "text-embedding-3-small";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> OpenAiEmbeddingClient {
        OpenAiEmbeddingClient::new(api_key.to_string(), model.to_string())
            .with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_embed_texts_parses_response() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .and(header("Authorization", "Bearer test-key"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 1, "embedding": [0.3, 0.4]}
                ]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let result = client.embed_texts(&["alpha", "beta"]).await.unwrap();

        assert_eq!(result, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[tokio::test]
    async fn test_embed_texts_reorders_by_index() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]}
                ]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let result = client.embed_texts(&["alpha", "beta"]).await.unwrap();

        assert_eq!(result, vec![vec![0.1, 0.2], vec![0.3, 0.4]]);
    }

    #[tokio::test]
    async fn test_embed_texts_validates_response_count() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]}
                ]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha", "beta"]).await.unwrap_err();

        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_texts_rejects_non_contiguous_indices() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]},
                    {"index": 99, "embedding": [0.3, 0.4]}
                ]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha", "beta"]).await.unwrap_err();

        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_texts_sends_configured_model() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .and(body_string_contains("\"model\":\"custom-embedding-model\""))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2]}
                ]
            })))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", "custom-embedding-model");

        client.embed_texts(&["alpha"]).await.unwrap();
    }

    #[tokio::test]
    async fn test_embed_texts_empty_input() {
        let client = OpenAiEmbeddingClient::new("test-key".to_string(), DEFAULT_MODEL.to_string());

        let result = client.embed_texts(&[]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_embed_texts_propagates_http_error_status() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .respond_with(ResponseTemplate::new(429).set_body_string("rate limit"))
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha"]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_texts_propagates_malformed_json_error() {
        let server = MockServer::start().await;

        test_support::post(test_support::EMBEDDINGS_PATH)
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("{\"data\":[{\"index\":0,\"embedding\": [1.0]}"),
            )
            .mount(&server)
            .await;

        let client = make_client(&server, "test-key", DEFAULT_MODEL);

        let err = client.embed_texts(&["alpha"]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
