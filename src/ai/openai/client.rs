use super::types::{ChatCompletionRequest, ChatCompletionResponse};
use crate::{Error, Result};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

/// Lightweight OpenAI REST client used by chat/image/embedding modules.
pub struct OpenAiHttpClient {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
    timeout: Duration,
}

impl OpenAiHttpClient {
    pub fn new(api_key: String, timeout: Duration) -> Self {
        Self::new_with_client(api_key, timeout, Client::new())
    }

    pub fn new_with_client(api_key: String, timeout: Duration, client: Client) -> Self {
        Self {
            client,
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout,
        }
    }

    #[cfg(test)]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub(crate) fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Issue a POST request against the OpenAI REST API and deserialize JSON.
    pub async fn post<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        request: &Req,
    ) -> Result<Resp> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .client
            .post(&url)
            .timeout(self.timeout)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(request)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("Failed to send request to OpenAI: {}", e);
                e
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            tracing::error!("OpenAI API error (status {}): {}", status, error_text);
            return Err(Error::AiProvider(format!(
                "OpenAI API error (status {}): {}",
                status, error_text
            )));
        }

        let body = response.text().await?;
        serde_json::from_str(&body).map_err(|e| {
            tracing::error!("Failed to parse OpenAI response: {}\nBody: {}", e, body);
            Error::AiProvider(format!("Failed to parse OpenAI response: {}", e))
        })
    }

    /// Convenience wrapper for `/v1/chat/completions`.
    pub async fn chat_completion(
        &self,
        request: &ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        self.post("/v1/chat/completions", request).await
    }

    /// Convenience wrapper for `/v1/embeddings`.
    pub async fn embeddings<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        request: &Req,
    ) -> Result<Resp> {
        self.post("/v1/embeddings", request).await
    }

    /// Convenience wrapper for `/v1/images/generations`.
    pub async fn image_generation<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        request: &Req,
    ) -> Result<Resp> {
        self.post("/v1/images/generations", request).await
    }
}
