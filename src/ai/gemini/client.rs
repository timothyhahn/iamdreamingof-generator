use crate::{Error, Result};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

/// Lightweight Gemini REST client used by chat/image/embedding modules.
pub struct GeminiHttpClient {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    model: String,
    pub(crate) base_url: String,
    timeout: Duration,
}

impl GeminiHttpClient {
    /// Construct a Gemini client.
    ///
    /// `model` should be the bare model ID (for example `gemini-embedding-001`),
    /// not a `models/...`-prefixed path segment.
    pub fn new(api_key: String, model: String, timeout: Duration) -> Self {
        Self::new_with_client(api_key, model, timeout, Client::new())
    }

    pub fn new_with_client(
        api_key: String,
        model: String,
        timeout: Duration,
        client: Client,
    ) -> Self {
        let model = model.strip_prefix("models/").unwrap_or(&model).to_string();

        Self {
            client,
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_string(),
            timeout,
        }
    }

    #[cfg(test)]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Returns the configured model ID without the `models/` prefix.
    pub fn model(&self) -> &str {
        &self.model
    }

    async fn post_to_url<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        url: String,
        request: &Req,
    ) -> Result<Resp> {
        let response = self
            .client
            .post(&url)
            .timeout(self.timeout)
            .header("x-goog-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(request)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("Failed to send request to Gemini: {}", e);
                e
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            tracing::error!("Gemini API error (status {}): {}", status, error_text);
            return Err(Error::AiProvider(format!(
                "Gemini API error (status {}): {}",
                status, error_text
            )));
        }

        let body = response.text().await?;
        serde_json::from_str(&body).map_err(|e| {
            tracing::error!("Failed to parse Gemini response: {}\nBody: {}", e, body);
            Error::AiProvider(format!("Failed to parse Gemini response: {}", e))
        })
    }

    /// Calls Gemini's `generateContent` endpoint for chat/image/qa requests.
    pub async fn generate_content<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        request: &Req,
    ) -> Result<Resp> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.model
        );
        self.post_to_url(url, request).await
    }

    /// Calls Gemini's `batchEmbedContents` endpoint for embedding requests.
    pub async fn batch_embed_contents<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        request: &Req,
    ) -> Result<Resp> {
        let url = format!(
            "{}/v1beta/models/{}:batchEmbedContents",
            self.base_url, self.model
        );
        self.post_to_url(url, request).await
    }
}
