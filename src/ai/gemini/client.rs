use crate::{Error, Result};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";

pub struct GeminiHttpClient {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) base_url: String,
}

impl GeminiHttpClient {
    pub fn new(api_key: String, model: String, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            api_key,
            model,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub async fn generate_content<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        request: &Req,
    ) -> Result<Resp> {
        let url = format!(
            "{}/v1beta/models/{}:generateContent",
            self.base_url, self.model
        );

        let response = self
            .client
            .post(&url)
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
}
