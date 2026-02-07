use crate::models::{ChatCompletionRequest, ChatCompletionResponse};
use crate::{Error, Result};
use reqwest::Client;
use serde::de::DeserializeOwned;
use serde::Serialize;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://api.openai.com";

pub struct OpenAiHttpClient {
    pub(crate) client: Client,
    pub(crate) api_key: String,
    pub(crate) base_url: String,
}

impl OpenAiHttpClient {
    pub fn new(api_key: String, timeout: Duration) -> Self {
        let client = Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to build HTTP client");

        Self {
            client,
            api_key,
            base_url: DEFAULT_BASE_URL.to_string(),
        }
    }

    #[cfg(test)]
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub async fn post<Req: Serialize, Resp: DeserializeOwned>(
        &self,
        path: &str,
        request: &Req,
    ) -> Result<Resp> {
        let url = format!("{}{}", self.base_url, path);
        let response = self
            .client
            .post(&url)
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

    pub async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        self.post("/v1/chat/completions", &request).await
    }
}
