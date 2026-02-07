use super::client::OpenAiHttpClient;
use super::types::{ImageGenerationRequest, ImageGenerationResponse};
use crate::ai::{words_to_csv, ImageGenerationService};
use crate::models::Word;
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use std::time::Duration;

pub struct OpenAiImageClient {
    http: OpenAiHttpClient,
    model: String,
}

impl OpenAiImageClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self::new_with_client(api_key, model, reqwest::Client::new())
    }

    pub fn new_with_client(api_key: String, model: String, client: reqwest::Client) -> Self {
        Self {
            http: OpenAiHttpClient::new_with_client(api_key, Duration::from_secs(60), client),
            model,
        }
    }

    async fn fetch_image_url(&self, url: &str) -> Result<Vec<u8>> {
        let response = self
            .http
            .client
            .get(url)
            .timeout(self.http.timeout())
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_else(|_| "".to_string());
            return Err(Error::AiProvider(format!(
                "OpenAI image download error (status {}): {}",
                status, error_text
            )));
        }

        Ok(response.bytes().await?.to_vec())
    }
}

#[cfg(test)]
super::impl_with_openai_base_url!(OpenAiImageClient);

#[async_trait]
impl ImageGenerationService for OpenAiImageClient {
    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>> {
        let words_str = words_to_csv(words);

        let enhanced_prompt = prompts::render(
            prompts::IMAGE_ENHANCEMENT,
            &[("prompt", prompt), ("words", &words_str)],
        );

        let request = ImageGenerationRequest {
            model: self.model.clone(),
            prompt: enhanced_prompt,
            n: 1,
            size: "1024x1024".to_string(),
            quality: "medium".to_string(),
        };

        let response: ImageGenerationResponse = self.http.image_generation(&request).await?;

        let image_data = response
            .data
            .first()
            .ok_or_else(|| Error::AiProvider("No image data in OpenAI response".to_string()))?;

        let image_bytes = if let Some(b64_json) = &image_data.b64_json {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(b64_json)
                .map_err(|e| {
                    Error::AiProvider(format!("Failed to decode OpenAI base64 image: {}", e))
                })?
        } else if let Some(url) = &image_data.url {
            self.fetch_image_url(url).await?
        } else {
            return Err(Error::AiProvider(
                "No image data (neither base64 nor URL) in response".to_string(),
            ));
        };

        Ok(image_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::openai::test_support;
    use wiremock::matchers::{method, path};
    use wiremock::Mock;
    use wiremock::{MockServer, ResponseTemplate};

    const DEFAULT_MODEL: &str = "gpt-image-1";

    fn make_client(server: &MockServer, api_key: &str, model: &str) -> OpenAiImageClient {
        OpenAiImageClient::new(api_key.to_string(), model.to_string()).with_base_url(server.uri())
    }

    #[tokio::test]
    async fn test_generate_image_handles_b64_response() {
        let server = MockServer::start().await;

        use base64::Engine as _;
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&fake_image);

        test_support::post(test_support::IMAGES_GENERATIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "b64_json": b64 }]
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        let result = client.generate_image("a dream", &[]).await.unwrap();
        assert_eq!(result, fake_image);
    }

    #[tokio::test]
    async fn test_generate_image_api_error() {
        let server = MockServer::start().await;

        test_support::post(test_support::IMAGES_GENERATIONS_PATH)
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);

        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_generate_image_handles_url_response() {
        let server = MockServer::start().await;
        let url = format!("{}/image.png", server.uri());
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];

        test_support::post(test_support::IMAGES_GENERATIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "url": url }]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/image.png"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(fake_image.clone()))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);
        let result = client.generate_image("a dream", &[]).await.unwrap();
        assert_eq!(result, fake_image);
    }

    #[tokio::test]
    async fn test_generate_image_url_download_status_error_is_ai_provider_error() {
        let server = MockServer::start().await;
        let url = format!("{}/missing.png", server.uri());

        test_support::post(test_support::IMAGES_GENERATIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "url": url }]
            })))
            .mount(&server)
            .await;

        Mock::given(method("GET"))
            .and(path("/missing.png"))
            .respond_with(ResponseTemplate::new(404).set_body_string("not found"))
            .expect(1)
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);
        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        match err {
            Error::AiProvider(message) => {
                assert!(message.contains("status 404"));
                assert!(message.contains("not found"));
            }
            other => panic!("Expected Error::AiProvider, got {}", other),
        }
    }

    #[tokio::test]
    async fn test_generate_image_rejects_empty_data() {
        let server = MockServer::start().await;

        test_support::post(test_support::IMAGES_GENERATIONS_PATH)
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": []
            })))
            .mount(&server)
            .await;

        let client = make_client(&server, "key", DEFAULT_MODEL);
        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
