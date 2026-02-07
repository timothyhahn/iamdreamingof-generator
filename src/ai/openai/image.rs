use super::client::OpenAiHttpClient;
use crate::ai::ImageGenerationService;
use crate::models::{ImageGenerationRequest, ImageGenerationResponse, Word};
use crate::{prompts, Error, Result};
use async_trait::async_trait;
use std::time::Duration;

pub struct OpenAiImageClient {
    http: OpenAiHttpClient,
    model: String,
}

impl OpenAiImageClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            http: OpenAiHttpClient::new(api_key, Duration::from_secs(60)),
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
impl ImageGenerationService for OpenAiImageClient {
    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>> {
        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        let words_str = word_list.join(", ");

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

        let response: ImageGenerationResponse =
            self.http.post("/v1/images/generations", &request).await?;

        let image_data = response
            .data
            .first()
            .ok_or_else(|| Error::AiProvider("No image data in OpenAI response".to_string()))?;

        let image_bytes = if let Some(b64_json) = &image_data.b64_json {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(b64_json)
                .map_err(|e| Error::Generic(format!("Failed to decode base64 image: {}", e)))?
        } else if let Some(url) = &image_data.url {
            self.http
                .client
                .get(url)
                .send()
                .await?
                .bytes()
                .await?
                .to_vec()
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
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn test_generate_image_handles_b64_response() {
        let server = MockServer::start().await;

        use base64::Engine as _;
        let fake_image = vec![0x89, 0x50, 0x4E, 0x47];
        let b64 = base64::engine::general_purpose::STANDARD.encode(&fake_image);

        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                "data": [{ "b64_json": b64 }]
            })))
            .mount(&server)
            .await;

        let client = OpenAiImageClient::new("key".to_string(), "gpt-image-1".to_string())
            .with_base_url(server.uri());

        let result = client.generate_image("a dream", &[]).await.unwrap();
        assert_eq!(result, fake_image);
    }

    #[tokio::test]
    async fn test_generate_image_api_error() {
        let server = MockServer::start().await;

        Mock::given(method("POST"))
            .and(path("/v1/images/generations"))
            .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
            .mount(&server)
            .await;

        let client = OpenAiImageClient::new("key".to_string(), "gpt-image-1".to_string())
            .with_base_url(server.uri());

        let err = client.generate_image("a dream", &[]).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }
}
