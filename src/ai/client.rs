use super::AiService;
use crate::models::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ImageGenerationRequest,
    ImageGenerationResponse, Word,
};
use crate::{Error, Result};
use async_trait::async_trait;
use reqwest::Client;
use std::time::Duration;

pub struct AiClient {
    client: Client,
    api_key: String,
}

impl AiClient {
    pub fn new(api_key: String) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30)) // 30 second timeout
            .build()
            .expect("Failed to build HTTP client");

        Self { client, api_key }
    }

    async fn chat_completion(
        &self,
        request: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        tracing::debug!("Sending chat completion request to OpenAI");

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
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
            return Err(Error::OpenAI(format!(
                "API error (status {}): {}",
                status, error_text
            )));
        }

        Ok(response.json().await?)
    }

    async fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse> {
        tracing::debug!("Sending image generation request to OpenAI");

        let response = self
            .client
            .post("https://api.openai.com/v1/images/generations")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                tracing::error!("Failed to send image request to OpenAI: {}", e);
                e
            })?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            tracing::error!("OpenAI API image error (status {}): {}", status, error_text);
            return Err(Error::OpenAI(format!(
                "API error (status {}): {}",
                status, error_text
            )));
        }

        Ok(response.json().await?)
    }
}

#[async_trait]
impl AiService for AiClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        let words_str = word_list.join(", ");

        let system_message = ChatMessage {
            role: "system".to_string(),
            content: Some("You're a specialist in dreams. Answer in under 250 characters. Never mention race, ethnicity, sex, or gender. Use simple English. Each of the three words in the brackets ([]) must be in the final output somewhere.".to_string()),
        };

        let user_message = ChatMessage {
            role: "user".to_string(),
            content: Some(format!(
                "Describe me a dreamlike scene involving [{}]. DO NOT PUT QUOTES AROUND YOUR ANSWER.",
                words_str
            )),
        };

        let request = ChatCompletionRequest {
            model: "gpt-5".to_string(),
            messages: vec![system_message, user_message],
            max_completion_tokens: 3000,
        };

        let response = self.chat_completion(request).await?;

        let prompt = response
            .choices
            .first()
            .and_then(|choice| choice.message.content.clone())
            .ok_or_else(|| Error::OpenAI("No response from API".to_string()))?;

        Ok(prompt)
    }

    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>> {
        // Build list of words that must be visually represented
        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        let words_str = word_list.join(", ");

        let enhanced_prompt = format!(
            "Create a surreal, dreamlike digital artwork based on this scene: {}

            VISUAL REQUIREMENTS:
            - Must include visual representations of: [{}]
            - Each word must be clearly identifiable in the image
            - Style: Ethereal, soft lighting, mystical atmosphere, dreamlike quality
            - Composition: Balanced, visually cohesive
            - Color palette: Otherworldly, harmonious

            STRICT RULES:
            - DO NOT include any text, words, letters, or writing in the image
            - NO TEXT OVERLAYS OR LABELS
            - Visual elements only",
            prompt, words_str
        );

        let request = ImageGenerationRequest {
            model: "gpt-image-1".to_string(),
            prompt: enhanced_prompt,
            n: 1,
            size: "1024x1024".to_string(),
            quality: "medium".to_string(),
        };

        let response = self.image_generation(request).await?;

        // Check if we got base64 or URL
        let image_data = response
            .data
            .first()
            .ok_or_else(|| Error::OpenAI("No image data in response".to_string()))?;

        let image_bytes = if let Some(b64_json) = &image_data.b64_json {
            // Handle base64 response
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(b64_json)
                .map_err(|e| Error::Generic(format!("Failed to decode base64 image: {}", e)))?
        } else if let Some(url) = &image_data.url {
            // Handle URL response
            self.client.get(url).send().await?.bytes().await?.to_vec()
        } else {
            return Err(Error::OpenAI(
                "No image data (neither base64 nor URL) in response".to_string(),
            ));
        };

        Ok(image_bytes)
    }
}
