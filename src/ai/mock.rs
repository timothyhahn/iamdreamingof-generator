use super::AiService;
use crate::models::Word;
use crate::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

pub struct MockAiClient {
    prompt_responses: Arc<Mutex<Vec<String>>>,
    image_responses: Arc<Mutex<Vec<Vec<u8>>>>,
    text_detection_responses: Arc<Mutex<Vec<bool>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockAiClient {
    pub fn new() -> Self {
        Self {
            prompt_responses: Arc::new(Mutex::new(Vec::new())),
            image_responses: Arc::new(Mutex::new(Vec::new())),
            text_detection_responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_prompt_response(self, response: String) -> Self {
        self.prompt_responses.lock().unwrap().push(response);
        self
    }

    pub fn with_image_response(self, response: Vec<u8>) -> Self {
        self.image_responses.lock().unwrap().push(response);
        self
    }

    pub fn with_text_detection_response(self, has_text: bool) -> Self {
        self.text_detection_responses.lock().unwrap().push(has_text);
        self
    }

    pub fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl Default for MockAiClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AiService for MockAiClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let responses = self.prompt_responses.lock().unwrap();
        if responses.is_empty() {
            // Default mock response
            let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
            Ok(format!("A dreamlike scene with {}", word_list.join(", ")))
        } else {
            let index = (*count - 1) % responses.len();
            Ok(responses[index].clone())
        }
    }

    async fn generate_image(&self, _prompt: &str, _words: &[Word]) -> Result<Vec<u8>> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let responses = self.image_responses.lock().unwrap();
        if responses.is_empty() {
            // Return a tiny valid PNG as default
            Ok(vec![
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG signature
                0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52, // IHDR chunk
                0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // 1x1 pixel
                0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49,
                0x44, 0x41, // IDAT chunk
                0x54, 0x08, 0x99, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0xE2,
                0x25, 0x00, 0xBC, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, // IEND chunk
                0x44, 0xAE, 0x42, 0x60, 0x82,
            ])
        } else {
            let index = (*count - 1) % responses.len();
            Ok(responses[index].clone())
        }
    }

    async fn detect_text(&self, _image_bytes: &[u8]) -> Result<bool> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let responses = self.text_detection_responses.lock().unwrap();
        if responses.is_empty() {
            // Default to no text detected
            Ok(false)
        } else {
            let index = (*count - 1) % responses.len();
            Ok(responses[index])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::WordType;

    #[tokio::test]
    async fn test_mock_ai_client_default_prompt() {
        let client = MockAiClient::new();
        let words = vec![
            Word {
                word: "apple".to_string(),
                word_type: WordType::Object,
            },
            Word {
                word: "running".to_string(),
                word_type: WordType::Gerund,
            },
        ];

        let prompt = client.generate_prompt(&words).await.unwrap();
        assert!(prompt.contains("apple"));
        assert!(prompt.contains("running"));
    }

    #[tokio::test]
    async fn test_mock_ai_client_custom_responses() {
        let client = MockAiClient::new()
            .with_prompt_response("Custom dream scene 1".to_string())
            .with_prompt_response("Custom dream scene 2".to_string());

        let words = vec![];
        let prompt1 = client.generate_prompt(&words).await.unwrap();
        assert_eq!(prompt1, "Custom dream scene 1");

        let prompt2 = client.generate_prompt(&words).await.unwrap();
        assert_eq!(prompt2, "Custom dream scene 2");

        // Should cycle back
        let prompt3 = client.generate_prompt(&words).await.unwrap();
        assert_eq!(prompt3, "Custom dream scene 1");
    }

    #[tokio::test]
    async fn test_mock_ai_client_call_count() {
        let client = MockAiClient::new();

        assert_eq!(client.get_call_count(), 0);

        client.generate_prompt(&[]).await.unwrap();
        assert_eq!(client.get_call_count(), 1);

        client.generate_image("test", &[]).await.unwrap();
        assert_eq!(client.get_call_count(), 2);
    }

    #[tokio::test]
    async fn test_mock_ai_client_text_detection_default() {
        let client = MockAiClient::new();

        // Default should be no text detected
        let has_text = client.detect_text(&[]).await.unwrap();
        assert!(!has_text);
    }

    #[tokio::test]
    async fn test_mock_ai_client_text_detection_custom() {
        let client = MockAiClient::new()
            .with_text_detection_response(true)
            .with_text_detection_response(false)
            .with_text_detection_response(true);

        // Should return configured responses in order
        assert!(client.detect_text(&[]).await.unwrap());
        assert!(!client.detect_text(&[]).await.unwrap());
        assert!(client.detect_text(&[]).await.unwrap());

        // Should cycle back
        assert!(client.detect_text(&[]).await.unwrap());
    }
}
