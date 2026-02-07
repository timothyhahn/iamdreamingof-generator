use super::{ChatService, ImageGenerationService, ImageQaService};
use crate::models::Word;
use crate::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

pub struct MockChatClient {
    prompt_responses: Arc<Mutex<Vec<String>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockChatClient {
    pub fn new() -> Self {
        Self {
            prompt_responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_prompt_response(self, response: String) -> Self {
        self.prompt_responses.lock().unwrap().push(response);
        self
    }

    pub fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl Default for MockChatClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChatService for MockChatClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let responses = self.prompt_responses.lock().unwrap();
        if responses.is_empty() {
            let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
            Ok(format!("A dreamlike scene with {}", word_list.join(", ")))
        } else {
            let index = (*count - 1) % responses.len();
            Ok(responses[index].clone())
        }
    }
}

pub struct MockImageGenerationClient {
    image_responses: Arc<Mutex<Vec<Vec<u8>>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockImageGenerationClient {
    pub fn new() -> Self {
        Self {
            image_responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_image_response(self, response: Vec<u8>) -> Self {
        self.image_responses.lock().unwrap().push(response);
        self
    }

    pub fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl Default for MockImageGenerationClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageGenerationService for MockImageGenerationClient {
    async fn generate_image(&self, _prompt: &str, _words: &[Word]) -> Result<Vec<u8>> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let responses = self.image_responses.lock().unwrap();
        if responses.is_empty() {
            // Minimal valid PNG
            Ok(vec![
                0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
                0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
                0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
                0x99, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0xE2, 0x25, 0x00,
                0xBC, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
            ])
        } else {
            let index = (*count - 1) % responses.len();
            Ok(responses[index].clone())
        }
    }
}

pub struct MockImageQaClient {
    text_detection_responses: Arc<Mutex<Vec<bool>>>,
    call_count: Arc<Mutex<usize>>,
}

impl MockImageQaClient {
    pub fn new() -> Self {
        Self {
            text_detection_responses: Arc::new(Mutex::new(Vec::new())),
            call_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_text_detection_response(self, has_text: bool) -> Self {
        self.text_detection_responses.lock().unwrap().push(has_text);
        self
    }

    pub fn get_call_count(&self) -> usize {
        *self.call_count.lock().unwrap()
    }
}

impl Default for MockImageQaClient {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ImageQaService for MockImageQaClient {
    async fn detect_text(&self, _image_bytes: &[u8]) -> Result<bool> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        let responses = self.text_detection_responses.lock().unwrap();
        if responses.is_empty() {
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
    async fn test_chat_default_includes_words() {
        let client = MockChatClient::new();
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
    async fn test_chat_custom_responses_cycle() {
        let client = MockChatClient::new()
            .with_prompt_response("first".to_string())
            .with_prompt_response("second".to_string());

        assert_eq!(client.generate_prompt(&[]).await.unwrap(), "first");
        assert_eq!(client.generate_prompt(&[]).await.unwrap(), "second");
        assert_eq!(client.generate_prompt(&[]).await.unwrap(), "first");
    }

    #[tokio::test]
    async fn test_chat_call_count() {
        let client = MockChatClient::new();
        assert_eq!(client.get_call_count(), 0);
        client.generate_prompt(&[]).await.unwrap();
        assert_eq!(client.get_call_count(), 1);
        client.generate_prompt(&[]).await.unwrap();
        assert_eq!(client.get_call_count(), 2);
    }

    #[tokio::test]
    async fn test_image_gen_default_returns_png() {
        let client = MockImageGenerationClient::new();
        let data = client.generate_image("test", &[]).await.unwrap();
        assert_eq!(&data[..4], &[0x89, 0x50, 0x4E, 0x47]);
    }

    #[tokio::test]
    async fn test_image_gen_call_count() {
        let client = MockImageGenerationClient::new();
        assert_eq!(client.get_call_count(), 0);
        client.generate_image("test", &[]).await.unwrap();
        assert_eq!(client.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_qa_default_no_text() {
        let client = MockImageQaClient::new();
        assert!(!client.detect_text(&[]).await.unwrap());
    }

    #[tokio::test]
    async fn test_qa_custom_responses_cycle() {
        let client = MockImageQaClient::new()
            .with_text_detection_response(true)
            .with_text_detection_response(false);

        assert!(client.detect_text(&[]).await.unwrap());
        assert!(!client.detect_text(&[]).await.unwrap());
        assert!(client.detect_text(&[]).await.unwrap());
    }

    #[tokio::test]
    async fn test_counters_are_independent() {
        let chat = MockChatClient::new();
        let image = MockImageGenerationClient::new();
        let qa = MockImageQaClient::new();

        chat.generate_prompt(&[]).await.unwrap();
        chat.generate_prompt(&[]).await.unwrap();
        image.generate_image("test", &[]).await.unwrap();
        qa.detect_text(&[]).await.unwrap();

        assert_eq!(chat.get_call_count(), 2);
        assert_eq!(image.get_call_count(), 1);
        assert_eq!(qa.get_call_count(), 1);
    }
}
