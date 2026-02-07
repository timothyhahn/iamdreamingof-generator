use super::{ChatService, EmbeddingService, ImageGenerationService, ImageQaService};
use crate::models::Word;
use crate::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct MockState<T> {
    responses: Vec<T>,
    call_count: usize,
}

#[derive(Clone, Default)]
struct MockResponses<T> {
    state: Arc<Mutex<MockState<T>>>,
}

impl<T: Clone> MockResponses<T> {
    fn with_response(self, response: T) -> Self {
        self.state.lock().unwrap().responses.push(response);
        self
    }

    fn next(&self) -> Option<T> {
        let mut state = self.state.lock().unwrap();
        state.call_count += 1;
        if state.responses.is_empty() {
            None
        } else {
            let index = (state.call_count - 1) % state.responses.len();
            Some(state.responses[index].clone())
        }
    }

    fn call_count(&self) -> usize {
        self.state.lock().unwrap().call_count
    }
}

#[derive(Clone, Default)]
/// Mock chat client with canned response cycling and call-count tracking.
pub struct MockChatClient {
    state: MockResponses<String>,
}

impl MockChatClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_prompt_response(self, response: String) -> Self {
        Self {
            state: self.state.with_response(response),
        }
    }

    pub fn get_call_count(&self) -> usize {
        self.state.call_count()
    }
}

#[async_trait]
impl ChatService for MockChatClient {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String> {
        if let Some(response) = self.state.next() {
            return Ok(response);
        }

        let word_list: Vec<String> = words.iter().map(|w| w.word.clone()).collect();
        Ok(format!("A dreamlike scene with {}", word_list.join(", ")))
    }
}

#[derive(Clone, Default)]
/// Mock image generation client with canned byte responses and call counting.
pub struct MockImageGenerationClient {
    state: MockResponses<Vec<u8>>,
}

impl MockImageGenerationClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_image_response(self, response: Vec<u8>) -> Self {
        Self {
            state: self.state.with_response(response),
        }
    }

    pub fn get_call_count(&self) -> usize {
        self.state.call_count()
    }
}

#[async_trait]
impl ImageGenerationService for MockImageGenerationClient {
    async fn generate_image(&self, _prompt: &str, _words: &[Word]) -> Result<Vec<u8>> {
        if let Some(response) = self.state.next() {
            return Ok(response);
        }

        // Minimal valid PNG
        Ok(vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00,
            0x00, 0x90, 0x77, 0x53, 0xDE, 0x00, 0x00, 0x00, 0x0C, 0x49, 0x44, 0x41, 0x54, 0x08,
            0x99, 0x63, 0xF8, 0xCF, 0xC0, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0xE2, 0x25, 0x00,
            0xBC, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ])
    }
}

#[derive(Clone, Default)]
/// Mock image QA client that cycles configured boolean detections.
pub struct MockImageQaClient {
    state: MockResponses<bool>,
}

impl MockImageQaClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_text_detection_response(self, has_text: bool) -> Self {
        Self {
            state: self.state.with_response(has_text),
        }
    }

    pub fn get_call_count(&self) -> usize {
        self.state.call_count()
    }
}

#[async_trait]
impl ImageQaService for MockImageQaClient {
    async fn detect_text(&self, _image_bytes: &[u8]) -> Result<bool> {
        Ok(self.state.next().unwrap_or(false))
    }
}

#[derive(Clone, Default)]
/// Mock embedding client with canned vector responses and deterministic fallback.
pub struct MockEmbeddingClient {
    state: MockResponses<Vec<Vec<f32>>>,
}

impl MockEmbeddingClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_embedding_response(self, response: Vec<Vec<f32>>) -> Self {
        // Intentionally does not validate response length against future request input length.
        // Callers that care about count mismatch should assert at the call site.
        Self {
            state: self.state.with_response(response),
        }
    }

    pub fn get_call_count(&self) -> usize {
        self.state.call_count()
    }
}

#[async_trait]
impl EmbeddingService for MockEmbeddingClient {
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if let Some(response) = self.state.next() {
            return Ok(response);
        }

        // Deterministic fallback vectors keep tests reproducible without asserting exact model values.
        Ok(texts
            .iter()
            .map(|text| {
                let len = text.len() as f32;
                vec![len, len + 1.0, len + 2.0]
            })
            .collect())
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
        let embeddings = MockEmbeddingClient::new();

        chat.generate_prompt(&[]).await.unwrap();
        chat.generate_prompt(&[]).await.unwrap();
        image.generate_image("test", &[]).await.unwrap();
        qa.detect_text(&[]).await.unwrap();
        embeddings.embed_texts(&["alpha", "beta"]).await.unwrap();

        assert_eq!(chat.get_call_count(), 2);
        assert_eq!(image.get_call_count(), 1);
        assert_eq!(qa.get_call_count(), 1);
        assert_eq!(embeddings.get_call_count(), 1);
    }

    #[tokio::test]
    async fn test_embedding_default_is_deterministic_shape() {
        let client = MockEmbeddingClient::new();
        let vectors = client.embed_texts(&["cat", "elephant"]).await.unwrap();

        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0].len(), 3);
        assert_eq!(vectors[0], vec![3.0, 4.0, 5.0]);
        assert_eq!(vectors[1], vec![8.0, 9.0, 10.0]);
    }

    #[tokio::test]
    async fn test_embedding_custom_responses_cycle() {
        let client = MockEmbeddingClient::new()
            .with_embedding_response(vec![vec![0.1, 0.2]])
            .with_embedding_response(vec![vec![0.3, 0.4]]);

        assert_eq!(
            client.embed_texts(&["a"]).await.unwrap(),
            vec![vec![0.1, 0.2]]
        );
        assert_eq!(
            client.embed_texts(&["b"]).await.unwrap(),
            vec![vec![0.3, 0.4]]
        );
        assert_eq!(
            client.embed_texts(&["c"]).await.unwrap(),
            vec![vec![0.1, 0.2]]
        );
    }
}
