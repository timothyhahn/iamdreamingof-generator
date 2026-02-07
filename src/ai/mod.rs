//! AI service integration for prompt, image, and embedding generation
//!
//! Provides trait abstractions for chat completion, image generation, image QA, and embeddings,
//! with implementations for OpenAI and Gemini.

pub mod gemini;
pub mod mime;
pub mod mock;
pub mod openai;

pub use gemini::GeminiEmbeddingClient;
pub use gemini::GEMINI_MAX_BATCH_EMBED_ITEMS;
pub use gemini::{GeminiChatClient, GeminiImageClient, GeminiImageQaClient};
pub use mock::{MockChatClient, MockEmbeddingClient, MockImageGenerationClient, MockImageQaClient};
pub use openai::{OpenAiChatClient, OpenAiEmbeddingClient, OpenAiImageClient, OpenAiImageQaClient};

use crate::models::Word;
use crate::Result;
use async_trait::async_trait;

/// Chat completion: prompt generation from words.
#[async_trait]
pub trait ChatService: Send + Sync {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String>;
}

/// Image generation from a text prompt.
#[async_trait]
pub trait ImageGenerationService: Send + Sync {
    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>>;
}

/// Vision-based QA on generated images (e.g. text detection).
#[async_trait]
pub trait ImageQaService: Send + Sync {
    async fn detect_text(&self, image_bytes: &[u8]) -> Result<bool>;
}

/// Batch text embeddings for semantic similarity.
#[async_trait]
pub trait EmbeddingService: Send + Sync {
    /// Embed a batch of texts in input order.
    ///
    /// Implementations should return `Ok(vec![])` for empty input and must
    /// preserve 1:1 positional alignment with `texts`.
    ///
    /// Implementations may return:
    /// - `Error::Config` for invalid request constraints (for example provider
    ///   batch limits)
    /// - `Error::AiProvider` for HTTP/response-shape failures
    async fn embed_texts(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
}

/// Join challenge words into a comma-separated list for prompt templates.
pub(crate) fn words_to_csv(words: &[Word]) -> String {
    words
        .iter()
        .map(|word| word.word.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}
