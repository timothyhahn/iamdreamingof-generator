//! AI service integration for prompt and image generation
//!
//! Provides trait abstractions for chat completion, image generation, and image QA,
//! with implementations for OpenAI and Gemini.

pub mod gemini;
pub mod mime;
pub mod mock;
pub mod openai;

pub use gemini::{GeminiChatClient, GeminiImageClient, GeminiImageQaClient};
pub use mock::{MockChatClient, MockImageGenerationClient, MockImageQaClient};
pub use openai::{OpenAiChatClient, OpenAiImageClient, OpenAiImageQaClient};

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
