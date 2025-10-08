//! AI service integration for prompt and image generation
//!
//! Provides interfaces to OpenAI's Completions and Image APIs for generating
//! dream descriptions and corresponding images.

pub mod client;
pub mod mock;

pub use client::AiClient;
pub use mock::MockAiClient;

use crate::models::Word;
use crate::Result;
use async_trait::async_trait;

#[async_trait]
pub trait AiService: Send + Sync {
    async fn generate_prompt(&self, words: &[Word]) -> Result<String>;
    async fn generate_image(&self, prompt: &str, words: &[Word]) -> Result<Vec<u8>>;
}
