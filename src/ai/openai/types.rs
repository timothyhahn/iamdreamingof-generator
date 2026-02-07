//! OpenAI-specific request/response payloads used by provider modules.

use serde::{Deserialize, Serialize};

/// Request body for OpenAI chat completions.
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

/// Structured response-format directive for chat completions.
#[derive(Debug, Serialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String,
    pub json_schema: JsonSchema,
}

/// JSON-schema payload for structured output mode.
#[derive(Debug, Serialize, Clone)]
pub struct JsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

/// OpenAI message content union used across chat and QA.
///
/// Variant order matters for `#[serde(untagged)]` decoding.
#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    ImageContent(Vec<MessagePart>),
}

/// One content segment in multipart message input.
#[derive(Debug, Serialize, Deserialize)]
pub struct MessagePart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

/// Image URL wrapper for OpenAI message payloads.
#[derive(Debug, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

/// Chat message object.
#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
}

/// Top-level chat completion response.
#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
}

/// Single choice item returned by chat completions.
#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

/// Request body for image generation.
#[derive(Debug, Serialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub n: u32,
    pub size: String,
    pub quality: String,
}

/// Top-level image generation response.
#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub data: Vec<ImageData>,
}

/// One generated image item (URL or base64).
#[derive(Debug, Deserialize)]
pub struct ImageData {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}
