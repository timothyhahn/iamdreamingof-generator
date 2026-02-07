//! Shared Gemini payload types used across chat, image, and QA modules.

use serde::{Deserialize, Serialize};

/// Gemini content container used in both requests and responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    pub parts: Vec<Part>,
}

/// Untagged union of text and inline media content parts.
///
/// Variant order matters for `#[serde(untagged)]` decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: InlineData,
    },
}

/// Base64 inline payload used for image/vision requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InlineData {
    pub mime_type: String,
    pub data: String,
}

/// Top-level `generateContent` response envelope.
#[derive(Debug, Deserialize)]
pub struct GenerateContentResponse {
    pub candidates: Vec<Candidate>,
}

/// Candidate completion item returned by Gemini.
#[derive(Debug, Deserialize)]
pub struct Candidate {
    pub content: Content,
}
