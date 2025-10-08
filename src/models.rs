//! Data models and structures
//!
//! Defines the core data structures for challenges, words, and API
//! interactions with OpenAI and CDN services.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WordType {
    Object,
    Gerund,
    Concept,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    #[serde(rename = "type")]
    pub word_type: WordType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    pub words: Vec<Word>,
    pub image_path: String,
    pub image_url_jpg: String,
    pub image_url_webp: String,
    pub prompt: String,
}

impl Challenge {
    pub fn new(
        words: Vec<Word>,
        image_path: String,
        image_url_jpg: String,
        image_url_webp: String,
        prompt: String,
    ) -> Self {
        Self {
            words,
            image_path,
            image_url_jpg,
            image_url_webp,
            prompt,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenges {
    pub easy: Challenge,
    pub medium: Challenge,
    pub hard: Challenge,
    pub dreaming: Challenge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Day {
    pub date: String, // Format: YYYY-MM-DD
    pub id: i32,
    pub challenges: Challenges,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateEntry {
    pub date: String,
    pub id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Days {
    pub days: Vec<DateEntry>,
}

impl Default for Days {
    fn default() -> Self {
        Self::new()
    }
}

impl Days {
    pub fn new() -> Self {
        Self { days: Vec::new() }
    }

    pub fn add_day(&mut self, date: String, id: i32) {
        self.days.push(DateEntry { date, id });
    }

    pub fn find_by_date(&self, date: &str) -> Option<&DateEntry> {
        self.days.iter().find(|d| d.date == date)
    }

    pub fn max_id(&self) -> Option<i32> {
        self.days.iter().map(|d| d.id).max()
    }
}

// OpenAI API Request/Response models
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionResponse {
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub n: u32,
    pub size: String,
    pub quality: String,
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationResponse {
    pub data: Vec<ImageData>,
}

#[derive(Debug, Deserialize)]
pub struct ImageData {
    pub url: Option<String>,
    pub b64_json: Option<String>,
}

// Configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub openai_api_key: String,
    pub cdn_access_key_id: String,
    pub cdn_secret_access_key: String,
    pub cdn_endpoint: String,
    pub cdn_bucket: String,
    pub cdn_base_url: String,
}

impl Config {
    pub fn from_env() -> crate::Result<Self> {
        dotenvy::dotenv().ok();

        Ok(Self {
            openai_api_key: std::env::var("AI_API_KEY")
                .map_err(|_| crate::Error::Generic("AI_API_KEY not set".to_string()))?,
            cdn_access_key_id: std::env::var("CDN_ACCESS_KEY_ID")
                .map_err(|_| crate::Error::Generic("CDN_ACCESS_KEY_ID not set".to_string()))?,
            cdn_secret_access_key: std::env::var("CDN_SECRET_ACCESS_KEY")
                .map_err(|_| crate::Error::Generic("CDN_SECRET_ACCESS_KEY not set".to_string()))?,
            cdn_endpoint: std::env::var("CDN_ENDPOINT")
                .unwrap_or_else(|_| "https://nyc3.digitaloceanspaces.com".to_string()),
            cdn_bucket: std::env::var("CDN_BUCKET").unwrap_or_else(|_| "iamdreamingof".to_string()),
            cdn_base_url: std::env::var("CDN_BASE_URL")
                .unwrap_or_else(|_| "https://cdn.iamdreamingof.com".to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_serialization() {
        let word = Word {
            word: "apple".to_string(),
            word_type: WordType::Object,
        };

        let json = serde_json::to_string(&word).unwrap();
        assert!(json.contains("\"type\":\"object\""));

        let deserialized: Word = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.word, "apple");
        assert_eq!(deserialized.word_type, WordType::Object);
    }

    #[test]
    fn test_days_operations() {
        let mut days = Days::new();
        days.add_day("2024-01-01".to_string(), 1);
        days.add_day("2024-01-02".to_string(), 2);

        assert_eq!(days.max_id(), Some(2));
        assert!(days.find_by_date("2024-01-01").is_some());
        assert!(days.find_by_date("2024-01-03").is_none());
    }
}
