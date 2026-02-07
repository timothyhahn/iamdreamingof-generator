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

// Provider enum
#[derive(Debug, Clone, PartialEq)]
pub enum AiProvider {
    OpenAi,
    Gemini,
}

impl std::fmt::Display for AiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AiProvider::OpenAi => write!(f, "openai"),
            AiProvider::Gemini => write!(f, "gemini"),
        }
    }
}

impl std::str::FromStr for AiProvider {
    type Err = crate::Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(AiProvider::OpenAi),
            "gemini" => Ok(AiProvider::Gemini),
            other => Err(crate::Error::Config(format!(
                "Unknown AI provider '{}'. Must be 'openai' or 'gemini'",
                other
            ))),
        }
    }
}

// OpenAI-format API request/response models
#[derive(Debug, Serialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_completion_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,
}

#[derive(Debug, Serialize, Clone)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub format_type: String, // "json_schema" for structured output
    pub json_schema: JsonSchema,
}

#[derive(Debug, Serialize, Clone)]
pub struct JsonSchema {
    pub name: String,
    pub schema: serde_json::Value,
    pub strict: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    ImageContent(Vec<MessagePart>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessagePart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_url: Option<ImageUrl>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatMessageContent>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextDetectionResponse {
    pub includes_text: bool,
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
    pub openai_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
    pub chat_provider: AiProvider,
    pub image_provider: AiProvider,
    pub qa_provider: AiProvider,
    pub chat_model: String,
    pub image_model: String,
    pub qa_model: String,
    pub cdn_access_key_id: Option<String>,
    pub cdn_secret_access_key: Option<String>,
    pub cdn_endpoint: String,
    pub cdn_bucket: String,
    pub cdn_base_url: String,
    pub dry_run: bool,
}

fn required_env(name: &str) -> crate::Result<String> {
    std::env::var(name)
        .map_err(|_| crate::Error::Config(format!("{} environment variable is required", name)))
}

impl Config {
    pub fn from_env() -> crate::Result<Self> {
        dotenvy::dotenv().ok();

        let chat_provider: AiProvider = required_env("CHAT_PROVIDER")?.parse()?;
        let image_provider: AiProvider = required_env("IMAGE_PROVIDER")?.parse()?;
        let qa_provider: AiProvider = required_env("QA_PROVIDER")?.parse()?;

        let chat_model = required_env("CHAT_MODEL")?;
        let image_model = required_env("IMAGE_MODEL")?;
        let qa_model = required_env("QA_MODEL")?;

        let dry_run = std::env::var("DRY_RUN")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(false);

        let openai_api_key = std::env::var("OPENAI_API_KEY").ok();
        let gemini_api_key = std::env::var("GEMINI_API_KEY").ok();

        let needs_openai = chat_provider == AiProvider::OpenAi
            || image_provider == AiProvider::OpenAi
            || qa_provider == AiProvider::OpenAi;
        let needs_gemini = chat_provider == AiProvider::Gemini
            || image_provider == AiProvider::Gemini
            || qa_provider == AiProvider::Gemini;

        if needs_openai && openai_api_key.is_none() {
            return Err(crate::Error::Config(
                "OPENAI_API_KEY is required when using OpenAI as a provider".to_string(),
            ));
        }
        if needs_gemini && gemini_api_key.is_none() {
            return Err(crate::Error::Config(
                "GEMINI_API_KEY is required when using Gemini as a provider".to_string(),
            ));
        }

        let cdn_access_key_id = std::env::var("CDN_ACCESS_KEY_ID").ok();
        let cdn_secret_access_key = std::env::var("CDN_SECRET_ACCESS_KEY").ok();

        if !dry_run && (cdn_access_key_id.is_none() || cdn_secret_access_key.is_none()) {
            return Err(crate::Error::Config(
                "CDN_ACCESS_KEY_ID and CDN_SECRET_ACCESS_KEY are required when DRY_RUN is not enabled".to_string(),
            ));
        }

        Ok(Self {
            openai_api_key,
            gemini_api_key,
            chat_provider,
            image_provider,
            qa_provider,
            chat_model,
            image_model,
            qa_model,
            cdn_access_key_id,
            cdn_secret_access_key,
            cdn_endpoint: std::env::var("CDN_ENDPOINT")
                .unwrap_or_else(|_| "https://nyc3.digitaloceanspaces.com".to_string()),
            cdn_bucket: std::env::var("CDN_BUCKET").unwrap_or_else(|_| "iamdreamingof".to_string()),
            cdn_base_url: std::env::var("CDN_BASE_URL")
                .unwrap_or_else(|_| "https://cdn.iamdreamingof.com".to_string()),
            dry_run,
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

    #[test]
    fn test_ai_provider_from_str() {
        assert_eq!("openai".parse::<AiProvider>().unwrap(), AiProvider::OpenAi);
        assert_eq!("gemini".parse::<AiProvider>().unwrap(), AiProvider::Gemini);
        assert_eq!("OpenAI".parse::<AiProvider>().unwrap(), AiProvider::OpenAi);
        assert_eq!("GEMINI".parse::<AiProvider>().unwrap(), AiProvider::Gemini);
    }

    #[test]
    fn test_ai_provider_from_str_unknown() {
        let err = "opneai".parse::<AiProvider>().unwrap_err();
        assert!(matches!(err, crate::Error::Config(_)));
    }

    #[test]
    fn test_ai_provider_display_roundtrip() {
        let openai = AiProvider::OpenAi;
        let gemini = AiProvider::Gemini;
        assert_eq!(openai.to_string().parse::<AiProvider>().unwrap(), openai);
        assert_eq!(gemini.to_string().parse::<AiProvider>().unwrap(), gemini);
    }
}
