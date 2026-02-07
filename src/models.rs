//! Data models and structures
//!
//! Defines the core data structures for challenges, words, and app
//! configuration.

use serde::{Deserialize, Serialize};

/// Word category used for challenge balancing and prompt constraints.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WordType {
    Object,
    Gerund,
    Concept,
}

/// A single selected word and its category label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub word: String,
    #[serde(rename = "type")]
    pub word_type: WordType,
}

/// A generated challenge payload published to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenge {
    pub words: Vec<Word>,
    pub image_path: String,
    pub image_url_jpg: String,
    pub image_url_webp: String,
    pub prompt: String,
}

impl Challenge {
    /// Build a challenge object from generated assets and selected words.
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

/// Challenge set for all supported difficulties on a day.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Challenges {
    pub easy: Challenge,
    pub medium: Challenge,
    pub hard: Challenge,
    pub dreaming: Challenge,
}

/// Daily record containing the generated challenge set and stable numeric id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Day {
    pub date: String, // Format: YYYY-MM-DD
    pub id: u32,
    pub challenges: Challenges,
}

/// Index entry linking a date string to its numeric day id.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DateEntry {
    pub date: String,
    pub id: u32,
}

/// `days.json` index structure used to map dates to generated payload ids.
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
    /// Create an empty days index.
    pub fn new() -> Self {
        Self { days: Vec::new() }
    }

    /// Append a date/id mapping to the index.
    pub fn add_day(&mut self, date: String, id: u32) {
        self.days.push(DateEntry { date, id });
    }

    /// Find an existing day entry by date.
    pub fn find_by_date(&self, date: &str) -> Option<&DateEntry> {
        self.days.iter().find(|d| d.date == date)
    }

    /// Return the current maximum assigned day id, if any.
    pub fn max_id(&self) -> Option<u32> {
        self.days.iter().map(|d| d.id).max()
    }
}

/// Supported AI providers for chat/image/QA/embedding operations.
#[derive(Debug, Clone, Copy, PartialEq)]
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

/// Structured response returned by vision QA prompts.
#[derive(Debug, Serialize, Deserialize)]
pub struct TextDetectionResponse {
    pub includes_text: bool,
}

/// Runtime configuration resolved from environment variables.
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
    /// Load configuration from environment variables and validate provider key requirements.
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
