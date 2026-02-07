//! Application orchestration for generating and publishing daily challenges.

use crate::ai::{
    ChatService, GeminiChatClient, GeminiImageClient, GeminiImageQaClient, ImageGenerationService,
    ImageQaService, OpenAiChatClient, OpenAiImageClient, OpenAiImageQaClient,
};
use crate::cdn::{CdnClient, CdnService, MockCdnClient};
use crate::image::{ImageProcessor, ImageService};
use crate::models::{AiProvider, Challenge, Challenges, Config, Day, Days, Word};
use crate::words::WordSelector;
use crate::{Error, Result};
use chrono::{Local, NaiveDate};
use std::fs;
use std::path::{Path, PathBuf};
use tokio_retry::{strategy::FixedInterval, Retry};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Coordinates AI generation, image processing, and CDN publishing for a day.
pub struct App {
    chat: Box<dyn ChatService>,
    image_gen: Box<dyn ImageGenerationService>,
    image_qa: Box<dyn ImageQaService>,
    cdn: Box<dyn CdnService>,
    image: Box<dyn ImageService>,
    word_selector: WordSelector,
    output_dir: PathBuf,
    dry_run: bool,
}

/// Injectable service bundle used to construct [`App`] in tests/harnesses.
pub struct AppServices {
    pub chat: Box<dyn ChatService>,
    pub image_gen: Box<dyn ImageGenerationService>,
    pub image_qa: Box<dyn ImageQaService>,
    pub cdn: Box<dyn CdnService>,
    pub image: Box<dyn ImageService>,
    pub word_selector: WordSelector,
}

impl App {
    /// Build an app from concrete service dependencies.
    ///
    /// This is primarily useful for integration tests and local harnesses that
    /// need to inject mocks.
    pub fn with_services(services: AppServices, output_dir: PathBuf, dry_run: bool) -> Self {
        Self {
            chat: services.chat,
            image_gen: services.image_gen,
            image_qa: services.image_qa,
            cdn: services.cdn,
            image: services.image,
            word_selector: services.word_selector,
            output_dir,
            dry_run,
        }
    }

    fn api_key_for_provider(config: &Config, provider: AiProvider) -> String {
        match provider {
            AiProvider::OpenAi => config
                .openai_api_key
                .as_ref()
                .expect("OPENAI_API_KEY validated in Config::from_env")
                .clone(),
            AiProvider::Gemini => config
                .gemini_api_key
                .as_ref()
                .expect("GEMINI_API_KEY validated in Config::from_env")
                .clone(),
        }
    }

    fn build_ai_client<T, FOpenAi, FGemini>(
        provider: AiProvider,
        model: &str,
        api_key: String,
        http_client: reqwest::Client,
        capability: &str,
        openai_builder: FOpenAi,
        gemini_builder: FGemini,
    ) -> T
    where
        FOpenAi: FnOnce(String, String, reqwest::Client) -> T,
        FGemini: FnOnce(String, String, reqwest::Client) -> T,
    {
        match provider {
            AiProvider::OpenAi => {
                info!("{} provider: OpenAI (model: {})", capability, model);
                openai_builder(api_key, model.to_string(), http_client)
            }
            AiProvider::Gemini => {
                info!("{} provider: Gemini (model: {})", capability, model);
                gemini_builder(api_key, model.to_string(), http_client)
            }
        }
    }

    /// Construct an app from environment configuration (`Config::from_env`).
    pub async fn new() -> Result<Self> {
        let config = Config::from_env()?;

        let date = Local::now().format("%Y-%m-%d").to_string();
        let session_id = Uuid::new_v4();
        let output_dir = PathBuf::from("output").join(format!("{}_{}", date, session_id));

        fs::create_dir_all(&output_dir)?;
        info!("Created output directory: {}", output_dir.display());

        // Reuse one HTTP connection pool across provider clients.
        let http_client = reqwest::Client::new();

        let chat: Box<dyn ChatService> = Self::build_ai_client(
            config.chat_provider,
            &config.chat_model,
            Self::api_key_for_provider(&config, config.chat_provider),
            http_client.clone(),
            "Chat",
            |api_key, model, client| {
                Box::new(OpenAiChatClient::new_with_client(api_key, model, client))
                    as Box<dyn ChatService>
            },
            |api_key, model, client| {
                Box::new(GeminiChatClient::new_with_client(api_key, model, client))
                    as Box<dyn ChatService>
            },
        );

        let image_gen: Box<dyn ImageGenerationService> = Self::build_ai_client(
            config.image_provider,
            &config.image_model,
            Self::api_key_for_provider(&config, config.image_provider),
            http_client.clone(),
            "Image",
            |api_key, model, client| {
                Box::new(OpenAiImageClient::new_with_client(api_key, model, client))
                    as Box<dyn ImageGenerationService>
            },
            |api_key, model, client| {
                Box::new(GeminiImageClient::new_with_client(api_key, model, client))
                    as Box<dyn ImageGenerationService>
            },
        );

        let image_qa: Box<dyn ImageQaService> = Self::build_ai_client(
            config.qa_provider,
            &config.qa_model,
            Self::api_key_for_provider(&config, config.qa_provider),
            http_client,
            "QA",
            |api_key, model, client| {
                Box::new(OpenAiImageQaClient::new_with_client(api_key, model, client))
                    as Box<dyn ImageQaService>
            },
            |api_key, model, client| {
                Box::new(GeminiImageQaClient::new_with_client(api_key, model, client))
                    as Box<dyn ImageQaService>
            },
        );

        let cdn: Box<dyn CdnService> = if config.dry_run {
            info!("DRY_RUN enabled — CDN uploads will be skipped");
            Box::new(MockCdnClient::new().with_base_url(config.cdn_base_url.clone()))
        } else {
            Box::new(
                CdnClient::new(
                    config
                        .cdn_access_key_id
                        .clone()
                        .expect("CDN_ACCESS_KEY_ID validated in Config::from_env"),
                    config
                        .cdn_secret_access_key
                        .clone()
                        .expect("CDN_SECRET_ACCESS_KEY validated in Config::from_env"),
                    config.cdn_endpoint.clone(),
                    config.cdn_bucket.clone(),
                    config.cdn_base_url.clone(),
                )
                .await?,
            )
        };

        let image = Box::new(ImageProcessor::new(&output_dir)?);
        let word_selector = WordSelector::from_files(Path::new("data"))?;

        Ok(Self::with_services(
            AppServices {
                chat,
                image_gen,
                image_qa,
                cdn,
                image,
                word_selector,
            },
            output_dir,
            config.dry_run,
        ))
    }

    /// Run generation for a target date (or today when `None`).
    pub async fn run(&self, target_date: Option<NaiveDate>) -> Result<()> {
        let date = target_date.unwrap_or_else(|| Local::now().date_naive());
        let date_str = date.format("%Y-%m-%d").to_string();

        info!("Generating content for date: {}", date_str);

        // In dry-run mode, start with fresh days
        let mut days = if self.dry_run {
            Days::new()
        } else {
            self.fetch_days().await.unwrap_or_else(|e| {
                warn!("Could not fetch existing days.json: {}. Starting fresh.", e);
                Days::new()
            })
        };

        let id = if let Some(existing) = days.find_by_date(&date_str) {
            info!("Reusing existing ID {} for date {}", existing.id, date_str);
            existing.id
        } else {
            let new_id = days
                .max_id()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| Error::Invariant("Day ID overflow".to_string()))?;
            info!("Using new ID {} for date {}", new_id, date_str);
            new_id
        };

        let day = match self.generate_day(&date_str, id).await {
            Ok(day) => day,
            Err(e) => {
                error!("Failed to generate day: {}", e);
                return Err(e);
            }
        };

        let day_json = serde_json::to_string_pretty(&day)?;
        let day_key = format!("days/{}.json", date_str);
        self.cdn
            .upload_file(&day_key, day_json.as_bytes(), "application/json")
            .await?;
        info!("Uploaded day data to {}", day_key);

        let json_path = self.output_dir.join(format!("{}.json", date_str));
        fs::write(&json_path, &day_json)?;
        info!("Saved JSON locally at: {}", json_path.display());

        // Update days index if this is a new day
        if days.find_by_date(&date_str).is_none() {
            days.add_day(date_str.clone(), id);
            let days_json = serde_json::to_string_pretty(&days)?;
            self.cdn
                .upload_file("days.json", days_json.as_bytes(), "application/json")
                .await?;
            info!("Updated days.json index");
        }

        let today = Local::now().date_naive();
        if date == today {
            self.cdn
                .upload_file("today.json", day_json.as_bytes(), "application/json")
                .await?;
            info!("Updated today.json");
        }

        info!("Generation complete for {}", date_str);
        Ok(())
    }

    async fn fetch_days(&self) -> Result<Days> {
        let json = self.cdn.read_json("days.json").await?;
        Ok(serde_json::from_str(&json)?)
    }

    async fn generate_day(&self, date: &str, id: u32) -> Result<Day> {
        info!("Generating challenges for date {}", date);

        let word_sets = self.word_selector.select_words()?;

        let (easy, medium, hard, dreaming) = tokio::join!(
            self.create_challenge_with_retry(word_sets.easy, "easy"),
            self.create_challenge_with_retry(word_sets.medium, "medium"),
            self.create_challenge_with_retry(word_sets.hard, "hard"),
            self.create_challenge_with_retry(word_sets.dreaming, "dreaming")
        );

        Ok(Day {
            date: date.to_string(),
            id,
            challenges: Challenges {
                easy: easy?,
                medium: medium?,
                hard: hard?,
                dreaming: dreaming?,
            },
        })
    }

    async fn create_challenge_with_retry(
        &self,
        words: Vec<Word>,
        difficulty: &str,
    ) -> Result<Challenge> {
        let retry_strategy = FixedInterval::from_millis(2000).take(3);

        Retry::spawn(retry_strategy, move || {
            let words_clone = words.clone();
            let difficulty = difficulty.to_string();
            async move {
                info!("[{}] Generating challenge...", difficulty);
                match self.create_challenge(&words_clone, &difficulty).await {
                    Ok(challenge) => Ok(challenge),
                    Err(e) => {
                        warn!(
                            "[{}] Challenge attempt failed: {}. Will retry...",
                            difficulty, e
                        );
                        Err(e)
                    }
                }
            }
        })
        .await
        .map_err(|e| {
            error!(
                "[{}] Failed to create challenge after retries: {}",
                difficulty, e
            );
            e
        })
    }

    async fn create_challenge(&self, words: &[Word], difficulty: &str) -> Result<Challenge> {
        info!("[{}] Creating challenge", difficulty);

        let prompt = self.chat.generate_prompt(words).await?;
        info!(
            "[{}] Generated prompt ({} chars): {}",
            difficulty,
            prompt.len(),
            prompt
        );

        const MAX_IMAGE_ATTEMPTS: usize = 3;
        let mut image_data;
        let mut attempt = 0;

        loop {
            attempt += 1;
            image_data = self.image_gen.generate_image(&prompt, words).await?;
            info!(
                "[{}] Generated image ({} bytes) - attempt {}/{}",
                difficulty,
                image_data.len(),
                attempt,
                MAX_IMAGE_ATTEMPTS
            );

            let has_text = self.image_qa.detect_text(&image_data).await?;

            if !has_text {
                info!("[{}] Image passed text detection check", difficulty);
                break;
            } else if attempt >= MAX_IMAGE_ATTEMPTS {
                warn!(
                    "[{}] Image contains text after {} attempts, proceeding anyway",
                    difficulty, MAX_IMAGE_ATTEMPTS
                );
                break;
            } else {
                warn!(
                    "[{}] Image contains text, retrying generation (attempt {}/{})",
                    difficulty, attempt, MAX_IMAGE_ATTEMPTS
                );
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
            }
        }

        let processed = self.image.process_image(&image_data, difficulty).await?;

        let jpeg_data = std::fs::read(&processed.jpeg_path)?;
        let webp_data = std::fs::read(&processed.webp_path)?;

        let jpeg_filename = processed
            .jpeg_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "Invalid JPEG output path: {}",
                    processed.jpeg_path.display()
                ))
            })?
            .to_string();
        let webp_filename = processed
            .webp_path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| {
                Error::Invariant(format!(
                    "Invalid WebP output path: {}",
                    processed.webp_path.display()
                ))
            })?
            .to_string();

        let jpeg_key = format!("images/{}", jpeg_filename);
        let webp_key = format!("images/{}", webp_filename);

        let jpeg_url = self
            .cdn
            .upload_file(&jpeg_key, &jpeg_data, "image/jpeg")
            .await?;
        let webp_url = self
            .cdn
            .upload_file(&webp_key, &webp_data, "image/webp")
            .await?;

        info!("[{}] Uploaded images to CDN", difficulty);
        info!(
            "[{}] Images saved locally at: {} and {}",
            difficulty,
            processed.jpeg_path.display(),
            processed.webp_path.display()
        );

        Ok(Challenge::new(
            words.to_vec(),
            jpeg_key,
            jpeg_url,
            webp_url,
            prompt,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{App, AppServices};
    use crate::ai::{MockChatClient, MockImageGenerationClient, MockImageQaClient};
    use crate::cdn::MockCdnClient;
    use crate::image::MockImageProcessor;
    use crate::models::{Challenge, Day, Days, Word, WordType};
    use crate::words::WordSelector;
    use chrono::{Local, NaiveDate};
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    const TEST_PROMPT: &str = "A surreal dream scene";
    const TEST_CDN_BASE_URL: &str = "https://cdn.test";

    fn setup_test_dirs() -> (tempfile::TempDir, PathBuf) {
        let dir = tempdir().unwrap();
        let output_dir = dir.path().join("output");
        fs::create_dir_all(&output_dir).unwrap();
        (dir, output_dir)
    }

    fn build_test_app(
        output_dir: &Path,
        image_gen: MockImageGenerationClient,
        image_qa: MockImageQaClient,
    ) -> App {
        App::with_services(
            AppServices {
                chat: Box::new(MockChatClient::new().with_prompt_response(TEST_PROMPT.to_string())),
                image_gen: Box::new(image_gen),
                image_qa: Box::new(image_qa),
                cdn: Box::new(MockCdnClient::new().with_base_url(TEST_CDN_BASE_URL.to_string())),
                image: Box::new(
                    MockImageProcessor::new()
                        .with_base_path(output_dir.to_string_lossy().to_string()),
                ),
                word_selector: WordSelector::from_files(Path::new("data"))
                    .expect("load real word lists"),
            },
            output_dir.to_path_buf(),
            true,
        )
    }

    fn assert_valid_challenge(
        challenge: &Challenge,
        expected_objects: usize,
        expected_gerunds: usize,
        expected_concepts: usize,
    ) {
        assert_eq!(challenge.words.len(), 3);
        assert!(!challenge.prompt.trim().is_empty());
        assert!(challenge.image_path.starts_with("images/"));
        assert!(challenge
            .image_url_jpg
            .starts_with("https://cdn.test/images/"));
        assert!(challenge
            .image_url_webp
            .starts_with("https://cdn.test/images/"));

        let objects = challenge
            .words
            .iter()
            .filter(|word| matches!(word.word_type, WordType::Object))
            .count();
        let gerunds = challenge
            .words
            .iter()
            .filter(|word| matches!(word.word_type, WordType::Gerund))
            .count();
        let concepts = challenge
            .words
            .iter()
            .filter(|word| matches!(word.word_type, WordType::Concept))
            .count();
        assert_eq!(objects, expected_objects);
        assert_eq!(gerunds, expected_gerunds);
        assert_eq!(concepts, expected_concepts);
    }

    #[tokio::test]
    async fn test_app_run_with_mocks_writes_day_and_updates_days_index() {
        let (_dir, output_dir) = setup_test_dirs();

        let app = build_test_app(
            &output_dir,
            MockImageGenerationClient::new().with_image_response(vec![1, 2, 3]),
            MockImageQaClient::new().with_text_detection_response(false),
        );

        let date = NaiveDate::from_ymd_opt(2099, 1, 1).unwrap();
        app.run(Some(date)).await.unwrap();

        assert!(output_dir.join("2099-01-01.json").exists());
        assert!(app.cdn.read_json("days/2099-01-01.json").await.is_ok());
        assert!(app.cdn.read_json("days.json").await.is_ok());
        assert!(app.cdn.read_json("today.json").await.is_err());

        let days_json = app.cdn.read_json("days.json").await.unwrap();
        let days: Days = serde_json::from_str(&days_json).unwrap();
        assert_eq!(days.days.len(), 1);
        assert_eq!(days.days[0].date, "2099-01-01");

        let day_json = app.cdn.read_json("days/2099-01-01.json").await.unwrap();
        let day: Day = serde_json::from_str(&day_json).unwrap();
        assert_eq!(day.date, "2099-01-01");
        assert_valid_challenge(&day.challenges.easy, 3, 0, 0);
        assert_valid_challenge(&day.challenges.medium, 2, 1, 0);
        assert_valid_challenge(&day.challenges.hard, 1, 2, 0);
        assert_valid_challenge(&day.challenges.dreaming, 1, 1, 1);
    }

    #[tokio::test]
    async fn test_app_run_for_today_uploads_today_json() {
        let (_dir, output_dir) = setup_test_dirs();
        let before = Local::now().date_naive();

        let app = build_test_app(
            &output_dir,
            MockImageGenerationClient::new().with_image_response(vec![1, 2, 3]),
            MockImageQaClient::new().with_text_detection_response(false),
        );

        app.run(Some(before)).await.unwrap();
        let after = Local::now().date_naive();

        if before == after {
            let today_json = app.cdn.read_json("today.json").await.unwrap();
            let day: Day = serde_json::from_str(&today_json).unwrap();
            assert_eq!(day.date, before.to_string());
        } else {
            // Midnight boundary case: `run` may skip today.json depending on
            // when the internal comparison executes.
            let dated_key = format!("days/{}.json", before);
            assert!(app.cdn.read_json(&dated_key).await.is_ok());
        }
    }

    #[tokio::test]
    async fn test_create_challenge_retries_until_image_qa_passes() {
        let (_dir, output_dir) = setup_test_dirs();

        let image_gen = MockImageGenerationClient::new().with_image_response(vec![1, 2, 3]);
        let image_gen_probe = image_gen.clone();
        let image_qa = MockImageQaClient::new()
            .with_text_detection_response(true)
            .with_text_detection_response(true)
            .with_text_detection_response(false);
        let image_qa_probe = image_qa.clone();

        let app = build_test_app(&output_dir, image_gen, image_qa);

        let words = vec![
            Word {
                word: "clock".to_string(),
                word_type: WordType::Object,
            },
            Word {
                word: "running".to_string(),
                word_type: WordType::Gerund,
            },
            Word {
                word: "wonder".to_string(),
                word_type: WordType::Concept,
            },
        ];

        let challenge = app.create_challenge(&words, "easy").await.unwrap();
        assert_valid_challenge(&challenge, 1, 1, 1);
        assert_eq!(image_gen_probe.get_call_count(), 3);
        assert_eq!(image_qa_probe.get_call_count(), 3);
    }
}
