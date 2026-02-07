use anyhow::Result;
use chrono::{Local, NaiveDate};
use iamdreamingof_generator::{
    ai::{
        ChatService, GeminiChatClient, GeminiImageClient, GeminiImageQaClient,
        ImageGenerationService, ImageQaService, OpenAiChatClient, OpenAiImageClient,
        OpenAiImageQaClient,
    },
    cdn::{CdnClient, CdnService, MockCdnClient},
    image::{ImageProcessor, ImageService},
    models::{AiProvider, Challenge, Challenges, Config, Day, Days, Word},
    words::WordSelector,
};
use std::fs;
use std::path::{Path, PathBuf};
use tokio_retry::{strategy::FixedInterval, Retry};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use uuid::Uuid;

struct App {
    chat: Box<dyn ChatService>,
    image_gen: Box<dyn ImageGenerationService>,
    image_qa: Box<dyn ImageQaService>,
    cdn: Box<dyn CdnService>,
    image: Box<dyn ImageService>,
    word_selector: WordSelector,
    output_dir: PathBuf,
    dry_run: bool,
}

impl App {
    async fn new() -> Result<Self> {
        let config = Config::from_env()?;

        let date = Local::now().format("%Y-%m-%d").to_string();
        let session_id = Uuid::new_v4();
        let output_dir = PathBuf::from("output").join(format!("{}_{}", date, session_id));

        fs::create_dir_all(&output_dir)?;
        info!("Created output directory: {}", output_dir.display());

        let chat: Box<dyn ChatService> = match config.chat_provider {
            AiProvider::OpenAi => {
                let key = config.openai_api_key.as_ref().unwrap();
                info!("Chat provider: OpenAI (model: {})", config.chat_model);
                Box::new(OpenAiChatClient::new(
                    key.clone(),
                    config.chat_model.clone(),
                ))
            }
            AiProvider::Gemini => {
                let key = config.gemini_api_key.as_ref().unwrap();
                info!("Chat provider: Gemini (model: {})", config.chat_model);
                Box::new(GeminiChatClient::new(
                    key.clone(),
                    config.chat_model.clone(),
                ))
            }
        };

        let image_gen: Box<dyn ImageGenerationService> = match config.image_provider {
            AiProvider::OpenAi => {
                let key = config.openai_api_key.as_ref().unwrap();
                info!("Image provider: OpenAI (model: {})", config.image_model);
                Box::new(OpenAiImageClient::new(
                    key.clone(),
                    config.image_model.clone(),
                ))
            }
            AiProvider::Gemini => {
                let key = config.gemini_api_key.as_ref().unwrap();
                info!("Image provider: Gemini (model: {})", config.image_model);
                Box::new(GeminiImageClient::new(
                    key.clone(),
                    config.image_model.clone(),
                ))
            }
        };

        let image_qa: Box<dyn ImageQaService> = match config.qa_provider {
            AiProvider::OpenAi => {
                let key = config.openai_api_key.as_ref().unwrap();
                info!("QA provider: OpenAI (model: {})", config.qa_model);
                Box::new(OpenAiImageQaClient::new(
                    key.clone(),
                    config.qa_model.clone(),
                ))
            }
            AiProvider::Gemini => {
                let key = config.gemini_api_key.as_ref().unwrap();
                info!("QA provider: Gemini (model: {})", config.qa_model);
                Box::new(GeminiImageQaClient::new(
                    key.clone(),
                    config.qa_model.clone(),
                ))
            }
        };

        let cdn: Box<dyn CdnService> = if config.dry_run {
            info!("DRY_RUN enabled — CDN uploads will be skipped");
            Box::new(MockCdnClient::new().with_base_url(config.cdn_base_url.clone()))
        } else {
            Box::new(
                CdnClient::new(
                    config.cdn_access_key_id.clone().unwrap(),
                    config.cdn_secret_access_key.clone().unwrap(),
                    config.cdn_endpoint.clone(),
                    config.cdn_bucket.clone(),
                    config.cdn_base_url.clone(),
                )
                .await?,
            )
        };

        let image = Box::new(ImageProcessor::new(&output_dir)?);
        let word_selector = WordSelector::from_files(Path::new("data"))?;

        Ok(Self {
            chat,
            image_gen,
            image_qa,
            cdn,
            image,
            word_selector,
            output_dir,
            dry_run: config.dry_run,
        })
    }

    async fn run(&self, target_date: Option<NaiveDate>) -> Result<()> {
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
            let new_id = days.max_id().unwrap_or(0) + 1;
            info!("Using new ID {} for date {}", new_id, date_str);
            new_id
        };

        let retry_strategy = FixedInterval::from_millis(2000).take(3);

        let day = match Retry::spawn(retry_strategy.clone(), || async {
            info!("Attempting to generate day content...");
            match self.generate_day(&date_str, id).await {
                Ok(day) => Ok(day),
                Err(e) => {
                    warn!("Generation attempt failed: {}. Will retry...", e);
                    Err(e)
                }
            }
        })
        .await
        {
            Ok(day) => day,
            Err(e) => {
                error!("Failed to generate day after all retries: {}", e);
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

    async fn generate_day(&self, date: &str, id: i32) -> Result<Day> {
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

        let jpeg_filename = Path::new(&processed.jpeg_path)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();
        let webp_filename = Path::new(&processed.webp_path)
            .file_name()
            .unwrap()
            .to_string_lossy()
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
            difficulty, processed.jpeg_path, processed.webp_path
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

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "iamdreamingof_generator=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting iamdreamingof-generator");

    let args: Vec<String> = std::env::args().collect();
    let target_date = if args.len() > 1 {
        Some(NaiveDate::parse_from_str(&args[1], "%Y-%m-%d")?)
    } else {
        None
    };

    match App::new().await {
        Ok(app) => match app.run(target_date).await {
            Ok(_) => {
                info!("Generation completed successfully");
                Ok(())
            }
            Err(e) => {
                error!("Generation failed: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to initialize application: {}", e);
            std::process::exit(1);
        }
    }
}
