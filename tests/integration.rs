use chrono::NaiveDate;
use iamdreamingof_generator::{
    ai::{
        ChatService, ImageGenerationService, ImageQaService, MockChatClient,
        MockImageGenerationClient, MockImageQaClient,
    },
    app::{App, AppServices},
    cdn::{CdnService, MockCdnClient},
    image::{ImageService, MockImageProcessor},
    models::{Challenge, Challenges, Day, Days, Word, WordType},
    words::WordSelector,
};
use std::fs;
use std::path::Path;

#[tokio::test]
async fn test_full_workflow_with_mocks() {
    let chat = MockChatClient::new()
        .with_prompt_response("A dreamlike scene with floating apples".to_string());
    let image_gen =
        MockImageGenerationClient::new().with_image_response(vec![0x89, 0x50, 0x4E, 0x47]);
    let image_qa = MockImageQaClient::new();

    let cdn = MockCdnClient::new().with_base_url("https://test-cdn.com".to_string());
    let dir = tempfile::tempdir().unwrap();
    let image_processor =
        MockImageProcessor::new().with_base_path(dir.path().to_string_lossy().to_string());

    let words = vec![
        Word {
            word: "apple".to_string(),
            word_type: WordType::Object,
        },
        Word {
            word: "running".to_string(),
            word_type: WordType::Gerund,
        },
    ];

    // Chat generates prompt
    let prompt = chat.generate_prompt(&words).await.unwrap();
    assert!(prompt.contains("floating apples"));

    // Image gen creates image bytes
    let image_data = image_gen.generate_image(&prompt, &words).await.unwrap();
    assert!(!image_data.is_empty());

    // QA checks for text
    let has_text = image_qa.detect_text(&image_data).await.unwrap();
    assert!(!has_text);

    // Image processor resizes/converts
    let processed = image_processor
        .process_image(&image_data, "test")
        .await
        .unwrap();
    assert!(processed.jpeg_path.to_string_lossy().ends_with(".jpg"));
    assert!(processed.webp_path.to_string_lossy().ends_with(".webp"));

    // CDN upload
    let url = cdn
        .upload_file("test.jpg", b"fake image", "image/jpeg")
        .await
        .unwrap();
    assert_eq!(url, "https://test-cdn.com/test.jpg");
}

#[tokio::test]
async fn test_days_json_handling() {
    let cdn = MockCdnClient::new();

    let mut days = Days::new();
    days.add_day("2024-01-01".to_string(), 1);
    days.add_day("2024-01-02".to_string(), 2);

    let days_json = serde_json::to_string(&days).unwrap();
    cdn.upload_file("days.json", days_json.as_bytes(), "application/json")
        .await
        .unwrap();

    let retrieved_json = cdn.read_json("days.json").await.unwrap();
    let retrieved_days: Days = serde_json::from_str(&retrieved_json).unwrap();

    assert_eq!(retrieved_days.days.len(), 2);
    assert_eq!(retrieved_days.max_id(), Some(2));
}

#[tokio::test]
async fn test_challenge_serialization() {
    let words = vec![
        Word {
            word: "mountain".to_string(),
            word_type: WordType::Object,
        },
        Word {
            word: "flying".to_string(),
            word_type: WordType::Gerund,
        },
        Word {
            word: "freedom".to_string(),
            word_type: WordType::Concept,
        },
    ];

    let challenge = Challenge::new(
        words.clone(),
        "images/test.jpg".to_string(),
        "https://cdn.example.com/images/test.jpg".to_string(),
        "https://cdn.example.com/images/test.webp".to_string(),
        "A dreamlike scene of flying mountains symbolizing freedom".to_string(),
    );

    let json = serde_json::to_string(&challenge).unwrap();
    let deserialized: Challenge = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.words.len(), 3);
    assert_eq!(deserialized.prompt, challenge.prompt);
    assert_eq!(deserialized.image_url_jpg, challenge.image_url_jpg);
}

#[tokio::test]
async fn test_chat_returns_configured_response() {
    let chat = MockChatClient::new().with_prompt_response("Custom dream scene".to_string());

    let words = vec![Word {
        word: "test".to_string(),
        word_type: WordType::Object,
    }];

    let prompt = chat.generate_prompt(&words).await.unwrap();
    assert_eq!(prompt, "Custom dream scene");
}

#[tokio::test]
async fn test_cdn_file_existence_check() {
    let cdn = MockCdnClient::new().with_file("existing.json".to_string(), b"content".to_vec());

    assert!(cdn.file_exists("existing.json").await.unwrap());
    assert!(!cdn.file_exists("missing.json").await.unwrap());
}

#[tokio::test]
async fn test_multiple_prompts_generation() {
    let chat = MockChatClient::new()
        .with_prompt_response("Easy dream".to_string())
        .with_prompt_response("Medium dream".to_string())
        .with_prompt_response("Hard dream".to_string())
        .with_prompt_response("Dreaming dream".to_string());

    let easy_prompt = chat.generate_prompt(&[]).await.unwrap();
    let medium_prompt = chat.generate_prompt(&[]).await.unwrap();
    let hard_prompt = chat.generate_prompt(&[]).await.unwrap();
    let dreaming_prompt = chat.generate_prompt(&[]).await.unwrap();

    assert_eq!(easy_prompt, "Easy dream");
    assert_eq!(medium_prompt, "Medium dream");
    assert_eq!(hard_prompt, "Hard dream");
    assert_eq!(dreaming_prompt, "Dreaming dream");
    assert_eq!(chat.get_call_count(), 4);
}

#[tokio::test]
async fn test_day_json_structure() {
    let day = Day {
        date: "2024-01-15".to_string(),
        id: 42,
        challenges: Challenges {
            easy: create_test_challenge("easy"),
            medium: create_test_challenge("medium"),
            hard: create_test_challenge("hard"),
            dreaming: create_test_challenge("dreaming"),
        },
    };

    let json = serde_json::to_string_pretty(&day).unwrap();

    assert!(json.contains("\"date\": \"2024-01-15\""));
    assert!(json.contains("\"id\": 42"));
    assert!(json.contains("\"easy\""));
    assert!(json.contains("\"medium\""));
    assert!(json.contains("\"hard\""));
    assert!(json.contains("\"dreaming\""));
}

/// Simulate the image QA retry loop: first image has text, second is clean.
#[tokio::test]
async fn test_image_qa_retry_flow() {
    let chat = MockChatClient::new().with_prompt_response("A dreamy scene".to_string());
    let image_gen = MockImageGenerationClient::new();
    let qa = MockImageQaClient::new()
        .with_text_detection_response(true)
        .with_text_detection_response(false);

    let prompt = chat.generate_prompt(&[]).await.unwrap();

    let img1 = image_gen.generate_image(&prompt, &[]).await.unwrap();
    assert!(qa.detect_text(&img1).await.unwrap());

    let img2 = image_gen.generate_image(&prompt, &[]).await.unwrap();
    assert!(!qa.detect_text(&img2).await.unwrap());

    assert_eq!(image_gen.get_call_count(), 2);
    assert_eq!(qa.get_call_count(), 2);
}

fn create_test_challenge(difficulty: &str) -> Challenge {
    Challenge::new(
        vec![Word {
            word: format!("{}_word", difficulty),
            word_type: WordType::Object,
        }],
        format!("images/{}.jpg", difficulty),
        format!("https://cdn.example.com/images/{}.jpg", difficulty),
        format!("https://cdn.example.com/images/{}.webp", difficulty),
        format!("Test prompt for {} difficulty", difficulty),
    )
}

#[tokio::test]
async fn test_app_with_services_is_usable_from_integration_tests() {
    let temp = tempfile::tempdir().unwrap();
    let output_dir = temp.path().join("output");
    fs::create_dir_all(&output_dir).unwrap();

    let app = App::with_services(
        AppServices {
            chat: Box::new(MockChatClient::new().with_prompt_response("A dream scene".to_string())),
            image_gen: Box::new(
                MockImageGenerationClient::new().with_image_response(vec![1, 2, 3]),
            ),
            image_qa: Box::new(MockImageQaClient::new().with_text_detection_response(false)),
            cdn: Box::new(MockCdnClient::new().with_base_url("https://cdn.test".to_string())),
            image: Box::new(
                MockImageProcessor::new().with_base_path(output_dir.to_string_lossy().to_string()),
            ),
            word_selector: WordSelector::from_files(Path::new("data")).unwrap(),
        },
        output_dir.clone(),
        true,
    );

    app.run(Some(NaiveDate::from_ymd_opt(2099, 2, 7).unwrap()))
        .await
        .unwrap();

    assert!(output_dir.join("2099-02-07.json").exists());
}
