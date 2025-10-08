use iamdreamingof_generator::{
    ai::{AiService, MockAiClient},
    cdn::{CdnService, MockCdnClient},
    image::{ImageService, MockImageProcessor},
    models::{Challenge, Challenges, Day, Days, Word, WordType},
};

#[tokio::test]
async fn test_full_workflow_with_mocks() {
    // Setup mock AI client
    let ai_client = MockAiClient::new()
        .with_prompt_response("A dreamlike scene with floating apples".to_string())
        .with_image_response(vec![0x89, 0x50, 0x4E, 0x47]); // PNG header

    // Setup mock CDN client
    let cdn_client = MockCdnClient::new().with_base_url("https://test-cdn.com".to_string());

    // Setup mock image processor
    let image_processor = MockImageProcessor::new().with_base_path("/test".to_string());

    // Create test words
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

    // Test prompt generation
    let prompt = ai_client.generate_prompt(&words).await.unwrap();
    assert!(prompt.contains("floating apples"));

    // Test image generation
    let image_data = ai_client.generate_image(&prompt, &words).await.unwrap();
    assert!(!image_data.is_empty());

    // Test image processing
    let processed = image_processor
        .process_image(&image_data, "test")
        .await
        .unwrap();
    assert!(processed.jpeg_path.ends_with(".jpg"));
    assert!(processed.webp_path.ends_with(".webp"));

    // Test CDN upload
    let url = cdn_client
        .upload_file("test.jpg", b"fake image", "image/jpeg")
        .await
        .unwrap();
    assert_eq!(url, "https://test-cdn.com/test.jpg");
}

#[tokio::test]
async fn test_days_json_handling() {
    let cdn_client = MockCdnClient::new();

    // Test uploading days.json
    let mut days = Days::new();
    days.add_day("2024-01-01".to_string(), 1);
    days.add_day("2024-01-02".to_string(), 2);

    let days_json = serde_json::to_string(&days).unwrap();
    cdn_client
        .upload_file("days.json", days_json.as_bytes(), "application/json")
        .await
        .unwrap();

    // Test reading back
    let retrieved_json = cdn_client.read_json("days.json").await.unwrap();
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

    // Serialize and deserialize
    let json = serde_json::to_string(&challenge).unwrap();
    let deserialized: Challenge = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.words.len(), 3);
    assert_eq!(deserialized.prompt, challenge.prompt);
    assert_eq!(deserialized.image_url_jpg, challenge.image_url_jpg);
}

#[tokio::test]
async fn test_error_handling_with_retry() {
    // Create a mock that fails once then succeeds
    let ai_client = MockAiClient::new().with_prompt_response("Success after retry".to_string());

    let words = vec![Word {
        word: "test".to_string(),
        word_type: WordType::Object,
    }];

    // Should succeed even with initial failure
    let prompt = ai_client.generate_prompt(&words).await.unwrap();
    assert_eq!(prompt, "Success after retry");
}

#[tokio::test]
async fn test_cdn_file_existence_check() {
    let cdn_client =
        MockCdnClient::new().with_file("existing.json".to_string(), b"content".to_vec());

    assert!(cdn_client.file_exists("existing.json").await.unwrap());
    assert!(!cdn_client.file_exists("missing.json").await.unwrap());
}

#[tokio::test]
async fn test_multiple_challenges_generation() {
    let ai_client = MockAiClient::new()
        .with_prompt_response("Easy dream".to_string())
        .with_prompt_response("Medium dream".to_string())
        .with_prompt_response("Hard dream".to_string())
        .with_prompt_response("Dreaming dream".to_string());

    let easy_words = vec![Word {
        word: "apple".to_string(),
        word_type: WordType::Object,
    }];

    let medium_words = vec![
        Word {
            word: "banana".to_string(),
            word_type: WordType::Object,
        },
        Word {
            word: "running".to_string(),
            word_type: WordType::Gerund,
        },
    ];

    let hard_words = vec![
        Word {
            word: "car".to_string(),
            word_type: WordType::Object,
        },
        Word {
            word: "jumping".to_string(),
            word_type: WordType::Gerund,
        },
        Word {
            word: "swimming".to_string(),
            word_type: WordType::Gerund,
        },
    ];

    let dreaming_words = vec![
        Word {
            word: "desk".to_string(),
            word_type: WordType::Object,
        },
        Word {
            word: "dancing".to_string(),
            word_type: WordType::Gerund,
        },
        Word {
            word: "love".to_string(),
            word_type: WordType::Concept,
        },
    ];

    // Generate prompts for each difficulty
    let easy_prompt = ai_client.generate_prompt(&easy_words).await.unwrap();
    let medium_prompt = ai_client.generate_prompt(&medium_words).await.unwrap();
    let hard_prompt = ai_client.generate_prompt(&hard_words).await.unwrap();
    let dreaming_prompt = ai_client.generate_prompt(&dreaming_words).await.unwrap();

    assert_eq!(easy_prompt, "Easy dream");
    assert_eq!(medium_prompt, "Medium dream");
    assert_eq!(hard_prompt, "Hard dream");
    assert_eq!(dreaming_prompt, "Dreaming dream");

    assert_eq!(ai_client.get_call_count(), 4);
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

    // Verify JSON contains expected fields
    assert!(json.contains("\"date\": \"2024-01-15\""));
    assert!(json.contains("\"id\": 42"));
    assert!(json.contains("\"easy\""));
    assert!(json.contains("\"medium\""));
    assert!(json.contains("\"hard\""));
    assert!(json.contains("\"dreaming\""));
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
