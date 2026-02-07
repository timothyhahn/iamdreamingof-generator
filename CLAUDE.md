# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the generator code for iamdreamingof.com, a daily AI-generated dream challenge service written in Rust. The application:
- Generates sets of words with varying difficulty levels (Easy, Medium, Hard, Dreaming)
- Creates dream-like prompts using a configurable AI provider (OpenAI or Gemini)
- Generates images using a configurable AI provider (OpenAI or Gemini)
- Runs image QA (text detection) via a configurable AI provider
- Processes and optimizes images for web (JPEG and WebP formats)
- Uploads content to DigitalOcean Spaces CDN
- Runs as a scheduled job on Fly.io

## Development Commands

### Building and Running

```bash
# Build the project
cargo build

# Build for release
cargo build --release

# Run the application
cargo run

# Run with a specific date
cargo run -- 2024-01-15

# Check code without building
cargo check
```

### Testing

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run unit tests only
cargo test --lib

# Run integration tests only
cargo test --test integration

# Run specific test
cargo test test_word_selection

# Run tests with coverage (requires cargo-tarpaulin)
cargo tarpaulin
```

### Code Quality

```bash
# Format code
cargo fmt

# Run linter (strict — treat warnings as errors)
cargo clippy --all-targets --all-features -- -D warnings

# Fix linting issues
cargo clippy --fix
```

### Docker Commands

```bash
# Build the Docker image
docker build -t iamdreamingof-generator .

# Run the container locally (requires environment variables)
docker run --env-file .env iamdreamingof-generator
```

## Architecture

### Project Structure
```
src/
├── main.rs           # Entry point and orchestration
├── lib.rs            # Library root
├── models.rs         # Data structures (Day, Challenge, Word, Config, AiProvider, etc.)
├── prompts.rs        # Shared prompt templates (compile-time includes from data/prompts/)
├── words.rs          # Word selection logic
├── error.rs          # Error types (AiProvider, Config, Cdn, etc.)
├── ai/
│   ├── mod.rs        # Trait definitions (ChatService, ImageGenerationService, ImageQaService)
│   ├── mock.rs       # Mock implementations for testing
│   ├── openai/
│   │   ├── mod.rs    # OpenAI module exports
│   │   ├── client.rs # Shared OpenAI HTTP client (chat completions endpoint)
│   │   ├── chat.rs   # OpenAI chat/prompt generation
│   │   ├── image.rs  # OpenAI image generation
│   │   └── qa.rs     # OpenAI image QA (text detection via vision)
│   └── gemini/
│       ├── mod.rs    # Gemini module exports
│       ├── client.rs # Shared Gemini HTTP client (generateContent endpoint)
│       ├── chat.rs   # Gemini chat/prompt generation
│       ├── image.rs  # Gemini image generation
│       └── qa.rs     # Gemini image QA (text detection via vision)
├── cdn/
│   ├── mod.rs        # CDN module interface
│   ├── client.rs     # S3/Spaces client implementation
│   └── mock.rs       # Mock implementation for testing
└── image/
    ├── mod.rs        # Image processing interface
    ├── processor.rs  # Image processing implementation
    └── mock.rs       # Mock implementation for testing

data/
├── prompts/          # Prompt templates (embedded at compile time)
│   ├── chat_system.txt
│   ├── chat_user.txt
│   ├── image_enhancement.txt
│   ├── qa_system.txt
│   └── qa_user.txt
├── objects.json      # Word lists
├── gerunds.json
└── concepts.json
```

### Key Design Principles

1. **Dependency Injection**: All external services (AI providers, S3, ImageMagick) are behind traits, allowing for easy mocking in tests
2. **Type Safety**: Strong typing throughout — `AiProvider` enum for provider routing, no string-based fallbacks
3. **Error Handling**: Comprehensive error types with thiserror (`Error::Config`, `Error::AiProvider`, etc.)
4. **Async/Await**: Fully async implementation using Tokio
5. **Testability**: Every module includes unit tests with wiremock for HTTP-level assertions, mocks for external dependencies
6. **Shared HTTP Clients**: `OpenAiHttpClient` and `GeminiHttpClient` eliminate duplication across providers
7. **Prompt Templates**: All prompts in `data/prompts/` with `include_str!` embedding and `{{var}}` substitution

### Main Components

1. **main.rs**: Entry point that orchestrates the entire generation process
   - Fetches existing days.json from CDN
   - Generates unique ID for the day
   - Creates challenges for all difficulty levels
   - Uploads results to CDN
   - Updates today.json if generating for current day

2. **words.rs**: Word selection logic
   - Easy: 3 random objects
   - Medium: 2 objects + 1 gerund
   - Hard: 1 object + 2 gerunds
   - Dreaming: 1 object + 1 gerund + 1 concept
   - Ensures all 12 words across difficulties are unique

3. **ai module**: Multi-provider AI integration (OpenAI + Gemini)
   - `ChatService` trait: Prompt generation from words
   - `ImageGenerationService` trait: Image generation from prompts
   - `ImageQaService` trait: Vision-based text detection on generated images
   - Each provider has a shared HTTP client (`OpenAiHttpClient`, `GeminiHttpClient`)

4. **image module**: Image processing
   - Resizes images to 800x800
   - Converts to both JPEG and WebP formats
   - Returns paths for CDN upload

5. **cdn module**: DigitalOcean Spaces integration
   - Uploads files to CDN with public-read ACL
   - Reads JSON files from CDN

### Data Flow

1. Word selection from JSON files (objects.json, gerunds.json, concepts.json)
2. Prompt generation via configured chat provider (OpenAI or Gemini)
3. Image generation via configured image provider (OpenAI or Gemini)
4. Image QA (text detection) via configured QA provider — regenerates if text found
5. Image processing (resize, format conversion to JPEG + WebP)
6. Upload to CDN (images and JSON data)
7. Update days.json index and today.json if applicable

## Environment Variables Required

### AI Providers (all required)
- `CHAT_PROVIDER`: Chat/prompt generation provider (`openai` or `gemini`)
- `CHAT_MODEL`: Chat model name (e.g. `gpt-5`, `gemini-3-flash-preview`)
- `IMAGE_PROVIDER`: Image generation provider (`openai` or `gemini`)
- `IMAGE_MODEL`: Image model name (e.g. `gpt-image-1`, `gemini-2.5-flash-preview-image-generation`)
- `QA_PROVIDER`: Image QA provider (`openai` or `gemini`)
- `QA_MODEL`: QA model name (e.g. `gpt-4o-mini`, `gemini-3-flash-preview`)

### API Keys (required based on which providers are configured)
- `OPENAI_API_KEY`: Required if any provider is set to `openai`
- `GEMINI_API_KEY`: Required if any provider is set to `gemini`

### CDN
- `CDN_ACCESS_KEY_ID`: DigitalOcean Spaces access key (required unless `DRY_RUN=true`)
- `CDN_SECRET_ACCESS_KEY`: DigitalOcean Spaces secret key (required unless `DRY_RUN=true`)
- `CDN_ENDPOINT`: (Optional) DigitalOcean Spaces endpoint (defaults to https://nyc3.digitaloceanspaces.com)
- `CDN_BUCKET`: (Optional) S3 bucket name (defaults to "iamdreamingof")
- `CDN_BASE_URL`: (Optional) CDN base URL (defaults to https://cdn.iamdreamingof.com)

### Other
- `DRY_RUN`: Set to `true` to skip CDN uploads and use mock CDN

## Error Handling

- Uses tokio-retry for retry logic (3 attempts with 2-second waits)
- Comprehensive error types with thiserror (`Error::Config` for config, `Error::AiProvider` for AI failures)
- Structured logging with tracing
- All provider/model env vars are required — missing or unknown values fail immediately with clear error messages

## Testing Strategy

```bash
# Run unit tests only
cargo test --lib

# Run integration tests only
cargo test --test '*'

# Run specific test
cargo test test_word_selection

# Run tests with coverage (requires cargo-tarpaulin)
cargo tarpaulin
```

### Mock Implementations

Each external service has a mock implementation for testing:
- `MockChatClient`, `MockImageGenerationClient`, `MockImageQaClient`: Simulate AI provider responses
- `MockCdnClient`: Simulates S3/CDN operations in memory
- `MockImageProcessor`: Simulates image processing without actual file I/O

## Deployment

The application is deployed as a scheduled job on Fly.io. The Docker image uses Debian Trixie for ImageMagick 7 support.