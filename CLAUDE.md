# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the generator code for iamdreamingof.com, a daily AI-generated dream challenge service written in Rust. The application:
- Generates sets of words with varying difficulty levels (Easy, Medium, Hard, Dreaming)
- Creates dream-like prompts using GPT-4 based on word combinations
- Generates images using DALL-E 3 from the prompts
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

# Run linter
cargo clippy

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
├── models.rs         # Data structures (Day, Challenge, Word, etc.)
├── words.rs          # Word selection logic
├── error.rs          # Error types
├── ai/
│   ├── mod.rs        # AI module interface
│   ├── client.rs     # OpenAI client implementation
│   └── mock.rs       # Mock implementation for testing
├── cdn/
│   ├── mod.rs        # CDN module interface
│   ├── client.rs     # S3/Spaces client implementation
│   └── mock.rs       # Mock implementation for testing
└── image/
    ├── mod.rs        # Image processing interface
    ├── processor.rs  # Image processing implementation
    └── mock.rs       # Mock implementation for testing
```

### Key Design Principles

1. **Dependency Injection**: All external services (OpenAI, S3, ImageMagick) are behind traits, allowing for easy mocking in tests
2. **Type Safety**: Strong typing throughout with serde for serialization
3. **Error Handling**: Comprehensive error types with thiserror
4. **Async/Await**: Fully async implementation using Tokio
5. **Testability**: Every module includes unit tests, with mocks for external dependencies

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

3. **ai module**: OpenAI API integration
   - `generate_prompt()`: Creates dream descriptions using GPT-4
   - `generate_image()`: Generates images using DALL-E 3

4. **image module**: Image processing
   - Resizes images to 800x800
   - Converts to both JPEG and WebP formats
   - Returns paths for CDN upload

5. **cdn module**: DigitalOcean Spaces integration
   - Uploads files to CDN with public-read ACL
   - Reads JSON files from CDN

### Data Flow

1. Word selection from JSON files (objects.json, gerunds.json, concepts.json)
2. Prompt generation via OpenAI GPT-4
3. Image generation via DALL-E 3
4. Image download and processing (resize, format conversion)
5. Upload to CDN (images and JSON data)
6. Update days.json index and today.json if applicable

## Environment Variables Required

- `AI_API_KEY`: OpenAI API key
- `CDN_ACCESS_KEY_ID`: DigitalOcean Spaces access key
- `CDN_SECRET_ACCESS_KEY`: DigitalOcean Spaces secret key
- `CDN_ENDPOINT`: (Optional) DigitalOcean Spaces endpoint (defaults to https://nyc3.digitaloceanspaces.com)
- `CDN_BUCKET`: (Optional) S3 bucket name (defaults to "iamdreamingof")
- `CDN_BASE_URL`: (Optional) CDN base URL (defaults to https://cdn.iamdreamingof.com)

## Error Handling

- Uses tokio-retry for retry logic (3 attempts with 2-minute waits)
- Comprehensive error types with thiserror
- Structured logging with tracing
- Graceful fallbacks for missing configuration

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
- `MockAiClient`: Simulates OpenAI API responses
- `MockCdnClient`: Simulates S3/CDN operations in memory
- `MockImageProcessor`: Simulates image processing without actual file I/O

## Deployment

The application is deployed as a scheduled job on Fly.io. The Docker image uses Debian Trixie for ImageMagick 7 support.