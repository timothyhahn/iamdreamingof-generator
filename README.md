# IAmDreamingOf Generator

A Rust application that generates daily abstract dream challenges for iamdreamingof.com.

## Architecture

Modular architecture with dependency injection and multi-provider AI support:

```
src/
├── main.rs           # Entry point and orchestration
├── lib.rs            # Library root
├── semantic.rs       # Cosine similarity + pair scoring utilities
├── models.rs         # Data structures (Day, Challenge, Config, AiProvider)
├── prompts.rs        # Shared prompt templates (include_str! from data/prompts/)
├── words.rs          # Word selection logic
├── error.rs          # Error types
├── bin/
│   └── word_similarity_audit.rs # Embedding-based near-duplicate word audit
├── ai/
│   ├── mod.rs        # Trait definitions (chat, image, QA, embeddings)
│   ├── mime.rs       # Image MIME type detection from magic bytes
│   ├── mock.rs       # Mock implementations for testing
│   ├── openai/       # OpenAI provider (chat, image, qa, embeddings)
│   └── gemini/       # Gemini provider (chat, image, qa, embeddings)
├── cdn/              # S3/CDN integration
└── image/            # Image processing (resize, JPEG + WebP)

data/
├── prompts/          # Prompt templates (embedded at compile time)
├── objects.json      # Word lists
├── gerunds.json
└── concepts.json
```

## Quick Start

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Environment variables (see Configuration)

### Building

```bash
cargo build
cargo build --release
```

### Running

```bash
# Run for today
cargo run

# Run for specific date
cargo run -- 2024-01-15
```

### Word Similarity Audit

Use embeddings to flag semantically close words both within categories and
across category pairs:

```bash
# Gemini embeddings (default provider/model)
GEMINI_API_KEY=... cargo run --bin word_similarity_audit -- --threshold 0.75

# Gemini embeddings
GEMINI_API_KEY=... cargo run --bin word_similarity_audit -- --provider gemini

# Write machine-readable report
GEMINI_API_KEY=... cargo run --bin word_similarity_audit -- \
  --json-output output/word_similarity_report.json
```

Useful flags:
- `--provider openai|gemini`
- `--model <embedding-model-name>`
- `--threshold <0.0..1.0>`
- `--batch-size <n>`
- `--max-pairs <n>`
- `--data-dir <path>`

### Testing

```bash
cargo test
cargo test --lib              # unit tests only
cargo test --test integration # integration tests only
cargo clippy --all-targets --all-features -- -D warnings
```

## Configuration

### AI Providers (all required)

- `CHAT_PROVIDER`: `openai` or `gemini`
- `CHAT_MODEL`: e.g. `gpt-5`, `gemini-3-flash-preview`
- `IMAGE_PROVIDER`: `openai` or `gemini`
- `IMAGE_MODEL`: e.g. `gpt-image-1`, `gemini-3-pro-image-preview`
- `QA_PROVIDER`: `openai` or `gemini`
- `QA_MODEL`: e.g. `gpt-4o-mini`, `gemini-3-flash-preview`

### API Keys (required based on configured providers)

- `OPENAI_API_KEY`: Required if any provider is `openai`
- `GEMINI_API_KEY`: Required if any provider is `gemini`

### CDN

- `CDN_ACCESS_KEY_ID`: DigitalOcean Spaces access key (required unless `DRY_RUN=true`)
- `CDN_SECRET_ACCESS_KEY`: DigitalOcean Spaces secret key (required unless `DRY_RUN=true`)
- `CDN_ENDPOINT`: (Optional) defaults to https://nyc3.digitaloceanspaces.com
- `CDN_BUCKET`: (Optional) defaults to "iamdreamingof"
- `CDN_BASE_URL`: (Optional) defaults to https://cdn.iamdreamingof.com

### Other

- `DRY_RUN`: Set to `true` to skip CDN uploads

## Data Format

The application generates and maintains the following JSON structure:
- `days.json`: Index of all generated days with IDs
- `days/{date}.json`: Individual day data with challenges
- `today.json`: Copy of the current day's data for quick access

Each challenge includes:
- Word sets for different difficulty levels
- Dream descriptions
- Image based off dream descriptions
- Image URLs (both JPEG and WebP formats)
