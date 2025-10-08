# IAmDreamingOf Generator

A Rust application that generates daily abstract dream challenges for iamdreamingof.com.

## Architecture

The Rust implementation follows a modular architecture with dependency injection:

```
src/
├── main.rs           # Entry point and orchestration
├── lib.rs            # Library root
├── models.rs         # Data structures
├── words.rs          # Word selection logic
├── error.rs          # Error types
├── ai/              # OpenAI integration
│   ├── client.rs    # Real implementation
│   └── mock.rs      # Mock for testing
├── cdn/             # S3/CDN integration
│   ├── client.rs    # Real implementation
│   └── mock.rs      # Mock for testing
└── image/           # Image processing
    ├── processor.rs # Real implementation
    └── mock.rs      # Mock for testing
```

## Quick Start

### Prerequisites

- Rust 1.70+ (install via [rustup](https://rustup.rs/))
- Environment variables (see Configuration)

### Building

```bash
# Build debug version
cargo build

# Build optimized release version
cargo build --release
```

### Running

```bash
# Run for today
cargo run

# Run for specific date
cargo run -- 2024-01-15
```

### Testing

```bash
# Run all tests
cargo test

# Run unit tests only
cargo test --lib

# Run integration tests only
cargo test --test integration

# Run with coverage (requires cargo-tarpaulin)
cargo install cargo-tarpaulin
cargo tarpaulin
```

## Configuration

Required environment variables:

- `AI_API_KEY`: OpenAI API key
- `CDN_ACCESS_KEY_ID`: DigitalOcean Spaces access key
- `CDN_SECRET_ACCESS_KEY`: DigitalOcean Spaces secret key
- `CDN_ENDPOINT`: S3 endpoint (defaults to https://nyc3.digitaloceanspaces.com)
- `CDN_BUCKET`: Bucket name (defaults to "iamdreamingof")
- `CDN_BASE_URL`: CDN URL (defaults to https://cdn.iamdreamingof.com)

## Data Format

The application generates and maintains the following JSON structure:
- `days.json`: Index of all generated days with IDs
- `days/{date}.json`: Individual day data with challenges
- `today.json`: Copy of the current day's data for quick access

Each challenge includes:
- Word sets for different difficulty levels
- Dream descriptions
- Image based off dream descriptions.
- Image URLs (both JPEG and WebP formats)
