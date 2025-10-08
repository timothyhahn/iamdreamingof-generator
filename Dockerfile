# Builder stage
FROM rust:1.89-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy Cargo files first for better caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies
RUN cargo build --release

# Remove dummy source
RUN rm -rf src

# Copy actual source code
COPY src ./src
COPY data ./data

# Touch main.rs to ensure it's newer than the cached dependency build
RUN touch src/main.rs

# Build the actual application
RUN cargo build --release

# Runtime stage - using Debian Trixie for ImageMagick 7
FROM debian:trixie-slim

# Install runtime dependencies including ImageMagick 7
RUN apt-get update && apt-get install -y \
    ca-certificates \
    imagemagick \
    libssl3 \
    libjpeg62-turbo \
    libpng16-16 \
    libwebp7 \
    libheif1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -u 1001 appuser

# Create app directory
WORKDIR /app

# Copy the binary from builder
COPY --from=builder /app/target/release/iamdreamingof-generator /app/iamdreamingof-generator

# Copy data files
COPY --from=builder /app/data /app/data

# Create output directory
RUN mkdir -p /app/output

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Set environment to production
ENV RUST_LOG=info

# Run the binary
CMD ["./iamdreamingof-generator"]
