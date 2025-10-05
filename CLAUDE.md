# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the generator code for iamdreamingof.com, a daily AI-generated dream challenge service. The application:
- Generates sets of words with varying difficulty levels (Easy, Medium, Hard, Dreaming)
- Creates dream-like prompts using GPT-4 based on word combinations
- Generates images using DALL-E 3 from the prompts
- Processes and optimizes images for web (JPEG and WebP formats)
- Uploads content to DigitalOcean Spaces CDN
- Runs as a Kubernetes CronJob once daily

## Development Commands

### UV Commands
```bash
# Install dependencies (creates venv automatically)
uv sync

# Install with dev dependencies
uv sync --dev

# Run the main generator
uv run python main.py

# Add a new dependency
uv add <package-name>

# Add a new dev dependency
uv add --dev <package-name>

# Update all dependencies
uv lock --upgrade

# Update a specific dependency
uv lock --upgrade-package <package-name>

# Remove a dependency
uv remove <package-name>
```

### Linting and Type Checking
```bash
# Run Ruff linter and formatter (auto-fix)
ruff check --fix .
ruff format .

# Run type checker (ty)
ty check .

# Run both Ruff and ty
ruff check --fix . && ruff format . && ty check .
```

### Docker Commands
```bash
# Build the Docker image
docker build -t iamdreamingof-generator .

# Run the Docker container locally (requires environment variables)
docker run --env-file .env iamdreamingof-generator
```

### Kubernetes Deployment
```bash
# Apply the CronJob configuration
kubectl apply -f k8s/cronjob.yaml

# Check CronJob status
kubectl get cronjobs -n iamdreamingof

# View recent job runs
kubectl get jobs -n iamdreamingof
```

## Core Architecture

### Main Components

1. **main.py**: Entry point that orchestrates the entire generation process
   - Fetches existing days.json from CDN
   - Generates unique ID for the day
   - Creates challenges for all difficulty levels
   - Uploads results to CDN
   - Updates today.json if generating for current day

2. **words.py**: Word selection logic
   - Easy: 3 random objects
   - Medium: 2 objects + 1 gerund
   - Hard: 1 object + 2 gerunds
   - Dreaming: 1 object + 1 gerund + 1 concept
   - Ensures all 12 words across difficulties are unique

3. **ai.py**: OpenAI API integration
   - `generate_prompt()`: Creates dream descriptions using GPT-4
   - `generate_image()`: Generates images using DALL-E 3

4. **image.py**: Image processing
   - Resizes images to 800x800
   - Converts to both JPEG and WebP formats
   - Returns paths for CDN upload

5. **cdn.py**: DigitalOcean Spaces integration
   - Uploads files to CDN with public-read ACL
   - Reads JSON files from CDN

### Data Flow

1. Word selection from JSON files (objects.json, gerunds.json, concepts.json)
2. Prompt generation via OpenAI GPT-4
3. Image generation via DALL-E 3
4. Image download and processing (resize, format conversion)
5. Upload to CDN (images and JSON data)
6. Update days.json index and today.json if applicable

### Environment Variables Required

- `AI_API_KEY`: OpenAI API key
- `CDN_ACCESS_KEY_ID`: DigitalOcean Spaces access key
- `CDN_SECRET_ACCESS_KEY`: DigitalOcean Spaces secret key
- `LOGTAIL_SOURCE_TOKEN`: Logtail logging token
- `ROLLBAR_ACCESS_TOKEN`: Rollbar error tracking token
- `ROLLBAR_ENVIRONMENT`: Environment name for Rollbar
- `HONEYBADGER_API_KEY`: Honeybadger monitoring API key
- `HONEYBADGER_CHECKIN_ID`: Honeybadger check-in ID

### Error Handling

- Uses Tenacity for retry logic (3 attempts with 2-minute waits)
- Rollbar and Honeybadger for error tracking
- Logtail for centralized logging
- Honeybadger check-in on successful completion