FROM python:3.11-trixie
LABEL authors="tmhhn"

# Install image processing libraries
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends openssl ca-certificates curl build-essential clang pkg-config libjpeg-turbo-progs libpng-dev libjpeg-dev libjpeg62-turbo-dev libheif-dev libmagickwand-dev --no-install-recommends \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean -y

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./
COPY *.py *.json ./

RUN touch README.md # lol

# Install dependencies using uv sync
RUN uv sync --frozen --no-dev

# Use uv to run the app
CMD ["/usr/local/bin/uv", "run", "python", "main.py"]
