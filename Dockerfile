FROM python:3.11-buster
LABEL authors="tmhhn"

# Install image processing libraries
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends openssl ca-certificates curl build-essential clang pkg-config libjpeg-turbo-progs libpng-dev libjpeg-dev libjpeg62-turbo-dev libheif-dev libmagickwand-dev --no-install-recommends \
    # Cleanup
    && apt-get autoremove -y \
    && apt-get clean -y


# Actual Python stuff
RUN pip install poetry==1.7.0

WORKDIR /app

COPY pyproject.toml poetry.lock ./
COPY *.py *.json ./

# Poetry won't run without a README, let's create a fake one
RUN touch README.md
RUN poetry install --no-root

CMD ["poetry", "run", "python", "main.py"]
