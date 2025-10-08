//! Error handling and custom error types
//!
//! Provides unified error handling across the application using thiserror.

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("HTTP request error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("S3/CDN error: {0}")]
    S3(String),

    #[error("Image processing error: {0}")]
    Image(#[from] image::ImageError),

    #[error("OpenAI API error: {0}")]
    OpenAI(String),

    #[error("Word selection error: {0}")]
    WordSelection(String),

    #[error("Environment variable error: {0}")]
    EnvVar(#[from] dotenvy::Error),

    #[error("UUID error: {0}")]
    Uuid(#[from] uuid::Error),

    #[error("Date parsing error: {0}")]
    DateParse(String),

    #[error("Generic error: {0}")]
    Generic(String),
}

pub type Result<T> = std::result::Result<T, Error>;
