//! Generator for iamdreamingof.com - creates daily AI-generated dream challenges
//!
//! This application generates dream-like images and prompts based on word combinations
//! with varying difficulty levels, then uploads them to a CDN for web consumption.

pub mod ai;
pub mod cdn;
pub mod error;
pub mod image;
pub mod models;
pub mod prompts;
pub mod words;

pub use error::{Error, Result};
