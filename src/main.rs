use anyhow::Result;
use chrono::NaiveDate;
use clap::Parser;
use iamdreamingof_generator::app::App;
use tracing::{error, info};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Parser)]
#[command(name = "iamdreamingof-generator")]
#[command(about = "Generate daily dream challenges")]
struct CliArgs {
    /// Optional target date in YYYY-MM-DD format.
    #[arg(value_name = "DATE", value_parser = parse_date_arg)]
    target_date: Option<NaiveDate>,
}

fn parse_date_arg(input: &str) -> std::result::Result<NaiveDate, String> {
    NaiveDate::parse_from_str(input, "%Y-%m-%d")
        .map_err(|_| format!("Invalid date '{}'. Expected format: YYYY-MM-DD", input))
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "iamdreamingof_generator=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting iamdreamingof-generator");

    let args = CliArgs::parse();

    match App::new().await {
        Ok(app) => match app.run(args.target_date).await {
            Ok(_) => {
                info!("Generation completed successfully");
                Ok(())
            }
            Err(e) => {
                error!("Generation failed: {}", e);
                std::process::exit(1);
            }
        },
        Err(e) => {
            error!("Failed to initialize application: {}", e);
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_date_arg;

    #[test]
    fn test_parse_date_arg_valid() {
        let parsed = parse_date_arg("2026-02-07").unwrap();
        assert_eq!(parsed.to_string(), "2026-02-07");
    }

    #[test]
    fn test_parse_date_arg_invalid() {
        let err = parse_date_arg("02/07/2026").unwrap_err();
        assert!(err.contains("YYYY-MM-DD"));
    }
}
