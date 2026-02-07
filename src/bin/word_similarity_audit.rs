//! Command-line audit for semantic overlap in category word lists.
//!
//! The tool embeds all words (deduplicated across categories), computes cosine
//! similarity within categories and across category pairs, then reports word
//! pairs above a threshold.

use anyhow::Result as AnyResult;
use clap::Parser;
use iamdreamingof_generator::ai::{
    EmbeddingService, GeminiEmbeddingClient, OpenAiEmbeddingClient, GEMINI_MAX_BATCH_EMBED_ITEMS,
};
use iamdreamingof_generator::models::AiProvider;
use iamdreamingof_generator::semantic::{
    find_similar_pairs, find_similar_pairs_between, SimilarPair,
};
use iamdreamingof_generator::words::load_word_list;
use iamdreamingof_generator::{Error, Result};
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Parser)]
#[command(name = "word_similarity_audit")]
#[command(about = "Find semantically similar word pairs inside and across category lists")]
struct CliArgs {
    /// Embedding provider to use for similarity calculations.
    #[arg(long, default_value = "gemini", value_parser = parse_ai_provider)]
    provider: AiProvider,

    /// Optional model override for the selected provider.
    #[arg(long)]
    model: Option<String>,

    /// Similarity threshold in [0.0, 1.0].
    #[arg(long, default_value_t = 0.75)]
    threshold: f32,

    /// Number of words per embedding request.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,

    /// Max pairs to include per category section.
    #[arg(long = "max-pairs", default_value_t = 50)]
    max_pairs_per_category: usize,

    /// Directory containing objects/gerunds/concepts JSON files.
    #[arg(long, default_value = "data")]
    data_dir: PathBuf,

    /// Optional path to write a machine-readable JSON report.
    #[arg(long)]
    json_output: Option<PathBuf>,
}

impl CliArgs {
    fn parse_for_app() -> Result<Self> {
        let args = Self::try_parse().map_err(|e| Error::Config(e.to_string()))?;
        args.validate()
    }

    #[cfg(test)]
    fn parse_from_for_test<I, S>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let collected: Vec<String> = args.into_iter().map(Into::into).collect();
        let args = Self::try_parse_from(collected).map_err(|e| Error::Config(e.to_string()))?;
        args.validate()
    }

    fn resolved_model(&self) -> String {
        let model = self
            .model
            .as_deref()
            .unwrap_or_else(|| default_embedding_model(&self.provider));

        // Gemini endpoint URLs are composed as `/models/{model}:...`, so we
        // normalize to a bare model ID to avoid `models/models/...`.
        // OpenAI sends the model as a JSON field and accepts the literal value.
        if self.provider == AiProvider::Gemini {
            model.strip_prefix("models/").unwrap_or(model).to_string()
        } else {
            model.to_string()
        }
    }

    fn validate(self) -> Result<Self> {
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(Error::Config(
                "--threshold must be between 0.0 and 1.0".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(Error::Config("--batch-size must be >= 1".to_string()));
        }
        if self.max_pairs_per_category == 0 {
            return Err(Error::Config("--max-pairs must be >= 1".to_string()));
        }
        if self.provider == AiProvider::Gemini && self.batch_size > GEMINI_MAX_BATCH_EMBED_ITEMS {
            return Err(Error::Config(format!(
                "--batch-size must be <= {} for provider gemini",
                GEMINI_MAX_BATCH_EMBED_ITEMS
            )));
        }
        Ok(self)
    }
}

#[derive(Debug, Serialize)]
struct PairReport {
    /// Total matches found before any top-N truncation.
    flagged_pairs: usize,
    /// Number of pair entries included in `pairs`.
    reported_pairs: usize,
    /// True when `pairs` was truncated to `--max-pairs`.
    truncated: bool,
    pairs: Vec<SimilarPair>,
}

#[derive(Debug, Serialize)]
struct CategoryReport {
    category: String,
    total_words: usize,
    #[serde(flatten)]
    pair_report: PairReport,
}

#[derive(Debug, Serialize)]
struct CrossCategoryReport {
    left_category: String,
    right_category: String,
    #[serde(flatten)]
    pair_report: PairReport,
}

#[derive(Debug, Serialize)]
struct AuditReport {
    provider: String,
    model: String,
    threshold: f32,
    batch_size: usize,
    categories: Vec<CategoryReport>,
    cross_category: Vec<CrossCategoryReport>,
}

#[tokio::main]
async fn main() -> AnyResult<()> {
    run().await.map_err(Into::into)
}

async fn run() -> Result<()> {
    let args = CliArgs::parse_for_app()?;
    let service = build_embedding_service(args.provider, args.resolved_model())?;
    run_with_embedding_service(args, service.as_ref()).await?;
    Ok(())
}

async fn run_with_embedding_service(
    args: CliArgs,
    embedding_service: &dyn EmbeddingService,
) -> Result<AuditReport> {
    let model = args.resolved_model();
    let categories = load_categories(&args.data_dir)?;
    let unique_words = collect_unique_words(&categories);
    let embeddings = embed_all_words(embedding_service, &unique_words, args.batch_size).await?;
    let category_vectors: Vec<Vec<&[f32]>> = categories
        .iter()
        .map(|(_, words)| resolve_vectors(words, &embeddings))
        .collect::<Result<Vec<_>>>()?;

    let mut category_reports = Vec::new();
    for ((name, words), vectors) in categories.iter().zip(category_vectors.iter()) {
        let pairs = find_similar_pairs(words, vectors, args.threshold)?;
        let pair_report =
            build_pair_report(name, args.threshold, args.max_pairs_per_category, pairs);

        category_reports.push(CategoryReport {
            category: name.clone(),
            total_words: words.len(),
            pair_report,
        });
    }

    let mut cross_reports = Vec::new();
    for i in 0..categories.len() {
        for j in (i + 1)..categories.len() {
            let (left_name, left_words) = &categories[i];
            let (right_name, right_words) = &categories[j];
            let left_vectors = &category_vectors[i];
            let right_vectors = &category_vectors[j];

            let pairs = find_similar_pairs_between(
                left_words,
                left_vectors,
                right_words,
                right_vectors,
                args.threshold,
            )?;
            let label = format!("cross:{} vs {}", left_name, right_name);
            let pair_report =
                build_pair_report(&label, args.threshold, args.max_pairs_per_category, pairs);

            cross_reports.push(CrossCategoryReport {
                left_category: left_name.clone(),
                right_category: right_name.clone(),
                pair_report,
            });
        }
    }

    let report = AuditReport {
        provider: args.provider.to_string(),
        model,
        threshold: args.threshold,
        batch_size: args.batch_size,
        categories: category_reports,
        cross_category: cross_reports,
    };

    if let Some(path) = &args.json_output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(&report)?)?;
        println!("Wrote JSON report to {}", path.display());
    }

    Ok(report)
}

/// Parse `--provider` values into the internal provider enum.
fn parse_ai_provider(input: &str) -> std::result::Result<AiProvider, String> {
    input.parse::<AiProvider>().map_err(|e| format!("{}", e))
}

/// Default embedding model per provider.
fn default_embedding_model(provider: &AiProvider) -> &'static str {
    match provider {
        AiProvider::OpenAi => "text-embedding-3-small",
        AiProvider::Gemini => "gemini-embedding-001", // Keep in sync with provider docs.
    }
}

fn canonical_word_key(word: &str) -> String {
    word.to_lowercase()
}

/// Load the three category files used by the game.
fn load_categories(data_dir: &Path) -> Result<Vec<(String, Vec<String>)>> {
    Ok(vec![
        (
            "objects".to_string(),
            load_word_list(&data_dir.join("objects.json"))?,
        ),
        (
            "gerunds".to_string(),
            load_word_list(&data_dir.join("gerunds.json"))?,
        ),
        (
            "concepts".to_string(),
            load_word_list(&data_dir.join("concepts.json"))?,
        ),
    ])
}

/// Build one deduplicated list across all categories so embeddings are fetched
/// once per unique surface form and reused for all overlap checks.
fn collect_unique_words(categories: &[(String, Vec<String>)]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut unique = Vec::new();

    for (_, words) in categories {
        for word in words {
            if seen.insert(canonical_word_key(word)) {
                unique.push(word.clone());
            }
        }
    }

    unique
}

fn resolve_vectors<'a>(
    words: &[String],
    embeddings: &'a HashMap<String, Vec<f32>>,
) -> Result<Vec<&'a [f32]>> {
    // Resolve each word back to its shared embedding vector.
    words
        .iter()
        .map(|word| {
            let key = canonical_word_key(word);
            embeddings
                .get(&key)
                .map(Vec::as_slice)
                .ok_or_else(|| Error::Invariant(format!("Missing embedding for word '{}'", word)))
        })
        .collect()
}

async fn embed_all_words(
    service: &dyn EmbeddingService,
    words: &[String],
    batch_size: usize,
) -> Result<HashMap<String, Vec<f32>>> {
    let mut map = HashMap::with_capacity(words.len());
    let mut expected_dimensions: Option<usize> = None;

    for chunk in words.chunks(batch_size) {
        let chunk_refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
        let vectors = service.embed_texts(&chunk_refs).await?;

        if vectors.len() != chunk.len() {
            return Err(Error::AiProvider(format!(
                "Embedding response length mismatch: requested {}, got {}",
                chunk.len(),
                vectors.len()
            )));
        }

        // EmbeddingService guarantees vectors align with input order.
        for (word, vector) in chunk.iter().zip(vectors.into_iter()) {
            let dims = vector.len();
            if let Some(expected) = expected_dimensions {
                if expected != dims {
                    return Err(Error::AiProvider(format!(
                        "Embedding dimension mismatch for '{}': expected {}, got {}",
                        word, expected, dims
                    )));
                }
            } else {
                expected_dimensions = Some(dims);
            }
            map.insert(canonical_word_key(word), vector);
        }
    }

    Ok(map)
}

fn build_embedding_service(
    provider: AiProvider,
    model: String,
) -> Result<Box<dyn EmbeddingService>> {
    build_embedding_service_with_keys(
        provider,
        model,
        std::env::var("OPENAI_API_KEY").ok(),
        std::env::var("GEMINI_API_KEY").ok(),
    )
}

fn build_embedding_service_with_keys(
    provider: AiProvider,
    model: String,
    openai_key: Option<String>,
    gemini_key: Option<String>,
) -> Result<Box<dyn EmbeddingService>> {
    match provider {
        AiProvider::OpenAi => {
            let api_key = openai_key.ok_or_else(|| {
                Error::Config(
                    "OPENAI_API_KEY environment variable is required for --provider openai"
                        .to_string(),
                )
            })?;
            Ok(Box::new(OpenAiEmbeddingClient::new(api_key, model)))
        }
        AiProvider::Gemini => {
            let api_key = gemini_key.ok_or_else(|| {
                Error::Config(
                    "GEMINI_API_KEY environment variable is required for --provider gemini"
                        .to_string(),
                )
            })?;
            Ok(Box::new(GeminiEmbeddingClient::new(api_key, model)))
        }
    }
}

fn cap_pairs(mut pairs: Vec<SimilarPair>, max_pairs: usize) -> (usize, bool, Vec<SimilarPair>) {
    let flagged_pairs = pairs.len();
    if flagged_pairs > max_pairs {
        pairs.truncate(max_pairs);
        (flagged_pairs, true, pairs)
    } else {
        (flagged_pairs, false, pairs)
    }
}

fn build_pair_report(
    label: &str,
    threshold: f32,
    max_pairs: usize,
    pairs: Vec<SimilarPair>,
) -> PairReport {
    let (flagged_pairs, truncated, pairs) = cap_pairs(pairs, max_pairs);
    print_pair_report(
        label,
        threshold,
        max_pairs,
        flagged_pairs,
        truncated,
        &pairs,
    );

    PairReport {
        flagged_pairs,
        reported_pairs: pairs.len(),
        truncated,
        pairs,
    }
}

fn print_pair_report(
    label: &str,
    threshold: f32,
    max_pairs: usize,
    flagged_pairs: usize,
    truncated: bool,
    pairs: &[SimilarPair],
) {
    for line in
        format_pair_report_lines(label, threshold, max_pairs, flagged_pairs, truncated, pairs)
    {
        println!("{}", line);
    }
}

fn format_pair_report_lines(
    label: &str,
    threshold: f32,
    max_pairs: usize,
    flagged_pairs: usize,
    truncated: bool,
    pairs: &[SimilarPair],
) -> Vec<String> {
    let mut lines = Vec::with_capacity(pairs.len() + 1);
    if truncated {
        lines.push(format!(
            "[{}] {} potential overlaps, showing top {} (threshold: {:.2})",
            label, flagged_pairs, max_pairs, threshold
        ));
    } else {
        lines.push(format!(
            "[{}] {} potential overlaps (threshold: {:.2})",
            label, flagged_pairs, threshold
        ));
    }

    lines.extend(pairs.iter().map(|pair| {
        format!(
            "  {:.3}  {}  <->  {}",
            pair.similarity, pair.left, pair.right
        )
    }));

    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use iamdreamingof_generator::ai::MockEmbeddingClient;
    use tempfile::tempdir;

    #[test]
    fn test_cli_defaults() {
        let args = CliArgs::parse_from_for_test(vec!["word_similarity_audit"]).unwrap();

        assert_eq!(args.provider, AiProvider::Gemini);
        assert_eq!(args.resolved_model(), "gemini-embedding-001");
        assert_eq!(args.threshold, 0.75);
        assert_eq!(args.batch_size, 64);
        assert_eq!(args.max_pairs_per_category, 50);
        assert_eq!(args.data_dir, PathBuf::from("data"));
        assert!(args.json_output.is_none());
    }

    #[test]
    fn test_cli_provider_override_sets_matching_default_model() {
        let args =
            CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--provider", "openai"])
                .unwrap();

        assert_eq!(args.provider, AiProvider::OpenAi);
        assert_eq!(args.resolved_model(), "text-embedding-3-small");
    }

    #[test]
    fn test_cli_rejects_out_of_range_threshold() {
        let err = CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--threshold", "1.1"])
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn test_cli_threshold_boundaries_are_allowed() {
        let zero =
            CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--threshold", "0.0"])
                .unwrap();
        let one = CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--threshold", "1.0"])
            .unwrap();

        assert_eq!(zero.threshold, 0.0);
        assert_eq!(one.threshold, 1.0);
    }

    #[test]
    fn test_cli_rejects_negative_threshold() {
        let err =
            CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--threshold", "-0.1"])
                .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn test_cli_rejects_zero_batch_size() {
        let err = CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--batch-size", "0"])
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn test_cli_rejects_batch_size_above_gemini_limit() {
        let err =
            CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--batch-size", "101"])
                .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn test_cli_allows_large_batch_size_for_openai() {
        let args = CliArgs::parse_from_for_test(vec![
            "word_similarity_audit",
            "--provider",
            "openai",
            "--batch-size",
            "512",
        ])
        .unwrap();
        assert_eq!(args.batch_size, 512);
    }

    #[test]
    fn test_cli_rejects_zero_max_pairs() {
        let err = CliArgs::parse_from_for_test(vec!["word_similarity_audit", "--max-pairs", "0"])
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn test_cli_model_override_wins() {
        let args = CliArgs::parse_from_for_test(vec![
            "word_similarity_audit",
            "--provider",
            "gemini",
            "--model",
            "custom-embed-model",
        ])
        .unwrap();

        assert_eq!(args.resolved_model(), "custom-embed-model");
    }

    #[test]
    fn test_cli_strips_gemini_models_prefix() {
        let args = CliArgs::parse_from_for_test(vec![
            "word_similarity_audit",
            "--provider",
            "gemini",
            "--model",
            "models/gemini-embedding-001",
        ])
        .unwrap();

        assert_eq!(args.resolved_model(), "gemini-embedding-001");
    }

    #[test]
    fn test_collect_unique_words_preserves_first_seen_order() {
        let categories = vec![
            (
                "a".to_string(),
                vec!["clock".to_string(), "apple".to_string()],
            ),
            (
                "b".to_string(),
                vec!["apple".to_string(), "watch".to_string()],
            ),
        ];

        let unique = collect_unique_words(&categories);
        assert_eq!(unique, vec!["clock", "apple", "watch"]);
    }

    #[test]
    fn test_collect_unique_words_is_case_insensitive() {
        let categories = vec![
            (
                "a".to_string(),
                vec!["Apple".to_string(), "clock".to_string()],
            ),
            (
                "b".to_string(),
                vec!["apple".to_string(), "watch".to_string()],
            ),
        ];

        let unique = collect_unique_words(&categories);
        assert_eq!(unique, vec!["Apple", "clock", "watch"]);
    }

    #[test]
    fn test_cap_pairs_reports_total_and_truncation() {
        let pairs = vec![
            SimilarPair {
                left: "a".to_string(),
                right: "b".to_string(),
                similarity: 0.9,
            },
            SimilarPair {
                left: "a".to_string(),
                right: "c".to_string(),
                similarity: 0.8,
            },
            SimilarPair {
                left: "b".to_string(),
                right: "c".to_string(),
                similarity: 0.7,
            },
        ];

        let (flagged_pairs, truncated, reported) = cap_pairs(pairs, 2);
        assert_eq!(flagged_pairs, 3);
        assert!(truncated);
        assert_eq!(reported.len(), 2);
    }

    #[test]
    fn test_cap_pairs_keeps_all_when_under_limit() {
        let pairs = vec![SimilarPair {
            left: "a".to_string(),
            right: "b".to_string(),
            similarity: 0.9,
        }];

        let (flagged_pairs, truncated, reported) = cap_pairs(pairs, 2);
        assert_eq!(flagged_pairs, 1);
        assert!(!truncated);
        assert_eq!(reported.len(), 1);
    }

    #[test]
    fn test_cap_pairs_keeps_all_at_exact_limit() {
        let pairs = vec![
            SimilarPair {
                left: "a".to_string(),
                right: "b".to_string(),
                similarity: 0.9,
            },
            SimilarPair {
                left: "a".to_string(),
                right: "c".to_string(),
                similarity: 0.8,
            },
        ];

        let (flagged_pairs, truncated, reported) = cap_pairs(pairs, 2);
        assert_eq!(flagged_pairs, 2);
        assert!(!truncated);
        assert_eq!(reported.len(), 2);
    }

    #[test]
    fn test_format_pair_report_lines_non_truncated() {
        let pairs = vec![SimilarPair {
            left: "a".to_string(),
            right: "b".to_string(),
            similarity: 0.9,
        }];

        let lines = format_pair_report_lines("objects", 0.75, 50, 1, false, &pairs);
        assert_eq!(lines[0], "[objects] 1 potential overlaps (threshold: 0.75)");
        assert_eq!(lines[1], "  0.900  a  <->  b");
    }

    #[test]
    fn test_format_pair_report_lines_truncated() {
        let lines = format_pair_report_lines("objects", 0.8, 10, 32, true, &[]);
        assert_eq!(
            lines[0],
            "[objects] 32 potential overlaps, showing top 10 (threshold: 0.80)"
        );
    }

    #[test]
    fn test_load_word_list_reads_json_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("objects.json");
        fs::write(&path, "[\"apple\",\"banana\"]").unwrap();

        let list = load_word_list(&path).unwrap();
        assert_eq!(list, vec!["apple", "banana"]);
    }

    #[test]
    fn test_load_categories_reads_all_files() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("objects.json"), "[\"apple\"]").unwrap();
        fs::write(dir.path().join("gerunds.json"), "[\"running\"]").unwrap();
        fs::write(dir.path().join("concepts.json"), "[\"joy\"]").unwrap();

        let categories = load_categories(dir.path()).unwrap();

        assert_eq!(categories.len(), 3);
        assert_eq!(categories[0].0, "objects");
        assert_eq!(categories[0].1, vec!["apple"]);
        assert_eq!(categories[1].0, "gerunds");
        assert_eq!(categories[1].1, vec!["running"]);
        assert_eq!(categories[2].0, "concepts");
        assert_eq!(categories[2].1, vec!["joy"]);
    }

    #[tokio::test]
    async fn test_embed_all_words_batches_and_maps_by_index() {
        let service = MockEmbeddingClient::new()
            .with_embedding_response(vec![vec![1.0, 0.0], vec![0.0, 1.0]])
            .with_embedding_response(vec![vec![0.5, 0.5]]);

        let words = vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()];
        let embeddings = embed_all_words(&service, &words, 2).await.unwrap();

        assert_eq!(service.get_call_count(), 2);
        assert_eq!(embeddings.get("alpha").unwrap(), &vec![1.0, 0.0]);
        assert_eq!(embeddings.get("beta").unwrap(), &vec![0.0, 1.0]);
        assert_eq!(embeddings.get("gamma").unwrap(), &vec![0.5, 0.5]);
    }

    #[tokio::test]
    async fn test_embed_all_words_normalizes_keys_to_lowercase() {
        let service = MockEmbeddingClient::new().with_embedding_response(vec![vec![1.0, 0.0]]);
        let words = vec!["Apple".to_string()];

        let embeddings = embed_all_words(&service, &words, 64).await.unwrap();
        assert!(embeddings.contains_key("apple"));
        assert!(!embeddings.contains_key("Apple"));
    }

    #[tokio::test]
    async fn test_embed_all_words_rejects_length_mismatch() {
        let service = MockEmbeddingClient::new().with_embedding_response(vec![vec![1.0, 0.0]]);

        let words = vec!["alpha".to_string(), "beta".to_string()];
        let err = embed_all_words(&service, &words, 2).await.unwrap_err();

        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[tokio::test]
    async fn test_embed_all_words_empty_input_is_noop() {
        let service = MockEmbeddingClient::new();
        let embeddings = embed_all_words(&service, &[], 64).await.unwrap();

        assert!(embeddings.is_empty());
        assert_eq!(service.get_call_count(), 0);
    }

    #[tokio::test]
    async fn test_embed_all_words_single_batch_when_batch_size_exceeds_word_count() {
        let service =
            MockEmbeddingClient::new().with_embedding_response(vec![vec![1.0], vec![2.0]]);
        let words = vec!["a".to_string(), "b".to_string()];

        let embeddings = embed_all_words(&service, &words, 99).await.unwrap();

        assert_eq!(service.get_call_count(), 1);
        assert_eq!(embeddings.get("a"), Some(&vec![1.0]));
        assert_eq!(embeddings.get("b"), Some(&vec![2.0]));
    }

    #[tokio::test]
    async fn test_embed_all_words_batches_when_batch_size_is_one() {
        let service = MockEmbeddingClient::new()
            .with_embedding_response(vec![vec![1.0]])
            .with_embedding_response(vec![vec![2.0]])
            .with_embedding_response(vec![vec![3.0]]);
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let embeddings = embed_all_words(&service, &words, 1).await.unwrap();

        assert_eq!(service.get_call_count(), 3);
        assert_eq!(embeddings.get("a"), Some(&vec![1.0]));
        assert_eq!(embeddings.get("b"), Some(&vec![2.0]));
        assert_eq!(embeddings.get("c"), Some(&vec![3.0]));
    }

    #[tokio::test]
    async fn test_embed_all_words_rejects_inconsistent_dimensions_across_batches() {
        let service = MockEmbeddingClient::new()
            .with_embedding_response(vec![vec![1.0, 2.0]])
            .with_embedding_response(vec![vec![3.0, 4.0, 5.0]]);
        let words = vec!["a".to_string(), "b".to_string()];

        let err = embed_all_words(&service, &words, 1).await.unwrap_err();
        assert!(matches!(err, Error::AiProvider(_)));
    }

    #[test]
    fn test_resolve_vectors_errors_when_embedding_missing() {
        let words = vec!["alpha".to_string(), "beta".to_string()];
        let mut embeddings = HashMap::new();
        embeddings.insert("alpha".to_string(), vec![1.0, 0.0]);

        let err = resolve_vectors(&words, &embeddings).unwrap_err();
        assert!(matches!(err, Error::Invariant(_)));
    }

    #[test]
    fn test_resolve_vectors_is_case_insensitive() {
        let words = vec!["Apple".to_string(), "apple".to_string()];
        let mut embeddings = HashMap::new();
        embeddings.insert("apple".to_string(), vec![1.0, 0.0]);

        let vectors = resolve_vectors(&words, &embeddings).unwrap();
        assert_eq!(vectors.len(), 2);
        assert_eq!(vectors[0], &[1.0, 0.0]);
        assert_eq!(vectors[1], &[1.0, 0.0]);
    }

    #[test]
    fn test_build_embedding_service_requires_openai_env_var() {
        let result = build_embedding_service_with_keys(
            AiProvider::OpenAi,
            "text-embedding-3-small".to_string(),
            None,
            Some("test-gemini-key".to_string()),
        );
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_build_embedding_service_requires_gemini_env_var() {
        let result = build_embedding_service_with_keys(
            AiProvider::Gemini,
            "gemini-embedding-001".to_string(),
            Some("test-openai-key".to_string()),
            None,
        );
        assert!(matches!(result, Err(Error::Config(_))));
    }

    #[test]
    fn test_build_embedding_service_constructs_clients_when_env_present() {
        assert!(build_embedding_service_with_keys(
            AiProvider::OpenAi,
            "text-embedding-3-small".to_string(),
            Some("test-openai-key".to_string()),
            Some("test-gemini-key".to_string()),
        )
        .is_ok());
        assert!(build_embedding_service_with_keys(
            AiProvider::Gemini,
            "gemini-embedding-001".to_string(),
            Some("test-openai-key".to_string()),
            Some("test-gemini-key".to_string()),
        )
        .is_ok());
    }

    #[tokio::test]
    async fn test_run_with_embedding_service_builds_and_writes_report() {
        let dir = tempdir().unwrap();
        let data_dir = dir.path().join("data");
        fs::create_dir_all(&data_dir).unwrap();
        fs::write(data_dir.join("objects.json"), "[\"clock\",\"watch\"]").unwrap();
        fs::write(data_dir.join("gerunds.json"), "[\"running\",\"jogging\"]").unwrap();
        fs::write(data_dir.join("concepts.json"), "[\"time\"]").unwrap();
        let output_path = dir.path().join("report.json");

        let args = CliArgs {
            provider: AiProvider::Gemini,
            model: None,
            threshold: 0.75,
            batch_size: 64,
            max_pairs_per_category: 50,
            data_dir,
            json_output: Some(output_path.clone()),
        };

        let service = MockEmbeddingClient::new().with_embedding_response(vec![
            vec![1.0, 0.0],   // clock
            vec![0.99, 0.1],  // watch
            vec![0.0, 1.0],   // running
            vec![0.01, 0.99], // jogging
            vec![0.95, 0.05], // time
        ]);

        let report = run_with_embedding_service(args, &service).await.unwrap();

        assert_eq!(service.get_call_count(), 1);
        assert_eq!(report.provider, "gemini");
        assert_eq!(report.model, "gemini-embedding-001");
        assert_eq!(report.categories.len(), 3);
        assert_eq!(report.cross_category.len(), 3);

        let objects = report
            .categories
            .iter()
            .find(|category| category.category == "objects")
            .unwrap();
        assert_eq!(objects.pair_report.flagged_pairs, 1);

        let objects_vs_concepts = report
            .cross_category
            .iter()
            .find(|entry| entry.left_category == "objects" && entry.right_category == "concepts")
            .unwrap();
        assert_eq!(objects_vs_concepts.pair_report.flagged_pairs, 2);

        let written = fs::read_to_string(&output_path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&written).unwrap();
        assert_eq!(json["provider"], "gemini");
        assert_eq!(json["categories"].as_array().unwrap().len(), 3);
    }
}
