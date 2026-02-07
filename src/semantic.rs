//! Semantic similarity helpers used by the word-audit tooling.

use crate::{Error, Result};
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct SimilarPair {
    /// Left-side word in the compared pair.
    pub left: String,
    /// Right-side word in the compared pair.
    pub right: String,
    /// Cosine similarity score in [-1.0, 1.0] for well-formed finite vectors.
    pub similarity: f32,
}

/// Compute cosine similarity between two embedding vectors.
///
/// Returns `None` when vectors have different lengths, are empty, contain
/// non-finite values, or either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }

    // Dot product and both norms are computed in one pass for cache-friendly
    // linear runtime over the vector length.
    let (dot, norm_a_sq, norm_b_sq) =
        a.iter()
            .zip(b.iter())
            .fold((0.0f64, 0.0f64, 0.0f64), |(dot, na_sq, nb_sq), (x, y)| {
                let x = *x as f64;
                let y = *y as f64;
                (dot + (x * y), na_sq + (x * x), nb_sq + (y * y))
            });

    if !dot.is_finite() || !norm_a_sq.is_finite() || !norm_b_sq.is_finite() {
        return None;
    }

    if norm_a_sq == 0.0 || norm_b_sq == 0.0 {
        return None;
    }

    Some((dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt())) as f32)
}

/// Shared pair-collection engine for both within-group and cross-group modes.
///
/// When `same_group` is true, comparisons start at `i + 1` to avoid duplicate
/// and self-pairs; otherwise every left item is compared to every right item.
fn collect_pairs(
    left_words: &[String],
    left_embeddings: &[impl AsRef<[f32]>],
    right_words: &[String],
    right_embeddings: &[impl AsRef<[f32]>],
    threshold: f32,
    same_group: bool,
) -> Vec<SimilarPair> {
    let mut out = Vec::new();

    for i in 0..left_words.len() {
        let j_start = if same_group { i + 1 } else { 0 };
        for j in j_start..right_words.len() {
            let Some(similarity) =
                cosine_similarity(left_embeddings[i].as_ref(), right_embeddings[j].as_ref())
            else {
                continue;
            };
            if similarity >= threshold {
                out.push(SimilarPair {
                    left: left_words[i].clone(),
                    right: right_words[j].clone(),
                    similarity,
                });
            }
        }
    }

    out.sort_by(|a, b| b.similarity.total_cmp(&a.similarity));
    out
}

/// Find all unique word pairs with cosine similarity greater than or equal to
/// `threshold`, sorted from highest to lowest similarity.
///
/// Returns an error when `words` and `embeddings` are not the same length.
/// Returns an empty vector when fewer than two words are provided.
///
/// Pair enumeration is O(n^2) over the number of input words.
/// Pairs where cosine similarity cannot be computed (e.g. zero-magnitude
/// vectors, ragged embedding dimensions, or non-finite scores) are skipped.
pub fn find_similar_pairs(
    words: &[String],
    embeddings: &[impl AsRef<[f32]>],
    threshold: f32,
) -> Result<Vec<SimilarPair>> {
    if words.len() != embeddings.len() {
        return Err(Error::InvalidInput(format!(
            "words/embeddings length mismatch: words={}, embeddings={}",
            words.len(),
            embeddings.len()
        )));
    }

    let out = collect_pairs(words, embeddings, words, embeddings, threshold, true);
    Ok(out)
}

/// Find similar pairs across two different word groups.
///
/// Returns an error when either `words`/`embeddings` side has a length mismatch.
/// Pairs where cosine similarity cannot be computed or is non-finite are skipped.
pub fn find_similar_pairs_between(
    left_words: &[String],
    left_embeddings: &[impl AsRef<[f32]>],
    right_words: &[String],
    right_embeddings: &[impl AsRef<[f32]>],
    threshold: f32,
) -> Result<Vec<SimilarPair>> {
    if left_words.len() != left_embeddings.len() {
        return Err(Error::InvalidInput(format!(
            "left words/embeddings length mismatch: words={}, embeddings={}",
            left_words.len(),
            left_embeddings.len()
        )));
    }
    if right_words.len() != right_embeddings.len() {
        return Err(Error::InvalidInput(format!(
            "right words/embeddings length mismatch: words={}, embeddings={}",
            right_words.len(),
            right_embeddings.len()
        )));
    }

    let out = collect_pairs(
        left_words,
        left_embeddings,
        right_words,
        right_embeddings,
        threshold,
        false,
    );
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_basic() {
        let same = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        let orth = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();

        assert!((same - 1.0).abs() < 1e-6);
        assert!(orth.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_negative_correlation() {
        let anti = cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
        assert!((anti + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_rejects_mismatched_lengths() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), None);
    }

    #[test]
    fn test_cosine_similarity_rejects_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), None);
    }

    #[test]
    fn test_cosine_similarity_rejects_zero_norm_vectors() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None);
        assert_eq!(cosine_similarity(&[1.0, 0.0], &[0.0, 0.0]), None);
    }

    #[test]
    fn test_cosine_similarity_nan_input_returns_none() {
        assert_eq!(cosine_similarity(&[f32::NAN, 0.0], &[1.0, 0.0]), None);
    }

    #[test]
    fn test_cosine_similarity_infinite_input_returns_none() {
        assert_eq!(cosine_similarity(&[f32::INFINITY, 0.0], &[1.0, 0.0]), None);
    }

    #[test]
    fn test_cosine_similarity_handles_large_finite_values() {
        let sim = cosine_similarity(&[f32::MAX, 0.0], &[f32::MAX, 0.0]).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_find_similar_pairs_threshold() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let embeddings = vec![vec![1.0, 0.0], vec![0.99, 0.01], vec![0.0, 1.0]];

        let pairs = find_similar_pairs(&words, &embeddings, 0.9).unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, "a");
        assert_eq!(pairs[0].right, "b");
    }

    #[test]
    fn test_find_similar_pairs_includes_exact_word_matches() {
        let words = vec![
            "running".to_string(),
            "Running".to_string(),
            "jogging".to_string(),
        ];
        let embeddings = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![0.99, 0.01]];

        let pairs = find_similar_pairs(&words, &embeddings, 0.9).unwrap();

        assert_eq!(pairs.len(), 3);
        assert!(pairs
            .iter()
            .any(|p| p.left == "running" && p.right == "Running"));
        assert!(pairs
            .iter()
            .any(|p| p.left == "running" && p.right == "jogging"));
        assert!(pairs
            .iter()
            .any(|p| p.left == "Running" && p.right == "jogging"));
    }

    #[test]
    fn test_find_similar_pairs_rejects_mismatched_lengths() {
        let words = vec!["a".to_string(), "b".to_string()];
        let embeddings = vec![vec![1.0, 0.0]];

        let err = find_similar_pairs(&words, &embeddings, 0.0).unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn test_find_similar_pairs_empty_input() {
        let words: Vec<String> = Vec::new();
        let embeddings: Vec<Vec<f32>> = Vec::new();
        let pairs = find_similar_pairs(&words, &embeddings, 0.0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_similar_pairs_single_word() {
        let words = vec!["solo".to_string()];
        let embeddings = vec![vec![1.0, 0.0]];
        let pairs = find_similar_pairs(&words, &embeddings, 0.0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_similar_pairs_threshold_boundary_is_inclusive() {
        let words = vec!["a".to_string(), "b".to_string()];
        let embeddings = vec![vec![1.0, 0.0], vec![0.8, 0.6]];

        let pairs = find_similar_pairs(&words, &embeddings, 0.8).unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, "a");
        assert_eq!(pairs[0].right, "b");
        assert!((pairs[0].similarity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_find_similar_pairs_sorted_descending() {
        let words = vec![
            "a".to_string(),
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
        ];
        let embeddings = vec![
            vec![1.0, 0.0],
            vec![0.99, 0.1],
            vec![0.8, 0.6],
            vec![0.0, 1.0],
        ];

        let pairs = find_similar_pairs(&words, &embeddings, 0.5).unwrap();

        assert_eq!(pairs.len(), 4);
        for idx in 1..pairs.len() {
            assert!(pairs[idx - 1].similarity >= pairs[idx].similarity);
        }
    }

    #[test]
    fn test_find_similar_pairs_skips_ragged_dimensions() {
        let words = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let embeddings = vec![vec![1.0, 0.0], vec![0.9], vec![0.8, 0.6]];

        let pairs = find_similar_pairs(&words, &embeddings, 0.0).unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, "a");
        assert_eq!(pairs[0].right, "c");
    }

    #[test]
    fn test_find_similar_pairs_skips_non_finite_similarity() {
        let words = vec!["a".to_string(), "b".to_string()];
        let embeddings = vec![vec![f32::NAN, 0.0], vec![1.0, 0.0]];

        let pairs = find_similar_pairs(&words, &embeddings, -1.0).unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_similar_pairs_between_basic() {
        let left_words = vec!["watch".to_string(), "apple".to_string()];
        let left_embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let right_words = vec!["clock".to_string(), "pear".to_string()];
        let right_embeddings = vec![vec![0.99, 0.1], vec![0.1, 0.9]];

        let pairs = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            0.8,
        )
        .unwrap();

        assert_eq!(pairs.len(), 2);
        assert!(pairs
            .iter()
            .any(|p| p.left == "watch" && p.right == "clock"));
        assert!(pairs.iter().any(|p| p.left == "apple" && p.right == "pear"));
    }

    #[test]
    fn test_find_similar_pairs_between_rejects_left_length_mismatch() {
        let left_words = vec!["a".to_string(), "b".to_string()];
        let left_embeddings = vec![vec![1.0, 0.0]];
        let right_words = vec!["c".to_string()];
        let right_embeddings = vec![vec![1.0, 0.0]];

        let err = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert!(err
            .to_string()
            .contains("left words/embeddings length mismatch"));
    }

    #[test]
    fn test_find_similar_pairs_between_rejects_right_length_mismatch() {
        let left_words = vec!["a".to_string()];
        let left_embeddings = vec![vec![1.0, 0.0]];
        let right_words = vec!["c".to_string(), "d".to_string()];
        let right_embeddings = vec![vec![1.0, 0.0]];

        let err = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert!(err
            .to_string()
            .contains("right words/embeddings length mismatch"));
    }

    #[test]
    fn test_find_similar_pairs_between_empty_input() {
        let left_words: Vec<String> = Vec::new();
        let left_embeddings: Vec<Vec<f32>> = Vec::new();
        let right_words: Vec<String> = Vec::new();
        let right_embeddings: Vec<Vec<f32>> = Vec::new();
        let pairs = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            0.0,
        )
        .unwrap();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_find_similar_pairs_between_skips_ragged_dimensions() {
        let left_words = vec!["a".to_string(), "b".to_string()];
        let left_embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let right_words = vec!["c".to_string(), "d".to_string()];
        let right_embeddings = vec![vec![1.0], vec![0.0, 1.0]];

        let pairs = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            0.1,
        )
        .unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, "b");
        assert_eq!(pairs[0].right, "d");
    }

    #[test]
    fn test_find_similar_pairs_between_threshold_boundary_is_inclusive() {
        let left_words = vec!["a".to_string()];
        let left_embeddings = vec![vec![1.0, 0.0]];
        let right_words = vec!["b".to_string()];
        let right_embeddings = vec![vec![0.8, 0.6]];

        let pairs = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            0.8,
        )
        .unwrap();

        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].left, "a");
        assert_eq!(pairs[0].right, "b");
        assert!((pairs[0].similarity - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_find_similar_pairs_between_skips_non_finite_similarity() {
        let left_words = vec!["a".to_string()];
        let left_embeddings = vec![vec![f32::NAN, 0.0]];
        let right_words = vec!["b".to_string()];
        let right_embeddings = vec![vec![1.0, 0.0]];

        let pairs = find_similar_pairs_between(
            &left_words,
            &left_embeddings,
            &right_words,
            &right_embeddings,
            -1.0,
        )
        .unwrap();
        assert!(pairs.is_empty());
    }
}
