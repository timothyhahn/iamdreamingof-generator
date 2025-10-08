//! Word selection and combination logic
//!
//! Manages selection of random words from categorized lists to create
//! challenge sets with varying difficulty levels.

use crate::models::{Word, WordType};
use crate::Result;
use rand::prelude::*;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct WordList(Vec<String>);

pub struct WordSelector {
    objects: Vec<String>,
    gerunds: Vec<String>,
    concepts: Vec<String>,
}

impl WordSelector {
    pub fn from_files(data_dir: &Path) -> Result<Self> {
        let objects_path = data_dir.join("objects.json");
        let gerunds_path = data_dir.join("gerunds.json");
        let concepts_path = data_dir.join("concepts.json");

        let objects: WordList = serde_json::from_str(&fs::read_to_string(objects_path)?)?;
        let gerunds: WordList = serde_json::from_str(&fs::read_to_string(gerunds_path)?)?;
        let concepts: WordList = serde_json::from_str(&fs::read_to_string(concepts_path)?)?;

        Ok(Self {
            objects: objects.0,
            gerunds: gerunds.0,
            concepts: concepts.0,
        })
    }

    // Constructor for testing with provided word lists
    #[cfg(test)]
    pub fn new(objects: Vec<String>, gerunds: Vec<String>, concepts: Vec<String>) -> Self {
        Self {
            objects,
            gerunds,
            concepts,
        }
    }

    pub fn select_words(&self) -> Result<WordSets> {
        const MAX_ATTEMPTS: usize = 100;

        for _ in 0..MAX_ATTEMPTS {
            let sets = self.generate_word_sets()?;
            if self.all_words_unique(&sets) {
                return Ok(sets);
            }
        }

        Err(crate::Error::WordSelection(
            "Could not generate unique words after 100 attempts".to_string(),
        ))
    }

    fn generate_word_sets(&self) -> Result<WordSets> {
        let mut rng = thread_rng();

        // Easy: 3 objects
        let easy = self.select_random_words(&self.objects, 3, WordType::Object, &mut rng)?;

        // Medium: 2 objects, 1 gerund
        let medium = [
            self.select_random_words(&self.objects, 2, WordType::Object, &mut rng)?,
            self.select_random_words(&self.gerunds, 1, WordType::Gerund, &mut rng)?,
        ]
        .concat();

        // Hard: 1 object, 2 gerunds
        let hard = [
            self.select_random_words(&self.objects, 1, WordType::Object, &mut rng)?,
            self.select_random_words(&self.gerunds, 2, WordType::Gerund, &mut rng)?,
        ]
        .concat();

        // Dreaming: 1 object, 1 gerund, 1 concept
        let dreaming = [
            self.select_random_words(&self.objects, 1, WordType::Object, &mut rng)?,
            self.select_random_words(&self.gerunds, 1, WordType::Gerund, &mut rng)?,
            self.select_random_words(&self.concepts, 1, WordType::Concept, &mut rng)?,
        ]
        .concat();

        Ok(WordSets {
            easy,
            medium,
            hard,
            dreaming,
        })
    }

    fn select_random_words(
        &self,
        word_list: &[String],
        count: usize,
        word_type: WordType,
        rng: &mut impl Rng,
    ) -> Result<Vec<Word>> {
        if word_list.len() < count {
            return Err(crate::Error::WordSelection(format!(
                "Not enough words in {:?} list",
                word_type
            )));
        }

        let selected = word_list.choose_multiple(rng, count);
        Ok(selected
            .map(|word| Word {
                word: word.clone(),
                word_type: word_type.clone(),
            })
            .collect())
    }

    fn all_words_unique(&self, sets: &WordSets) -> bool {
        let mut seen = HashSet::new();

        for word in sets.all_words() {
            if !seen.insert(&word.word) {
                return false;
            }
        }

        true
    }
}

pub struct WordSets {
    pub easy: Vec<Word>,
    pub medium: Vec<Word>,
    pub hard: Vec<Word>,
    pub dreaming: Vec<Word>,
}

impl WordSets {
    fn all_words(&self) -> impl Iterator<Item = &Word> {
        self.easy
            .iter()
            .chain(self.medium.iter())
            .chain(self.hard.iter())
            .chain(self.dreaming.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn create_test_selector() -> WordSelector {
        // Load actual word files for testing to ensure enough unique words
        WordSelector::from_files(Path::new("data")).expect("Failed to load word files for testing")
    }

    #[test]
    fn test_word_selection_difficulty_counts() {
        let selector = create_test_selector();
        let word_sets = selector.select_words().unwrap();

        // Easy: 3 objects
        assert_eq!(word_sets.easy.len(), 3);
        assert!(word_sets
            .easy
            .iter()
            .all(|w| matches!(w.word_type, WordType::Object)));

        // Medium: 2 objects, 1 gerund
        assert_eq!(word_sets.medium.len(), 3);
        let medium_objects = word_sets
            .medium
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Object))
            .count();
        let medium_gerunds = word_sets
            .medium
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Gerund))
            .count();
        assert_eq!(medium_objects, 2);
        assert_eq!(medium_gerunds, 1);

        // Hard: 1 object, 2 gerunds
        assert_eq!(word_sets.hard.len(), 3);
        let hard_objects = word_sets
            .hard
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Object))
            .count();
        let hard_gerunds = word_sets
            .hard
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Gerund))
            .count();
        assert_eq!(hard_objects, 1);
        assert_eq!(hard_gerunds, 2);

        // Dreaming: 1 object, 1 gerund, 1 concept
        assert_eq!(word_sets.dreaming.len(), 3);
        let dream_objects = word_sets
            .dreaming
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Object))
            .count();
        let dream_gerunds = word_sets
            .dreaming
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Gerund))
            .count();
        let dream_concepts = word_sets
            .dreaming
            .iter()
            .filter(|w| matches!(w.word_type, WordType::Concept))
            .count();
        assert_eq!(dream_objects, 1);
        assert_eq!(dream_gerunds, 1);
        assert_eq!(dream_concepts, 1);
    }

    #[test]
    fn test_all_words_unique() {
        let selector = create_test_selector();

        // Run multiple times to ensure randomness doesn't break uniqueness
        for _ in 0..10 {
            let word_sets = selector.select_words().unwrap();
            let mut seen = HashSet::new();

            for word in word_sets.all_words() {
                assert!(
                    seen.insert(&word.word),
                    "Duplicate word found: {}",
                    word.word
                );
            }
        }
    }
}
