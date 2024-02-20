import json
import logging
import random

from models import Difficulty, Word, WordsForDay


def import_json_wordlist(filename: str) -> list[str]:
    with open(filename, "r") as file:
        return json.loads(file.read())


def generate_word_list(difficulty: Difficulty) -> list[Word]:
    objects = import_json_wordlist("objects.json")
    gerunds = import_json_wordlist("gerunds.json")
    concepts = import_json_wordlist("concepts.json")
    # If difficulty is Easy, return three random objects, tagged as objects
    if difficulty == Difficulty.EASY:
        words = random.choices(objects, k=3)
        return [Word(word=word, type="object") for word in words]

    # If difficulty is Medium, return two random objects and one random gerund
    elif difficulty == Difficulty.MEDIUM:
        objects = [
            Word(word=word, type="object") for word in random.choices(objects, k=2)
        ]
        gerund = [Word(word=random.choice(gerunds), type="gerund")]
        return objects + gerund

    # If difficult is Hard, return one random object and two gerunds
    elif difficulty == Difficulty.HARD:
        single_object = Word(word=random.choice(objects), type="object")
        gerunds = [
            Word(word=word, type="gerund") for word in random.choices(gerunds, k=2)
        ]
        return [single_object] + gerunds

    # If difficulty is Dreaming, return one random object, one random gerund, and one random concept
    elif difficulty == Difficulty.DREAMING:
        single_object = Word(word=random.choice(objects), type="object")
        single_gerund = Word(word=random.choice(gerunds), type="gerund")
        single_concept = Word(word=random.choice(concepts), type="concept")
        return [single_object, single_gerund, single_concept]


def get_total_word_count(words: list[Word]) -> int:
    return len(set([word.word for word in words]))


def generate_words_for_day(day: str) -> WordsForDay:
    easy = generate_word_list(Difficulty.EASY)
    medium = generate_word_list(Difficulty.MEDIUM)
    hard = generate_word_list(Difficulty.HARD)
    dreaming = generate_word_list(Difficulty.DREAMING)
    all_words = easy + medium + hard + dreaming

    while get_total_word_count(all_words) < 12:
        logging.info("Regenerating words list as we had non-unique words")

        easy = generate_word_list(Difficulty.EASY)
        medium = generate_word_list(Difficulty.MEDIUM)
        hard = generate_word_list(Difficulty.HARD)
        dreaming = generate_word_list(Difficulty.DREAMING)
        all_words = easy + medium + hard + dreaming

    return WordsForDay(day=day, easy=easy, medium=medium, hard=hard, dreaming=dreaming)
