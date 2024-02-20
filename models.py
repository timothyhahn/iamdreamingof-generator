from enum import Enum

from pydantic import BaseModel


class Difficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3
    DREAMING = 4


class Word(BaseModel):
    word: str
    type: str


class Challenge(BaseModel):
    words: list[Word]
    image_path: str
    image_url_jpg: str
    image_url_webp: str
    prompt: str


class Challenges(BaseModel):
    easy: Challenge
    medium: Challenge
    hard: Challenge
    dreaming: Challenge


class WordsForDay(BaseModel):
    day: str
    easy: list[Word]
    medium: list[Word]
    hard: list[Word]
    dreaming: list[Word]


class Day(BaseModel):
    date: str
    id: int
    challenges: Challenges


class DateEntry(BaseModel):
    date: str
    id: int


class Days(BaseModel):
    days: list[DateEntry]
