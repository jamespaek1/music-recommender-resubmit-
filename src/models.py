"""
Data models for the Music Discovery Agent.
Song and UserProfile are the core data structures used throughout the system.
"""

from dataclasses import dataclass, field
from typing import Optional
import csv
import os


@dataclass
class Song:
    """Represents a single song in the catalog with its audio features."""
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: int
    valence: float
    danceability: float
    acousticness: float

    def __str__(self):
        return f'"{self.title}" by {self.artist} [{self.genre}/{self.mood}]'


@dataclass
class UserProfile:
    """Represents a user's music taste preferences, parsed from natural language."""
    favorite_genre: str
    favorite_mood: str
    energy_target: float = 0.5
    prefers_acoustic: bool = False

    def to_dict(self) -> dict:
        return {
            "favorite_genre": self.favorite_genre,
            "favorite_mood": self.favorite_mood,
            "energy_target": self.energy_target,
            "prefers_acoustic": self.prefers_acoustic,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(
            favorite_genre=data.get("favorite_genre", "pop"),
            favorite_mood=data.get("favorite_mood", "happy"),
            energy_target=float(data.get("energy_target", 0.5)),
            prefers_acoustic=bool(data.get("prefers_acoustic", False)),
        )


@dataclass
class Recommendation:
    """A single recommendation with its score breakdown."""
    song: Song
    total_score: float
    breakdown: dict = field(default_factory=dict)
    explanation: str = ""


@dataclass
class AgentResult:
    """The full output of one agent run, including intermediate steps."""
    user_input: str
    parsed_profile: Optional[UserProfile]
    recommendations: list  # list of Recommendation
    natural_response: str
    confidence_score: float
    refinement_attempts: int = 0
    retrieved_context: str = ""  # RAG context used to augment response
    log_entries: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


def load_catalog(csv_path: str = None) -> list:
    """Load the song catalog from a CSV file."""
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")
    csv_path = os.path.abspath(csv_path)

    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            songs.append(Song(
                title=row["title"].strip(),
                artist=row["artist"].strip(),
                genre=row["genre"].strip().lower(),
                mood=row["mood"].strip().lower(),
                energy=float(row["energy"]),
                tempo_bpm=int(row["tempo_bpm"]),
                valence=float(row["valence"]),
                danceability=float(row["danceability"]),
                acousticness=float(row["acousticness"]),
            ))
    return songs


# Valid genres and moods in the catalog (used for validation)
VALID_GENRES = {"pop", "edm", "lofi", "rock", "jazz", "folk", "metal"}
VALID_MOODS = {"happy", "sad", "energetic", "chill"}
