"""
Unit tests for the recommender engine.

Tests the core scoring algorithm to ensure correctness and consistency.
"""

import pytest
from src.models import Song, UserProfile
from src.recommender import score_song, recommend, get_max_possible_score


# ── Test fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def sample_song():
    return Song(
        title="Test Song",
        artist="Test Artist",
        genre="pop",
        mood="happy",
        energy=0.7,
        tempo_bpm=120,
        valence=0.8,
        danceability=0.75,
        acousticness=0.1,
    )


@pytest.fixture
def pop_profile():
    return UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        energy_target=0.7,
        prefers_acoustic=False,
    )


@pytest.fixture
def acoustic_profile():
    return UserProfile(
        favorite_genre="folk",
        favorite_mood="chill",
        energy_target=0.3,
        prefers_acoustic=True,
    )


@pytest.fixture
def small_catalog():
    return [
        Song("Song A", "Artist A", "pop", "happy", 0.8, 120, 0.9, 0.85, 0.1),
        Song("Song B", "Artist B", "rock", "sad", 0.6, 100, 0.2, 0.3, 0.2),
        Song("Song C", "Artist C", "jazz", "chill", 0.3, 90, 0.5, 0.4, 0.8),
        Song("Song D", "Artist D", "edm", "energetic", 0.95, 140, 0.7, 0.9, 0.05),
        Song("Song E", "Artist E", "folk", "happy", 0.4, 100, 0.7, 0.5, 0.9),
    ]


# ── Scoring tests ──────────────────────────────────────────────────────

def test_genre_match_awards_points(sample_song, pop_profile):
    """A song matching the user's genre should get +2.0 points."""
    rec = score_song(sample_song, pop_profile)
    assert rec.breakdown["genre_match"] == 2.0


def test_genre_mismatch_zero(sample_song, acoustic_profile):
    """A song NOT matching the user's genre should get 0 for genre."""
    rec = score_song(sample_song, acoustic_profile)
    assert rec.breakdown["genre_match"] == 0.0


def test_mood_match_awards_points(sample_song, pop_profile):
    """A song matching the user's mood should get +1.0 points."""
    rec = score_song(sample_song, pop_profile)
    assert rec.breakdown["mood_match"] == 1.0


def test_energy_similarity_perfect_match(sample_song, pop_profile):
    """When song energy exactly matches target, energy similarity should be 1.5."""
    # song energy = 0.7, profile target = 0.7 → perfect match
    rec = score_song(sample_song, pop_profile)
    assert rec.breakdown["energy_similarity"] == 1.5


def test_energy_similarity_far_apart():
    """Distant energy levels should produce low energy similarity scores."""
    song = Song("X", "Y", "pop", "happy", 0.1, 100, 0.5, 0.5, 0.5)
    profile = UserProfile("pop", "happy", energy_target=0.9)
    rec = score_song(song, profile)
    assert rec.breakdown["energy_similarity"] < 0.5


def test_valence_bonus_only_for_happy_mood(sample_song):
    """Valence bonus should only apply when mood preference is 'happy'."""
    happy_profile = UserProfile("pop", "happy")
    sad_profile = UserProfile("pop", "sad")

    rec_happy = score_song(sample_song, happy_profile)
    rec_sad = score_song(sample_song, sad_profile)

    assert rec_happy.breakdown["valence_bonus"] > 0
    assert rec_sad.breakdown["valence_bonus"] == 0


def test_danceability_bonus_high_energy():
    """Danceability bonus should apply when energy target >= 0.7."""
    song = Song("X", "Y", "pop", "happy", 0.8, 120, 0.5, 0.9, 0.1)
    high_energy = UserProfile("pop", "happy", energy_target=0.8)
    low_energy = UserProfile("pop", "happy", energy_target=0.3)

    rec_high = score_song(song, high_energy)
    rec_low = score_song(song, low_energy)

    assert rec_high.breakdown["danceability_bonus"] > 0
    assert rec_low.breakdown["danceability_bonus"] == 0


def test_acoustic_bonus(sample_song):
    """Acoustic bonus should apply only when prefers_acoustic is True."""
    acoustic = UserProfile("pop", "happy", prefers_acoustic=True)
    not_acoustic = UserProfile("pop", "happy", prefers_acoustic=False)

    rec_yes = score_song(sample_song, acoustic)
    rec_no = score_song(sample_song, not_acoustic)

    assert rec_yes.breakdown["acoustic_bonus"] > 0
    assert rec_no.breakdown["acoustic_bonus"] == 0


def test_total_score_is_sum_of_parts(sample_song, pop_profile):
    """Total score should equal the sum of all breakdown values."""
    rec = score_song(sample_song, pop_profile)
    expected = sum(rec.breakdown.values())
    assert abs(rec.total_score - expected) < 0.01


# ── Recommendation ranking tests ───────────────────────────────────────

def test_recommend_returns_top_k(small_catalog, pop_profile):
    """recommend() should return exactly top_k results."""
    results = recommend(small_catalog, pop_profile, top_k=3)
    assert len(results) == 3


def test_recommend_sorted_descending(small_catalog, pop_profile):
    """Results should be sorted by score, highest first."""
    results = recommend(small_catalog, pop_profile, top_k=5)
    scores = [r.total_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recommend_best_match_is_genre_and_mood(small_catalog, pop_profile):
    """For a pop/happy profile, Song A (pop/happy) should be the top result."""
    results = recommend(small_catalog, pop_profile, top_k=1)
    assert results[0].song.title == "Song A"


def test_max_possible_score_basic():
    """Max possible score for a basic profile (no bonuses) should be 4.5."""
    profile = UserProfile("pop", "sad", energy_target=0.5, prefers_acoustic=False)
    assert get_max_possible_score(profile) == 4.5  # 2.0 + 1.0 + 1.5


def test_max_possible_score_all_bonuses():
    """Max possible score with all bonuses should be 6.3."""
    profile = UserProfile("pop", "happy", energy_target=0.8, prefers_acoustic=True)
    expected = 2.0 + 1.0 + 1.5 + 0.5 + 0.5 + 0.8  # = 6.3
    assert get_max_possible_score(profile) == expected
