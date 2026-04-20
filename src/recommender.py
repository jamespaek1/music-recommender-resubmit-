"""
Content-based recommender engine.

This is the core scoring algorithm from the original Module 3 project.
Each song is scored against a UserProfile using weighted feature matching.

Scoring formula:
  +2.0  genre match
  +1.0  mood match
  +1.5  energy similarity (1 - |song_energy - target|) * 1.5
  +0.5  valence bonus (if user prefers happy mood)
  +0.5  danceability bonus (if user energy target >= 0.7)
  +0.8  acoustic bonus (if user prefers acoustic)
"""

from src.models import Song, UserProfile, Recommendation


def score_song(song: Song, profile: UserProfile) -> Recommendation:
    """Score a single song against a user profile. Returns a Recommendation with breakdown."""
    breakdown = {}
    total = 0.0

    # Genre match: +2.0
    if song.genre == profile.favorite_genre:
        breakdown["genre_match"] = 2.0
        total += 2.0
    else:
        breakdown["genre_match"] = 0.0

    # Mood match: +1.0
    if song.mood == profile.favorite_mood:
        breakdown["mood_match"] = 1.0
        total += 1.0
    else:
        breakdown["mood_match"] = 0.0

    # Energy similarity: up to +1.5
    energy_sim = (1.0 - abs(song.energy - profile.energy_target)) * 1.5
    breakdown["energy_similarity"] = round(energy_sim, 3)
    total += energy_sim

    # Valence bonus: up to +0.5 (if user prefers happy mood)
    if profile.favorite_mood == "happy":
        valence_bonus = song.valence * 0.5
        breakdown["valence_bonus"] = round(valence_bonus, 3)
        total += valence_bonus
    else:
        breakdown["valence_bonus"] = 0.0

    # Danceability bonus: up to +0.5 (if high energy target >= 0.7)
    if profile.energy_target >= 0.7:
        dance_bonus = song.danceability * 0.5
        breakdown["danceability_bonus"] = round(dance_bonus, 3)
        total += dance_bonus
    else:
        breakdown["danceability_bonus"] = 0.0

    # Acoustic bonus: up to +0.8 (if user prefers acoustic)
    if profile.prefers_acoustic:
        acoustic_bonus = song.acousticness * 0.8
        breakdown["acoustic_bonus"] = round(acoustic_bonus, 3)
        total += acoustic_bonus
    else:
        breakdown["acoustic_bonus"] = 0.0

    return Recommendation(
        song=song,
        total_score=round(total, 3),
        breakdown=breakdown,
    )


def recommend(catalog: list, profile: UserProfile, top_k: int = 5) -> list:
    """
    Score every song in the catalog and return the top_k recommendations,
    sorted by score descending.
    """
    scored = [score_song(song, profile) for song in catalog]
    scored.sort(key=lambda r: r.total_score, reverse=True)
    return scored[:top_k]


def get_max_possible_score(profile: UserProfile) -> float:
    """Calculate the theoretical maximum score for a given profile."""
    max_score = 2.0 + 1.0 + 1.5  # genre + mood + perfect energy
    if profile.favorite_mood == "happy":
        max_score += 0.5
    if profile.energy_target >= 0.7:
        max_score += 0.5
    if profile.prefers_acoustic:
        max_score += 0.8
    return max_score
