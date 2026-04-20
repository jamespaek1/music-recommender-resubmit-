"""
Evaluator: Confidence scoring, guardrails, and output validation.

This module assesses the quality of recommendations and determines
whether the agent should refine its approach or deliver the results.

Confidence is computed from multiple signals:
  - Score coverage: how close are top scores to the theoretical max?
  - Genre availability: does the catalog have songs in the requested genre?
  - Score spread: are the top results clearly differentiated?
  - Profile coherence: do the parsed preferences make sense together?
"""

from src.models import (
    UserProfile, Recommendation, VALID_GENRES, VALID_MOODS, Song
)
from src.recommender import get_max_possible_score
from src.logger import AgentLogger


# Confidence threshold — below this, the agent will attempt refinement
CONFIDENCE_THRESHOLD = 0.5


def validate_input(user_input: str, logger: AgentLogger) -> list:
    """
    Validate the user's raw input. Returns a list of warning strings.
    Empty list means the input looks fine.
    """
    warnings = []

    if not user_input or not user_input.strip():
        warnings.append("Input is empty. Please describe what kind of music you'd like.")

    if len(user_input.strip()) < 5:
        warnings.append("Input is very short. More detail helps me find better matches.")

    if len(user_input) > 1000:
        warnings.append("Input is very long. I'll focus on the key details.")

    for w in warnings:
        logger.log_warning("VALIDATOR", w)

    return warnings


def validate_profile(profile: UserProfile, catalog: list, logger: AgentLogger) -> list:
    """
    Validate the parsed profile against the catalog.
    Returns a list of warnings.
    """
    warnings = []

    if profile.favorite_genre not in VALID_GENRES:
        warnings.append(f"Genre '{profile.favorite_genre}' is not in the catalog. "
                        f"Available: {', '.join(sorted(VALID_GENRES))}")

    if profile.favorite_mood not in VALID_MOODS:
        warnings.append(f"Mood '{profile.favorite_mood}' is not in the catalog. "
                        f"Available: {', '.join(sorted(VALID_MOODS))}")

    # Check if the catalog has songs in the requested genre
    genre_count = sum(1 for s in catalog if s.genre == profile.favorite_genre)
    if genre_count == 0:
        warnings.append(f"No songs found for genre '{profile.favorite_genre}'.")
    elif genre_count < 3:
        warnings.append(f"Only {genre_count} song(s) in genre '{profile.favorite_genre}' — "
                        "recommendations may be limited.")

    # Check for contradictory preferences
    if profile.prefers_acoustic and profile.favorite_genre in ("edm", "metal"):
        warnings.append(f"Acoustic preference conflicts with {profile.favorite_genre} genre — "
                        "results may have low scores.")

    if profile.favorite_mood == "chill" and profile.energy_target > 0.8:
        warnings.append("Chill mood with very high energy is unusual — scores may be lower.")

    for w in warnings:
        logger.log_warning("VALIDATOR", w)

    return warnings


def compute_confidence(
    profile: UserProfile,
    recommendations: list,
    catalog: list,
    logger: AgentLogger,
) -> float:
    """
    Compute a confidence score (0.0 to 1.0) for the recommendation set.

    Factors:
      1. Score coverage (40%): top score / max possible score
      2. Genre availability (20%): fraction of catalog in the requested genre
      3. Score spread (20%): gap between #1 and #5 (good differentiation)
      4. Top score absolute (20%): is the best score above a minimum threshold?
    """
    if not recommendations:
        logger.log_step("EVALUATOR", "No recommendations — confidence = 0.0")
        return 0.0

    max_possible = get_max_possible_score(profile)
    top_score = recommendations[0].total_score

    # Factor 1: Score coverage (how close to max?)
    score_coverage = top_score / max_possible if max_possible > 0 else 0
    score_coverage = min(score_coverage, 1.0)

    # Factor 2: Genre availability
    genre_count = sum(1 for s in catalog if s.genre == profile.favorite_genre)
    genre_availability = min(genre_count / 5.0, 1.0)  # 5+ songs = full score

    # Factor 3: Score spread (differentiation between top and 5th)
    if len(recommendations) >= 2:
        spread = recommendations[0].total_score - recommendations[-1].total_score
        score_spread = min(spread / 2.0, 1.0)
    else:
        score_spread = 0.5

    # Factor 4: Absolute top score threshold
    absolute_score = min(top_score / 3.5, 1.0)  # 3.5+ is a strong match

    # Weighted combination
    confidence = (
        0.40 * score_coverage +
        0.20 * genre_availability +
        0.20 * score_spread +
        0.20 * absolute_score
    )
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    logger.log_step("EVALUATOR", f"Confidence: {confidence:.3f}", {
        "score_coverage": round(score_coverage, 3),
        "genre_availability": round(genre_availability, 3),
        "score_spread": round(score_spread, 3),
        "absolute_score": round(absolute_score, 3),
        "top_score": top_score,
        "max_possible": max_possible,
    })

    return confidence


def suggest_refinement(
    profile: UserProfile,
    recommendations: list,
    catalog: list,
    logger: AgentLogger,
) -> UserProfile:
    """
    When confidence is low, suggest a refined profile.
    Strategy: relax the most restrictive constraint.
    """
    logger.log_step("EVALUATOR", "Attempting profile refinement")

    new_genre = profile.favorite_genre
    new_mood = profile.favorite_mood
    new_energy = profile.energy_target
    new_acoustic = profile.prefers_acoustic

    # If no songs in genre, pick the genre with most songs matching the mood
    genre_count = sum(1 for s in catalog if s.genre == profile.favorite_genre)
    if genre_count == 0:
        # Find the best alternative genre for this mood
        genre_mood_counts = {}
        for g in VALID_GENRES:
            genre_mood_counts[g] = sum(
                1 for s in catalog if s.genre == g and s.mood == profile.favorite_mood
            )
        new_genre = max(genre_mood_counts, key=genre_mood_counts.get)
        logger.log_step("EVALUATOR", f"Relaxed genre: {profile.favorite_genre} → {new_genre}")

    # If acoustic + EDM/metal contradiction, drop acoustic
    if profile.prefers_acoustic and profile.favorite_genre in ("edm", "metal"):
        new_acoustic = False
        logger.log_step("EVALUATOR", "Dropped acoustic preference (conflicts with genre)")

    # If energy is extreme, move toward 0.5
    if profile.energy_target > 0.9 or profile.energy_target < 0.1:
        new_energy = profile.energy_target * 0.7 + 0.5 * 0.3
        logger.log_step("EVALUATOR", f"Relaxed energy: {profile.energy_target} → {new_energy:.2f}")

    refined = UserProfile(
        favorite_genre=new_genre,
        favorite_mood=new_mood,
        energy_target=round(new_energy, 2),
        prefers_acoustic=new_acoustic,
    )
    logger.log_step("EVALUATOR", "Refined profile", refined.to_dict())
    return refined


def should_refine(confidence: float) -> bool:
    """Determine if the agent should attempt a refinement loop."""
    return confidence < CONFIDENCE_THRESHOLD
