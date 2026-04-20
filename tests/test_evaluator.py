"""
Tests for the evaluator module — confidence scoring and guardrails.
"""

import pytest
from src.models import Song, UserProfile, Recommendation
from src.evaluator import (
    validate_input,
    validate_profile,
    compute_confidence,
    suggest_refinement,
    should_refine,
    CONFIDENCE_THRESHOLD,
)
from src.logger import AgentLogger


@pytest.fixture
def logger():
    return AgentLogger(verbose=False)


@pytest.fixture
def catalog():
    return [
        Song("S1", "A1", "pop", "happy", 0.8, 120, 0.9, 0.85, 0.1),
        Song("S2", "A2", "pop", "sad", 0.4, 95, 0.2, 0.3, 0.5),
        Song("S3", "A3", "pop", "energetic", 0.9, 130, 0.7, 0.8, 0.05),
        Song("S4", "A4", "rock", "energetic", 0.85, 140, 0.4, 0.6, 0.1),
        Song("S5", "A5", "jazz", "chill", 0.3, 90, 0.5, 0.4, 0.75),
        Song("S6", "A6", "lofi", "chill", 0.2, 70, 0.45, 0.25, 0.85),
    ]


# ── Input validation ───────────────────────────────────────────────────

def test_validate_empty_input(logger):
    warnings = validate_input("", logger)
    assert len(warnings) > 0


def test_validate_short_input(logger):
    warnings = validate_input("hi", logger)
    assert len(warnings) > 0


def test_validate_normal_input(logger):
    warnings = validate_input("I want upbeat pop music for working out", logger)
    assert len(warnings) == 0


def test_validate_long_input(logger):
    warnings = validate_input("x" * 1001, logger)
    assert len(warnings) > 0


# ── Profile validation ─────────────────────────────────────────────────

def test_validate_good_profile(logger, catalog):
    profile = UserProfile("pop", "happy", 0.7, False)
    warnings = validate_profile(profile, catalog, logger)
    assert len(warnings) == 0


def test_validate_contradictory_acoustic_edm(logger, catalog):
    profile = UserProfile("edm", "energetic", 0.9, True)
    warnings = validate_profile(profile, catalog, logger)
    assert any("acoustic" in w.lower() or "conflicts" in w.lower() for w in warnings)


def test_validate_missing_genre(logger, catalog):
    profile = UserProfile("country", "happy", 0.5, False)
    warnings = validate_profile(profile, catalog, logger)
    assert any("country" in w.lower() for w in warnings)


def test_validate_chill_high_energy(logger, catalog):
    profile = UserProfile("pop", "chill", 0.95, False)
    warnings = validate_profile(profile, catalog, logger)
    assert any("chill" in w.lower() for w in warnings)


# ── Confidence scoring ─────────────────────────────────────────────────

def test_confidence_strong_match(logger, catalog):
    """A pop/happy profile should get high confidence with a pop-heavy catalog."""
    profile = UserProfile("pop", "happy", 0.8, False)
    from src.recommender import recommend
    recs = recommend(catalog, profile, top_k=5)
    conf = compute_confidence(profile, recs, catalog, logger)
    assert conf > 0.6


def test_confidence_weak_match(logger, catalog):
    """A genre with no songs should get low confidence."""
    profile = UserProfile("metal", "sad", 0.5, True)
    from src.recommender import recommend
    recs = recommend(catalog, profile, top_k=5)
    conf = compute_confidence(profile, recs, catalog, logger)
    assert conf < 0.5


def test_confidence_no_recommendations(logger, catalog):
    profile = UserProfile("pop", "happy", 0.5, False)
    conf = compute_confidence(profile, [], catalog, logger)
    assert conf == 0.0


# ── Refinement ─────────────────────────────────────────────────────────

def test_refinement_drops_acoustic_for_edm(logger, catalog):
    profile = UserProfile("edm", "energetic", 0.9, True)
    from src.recommender import recommend
    recs = recommend(catalog, profile, top_k=5)
    refined = suggest_refinement(profile, recs, catalog, logger)
    assert refined.prefers_acoustic is False


def test_should_refine_below_threshold():
    assert should_refine(CONFIDENCE_THRESHOLD - 0.1) is True


def test_should_not_refine_above_threshold():
    assert should_refine(CONFIDENCE_THRESHOLD + 0.1) is False
