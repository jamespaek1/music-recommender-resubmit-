"""
Planner Agent: Converts natural language music requests into structured UserProfiles.

Uses the OpenAI API when available, with a keyword-based fallback parser
so the system works even without an API key (useful for testing).
"""

import json
import os
from src.models import UserProfile, VALID_GENRES, VALID_MOODS
from src.logger import AgentLogger


# ── Keyword mappings for the fallback parser ──────────────────────────────

GENRE_KEYWORDS = {
    "pop": ["pop", "catchy", "mainstream", "top 40", "radio"],
    "rock": ["rock", "guitar", "band", "grunge", "punk", "alternative"],
    "edm": ["edm", "electronic", "dance", "rave", "techno", "house", "club", "dj"],
    "lofi": ["lofi", "lo-fi", "lo fi", "study", "background", "ambient"],
    "jazz": ["jazz", "saxophone", "swing", "smooth", "blues"],
    "folk": ["folk", "acoustic guitar", "campfire", "singer-songwriter", "indie folk"],
    "metal": ["metal", "heavy", "thrash", "headbang", "intense", "aggressive"],
}

MOOD_KEYWORDS = {
    "happy": ["happy", "upbeat", "cheerful", "joyful", "fun", "bright", "positive", "good vibes"],
    "sad": ["sad", "melancholy", "heartbreak", "lonely", "crying", "emotional", "somber"],
    "energetic": ["energetic", "pump", "workout", "running", "hype", "power", "intense", "adrenaline", "energy", "heavy", "hard", "loud", "maximum"],
    "chill": ["chill", "relax", "calm", "mellow", "peaceful", "wind down", "cozy", "soothing", "studying"],
}

ENERGY_HINTS = {
    "high": ["workout", "running", "gym", "pump", "party", "hype", "energetic", "intense", "power", "rave", "maximum", "max energy", "high energy"],
    "low": ["sleep", "study", "relax", "calm", "background", "wind down", "soothing", "quiet", "peaceful"],
    "mid": ["drive", "walk", "morning", "afternoon", "cooking", "vibes"],
}

ACOUSTIC_HINTS = ["acoustic", "unplugged", "folk", "campfire", "singer-songwriter", "piano", "guitar only"]


def _fallback_parse(user_input: str) -> UserProfile:
    """
    Rule-based keyword parser. Scans the user's message for genre, mood,
    energy, and acoustic hints. Returns a best-guess UserProfile.
    """
    text = user_input.lower()

    # Detect genre
    genre_scores = {}
    for genre, keywords in GENRE_KEYWORDS.items():
        genre_scores[genre] = sum(1 for kw in keywords if kw in text)
    best_genre = max(genre_scores, key=genre_scores.get)
    if genre_scores[best_genre] == 0:
        best_genre = "pop"  # default

    # Detect mood
    mood_scores = {}
    for mood, keywords in MOOD_KEYWORDS.items():
        mood_scores[mood] = sum(1 for kw in keywords if kw in text)
    best_mood = max(mood_scores, key=mood_scores.get)
    if mood_scores[best_mood] == 0:
        best_mood = "happy"  # default

    # Detect energy level
    energy = 0.5
    for kw in ENERGY_HINTS["high"]:
        if kw in text:
            energy = 0.85
            break
    for kw in ENERGY_HINTS["low"]:
        if kw in text:
            energy = 0.25
            break

    # Detect acoustic preference
    prefers_acoustic = any(kw in text for kw in ACOUSTIC_HINTS)

    return UserProfile(
        favorite_genre=best_genre,
        favorite_mood=best_mood,
        energy_target=energy,
        prefers_acoustic=prefers_acoustic,
    )


def _llm_parse(user_input: str, logger: AgentLogger) -> UserProfile:
    """
    Use OpenAI to parse natural language into a structured UserProfile.
    Falls back to keyword parser on any API error.
    """
    try:
        from openai import OpenAI
        client = OpenAI()  # reads OPENAI_API_KEY from env

        system_prompt = f"""You are a music preference parser. Given a user's natural language
description of what music they want, extract structured preferences.

Return ONLY valid JSON with these fields:
- "favorite_genre": one of {sorted(VALID_GENRES)}
- "favorite_mood": one of {sorted(VALID_MOODS)}
- "energy_target": float from 0.0 (very calm) to 1.0 (very intense)
- "prefers_acoustic": boolean

Pick the closest match. If the user's request is ambiguous, make your best guess.
Return raw JSON only, no markdown fences or extra text."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0.3,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

        data = json.loads(raw)
        logger.log_step("PLANNER", "LLM parsed profile successfully", data)

        # Validate and clamp values
        genre = data.get("favorite_genre", "pop").lower()
        if genre not in VALID_GENRES:
            logger.log_warning("PLANNER", f"LLM returned invalid genre '{genre}', defaulting to 'pop'")
            genre = "pop"

        mood = data.get("favorite_mood", "happy").lower()
        if mood not in VALID_MOODS:
            logger.log_warning("PLANNER", f"LLM returned invalid mood '{mood}', defaulting to 'happy'")
            mood = "happy"

        energy = float(data.get("energy_target", 0.5))
        energy = max(0.0, min(1.0, energy))

        return UserProfile(
            favorite_genre=genre,
            favorite_mood=mood,
            energy_target=energy,
            prefers_acoustic=bool(data.get("prefers_acoustic", False)),
        )

    except ImportError:
        logger.log_warning("PLANNER", "openai package not installed, using fallback parser")
        return _fallback_parse(user_input)
    except Exception as e:
        logger.log_warning("PLANNER", f"LLM parsing failed ({e}), using fallback parser")
        return _fallback_parse(user_input)


def parse_request(user_input: str, logger: AgentLogger, use_llm: bool = True) -> UserProfile:
    """
    Main entry point for the planner. Attempts LLM parsing first,
    falls back to keyword-based parsing on failure.
    """
    logger.log_step("PLANNER", f"Received request: '{user_input}'")

    if not user_input or not user_input.strip():
        logger.log_warning("PLANNER", "Empty input received, using default profile")
        return UserProfile(favorite_genre="pop", favorite_mood="happy")

    if use_llm and os.getenv("OPENAI_API_KEY"):
        logger.log_step("PLANNER", "Attempting LLM-based parsing")
        return _llm_parse(user_input, logger)
    else:
        logger.log_step("PLANNER", "Using keyword-based fallback parser")
        profile = _fallback_parse(user_input)
        logger.log_step("PLANNER", "Parsed profile", profile.to_dict())
        return profile
