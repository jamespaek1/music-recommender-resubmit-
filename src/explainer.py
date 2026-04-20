"""
Explainer Agent: Generates natural language explanations for recommendations.

Supports two personas:
  - BASELINE: Plain, factual template-based explanations
  - MUSIC_CRITIC: Specialized persona with vivid, editorial-style language
    modeled after music journalism (few-shot pattern specialization)

Uses the OpenAI API when available (with a few-shot specialized system prompt),
and falls back to persona-aware template generation otherwise.
"""

import os
from src.models import Recommendation, UserProfile
from src.logger import AgentLogger


# ── Persona definitions ─────────────────────────────────────────────────
# The MUSIC_CRITIC persona demonstrates specialization behavior:
# constrained tone/style that measurably differs from baseline output.

PERSONAS = {
    "baseline": {
        "name": "Baseline",
        "description": "Plain factual recommendations",
        "system_prompt": (
            "You are a music recommender. Given recommendations and scores, "
            "write a brief response listing the picks and why they match. "
            "Be factual and concise. Use plain language."
        ),
    },
    "music_critic": {
        "name": "Music Critic",
        "description": "Editorial music journalism style with vivid language",
        "system_prompt": (
            "You are a passionate music critic writing for a hip online magazine. "
            "Your tone is warm, opinionated, and vivid — like a trusted friend who "
            "happens to have encyclopedic music knowledge.\n\n"
            "Style rules:\n"
            "- Use sensory language: describe how music FEELS, not just what it IS\n"
            "- Reference the listener's situation (workout, rainy day, road trip)\n"
            "- Give each song a one-line 'hook' that captures its essence\n"
            "- End with a confident, specific recommendation to start with\n"
            "- Never use bullet points. Write flowing, magazine-style prose.\n"
            "- Keep it to 4-6 sentences.\n\n"
            "Here are examples of your style:\n\n"
            "EXAMPLE 1 (upbeat request):\n"
            "You're chasing that sunrise-run energy, and your playlist should match. "
            "\"Midnight Drive\" is pure neon-lit momentum — the kind of track that makes "
            "stoplights feel like starting gates. Pair it with \"Wildfire,\" which brings "
            "a punchier, festival-ready pulse that'll carry you through the last mile. "
            "Start with \"Midnight Drive\" and let the tempo do the work.\n\n"
            "EXAMPLE 2 (chill request):\n"
            "For those grey-sky afternoons where the rain is doing half the work, you want "
            "songs that lean into the quiet. \"Paper Hearts\" has this aching, stripped-back "
            "tenderness — like finding an old letter you forgot you kept. \"Rainy Window\" "
            "lives in the same emotional neighborhood, all soft piano and slow exhales. "
            "Put on \"Paper Hearts\" first, make some tea, and let the afternoon unspool."
        ),
    },
}

DEFAULT_PERSONA = "music_critic"


# ── Baseline template explainer (plain style) ───────────────────────────

def _baseline_explain(rec: Recommendation, profile: UserProfile) -> str:
    """Generate a plain, factual explanation from the score breakdown."""
    song = rec.song
    parts = []

    if rec.breakdown.get("genre_match", 0) > 0:
        parts.append(f"it's {song.genre}, which matches your taste")
    if rec.breakdown.get("mood_match", 0) > 0:
        parts.append(f"the {song.mood} mood fits what you're looking for")
    if rec.breakdown.get("energy_similarity", 0) > 1.0:
        parts.append(f"its energy level ({song.energy:.0%}) is close to your target")
    if rec.breakdown.get("valence_bonus", 0) > 0.3:
        parts.append("it has a bright, positive feel")
    if rec.breakdown.get("danceability_bonus", 0) > 0.3:
        parts.append("it's great for dancing")
    if rec.breakdown.get("acoustic_bonus", 0) > 0.3:
        parts.append("it has the acoustic vibe you like")

    if parts:
        reason = ", ".join(parts[:-1])
        if len(parts) > 1:
            reason += f", and {parts[-1]}"
        else:
            reason = parts[0]
        return f'"{song.title}" by {song.artist} — {reason}. (Score: {rec.total_score:.2f})'
    else:
        return f'"{song.title}" by {song.artist} — a decent match overall. (Score: {rec.total_score:.2f})'


def _baseline_response(
    user_input: str,
    profile: UserProfile,
    recommendations: list,
) -> str:
    """Build a plain, factual template response (baseline persona)."""
    lines = []
    lines.append(f"Based on your request for {profile.favorite_mood} "
                 f"{profile.favorite_genre} music, here are my top picks:\n")

    for i, rec in enumerate(recommendations, 1):
        explanation = _baseline_explain(rec, profile)
        lines.append(f"  {i}. {explanation}")

    top = recommendations[0].song if recommendations else None
    if top:
        lines.append(f"\nI'd start with \"{top.title}\" — it's the strongest match "
                     "for what you described.")

    return "\n".join(lines)


# ── Music critic template explainer (specialized style) ─────────────────

# Sensory descriptors mapped to song features for vivid language
_ENERGY_WORDS = {
    "high": ["driving", "electric", "pulse-pounding", "adrenaline-fueled"],
    "mid": ["steady", "grooving", "smooth-rolling", "effortless"],
    "low": ["hushed", "floating", "featherlight", "whisper-quiet"],
}

_MOOD_HOOKS = {
    "happy": ["sun-soaked", "grinning", "golden-hour", "windows-down"],
    "sad": ["aching", "bittersweet", "rain-streaked", "tender"],
    "energetic": ["thundering", "white-hot", "stadium-sized", "relentless"],
    "chill": ["velvet-smooth", "drifting", "late-night", "candle-lit"],
}

_GENRE_FLAVOR = {
    "pop": "polished hooks and sing-along melodies",
    "rock": "raw guitar crunch and attitude",
    "edm": "pulsing bass and synthesizer waves",
    "lofi": "dusty vinyl warmth and soft beats",
    "jazz": "smoky sophistication and swinging rhythm",
    "folk": "campfire-warm acoustics and honest storytelling",
    "metal": "crushing riffs and unrelenting power",
}


def _critic_explain(rec: Recommendation, profile: UserProfile, artist_context: dict = None) -> str:
    """Generate a vivid, music-critic-style explanation for one song."""
    song = rec.song
    energy_tier = "high" if song.energy >= 0.7 else ("low" if song.energy < 0.35 else "mid")
    energy_word = _ENERGY_WORDS[energy_tier][hash(song.title) % len(_ENERGY_WORDS[energy_tier])]
    mood_hook = _MOOD_HOOKS.get(song.mood, ["intriguing"])[hash(song.artist) % len(_MOOD_HOOKS.get(song.mood, ["intriguing"]))]
    genre_note = _GENRE_FLAVOR.get(song.genre, "a distinctive sonic palette")

    # If RAG provided artist context, add a detail from their bio
    artist_detail = ""
    if artist_context and song.artist in artist_context:
        artist_detail = f" {artist_context[song.artist]}"

    return (
        f'"{song.title}" by {song.artist} — {mood_hook} and {energy_word}, '
        f"built on {genre_note}.{artist_detail} (Score: {rec.total_score:.2f})"
    )


def _extract_artist_details(rag_context: str) -> dict:
    """Extract short artist details from RAG context for inline use."""
    details = {}
    if not rag_context:
        return details
    for block in rag_context.split("\n\n"):
        if block.startswith("[ARTIST:"):
            # Extract artist name and a useful snippet
            name_end = block.index("]")
            name = block[len("[ARTIST: "):name_end].strip()
            content = block[name_end + 1:].strip()
            # Find "Best for:" snippet or a comparison
            if "Best for:" in content:
                best_for = content.split("Best for:")[1].strip().rstrip(".")
                details[name] = f"(Best for: {best_for}.)"
            elif "compared to" in content.lower() or "comparisons to" in content.lower():
                details[name] = ""
    return details


def _extract_context_tip(rag_context: str, mood: str) -> str:
    """Extract a listening context tip from RAG context."""
    if not rag_context:
        return ""
    for block in rag_context.split("\n\n"):
        if block.startswith("[CONTEXT:"):
            name_end = block.index("]")
            content = block[name_end + 1:].strip()
            # Extract just the first sentence as a tip
            sentences = content.split(". ")
            if sentences:
                return sentences[0].split(": ", 1)[-1] if ": " in sentences[0] else sentences[0]
    return ""


def _critic_response(
    user_input: str,
    profile: UserProfile,
    recommendations: list,
    rag_context: str = "",
) -> str:
    """Build a music-critic-style template response (specialized persona)."""
    lines = []
    artist_details = _extract_artist_details(rag_context)
    context_tip = _extract_context_tip(rag_context, profile.favorite_mood)

    # Opening line references the listener's context
    context_map = {
        "happy": "You're chasing good vibes",
        "sad": "When the mood turns reflective",
        "energetic": "You want something with real momentum",
        "chill": "For those moments when you need the world to slow down",
    }
    opener = context_map.get(profile.favorite_mood, "Here's what I'd spin for you")
    lines.append(f"{opener}, and these tracks deliver:\n")

    for i, rec in enumerate(recommendations, 1):
        explanation = _critic_explain(rec, profile, artist_details)
        lines.append(f"  {i}. {explanation}")

    top = recommendations[0].song if recommendations else None
    if top:
        energy_feel = "let it build" if top.energy >= 0.6 else "let it wash over you"
        lines.append(f'\nDrop the needle on "{top.title}" first and {energy_feel}.')

    # Add RAG-sourced context tip if available
    if context_tip:
        lines.append(f"\nPro tip: {context_tip}.")

    return "\n".join(lines)


# ── LLM-powered response (uses persona system prompt) ───────────────────

def _llm_response(
    user_input: str,
    profile: UserProfile,
    recommendations: list,
    logger: AgentLogger,
    persona: str = DEFAULT_PERSONA,
    rag_context: str = "",
) -> str:
    """Use OpenAI to generate a response with the selected persona."""
    try:
        from openai import OpenAI
        client = OpenAI()

        recs_text = ""
        for i, rec in enumerate(recommendations, 1):
            recs_text += (
                f"{i}. \"{rec.song.title}\" by {rec.song.artist} "
                f"[{rec.song.genre}/{rec.song.mood}, energy={rec.song.energy}] "
                f"Score: {rec.total_score:.2f} "
                f"Breakdown: {rec.breakdown}\n"
            )

        persona_config = PERSONAS.get(persona, PERSONAS[DEFAULT_PERSONA])
        system_prompt = persona_config["system_prompt"]

        user_msg = (
            f"User request: \"{user_input}\"\n"
            f"Parsed profile: genre={profile.favorite_genre}, mood={profile.favorite_mood}, "
            f"energy={profile.energy_target}, acoustic={profile.prefers_acoustic}\n\n"
            f"Top recommendations:\n{recs_text}"
        )

        # Inject RAG context if available
        if rag_context:
            user_msg += (
                f"\nRelevant background knowledge (use to enrich your response):\n"
                f"{rag_context}\n"
            )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=400,
        )

        result = response.choices[0].message.content.strip()
        logger.log_step("EXPLAINER", f"LLM generated response (persona={persona})")
        return result

    except Exception as e:
        logger.log_warning("EXPLAINER", f"LLM failed ({e}), using template")
        if persona == "music_critic":
            return _critic_response(user_input, profile, recommendations,
                                    rag_context=rag_context)
        return _baseline_response(user_input, profile, recommendations)


# ── Public API ──────────────────────────────────────────────────────────

def generate_response(
    user_input: str,
    profile: UserProfile,
    recommendations: list,
    logger: AgentLogger,
    use_llm: bool = True,
    persona: str = DEFAULT_PERSONA,
    rag_context: str = "",
) -> str:
    """
    Main entry point for the explainer. Produces a natural-language response
    describing the recommendations and why they were chosen.

    Args:
        persona: "baseline" for plain output, "music_critic" for specialized style
        rag_context: Retrieved knowledge to augment the response
    """
    logger.log_step("EXPLAINER",
                    f"Generating response for {len(recommendations)} recs "
                    f"(persona={persona}, rag={'yes' if rag_context else 'no'})")

    if use_llm and os.getenv("OPENAI_API_KEY"):
        return _llm_response(user_input, profile, recommendations, logger, persona,
                             rag_context=rag_context)
    else:
        if persona == "music_critic":
            response = _critic_response(user_input, profile, recommendations,
                                        rag_context=rag_context)
            logger.log_step("EXPLAINER", "Generated music-critic template response")
        else:
            response = _baseline_response(user_input, profile, recommendations)
            logger.log_step("EXPLAINER", "Generated baseline template response")
        return response


def compare_personas(
    user_input: str,
    profile: UserProfile,
    recommendations: list,
    logger: AgentLogger,
    use_llm: bool = True,
) -> dict:
    """
    Generate responses from BOTH personas and return them side-by-side.
    Used to demonstrate that specialized output measurably differs from baseline.
    """
    baseline = generate_response(
        user_input, profile, recommendations, logger,
        use_llm=use_llm, persona="baseline"
    )
    critic = generate_response(
        user_input, profile, recommendations, logger,
        use_llm=use_llm, persona="music_critic"
    )

    # Measure stylistic differences
    baseline_words = set(baseline.lower().split())
    critic_words = set(critic.lower().split())
    overlap = len(baseline_words & critic_words)
    total = len(baseline_words | critic_words)
    word_overlap_pct = round(overlap / total * 100, 1) if total > 0 else 0

    return {
        "baseline": baseline,
        "music_critic": critic,
        "word_overlap_pct": word_overlap_pct,
        "baseline_length": len(baseline),
        "critic_length": len(critic),
    }
