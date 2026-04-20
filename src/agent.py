"""
Agent Orchestrator: The core agentic workflow.

This module implements the plan → retrieve → act → evaluate → refine loop:
  1. PLAN:     Parse the user's natural language request into a UserProfile
  2. ACT:      Run the recommender engine to score and rank songs
  3. RETRIEVE: Fetch relevant knowledge (artist bios, genre guides, context tips)
  4. EXPLAIN:  Generate a natural language response augmented with retrieved context
  5. EVALUATE: Score confidence and check guardrails
  6. REFINE:   If confidence is low, adjust the profile and loop back

The agent runs at most MAX_REFINEMENTS iterations before delivering
whatever results it has, with a warning about low confidence.
"""

from src.models import load_catalog, AgentResult
from src.planner import parse_request
from src.recommender import recommend
from src.explainer import generate_response
from src.retriever import KnowledgeBase
from src.evaluator import (
    validate_input,
    validate_profile,
    compute_confidence,
    suggest_refinement,
    should_refine,
)
from src.logger import AgentLogger


MAX_REFINEMENTS = 2  # Maximum retry loops before delivering results
TOP_K = 5  # Number of recommendations to return


def run_agent(
    user_input: str,
    catalog: list = None,
    use_llm: bool = True,
    verbose: bool = True,
) -> AgentResult:
    """
    Run the full agent pipeline on a single user request.

    Args:
        user_input: Natural language music request
        catalog: Song catalog (loaded from CSV if None)
        use_llm: Whether to attempt LLM-based parsing/explanation
        verbose: Whether to print step-by-step logs to console

    Returns:
        AgentResult with recommendations, confidence, and full log
    """
    logger = AgentLogger(verbose=verbose)
    logger.log_step("AGENT", "═" * 50)
    logger.log_step("AGENT", f"Starting agent for input: \"{user_input}\"")

    # Load catalog if not provided
    if catalog is None:
        catalog = load_catalog()
        logger.log_step("AGENT", f"Loaded {len(catalog)} songs from catalog")

    # Load knowledge base for RAG
    kb = KnowledgeBase()
    logger.log_step("AGENT", f"Knowledge base loaded: {kb.stats()['total_documents']} documents")

    # ── Step 1: Validate input ──────────────────────────────────────────
    input_warnings = validate_input(user_input, logger)

    # ── Step 2: Plan — parse request into a profile ─────────────────────
    profile = parse_request(user_input, logger, use_llm=use_llm)
    logger.log_step("AGENT", "Profile parsed", profile.to_dict())

    # ── Step 3: Validate profile against catalog ────────────────────────
    profile_warnings = validate_profile(profile, catalog, logger)
    all_warnings = input_warnings + profile_warnings

    # ── Step 4: Act — run recommender ───────────────────────────────────
    recommendations = recommend(catalog, profile, top_k=TOP_K)
    logger.log_step("AGENT", f"Generated {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations):
        logger.log_step("AGENT", f"  #{i+1}: {rec.song} → score={rec.total_score:.3f}")

    # ── Step 5: Evaluate — compute confidence ───────────────────────────
    confidence = compute_confidence(profile, recommendations, catalog, logger)

    # ── Step 6: Refine loop if needed ───────────────────────────────────
    refinement_count = 0
    while should_refine(confidence) and refinement_count < MAX_REFINEMENTS:
        refinement_count += 1
        logger.log_step("AGENT", f"Refinement attempt {refinement_count}/{MAX_REFINEMENTS} "
                        f"(confidence={confidence:.3f})")

        # Get a refined profile
        profile = suggest_refinement(profile, recommendations, catalog, logger)

        # Re-run recommender with refined profile
        recommendations = recommend(catalog, profile, top_k=TOP_K)
        logger.log_step("AGENT", f"Re-generated {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations):
            logger.log_step("AGENT", f"  #{i+1}: {rec.song} → score={rec.total_score:.3f}")

        # Re-evaluate
        confidence = compute_confidence(profile, recommendations, catalog, logger)

    if refinement_count > 0 and should_refine(confidence):
        all_warnings.append(
            f"Confidence remains low ({confidence:.2f}) after {refinement_count} refinements. "
            "Results may not match your request well."
        )

    # ── Step 7: Retrieve — fetch relevant knowledge (RAG) ───────────────
    rec_artists = list({rec.song.artist for rec in recommendations})
    rag_results = kb.retrieve_for_profile(
        user_input, profile, rec_artists, logger, top_k=5
    )
    retrieved_context = kb.format_context(rag_results)
    if retrieved_context:
        logger.log_step("AGENT", f"RAG context: {len(retrieved_context)} chars from "
                        f"{len(rag_results)} documents")

    # ── Step 8: Explain — generate response augmented with RAG context ──
    natural_response = generate_response(
        user_input, profile, recommendations, logger,
        use_llm=use_llm, rag_context=retrieved_context,
    )
    logger.log_step("AGENT", "Response generated")

    # ── Package result ──────────────────────────────────────────────────
    result = AgentResult(
        user_input=user_input,
        parsed_profile=profile,
        recommendations=recommendations,
        natural_response=natural_response,
        confidence_score=confidence,
        refinement_attempts=refinement_count,
        retrieved_context=retrieved_context,
        log_entries=logger.get_entries(),
        warnings=all_warnings,
    )

    logger.log_step("AGENT", f"Done — confidence={confidence:.3f}, "
                    f"refinements={refinement_count}, warnings={len(all_warnings)}")
    logger.log_step("AGENT", "═" * 50)

    return result


def format_result(result: AgentResult) -> str:
    """Format an AgentResult into a user-friendly string for display."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"🎵  MUSIC DISCOVERY AGENT")
    lines.append("=" * 60)
    lines.append(f"\n📝 Your request: \"{result.user_input}\"")
    lines.append(f"\n🔍 Parsed profile:")
    lines.append(f"   Genre: {result.parsed_profile.favorite_genre}")
    lines.append(f"   Mood:  {result.parsed_profile.favorite_mood}")
    lines.append(f"   Energy target: {result.parsed_profile.energy_target:.0%}")
    lines.append(f"   Acoustic: {'Yes' if result.parsed_profile.prefers_acoustic else 'No'}")

    if result.retrieved_context:
        # Show which knowledge sources were used
        sources = []
        for line in result.retrieved_context.split("\n\n"):
            if line.startswith("["):
                tag = line.split("]")[0] + "]"
                sources.append(tag)
        if sources:
            lines.append(f"\n📚 RAG context retrieved:")
            for s in sources[:5]:
                lines.append(f"   {s}")

    lines.append(f"\n🎶 Recommendations:")
    lines.append("-" * 40)
    lines.append(result.natural_response)
    lines.append("-" * 40)

    lines.append(f"\n📊 Confidence score: {result.confidence_score:.2f}")
    if result.refinement_attempts > 0:
        lines.append(f"🔄 Refinement attempts: {result.refinement_attempts}")

    if result.warnings:
        lines.append(f"\n⚠️  Warnings:")
        for w in result.warnings:
            lines.append(f"   • {w}")

    lines.append(f"\n📋 Score breakdown for top pick:")
    if result.recommendations:
        top = result.recommendations[0]
        for key, val in top.breakdown.items():
            bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
            lines.append(f"   {key:25s} {bar} {val:.3f}")

    lines.append("=" * 60)
    return "\n".join(lines)
