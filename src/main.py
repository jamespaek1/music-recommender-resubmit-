"""
Music Discovery Agent — CLI Entry Point

Run with:
    python -m src.main                  # demo mode (runs 3 example queries)
    python -m src.main --interactive    # interactive chat mode
    python -m src.main --query "..."    # single query mode
    python -m src.main --compare        # compare baseline vs music-critic personas
"""

import argparse
import sys
from src.agent import run_agent, format_result
from src.models import load_catalog


# ── Example queries for demo mode ───────────────────────────────────────

DEMO_QUERIES = [
    "I need something upbeat and energetic for my morning run — pop or EDM would be great!",
    "I'm feeling sad and want to listen to something chill and acoustic while it rains outside.",
    "Give me intense jazz with high energy for a late-night coding session.",
]


def run_demo():
    """Run the agent on predefined demo queries to showcase the system."""
    print("\n" + "🎵 " * 20)
    print("  MUSIC DISCOVERY AGENT — Demo Mode")
    print("🎵 " * 20 + "\n")

    catalog = load_catalog()
    print(f"Loaded {len(catalog)} songs from catalog.\n")

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{'━' * 60}")
        print(f"  Demo Query {i}/{len(DEMO_QUERIES)}")
        print(f"{'━' * 60}")

        result = run_agent(query, catalog=catalog, use_llm=True, verbose=True)
        print(format_result(result))
        print()


def run_interactive():
    """Run the agent in interactive chat mode."""
    print("\n" + "🎵 " * 20)
    print("  MUSIC DISCOVERY AGENT — Interactive Mode")
    print("🎵 " * 20)
    print("\nDescribe the music you're looking for, and I'll find the best matches.")
    print("Type 'quit' or 'exit' to stop.\n")

    catalog = load_catalog()

    while True:
        try:
            user_input = input("🎤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! 🎶")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! 🎶")
            break

        if not user_input:
            print("Please type something! Describe the music you want.\n")
            continue

        result = run_agent(user_input, catalog=catalog, use_llm=True, verbose=False)
        print(format_result(result))
        print()


def run_single(query: str):
    """Run the agent on a single query."""
    result = run_agent(query, use_llm=True, verbose=True)
    print(format_result(result))


def run_compare():
    """
    Compare baseline vs music-critic persona side-by-side.
    Demonstrates that specialized output measurably differs from baseline.
    """
    from src.planner import parse_request
    from src.recommender import recommend
    from src.explainer import compare_personas
    from src.logger import AgentLogger

    print("\n" + "=" * 65)
    print("  PERSONA COMPARISON — Baseline vs Music Critic")
    print("=" * 65)

    catalog = load_catalog()
    queries = [
        "I need something upbeat and energetic for my morning run!",
        "Sad acoustic music for a rainy evening at home.",
    ]

    for query in queries:
        logger = AgentLogger(verbose=False)
        profile = parse_request(query, logger, use_llm=False)
        recs = recommend(catalog, profile, top_k=3)

        comparison = compare_personas(query, profile, recs, logger, use_llm=False)

        print(f"\n{'─' * 65}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 65}")

        print(f"\n  ┌── BASELINE (plain, factual) ──")
        for line in comparison["baseline"].split("\n"):
            print(f"  │ {line}")
        print(f"  └── Length: {comparison['baseline_length']} chars")

        print(f"\n  ┌── MUSIC CRITIC (specialized persona) ──")
        for line in comparison["music_critic"].split("\n"):
            print(f"  │ {line}")
        print(f"  └── Length: {comparison['critic_length']} chars")

        print(f"\n  Word overlap: {comparison['word_overlap_pct']}% "
              "(lower = more stylistic difference)")

    print(f"\n{'=' * 65}")
    print("  The music-critic persona uses sensory language, situational")
    print("  hooks, and editorial flair that measurably differs from the")
    print("  factual baseline — demonstrating specialization behavior.")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Music Discovery Agent — AI-powered music recommendations"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive chat mode"
    )
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Run a single query"
    )
    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run demo mode with example queries (default)"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Compare baseline vs music-critic persona side-by-side"
    )

    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    elif args.query:
        run_single(args.query)
    elif args.compare:
        run_compare()
    else:
        run_demo()


if __name__ == "__main__":
    main()
