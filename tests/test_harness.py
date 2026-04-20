"""
Test Harness: Batch evaluation script for the Music Discovery Agent.

Runs the agent on a set of predefined test cases and prints a summary
with pass/fail results, confidence scores, and overall statistics.

Run with:
    python -m tests.test_harness
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.agent import run_agent
from src.models import load_catalog


# ── Test cases ──────────────────────────────────────────────────────────
# Each case: (description, query, expected_genre, expected_mood, min_confidence)

TEST_CASES = [
    (
        "Upbeat pop request",
        "I want happy upbeat pop music for a party!",
        "pop",
        "happy",
        0.6,
    ),
    (
        "Chill lofi for studying",
        "Something calm and chill for studying, maybe lofi beats",
        "lofi",
        "chill",
        0.5,
    ),
    (
        "Energetic workout music",
        "I need high energy EDM for my gym workout session",
        "edm",
        "energetic",
        0.5,
    ),
    (
        "Sad acoustic songs",
        "I'm feeling down and want sad acoustic music",
        None,  # multiple genres could match
        "sad",
        0.3,
    ),
    (
        "Rock road trip",
        "Give me some rock music with good energy for a road trip",
        "rock",
        None,  # could be happy or energetic
        0.4,
    ),
    (
        "Jazz evening",
        "Smooth jazz for a relaxing evening at home",
        "jazz",
        "chill",
        0.4,
    ),
    (
        "Contradictory request (edge case)",
        "I want intense acoustic EDM that's also really calm",
        None,
        None,
        0.0,  # low bar — this is expected to struggle
    ),
    (
        "Minimal input (edge case)",
        "music",
        None,
        None,
        0.0,
    ),
    (
        "Folk campfire",
        "Acoustic folk songs for sitting around a campfire, happy vibes",
        "folk",
        "happy",
        0.4,
    ),
    (
        "Heavy metal energy",
        "I want the most intense heavy metal you've got, maximum energy",
        "metal",
        "energetic",
        0.3,
    ),
]


def run_test_harness():
    """Run all test cases and print a summary report."""
    print("\n" + "=" * 70)
    print("  TEST HARNESS — Music Discovery Agent Batch Evaluation")
    print("=" * 70)

    catalog = load_catalog()
    results = []

    for i, (desc, query, exp_genre, exp_mood, min_conf) in enumerate(TEST_CASES, 1):
        print(f"\n--- Test {i}/{len(TEST_CASES)}: {desc} ---")
        print(f"    Query: \"{query}\"")

        result = run_agent(query, catalog=catalog, use_llm=True, verbose=False)

        # Check assertions
        genre_ok = (exp_genre is None) or (result.parsed_profile.favorite_genre == exp_genre)
        mood_ok = (exp_mood is None) or (result.parsed_profile.favorite_mood == exp_mood)
        conf_ok = result.confidence_score >= min_conf
        has_recs = len(result.recommendations) > 0

        passed = genre_ok and mood_ok and conf_ok and has_recs
        status = "PASS ✅" if passed else "FAIL ❌"

        print(f"    Parsed: genre={result.parsed_profile.favorite_genre}, "
              f"mood={result.parsed_profile.favorite_mood}, "
              f"energy={result.parsed_profile.energy_target:.2f}")
        print(f"    Confidence: {result.confidence_score:.3f} "
              f"(threshold: {min_conf:.2f}) {'✓' if conf_ok else '✗'}")
        print(f"    Genre check: {'✓' if genre_ok else '✗'} "
              f"(expected={exp_genre}, got={result.parsed_profile.favorite_genre})")
        print(f"    Mood check:  {'✓' if mood_ok else '✗'} "
              f"(expected={exp_mood}, got={result.parsed_profile.favorite_mood})")
        print(f"    Has recs:    {'✓' if has_recs else '✗'} ({len(result.recommendations)} songs)")
        if result.recommendations:
            top = result.recommendations[0]
            print(f"    Top pick: {top.song} (score={top.total_score:.3f})")
        if result.refinement_attempts > 0:
            print(f"    Refinements: {result.refinement_attempts}")
        print(f"    Result: {status}")

        results.append({
            "desc": desc,
            "passed": passed,
            "confidence": result.confidence_score,
            "genre_ok": genre_ok,
            "mood_ok": mood_ok,
            "conf_ok": conf_ok,
            "refinements": result.refinement_attempts,
        })

    # ── Summary report ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    avg_conf = sum(r["confidence"] for r in results) / total if total > 0 else 0
    total_refinements = sum(r["refinements"] for r in results)

    print(f"\n  Total tests:        {total}")
    print(f"  Passed:             {passed} ✅")
    print(f"  Failed:             {failed} ❌")
    print(f"  Pass rate:          {passed/total:.0%}")
    print(f"  Avg confidence:     {avg_conf:.3f}")
    print(f"  Total refinements:  {total_refinements}")

    print(f"\n  {'Test':<35s} {'Status':<10s} {'Confidence':<12s}")
    print(f"  {'-'*35} {'-'*10} {'-'*12}")
    for r in results:
        status = "PASS ✅" if r["passed"] else "FAIL ❌"
        print(f"  {r['desc']:<35s} {status:<10s} {r['confidence']:.3f}")

    print("\n" + "=" * 70)

    # Exit with non-zero code if any tests failed
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_test_harness()
    sys.exit(exit_code)
