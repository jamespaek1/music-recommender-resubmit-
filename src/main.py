"""
Command-line runner for the Music Recommender Simulation.
Run with:  python3 -m src.main

Features demonstrated:
  - Phase 3: Default recommendations with scoring explanations
  - Phase 4: Multiple user profiles with output comparison
  - Extension 1: Advanced song features (popularity, decade, mood_tag, instrumentalness, liveness)
  - Extension 2: Multiple scoring modes (balanced, genre_first, mood_first, energy_focused)
  - Extension 3: Diversity and fairness logic (artist/genre penalties)
  - Extension 4: Visual summary table (tabulate)
"""

from src.recommender import (
    load_songs,
    recommend_songs,
    recommend_songs_diverse,
    SCORING_MODES,
)

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


# -----------------------------------------------------------------------
# Extension 4: Visual Summary Table
# -----------------------------------------------------------------------
def print_table(results):
    """Print results as a formatted table using tabulate or ASCII fallback."""
    if HAS_TABULATE:
        rows = []
        for rank, (song, score, explanation) in enumerate(results, start=1):
            rows.append([
                rank,
                song["title"],
                song["artist"],
                song["genre"],
                song["mood"],
                song["energy"],
                song.get("popularity", ""),
                song.get("mood_tag", ""),
                song.get("instrumentalness", ""),
                song.get("liveness", ""),
                f"{score:.2f}",
            ])
        headers = ["#", "Title", "Artist", "Genre", "Mood", "Energy", "Pop", "MoodTag", "Instr", "Live", "Score"]
        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()
        for rank, (song, score, explanation) in enumerate(results, start=1):
            print(f"  #{rank} Reasons: {explanation}")
    else:
        print(f"  {'#':<4} {'Title':<25} {'Artist':<18} {'Genre':<10} {'Score':<8}")
        print("  " + "-" * 70)
        for rank, (song, score, explanation) in enumerate(results, start=1):
            print(f"  {rank:<4} {song['title']:<25} {song['artist']:<18} {song['genre']:<10} {score:<8.2f}")
        print()
        for rank, (song, score, explanation) in enumerate(results, start=1):
            print(f"  #{rank} Reasons: {explanation}")
    print()


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def main():
    songs = load_songs("data/songs.csv")

    # ==================================================================
    # Define all user profiles
    # ==================================================================
    profiles = {
        "High-Energy Pop Fan": {
            "genre": "pop",
            "mood": "happy",
            "energy": 0.8,
            "likes_acoustic": False,
            "preferred_mood_tag": "euphoric",
            "preferred_decade": "2020s",
            "likes_instrumental": False,
            "likes_live": False,
        },
        "Chill Lofi Listener": {
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.3,
            "likes_acoustic": True,
            "preferred_mood_tag": "nostalgic",
            "preferred_decade": "2020s",
            "likes_instrumental": True,
            "likes_live": False,
        },
        "Intense Rock Lover": {
            "genre": "rock",
            "mood": "energetic",
            "energy": 0.9,
            "likes_acoustic": False,
            "preferred_mood_tag": "aggressive",
            "preferred_decade": "2010s",
            "likes_instrumental": False,
            "likes_live": True,
        },
        "Edge Case: Sad EDM + Acoustic": {
            "genre": "edm",
            "mood": "sad",
            "energy": 0.9,
            "likes_acoustic": True,
            "preferred_mood_tag": "melancholic",
            "preferred_decade": "1990s",
            "likes_instrumental": True,
            "likes_live": False,
        },
    }

    # ==================================================================
    # Phase 3 & 4: Recommendations for each profile (balanced mode)
    # ==================================================================
    for name, prefs in profiles.items():
        print_section(f"Profile: {name}")
        print(f"  Preferences: {prefs}\n")
        results = recommend_songs(prefs, songs, k=5)
        print_table(results)

    # ==================================================================
    # Phase 4: Comments comparing outputs
    # ==================================================================
    print_section("COMPARING PROFILE OUTPUTS")

    print("  Pop vs Lofi:")
    print("    The Pop profile is dominated by pop/EDM tracks with high energy (0.75-0.9)")
    print("    and happy moods. The Lofi profile produces completely different results —")
    print("    low-energy tracks with high acousticness and instrumentalness like Chill")
    print("    Horizon and Quiet Library rise to the top. The acoustic and instrumental")
    print("    bonuses only activate for the Lofi listener, further separating the outputs.\n")

    print("  Rock vs Pop:")
    print("    Both have high energy targets (0.9 vs 0.8) but genre is the primary")
    print("    differentiator. Rock profile surfaces Thunder Road and Broken Strings;")
    print("    Pop profile surfaces Midnight Drive and Golden Hour. The Rock listener")
    print("    also gets a liveness bonus, which rewards live-sounding tracks like")
    print("    Thunder Road (0.70) and Campfire Song (0.80).\n")

    print("  Edge Case vs All Others:")
    print("    The contradictory profile (sad + EDM + acoustic + instrumental) shows")
    print("    that genre bonus (+2.0) overwhelms mood (+1.0). EDM tracks rank highest")
    print("    despite not being sad. The acoustic bonus barely helps since EDM songs")
    print("    have ~0.05 acousticness. The instrumental bonus adds some points to EDM")
    print("    tracks (0.55-0.70) but not enough to change the overall ranking pattern.")
    print("    This reveals the system cannot handle conflicting preferences gracefully.\n")

    # ==================================================================
    # Extension 1: Advanced Features Demo
    # ==================================================================
    print_section("EXTENSION 1: Advanced Song Features (5 new attributes)")
    print("  New attributes added to songs.csv and scoring logic:")
    print("    1. popularity (0-100)      — bonus: (popularity/100) * 0.3")
    print("    2. release_decade          — +0.5 if decade matches user preference")
    print("    3. mood_tag                — +0.7 if mood tag matches user preference")
    print("    4. instrumentalness (0-1)  — bonus: instrumentalness * 0.6 (if user prefers)")
    print("    5. liveness (0-1)          — bonus: liveness * 0.5 (if user prefers)")
    print()
    print("  Example: Lofi Listener prefers instrumental=True, decade=2020s, mood_tag=nostalgic")
    print("  Lofi tracks with high instrumentalness (0.85-0.90) get significant extra points.\n")
    results = recommend_songs(profiles["Chill Lofi Listener"], songs, k=5)
    print_table(results)

    # ==================================================================
    # Extension 2: Multiple Scoring Modes
    # ==================================================================
    pop_prefs = profiles["High-Energy Pop Fan"]

    for mode_name in ["genre_first", "mood_first", "energy_focused"]:
        print_section(f"EXTENSION 2: Scoring Mode — {mode_name.upper()}")
        weights = SCORING_MODES[mode_name]
        print(f"  Weights: {weights}\n")
        results = recommend_songs(pop_prefs, songs, k=5, weights=weights)
        print_table(results)

    print_section("EXTENSION 2: Mode Comparison Summary")
    print("  GENRE_FIRST:    Only pop songs appear in top 5 (genre weight = 4.0)")
    print("  MOOD_FIRST:     Happy songs from any genre rise (mood weight = 3.0)")
    print("  ENERGY_FOCUSED: High-energy songs regardless of genre (energy weight = 4.0)")
    print("  This demonstrates that switching modes produces meaningfully different")
    print("  recommendations for the exact same user profile.\n")

    # ==================================================================
    # Extension 3: Diversity and Fairness Logic
    # ==================================================================
    print_section("EXTENSION 3: Diversity Penalty")
    print("  Rules: max 1 song per artist, max 2 songs per genre")
    print("  Penalty: -1.5 per duplicate artist, -1.0 per excess genre\n")

    print("  --- Standard (no diversity penalty) ---")
    standard = recommend_songs(pop_prefs, songs, k=5)
    print_table(standard)

    print("  --- With Diversity Penalty ---")
    diverse = recommend_songs_diverse(
        pop_prefs, songs, k=5, max_per_artist=1, max_per_genre=2
    )
    print_table(diverse)

    print("  Comparison: With diversity enabled, duplicate artists like The Neons")
    print("  are penalized after their first appearance, and genres are capped at 2.")
    print("  This allows songs from different artists and genres to enter the top 5,")
    print("  reducing filter bubble effects.\n")

    # ==================================================================
    # Experiment: Weight Shift
    # ==================================================================
    print_section("EXPERIMENT: energy x2, genre x0.5")
    print("  Modified weights: genre=1.0 (was 2.0), energy=3.0 (was 1.5)\n")
    exp_weights = dict(SCORING_MODES["balanced"])
    exp_weights["genre"] = 1.0
    exp_weights["energy"] = 3.0
    results = recommend_songs(pop_prefs, songs, k=5, weights=exp_weights)
    print_table(results)
    print("  Result: High-energy songs from non-pop genres now compete with pop songs.")
    print("  Genre boundaries blur when energy weight dominates the calculation.\n")


if __name__ == "__main__":
    main()