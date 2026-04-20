# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

This recommender suggests 3–5 songs from a small catalog based on a user's preferred genre, mood, energy level, and additional attributes like mood tag, release decade, instrumentalness, and liveness. It assumes each user has a single, consistent taste profile and does not account for situational context (like time of day or current activity). This system is designed for classroom exploration of how content-based recommendation algorithms work. It is not intended for production use with real listeners, and should not be used to make commercial playlist or advertising decisions.

---

## 3. How the Model Works

The system works in three steps. First, it loads a catalog of songs from a CSV file, where each song is described by features like genre, mood, energy, danceability, acousticness, instrumentalness, liveness, popularity, release decade, and a detailed mood tag. Second, when given a user profile (which specifies a favorite genre, mood, target energy, and other preferences), it compares every song against that profile using a scoring function. The scoring function awards points for exact matches (like matching genre or mood) and calculates similarity scores for numerical features (like how close a song's energy is to the user's target). Different features have different weights — by default, a genre match is worth 2.0 points while a mood match is worth 1.0, reflecting the assumption that genre is a stronger signal of taste than mood.

Third, after scoring all songs, the system sorts them from highest to lowest score and returns the top results along with a breakdown of exactly which rules contributed to each score.

The system also supports four different scoring modes (balanced, genre_first, mood_first, energy_focused) that redistribute the weights to prioritize different aspects of musical preference. A diversity penalty feature can also be activated to prevent the same artist or genre from dominating the top results.

---

## 4. Data

The catalog contains 20 songs in `data/songs.csv`. Each song has 14 attributes: id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness, popularity, release_decade, mood_tag, instrumentalness, and liveness. Genres represented include pop (5 songs), rock (3), edm (3), lofi (3), folk (2), jazz (3), metal (1). Moods include happy, chill, sad, and energetic. Mood tags include euphoric, nostalgic, aggressive, and melancholic. The original starter dataset had 10 songs and was expanded with 10 additional tracks to improve diversity. Five new attributes were added beyond the original schema: popularity, release_decade, mood_tag, instrumentalness, and liveness.

The dataset does not include Latin, K-pop, classical, Afrobeats, or any non-English-language music. It mostly reflects a Western, English-language listening perspective, which limits the system's ability to serve users with global music tastes.

---

## 5. Strengths

The system works well for users whose taste aligns cleanly with a single genre and mood. A "happy pop" listener gets pop songs ranked at the top with clear explanations. A "chill lofi" listener sees lofi and folk tracks rise, with the instrumental and acoustic bonuses correctly boosting tracks that match those preferences. The scoring is fully transparent — every recommendation comes with a breakdown showing exactly which rules fired and how many points each contributed. This makes the system easy to understand, debug, and explain to non-programmers.

The modular scoring modes make it easy to experiment with different weighting strategies, and the diversity penalty effectively prevents repetitive results from the same artist or genre. The five additional attributes (popularity, decade, mood_tag, instrumentalness, liveness) add meaningful depth — for example, the Lofi listener's instrumental bonus correctly pushes high-instrumentalness tracks (0.85–0.90) above other chill songs that lack that quality.

---

## 6. Limitations and Bias

The system has several significant limitations. Genre matching is binary — "rock" and "metal" are treated as completely unrelated despite being closely related musically. The small catalog (20 songs) severely limits variety, especially for underrepresented genres like metal or country (1 song each). Pop has the most entries (5 songs), so fans of pop get better-differentiated results than fans of niche genres. The energy similarity calculation slightly favors mid-range energy songs (around 0.5) since they are "close enough" to both high and low targets, creating a subtle bias toward moderate-energy tracks.

The system also ignores feature interactions — it doesn't understand that "acoustic EDM" is contradictory, or that "sad" and "high energy" rarely coexist in the same song. It treats all users as if their taste can be captured by a single static profile, ignoring how preferences change by context (working out vs studying vs sleeping). If deployed in a real product, these limitations could create filter bubbles where users only see songs similar to what they already like, and users with underrepresented taste profiles would receive consistently worse recommendations than mainstream listeners.

---

## 7. Evaluation

Four user profiles were tested: High-Energy Pop Fan, Chill Lofi Listener, Intense Rock Lover, and an adversarial edge case combining sad mood with EDM genre, acoustic preference, and instrumental preference.

For the Pop profile, the top results were Midnight Drive, Golden Hour, and Wildfire — all matching the expected genre/mood combination. For the Lofi profile, Chill Horizon, Daydream, and Quiet Library ranked highest, correctly reflecting low-energy, acoustic, and instrumental preferences. The Rock profile surfaced Thunder Road as the clear #1, benefiting from genre match, mood match, energy match, liveness bonus, and decade match all firing simultaneously. These results matched musical intuition.

The edge-case profile exposed a limitation: EDM tracks still ranked highest despite the "sad" mood preference, because the +2.0 genre bonus overwhelmed the +1.0 mood bonus. The instrumental bonus did add points to EDM tracks (which have moderate instrumentalness) but was not enough to change the fundamental ranking.

One experiment doubled the energy weight and halved the genre weight. This caused genre boundaries to blur — high-energy songs from different genres started competing with genre-matched songs. Neon Jungle (EDM) rose from #4 to #3 for the Pop profile because its energy similarity was rewarded more heavily.

The scoring modes experiment showed clear differentiation: genre_first produced an all-pop top 5, mood_first brought in happy songs from jazz and folk, and energy_focused pulled in high-energy EDM alongside pop.

The diversity penalty successfully reduced The Neons from 3 appearances in the top 5 to just 2 (with a -1.5 penalty on the second), allowing Saxophone Nights (jazz) to enter the recommendations.

Unit tests (9 total, all passing) verified that genre matches increase scores, acoustic bonuses are correctly applied, instrumental and liveness bonuses fire when expected, different scoring modes produce different outputs, and the diversity penalty limits artist repetition.

---

## 8. Future Work

- Add partial genre matching using genre similarity scores (e.g., rock and metal share 0.7 similarity) instead of binary exact matching.
- Implement collaborative filtering by tracking multiple users' ratings and finding taste overlap.
- Expand the catalog to 100+ songs with better representation across global genres including Latin, K-pop, classical, and Afrobeats.
- Build a Streamlit web UI where users can adjust preferences with sliders and see results update in real time.
- Add contextual preferences (workout mode, study mode, sleep mode) that automatically shift weights.
- Implement feature interaction awareness so the system understands that certain preference combinations are contradictory.
- Use machine learning to learn optimal weights from user feedback rather than hard-coding them.

---

## 9. Personal Reflection

The most surprising thing was how much the dataset composition matters. Even with a well-designed scoring algorithm, the recommendations are only as diverse as the catalog allows. A perfectly tuned formula still fails jazz fans if there are only a few jazz songs to recommend. Building this also made me realize that the "magic" behind Spotify's Discover Weekly is less about a single clever formula and more about the massive scale of data and continuous feedback loops — millions of users generating behavioral signals that no 20-row CSV can replicate.

The weight system taught me that every default weight is a value judgment. Setting genre at 2.0 and mood at 1.0 implicitly says "genre matters twice as much as mood," which may not be true for every listener. In a real product, these weights would need to be learned from user behavior rather than hard-coded by a developer. Adding the five new attributes (especially instrumentalness and liveness) showed how each additional feature creates more ways to differentiate between songs — the Lofi listener's results improved noticeably because the instrumental bonus correctly separated "background lofi" from "vocal jazz" in ways that genre and mood alone couldn't capture.

Human judgment still matters in deciding what features to include, what weights to set as defaults, and how to handle edge cases where the algorithm produces technically correct but musically wrong results.