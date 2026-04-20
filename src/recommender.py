import csv
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float
    popularity: int = 0
    release_decade: str = "2020s"
    mood_tag: str = "nostalgic"
    instrumentalness: float = 0.0
    liveness: float = 0.0


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool
    preferred_mood_tag: str = "euphoric"
    preferred_decade: str = "2020s"
    min_popularity: int = 0
    likes_instrumental: bool = False
    likes_live: bool = False


SCORING_MODES = {
    "balanced": {
        "genre": 2.0, "mood": 1.0, "energy": 1.5, "valence": 0.5,
        "danceability": 0.5, "acousticness": 0.8, "popularity": 0.3,
        "decade": 0.5, "mood_tag": 0.7, "instrumentalness": 0.6,
        "liveness": 0.5,
    },
    "genre_first": {
        "genre": 4.0, "mood": 0.5, "energy": 0.5, "valence": 0.3,
        "danceability": 0.3, "acousticness": 0.3, "popularity": 0.2,
        "decade": 0.2, "mood_tag": 0.3, "instrumentalness": 0.2,
        "liveness": 0.2,
    },
    "mood_first": {
        "genre": 0.5, "mood": 3.0, "energy": 1.0, "valence": 1.5,
        "danceability": 0.5, "acousticness": 0.5, "popularity": 0.2,
        "decade": 0.2, "mood_tag": 2.0, "instrumentalness": 0.3,
        "liveness": 0.3,
    },
    "energy_focused": {
        "genre": 0.5, "mood": 0.5, "energy": 4.0, "valence": 0.5,
        "danceability": 2.0, "acousticness": 0.3, "popularity": 0.2,
        "decade": 0.2, "mood_tag": 0.3, "instrumentalness": 0.2,
        "liveness": 0.2,
    },
}


class Recommender:
    """OOP implementation of the recommendation logic."""

    def __init__(self, songs: List[Song], mode: str = "balanced"):
        self.songs = songs
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        if mode not in SCORING_MODES:
            raise ValueError(f"Unknown mode. Choose from: {list(SCORING_MODES.keys())}")
        self.mode = mode
        self.weights = SCORING_MODES[mode]

    def _score_song(self, user: UserProfile, song: Song) -> Tuple[float, List[str]]:
        w = self.weights
        score = 0.0
        reasons = []

        if song.genre.lower() == user.favorite_genre.lower():
            score += w["genre"]
            reasons.append(f"genre match: {song.genre} (+{w['genre']})")

        if song.mood.lower() == user.favorite_mood.lower():
            score += w["mood"]
            reasons.append(f"mood match: {song.mood} (+{w['mood']})")

        energy_sim = 1.0 - abs(song.energy - user.target_energy)
        energy_pts = round(energy_sim * w["energy"], 3)
        score += energy_pts
        reasons.append(f"energy similarity: {energy_sim:.2f} (+{energy_pts})")

        if user.favorite_mood.lower() == "happy":
            val_pts = round(song.valence * w["valence"], 3)
            score += val_pts
            reasons.append(f"valence bonus: {song.valence:.2f} (+{val_pts})")

        if user.target_energy >= 0.7:
            dance_pts = round(song.danceability * w["danceability"], 3)
            score += dance_pts
            reasons.append(f"danceability bonus: {song.danceability:.2f} (+{dance_pts})")

        if user.likes_acoustic:
            ac_pts = round(song.acousticness * w["acousticness"], 3)
            score += ac_pts
            reasons.append(f"acoustic bonus: {song.acousticness:.2f} (+{ac_pts})")

        pop_pts = round((song.popularity / 100.0) * w["popularity"], 3)
        score += pop_pts
        reasons.append(f"popularity: {song.popularity} (+{pop_pts})")

        if song.release_decade.lower() == user.preferred_decade.lower():
            score += w["decade"]
            reasons.append(f"decade match: {song.release_decade} (+{w['decade']})")

        if song.mood_tag.lower() == user.preferred_mood_tag.lower():
            score += w["mood_tag"]
            reasons.append(f"mood tag match: {song.mood_tag} (+{w['mood_tag']})")

        if user.likes_instrumental:
            inst_pts = round(song.instrumentalness * w["instrumentalness"], 3)
            score += inst_pts
            reasons.append(f"instrumental bonus: {song.instrumentalness:.2f} (+{inst_pts})")

        if user.likes_live:
            live_pts = round(song.liveness * w["liveness"], 3)
            score += live_pts
            reasons.append(f"liveness bonus: {song.liveness:.2f} (+{live_pts})")

        return round(score, 3), reasons

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored = [(song, self._score_song(user, song)[0]) for song in self.songs]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def recommend_with_details(self, user: UserProfile, k: int = 5) -> List[Tuple[Song, float, List[str]]]:
        scored = []
        for song in self.songs:
            s, reasons = self._score_song(user, song)
            scored.append((song, s, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def recommend_diverse(self, user: UserProfile, k: int = 5, max_per_artist: int = 1, max_per_genre: int = 2) -> List[Tuple[Song, float, List[str]]]:
        scored = []
        for song in self.songs:
            s, reasons = self._score_song(user, song)
            scored.append((song, s, reasons))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        artist_count = {}
        genre_count = {}

        for song, s, reasons in scored:
            a = song.artist.lower()
            g = song.genre.lower()
            artist_count.setdefault(a, 0)
            genre_count.setdefault(g, 0)

            penalty = 0.0
            penalty_reasons = list(reasons)

            if artist_count[a] >= max_per_artist:
                penalty += 1.5
                penalty_reasons.append(f"artist diversity penalty: {song.artist} (-1.5)")

            if genre_count[g] >= max_per_genre:
                penalty += 1.0
                penalty_reasons.append(f"genre diversity penalty: {song.genre} (-1.0)")

            adjusted = round(s - penalty, 3)
            results.append((song, adjusted, penalty_reasons))
            artist_count[a] += 1
            genre_count[g] += 1

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        score, reasons = self._score_song(user, song)
        lines = [f"Score: {score:.2f}"]
        for r in reasons:
            lines.append(f"  - {r}")
        return "\n".join(lines)


def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file and return a list of dictionaries."""
    songs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["id"] = int(row["id"])
            row["energy"] = float(row["energy"])
            row["tempo_bpm"] = float(row["tempo_bpm"])
            row["valence"] = float(row["valence"])
            row["danceability"] = float(row["danceability"])
            row["acousticness"] = float(row["acousticness"])
            row["popularity"] = int(row.get("popularity", 0))
            row["release_decade"] = row.get("release_decade", "2020s")
            row["mood_tag"] = row.get("mood_tag", "nostalgic")
            row["instrumentalness"] = float(row.get("instrumentalness", 0.0))
            row["liveness"] = float(row.get("liveness", 0.0))
            songs.append(row)
    print(f"Loaded songs: {len(songs)}")
    return songs


def score_song(user_prefs: Dict, song: Dict, weights: Optional[Dict] = None) -> Tuple[float, str]:
    """Score a single song dict against user preferences and return (score, explanation)."""
    if weights is None:
        weights = SCORING_MODES["balanced"]

    score = 0.0
    reasons = []

    if song.get("genre", "").lower() == user_prefs.get("genre", "").lower():
        score += weights["genre"]
        reasons.append(f"genre match: {song['genre']} (+{weights['genre']})")

    if song.get("mood", "").lower() == user_prefs.get("mood", "").lower():
        score += weights["mood"]
        reasons.append(f"mood match: {song['mood']} (+{weights['mood']})")

    if "energy" in user_prefs and "energy" in song:
        energy_sim = 1.0 - abs(song["energy"] - user_prefs["energy"])
        energy_pts = round(energy_sim * weights["energy"], 3)
        score += energy_pts
        reasons.append(f"energy similarity: {energy_sim:.2f} (+{energy_pts})")

    if user_prefs.get("mood", "").lower() == "happy" and "valence" in song:
        val_pts = round(song["valence"] * weights["valence"], 3)
        score += val_pts
        reasons.append(f"valence bonus: {song['valence']:.2f} (+{val_pts})")

    if user_prefs.get("energy", 0) >= 0.7 and "danceability" in song:
        dance_pts = round(song["danceability"] * weights["danceability"], 3)
        score += dance_pts
        reasons.append(f"danceability bonus: {song['danceability']:.2f} (+{dance_pts})")

    if user_prefs.get("likes_acoustic") and "acousticness" in song:
        ac_pts = round(song["acousticness"] * weights["acousticness"], 3)
        score += ac_pts
        reasons.append(f"acoustic bonus: {song['acousticness']:.2f} (+{ac_pts})")

    if "popularity" in song:
        pop_pts = round((song["popularity"] / 100.0) * weights.get("popularity", 0.3), 3)
        score += pop_pts
        reasons.append(f"popularity: {song['popularity']} (+{pop_pts})")

    if song.get("release_decade", "").lower() == user_prefs.get("preferred_decade", "").lower():
        dec_w = weights.get("decade", 0.5)
        score += dec_w
        reasons.append(f"decade match: {song['release_decade']} (+{dec_w})")

    if song.get("mood_tag", "").lower() == user_prefs.get("preferred_mood_tag", "").lower():
        mt_w = weights.get("mood_tag", 0.7)
        score += mt_w
        reasons.append(f"mood tag match: {song['mood_tag']} (+{mt_w})")

    if user_prefs.get("likes_instrumental") and "instrumentalness" in song:
        inst_w = weights.get("instrumentalness", 0.6)
        inst_pts = round(song["instrumentalness"] * inst_w, 3)
        score += inst_pts
        reasons.append(f"instrumental bonus: {song['instrumentalness']:.2f} (+{inst_pts})")

    if user_prefs.get("likes_live") and "liveness" in song:
        live_w = weights.get("liveness", 0.5)
        live_pts = round(song["liveness"] * live_w, 3)
        score += live_pts
        reasons.append(f"liveness bonus: {song['liveness']:.2f} (+{live_pts})")

    explanation = "; ".join(reasons) if reasons else "no matching features"
    return round(score, 3), explanation


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Optional[Dict] = None) -> List[Tuple[Dict, float, str]]:
    """Return the top k songs as (song_dict, score, explanation) tuples."""
    scored = []
    for song in songs:
        s, explanation = score_song(user_prefs, song, weights)
        scored.append((song, s, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def recommend_songs_diverse(user_prefs: Dict, songs: List[Dict], k: int = 5, weights: Optional[Dict] = None, max_per_artist: int = 1, max_per_genre: int = 2) -> List[Tuple[Dict, float, str]]:
    """Return top k songs with diversity penalties applied."""
    scored = []
    for song in songs:
        s, explanation = score_song(user_prefs, song, weights)
        scored.append((song, s, explanation))
    scored.sort(key=lambda x: x[1], reverse=True)

    results = []
    artist_count = {}
    genre_count = {}

    for song, s, explanation in scored:
        a = song.get("artist", "").lower()
        g = song.get("genre", "").lower()
        artist_count.setdefault(a, 0)
        genre_count.setdefault(g, 0)

        penalty = 0.0
        extra = ""

        if artist_count[a] >= max_per_artist:
            penalty += 1.5
            extra += f"; artist diversity penalty: {song['artist']} (-1.5)"

        if genre_count[g] >= max_per_genre:
            penalty += 1.0
            extra += f"; genre diversity penalty: {song['genre']} (-1.0)"

        adjusted = round(s - penalty, 3)
        results.append((song, adjusted, explanation + extra))
        artist_count[a] += 1
        genre_count[g] += 1

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:k]