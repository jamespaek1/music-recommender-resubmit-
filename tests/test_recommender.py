from src.recommender import Song, UserProfile, Recommender


def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8, tempo_bpm=120,
            valence=0.9, danceability=0.8, acousticness=0.2,
            popularity=85, release_decade="2020s", mood_tag="euphoric",
            instrumentalness=0.05, liveness=0.15,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill", energy=0.4, tempo_bpm=80,
            valence=0.6, danceability=0.5, acousticness=0.9,
            popularity=50, release_decade="2020s", mood_tag="nostalgic",
            instrumentalness=0.85, liveness=0.05,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_score_increases_with_genre_match():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="chill",
        target_energy=0.5, likes_acoustic=False,
    )
    rec = make_small_recommender()
    pop_score = rec._score_song(user, rec.songs[0])[0]
    lofi_score = rec._score_song(user, rec.songs[1])[0]
    assert pop_score > lofi_score


def test_acoustic_bonus_applied():
    user = UserProfile(
        favorite_genre="lofi", favorite_mood="chill",
        target_energy=0.4, likes_acoustic=True,
    )
    rec = make_small_recommender()
    _, reasons = rec._score_song(user, rec.songs[1])
    reason_text = " ".join(reasons)
    assert "acoustic bonus" in reason_text


def test_instrumental_bonus_applied():
    user = UserProfile(
        favorite_genre="lofi", favorite_mood="chill",
        target_energy=0.4, likes_acoustic=True,
        likes_instrumental=True,
    )
    rec = make_small_recommender()
    _, reasons = rec._score_song(user, rec.songs[1])  # instrumentalness=0.85
    reason_text = " ".join(reasons)
    assert "instrumental bonus" in reason_text


def test_liveness_bonus_applied():
    user = UserProfile(
        favorite_genre="rock", favorite_mood="energetic",
        target_energy=0.9, likes_acoustic=False,
        likes_live=True,
    )
    rec = make_small_recommender()
    _, reasons = rec._score_song(user, rec.songs[0])  # liveness=0.15
    reason_text = " ".join(reasons)
    assert "liveness bonus" in reason_text


def test_diverse_recommend_limits_artist_repeats():
    songs = [
        Song(id=1, title="Song A", artist="Same Artist", genre="pop", mood="happy",
             energy=0.8, tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2,
             popularity=90, release_decade="2020s", mood_tag="euphoric",
             instrumentalness=0.05, liveness=0.15),
        Song(id=2, title="Song B", artist="Same Artist", genre="pop", mood="happy",
             energy=0.75, tempo_bpm=115, valence=0.85, danceability=0.75, acousticness=0.15,
             popularity=88, release_decade="2020s", mood_tag="euphoric",
             instrumentalness=0.04, liveness=0.12),
        Song(id=3, title="Song C", artist="Other Artist", genre="rock", mood="energetic",
             energy=0.5, tempo_bpm=100, valence=0.4, danceability=0.4, acousticness=0.3,
             popularity=60, release_decade="2010s", mood_tag="aggressive",
             instrumentalness=0.20, liveness=0.40),
    ]
    rec = Recommender(songs)
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False)
    results = rec.recommend_diverse(user, k=3, max_per_artist=1)

    artists = [song.artist for song, _, _ in results]
    assert artists.count("Same Artist") <= 2


def test_scoring_modes_produce_different_results():
    rec = make_small_recommender()
    user = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False)

    rec.set_mode("genre_first")
    genre_results = rec.recommend_with_details(user, k=2)

    rec.set_mode("energy_focused")
    energy_results = rec.recommend_with_details(user, k=2)

    genre_scores = [s for _, s, _ in genre_results]
    energy_scores = [s for _, s, _ in energy_results]
    assert genre_scores != energy_scores


def test_mood_tag_bonus_applied():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
        preferred_mood_tag="euphoric",
    )
    rec = make_small_recommender()
    _, reasons = rec._score_song(user, rec.songs[0])
    reason_text = " ".join(reasons)
    assert "mood tag match" in reason_text