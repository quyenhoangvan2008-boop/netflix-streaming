"""
Automated Verification Tests for CineRecSys (Netflix-Style MovieLens-1M Demo).
Tests:
1. All 9 models load & produce clean Top-10 User Inspector recommendations (excluding watched items).
2. Interactive Cold-Start with 3-5 arbitrary seed movies works across all Stage-1 models.
3. Hybrid Search & Mood Discovery Agent distinguishes Title Search vs Vietnamese Mood query.
4. TMDB Client returns poster URL & embedded YouTube trailer link cleanly (live check when secrets present).
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.engine import RecSysEngine, translate_genres_vi
from core.tmdb_client import TMDBClient
from core.profile_store import build_cookie_sync_html, load_user_profile, save_user_profile, reset_user_profile


class TestNetflixRecSys(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RecSysEngine(BASE_DIR)

    def test_user_inspector_all_stage1_models(self):
        uid = 24
        hist = self.engine.get_user_watch_history(uid, limit=10)
        self.assertEqual(len(hist), 10)
        self.assertIn("user_rating", hist[0])

        seen = self.engine.seen_map[uid]
        for mk in ["popularity", "content_based", "user_cf", "item_cf", "svd", "lightgcn", "sasrec", "bert4rec"]:
            res = self.engine.recommend_for_user(uid, mk, top_k=10)
            self.assertEqual(len(res["items"]), 10, f"{mk} did not return 10 items")
            rec_ids = {it["movie_id"] for it in res["items"]}
            self.assertEqual(len(rec_ids), 10, f"{mk} returned duplicates")
            self.assertTrue(rec_ids.isdisjoint(seen), f"{mk} recommended already-seen items")

    def test_cold_start_custom_seeds_all_models(self):
        seeds = [1, 2571, 318, 858]  # Toy Story, The Matrix, Shawshank Redemption, The Godfather
        for mk in ["popularity", "content_based", "user_cf", "item_cf", "svd", "lightgcn", "sasrec", "bert4rec"]:
            res = self.engine.recommend_for_cold_seeds(seeds, mk, top_k=10)
            self.assertEqual(len(res["items"]), 10, f"Cold start failed on {mk}")
            rec_ids = {it["movie_id"] for it in res["items"]}
            self.assertTrue(rec_ids.isdisjoint(set(seeds)), f"Cold start {mk} leaked seed movies")

    def test_hybrid_search_and_mood_agent_local(self):
        # Title search mode
        r_title = self.engine.hybrid_search_agent("Toy Story", gemini_keys=None, top_n=5)
        self.assertEqual(r_title["intent_mode"], "title_search")
        self.assertTrue(any("Toy Story" in it["display_title"] for it in r_title["items"]))

        # Vietnamese mood mode (anti-false-positive check for Children's / low-rated "Life" titles)
        r_mood = self.engine.hybrid_search_agent(
            "hôm nay tôi buồn, tôi muốn xem phim nhẹ nhàng sâu lắng, có ý nghĩa",
            gemini_keys=None,
            top_n=8,
        )
        self.assertEqual(r_mood["intent_mode"], "mood_recommend")
        self.assertEqual(len(r_mood["items"]), 8)
        self.assertTrue(any("Drama" in it["genres"] for it in r_mood["items"]))
        for it in r_mood["items"]:
            self.assertGreaterEqual(it["ml_avg_rating"], 3.2, f"Low-rated title leaked: {it['display_title']}")
            self.assertNotIn("Children's", it["genres"], f"Children's title leaked: {it['display_title']}")
            self.assertNotIn("Animation", it["genres"], f"Animation title leaked: {it['display_title']}")

        self.assertEqual(translate_genres_vi(["Action", "Sci-Fi"]), "Hành động, Khoa học viễn tưởng")

    def test_tmdb_youtube_trailer_if_configured(self):
        secrets_file = os.path.join(BASE_DIR, ".streamlit", "secrets.toml")
        if not os.path.exists(secrets_file):
            self.skipTest("No local .streamlit/secrets.toml")
        import tomllib
        with open(secrets_file, "rb") as f:
            sec = tomllib.load(f)
        client = TMDBClient(
            read_token=sec.get("TMDB_READ_TOKEN", ""),
            api_key=sec.get("TMDB_API_KEY", ""),
        )
        info = client.get_movie_details(1, tmdb_id="862", title="Toy Story (1995)")
        self.assertTrue(info.get("poster_url", "").startswith("https://image.tmdb.org"))
        self.assertTrue(info.get("youtube_url", "").startswith("https://www.youtube.com/watch?v="))

    def test_realtime_stream_cold_and_active(self):
        # 1. Pure cold-start -> Popularity
        r_cold = self.engine.recommend_realtime_stream([], preferred_genres=[], current_movie_id=None, gemini_keys=None, top_k=12)
        self.assertEqual(r_cold["engine"], "Popularity")
        self.assertEqual(len(r_cold["items"]), 12)

        # 2. Genre-only cold-start -> Genre + Popularity
        r_genre = self.engine.recommend_realtime_stream([], preferred_genres=["Sci-Fi", "Action"], current_movie_id=None, gemini_keys=None, top_k=10)
        self.assertEqual(r_genre["engine"], "Genre + Popularity")
        self.assertEqual(len(r_genre["items"]), 10)

        # 3. Watching a movie (The Matrix = 2571) shifts LightGCN recommendations dynamically
        r_watch_1 = self.engine.recommend_realtime_stream([1], current_movie_id=2571, gemini_keys=None, top_k=10, use_llm=False)
        self.assertEqual(r_watch_1["engine"], "LightGCN")
        self.assertLess(r_watch_1["latency_ms"], 100.0)
        ids_matrix = [it["movie_id"] for it in r_watch_1["items"]]
        self.assertNotIn(2571, ids_matrix)
        self.assertNotIn(1, ids_matrix)

        # 4. Switching to Toy Story (1) shifts right-rail recommendations immediately
        r_watch_2 = self.engine.recommend_realtime_stream([2571], current_movie_id=1, gemini_keys=None, top_k=10)
        ids_toystory = [it["movie_id"] for it in r_watch_2["items"]]
        self.assertNotEqual(ids_matrix, ids_toystory)

    def test_profile_cookie_roundtrip(self):
        import tempfile, urllib.parse, json
        with tempfile.TemporaryDirectory() as tmpdir:
            prof = {"uid": "test123", "onboarded": True, "skipped": False, "genres": ["Drama"], "watched": [2571, 318]}
            save_user_profile(prof, tmpdir)
            loaded = load_user_profile({"cinerecsys_uid": "test123"}, tmpdir)
            self.assertTrue(loaded["onboarded"])
            self.assertEqual(loaded["watched"], [2571, 318])
            # Fallback from raw cookie without disk entry
            raw_cookie = urllib.parse.quote(json.dumps(prof))
            loaded2 = load_user_profile({"cinerecsys_profile": raw_cookie}, os.path.join(tmpdir, "empty"))
            self.assertTrue(loaded2["onboarded"])
            self.assertEqual(loaded2["genres"], ["Drama"])
            fresh = reset_user_profile("test123", tmpdir)
            self.assertFalse(fresh["onboarded"])
            self.assertIn("max-age=0", build_cookie_sync_html(fresh, clear=True))


if __name__ == "__main__":
    unittest.main()
