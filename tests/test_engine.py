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

from core.engine import RecSysEngine
from core.tmdb_client import TMDBClient


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

        # Vietnamese mood mode
        r_mood = self.engine.hybrid_search_agent(
            "hôm nay tôi muốn xem phim thể loại buồn, nhẹ nhàng, sâu lắng, ý nghĩa",
            gemini_keys=None,
            top_n=6,
        )
        self.assertEqual(r_mood["intent_mode"], "mood_recommend")
        self.assertEqual(len(r_mood["items"]), 6)
        self.assertTrue(any("Drama" in it["genres"] for it in r_mood["items"]))

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


if __name__ == "__main__":
    unittest.main()
