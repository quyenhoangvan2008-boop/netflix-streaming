"""
TMDB v3 API Client with thread-safe disk & memory caching for posters, backdrops,
metadata, and embedded YouTube trailers.
"""

import os
import re
import json
import time
import threading
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import requests

TMDB_IMG_POSTER = "https://image.tmdb.org/t/p/w342"
TMDB_IMG_BACKDROP = "https://image.tmdb.org/t/p/w780"


def _strip_year(title: str):
    s = str(title).strip()
    m_yr = re.search(r"\((\d{4})\)\s*$", s)
    year = m_yr.group(1) if m_yr else None
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", s).strip()
    # Convert "Matrix, The" -> "The Matrix"
    m_art = re.match(r"^(.*?),\s*(The|A|An)$", clean)
    if m_art:
        clean = f"{m_art.group(2)} {m_art.group(1)}"
    return clean, year


def _pick_best_youtube_video(videos: List[Dict]) -> Dict:
    yt = [v for v in videos if str(v.get("site", "")).lower() == "youtube" and v.get("key")]
    if not yt:
        return {}
    priority = {"Trailer": 0, "Teaser": 1, "Clip": 2, "Featurette": 3}
    yt.sort(
        key=lambda v: (
            0 if v.get("official") else 1,
            priority.get(v.get("type", ""), 9),
        )
    )
    best = yt[0]
    key = best["key"]
    return {
        "youtube_key": key,
        "youtube_url": f"https://www.youtube.com/watch?v={key}",
        "youtube_embed_url": f"https://www.youtube.com/embed/{key}",
        "video_name": best.get("name", "Trailer"),
        "video_type": best.get("type", "Trailer"),
    }


class TMDBClient:
    def __init__(
        self,
        read_token: str = "",
        api_key: str = "",
        cache_path: Optional[str] = None,
    ):
        self.read_token = (read_token or "").strip()
        self.api_key = (api_key or "").strip()
        self.cache_path = cache_path
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict] = {}
        self._dirty = False

        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    self._cache = loaded
            except Exception:
                self._cache = {}

    def update_credentials(self, read_token: str = "", api_key: str = ""):
        if read_token:
            self.read_token = read_token.strip()
        if api_key:
            self.api_key = api_key.strip()

    @property
    def is_configured(self) -> bool:
        return bool(self.read_token or self.api_key)

    def _headers(self) -> Dict[str, str]:
        h = {"Accept": "application/json"}
        if self.read_token:
            h["Authorization"] = f"Bearer {self.read_token}"
        return h

    def _get(self, path: str, params: Optional[Dict] = None) -> Optional[Dict]:
        if not self.is_configured:
            return None
        p = dict(params or {})
        if not self.read_token and self.api_key:
            p["api_key"] = self.api_key
        url = f"https://api.themoviedb.org/3{path}"
        try:
            resp = requests.get(url, headers=self._headers(), params=p, timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            return None
        return None

    def _save_cache(self):
        if not self.cache_path or not self._dirty:
            return
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False)
                self._dirty = False
            except Exception:
                pass

    def get_movie_details(
        self,
        movie_id: int,
        tmdb_id: Optional[str] = None,
        title: str = "",
        fetch_videos: bool = True,
    ) -> Dict:
        key = str(movie_id)
        with self._lock:
            cached = self._cache.get(key)
        if cached and (not fetch_videos or cached.get("videos_checked")):
            return cached

        if not self.is_configured:
            return cached or {}

        tid = str(tmdb_id or "").strip()
        if tid.endswith(".0"):
            tid = tid[:-2]

        data = None
        if tid and tid.isdigit():
            data = self._get(
                f"/movie/{tid}",
                params={"append_to_response": "videos,credits", "language": "en-US"},
            )

        if data is None and title:
            clean_title, year = _strip_year(title)
            search_params = {"query": clean_title, "language": "en-US"}
            if year:
                search_params["year"] = year
            sr = self._get("/search/movie", params=search_params)
            results = (sr or {}).get("results", [])
            if not results and year:
                sr = self._get("/search/movie", params={"query": clean_title, "language": "en-US"})
                results = (sr or {}).get("results", [])
            if results:
                found_id = str(results[0]["id"])
                data = self._get(
                    f"/movie/{found_id}",
                    params={"append_to_response": "videos,credits", "language": "en-US"},
                )

        if not data:
            fallback = cached or {"videos_checked": True}
            with self._lock:
                self._cache[key] = fallback
                self._dirty = True
            return fallback

        poster_path = data.get("poster_path")
        backdrop_path = data.get("backdrop_path")
        vid_info = _pick_best_youtube_video((data.get("videos") or {}).get("results", []))
        credits = data.get("credits") or {}
        cast = [c.get("name") for c in credits.get("cast", [])[:5] if c.get("name")]
        directors = [c.get("name") for c in credits.get("crew", []) if c.get("job") == "Director" and c.get("name")]

        parsed = {
            "tmdb_id": str(data.get("id", tid)),
            "tmdb_title": data.get("title", ""),
            "poster_url": f"{TMDB_IMG_POSTER}{poster_path}" if poster_path else None,
            "backdrop_url": f"{TMDB_IMG_BACKDROP}{backdrop_path}" if backdrop_path else None,
            "tmdb_overview": data.get("overview", ""),
            "tagline": data.get("tagline", ""),
            "vote_average": data.get("vote_average"),
            "vote_count": data.get("vote_count"),
            "runtime": data.get("runtime"),
            "release_date": data.get("release_date", ""),
            "cast": cast,
            "director": ", ".join(directors[:2]),
            "videos_checked": True,
            **vid_info,
        }
        with self._lock:
            self._cache[key] = parsed
            self._dirty = True
        return parsed

    def prefetch_movies(self, items: List[Dict], max_workers: int = 8) -> Dict[int, Dict]:
        """Batch fetches TMDB metadata + posters + YouTube trailer links concurrently."""
        if not self.is_configured or not items:
            return {int(it["movie_id"]): self._cache.get(str(it["movie_id"]), {}) for it in items}

        out: Dict[int, Dict] = {}
        missing = []
        for it in items:
            mid = int(it["movie_id"])
            cached = self._cache.get(str(mid))
            if cached and cached.get("videos_checked"):
                out[mid] = cached
            else:
                missing.append(it)

        if missing:
            def _fetch(it):
                mid = int(it["movie_id"])
                return mid, self.get_movie_details(
                    mid,
                    tmdb_id=it.get("tmdb_id"),
                    title=it.get("title", ""),
                    fetch_videos=True,
                )

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                for mid, info in pool.map(_fetch, missing):
                    out[mid] = info
            self._save_cache()

        return out
