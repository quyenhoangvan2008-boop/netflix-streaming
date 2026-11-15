"""
Gemini 3.5 Flash-Lite Reranker & Hybrid Search/Discovery Agent
with multi-key round-robin rotation and calibrated Stage-1 score fusion.
"""

import json
import time
import threading
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

GEMINI_MODEL_NAME = "gemini-3.5-flash-lite"


class KeyRotator:
    def __init__(self, api_keys: List[str]):
        cleaned = [k.strip() for k in api_keys if k and k.strip()]
        self._keys = list(dict.fromkeys(cleaned))
        self._idx = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        return len(self._keys)

    def next_key(self) -> Optional[str]:
        if not self._keys:
            return None
        with self._lock:
            k = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return k


def _call_gemini_json(api_keys: List[str], prompt: str, temperature: float = 0.15) -> Tuple[Dict, int]:
    from google import genai

    rotator = KeyRotator(api_keys)
    if rotator.count == 0:
        raise ValueError("Chưa cấu hình Gemini API Key.")

    last_err = None
    attempts = min(max(3, rotator.count * 2), 8)
    for attempt in range(attempts):
        key = rotator.next_key()
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": temperature,
                },
            )
            raw = (resp.text or "").strip()
            if not raw:
                raise ValueError("Empty response from Gemini")
            parsed = json.loads(raw)
            return parsed, attempt + 1
        except Exception as exc:
            last_err = exc
            time.sleep(min(2.0, 0.35 * (attempt + 1)))
    raise RuntimeError(f"Gemini API error after {attempts} attempts: {last_err}")


def build_rerank_prompt(
    history_titles: List[str],
    cand_items: List[Dict],
    retriever_name: str,
    top_n: int = 10,
) -> str:
    hist_txt = "\n".join([f"{i}. {t}" for i, t in enumerate(history_titles[-10:], 1)]) if history_titles else "(Cold-start / new user)"
    cand_lines = []
    for rank, it in enumerate(cand_items, 1):
        cats = "|".join(it.get("categories", [])[:3])
        desc = str(it.get("description", "")).strip()
        if len(desc) > 160:
            desc = desc[:160].rsplit(" ", 1)[0] + "..."
        cand_lines.append(f"#{rank} (ID={it['movie_id']}, score={it.get('score', 0.0):.2f}) {it.get('display_title', it.get('title'))} [{cats}] - {desc}")

    return (
        "You are an expert MovieLens Recommendation Reranker.\n"
        f"Analyze the user's liked watch history below and rerank the Top-{len(cand_items)} candidates from {retriever_name}.\n"
        "Calibration rule: preserve strong collaborative top picks (#1-#5) while promoting strong semantic, mood, and genre matches.\n\n"
        "[USER WATCH HISTORY]\n"
        f"{hist_txt}\n\n"
        f"[{retriever_name.upper()} CANDIDATES]\n"
        + "\n".join(cand_lines)
        + f'\n\nReturn ONLY valid JSON:\n{{"user_intent": "<concise 1-sentence taste analysis in Vietnamese>", "ranked_ids": [<exact {top_n} candidate integer IDs best-first>]}}'
    )


def rerank_with_gemini_calibrated(
    api_keys: List[str],
    history_titles: List[str],
    cand_items: List[Dict],
    retriever_name: str = "LightGCN",
    top_n: int = 10,
    alpha_fuse: float = 0.30,
) -> Dict:
    valid_ids: Set[int] = {int(c["movie_id"]) for c in cand_items}
    prompt = build_rerank_prompt(history_titles, cand_items, retriever_name, top_n=top_n)
    t0 = time.perf_counter()
    data, tries = _call_gemini_json(api_keys, prompt, temperature=0.15)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if isinstance(data, list):
        if len(data) == 1 and isinstance(data[0], dict):
            data = data[0]
            raw_list = data.get("ranked_ids") or data.get("reranked_ids") or data.get("ids") or []
            user_intent = str(data.get("user_intent", ""))
        else:
            raw_list = data
            user_intent = ""
    elif isinstance(data, dict):
        raw_list = data.get("ranked_ids") or data.get("reranked_ids") or data.get("ids") or []
        user_intent = str(data.get("user_intent", ""))
    else:
        raw_list = []
        user_intent = ""

    gemini_ids: List[int] = []
    for x in raw_list:
        if isinstance(x, dict):
            x = x.get("item_id") or x.get("movie_id") or x.get("id")
        try:
            v = int(x)
            if v in valid_ids and v not in gemini_ids:
                gemini_ids.append(v)
        except (TypeError, ValueError):
            continue

    if not gemini_ids:
        raise ValueError("Gemini returned no valid candidate IDs")

    # Calibrated fusion of Stage-1 scores and Gemini listwise ranking
    cids = [int(c["movie_id"]) for c in cand_items]
    s1 = np.array([float(c.get("score", len(cand_items) - i)) for i, c in enumerate(cand_items)], dtype=np.float32)
    s1_std = float(s1.std())
    s1_z = (s1 - s1.mean()) / (s1_std + 1e-6) if s1_std > 1e-6 else np.linspace(1.5, -1.5, len(s1), dtype=np.float32)
    s1_norm = np.tanh(0.6 * s1_z)

    gem_pos = {mid: idx for idx, mid in enumerate(gemini_ids)}
    gem_scores = np.zeros(len(cids), dtype=np.float32)
    max_r = max(1, len(gemini_ids) - 1)
    for i, cid in enumerate(cids):
        gem_scores[i] = (1.0 - 0.9 * (gem_pos[cid] / max_r)) if cid in gem_pos else -0.35

    comb = (1.0 - alpha_fuse) * s1_norm + alpha_fuse * gem_scores
    order = np.argsort(-comb)
    fused_ids = [cids[i] for i in order[:top_n]]

    return {
        "model": GEMINI_MODEL_NAME,
        "ranked_ids": fused_ids,
        "pure_gemini_ids": gemini_ids[:top_n],
        "user_intent": user_intent,
        "latency_ms": round(latency_ms, 1),
        "attempts": tries,
    }


def agent_curate_movies(
    api_keys: List[str],
    user_query: str,
    local_candidates: List[Dict],
    intent_mode: str = "mood",
    top_n: int = 8,
) -> Dict:
    """
    Hybrid Search + Mood Discovery Agent using Gemini 3.5 Flash-Lite over
    top MovieLens-1M candidate matches.
    """
    cand_lines = []
    valid_ids = set()
    for c in local_candidates:
        mid = int(c["movie_id"])
        valid_ids.add(mid)
        cats = "|".join(c.get("categories", [])[:3])
        desc = str(c.get("description", "")).strip()
        if len(desc) > 180:
            desc = desc[:180].rsplit(" ", 1)[0] + "..."
        cand_lines.append(
            f"ID={mid} | {c.get('display_title', c.get('title'))} [{cats}] "
            f"(ML ★{c.get('ml_avg_rating', 0):.1f}, {c.get('popularity_count', 0)} votes) — {desc}"
        )

    prompt = (
        "You are a warm, knowledgeable bilingual (Vietnamese & English) Netflix Movie Discovery Agent.\n"
        f"User query ({intent_mode} mode): \"{user_query}\"\n\n"
        "From the MovieLens-1M candidate pool below (already pre-filtered for relevance), select the best matching movies "
        f"(up to {top_n} movies) and explain concisely WHY each film fits the user's request.\n\n"
        "[MOVIELENS CANDIDATES]\n"
        + "\n".join(cand_lines)
        + "\n\nReturn ONLY valid JSON with exact schema:\n"
        '{"assistant_reply": "<friendly natural response in the same language as the user query>", '
        '"picks": [{"movie_id": <int ID from pool>, "reason": "<1 crisp sentence explaining why it fits>"}]}'
    )

    t0 = time.perf_counter()
    data, _ = _call_gemini_json(api_keys, prompt, temperature=0.25)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    reply = str(data.get("assistant_reply", "")) if isinstance(data, dict) else ""
    picks_raw = data.get("picks", []) if isinstance(data, dict) else []

    picks: List[Dict] = []
    seen = set()
    for p in picks_raw:
        if not isinstance(p, dict):
            continue
        try:
            mid = int(p.get("movie_id"))
        except (TypeError, ValueError):
            continue
        if mid in valid_ids and mid not in seen:
            seen.add(mid)
            picks.append({"movie_id": mid, "reason": str(p.get("reason", "")).strip()})

    if not picks:
        raise ValueError("Agent returned no valid candidate picks")

    return {
        "mode": "gemini_agent",
        "model": GEMINI_MODEL_NAME,
        "assistant_reply": reply,
        "picks": picks[:top_n],
        "latency_ms": round(latency_ms, 1),
    }
