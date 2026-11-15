"""
Cookie & Local Profile Store for New vs. Returning Users.
Reads browser cookies via `st.context.cookies`, persists full profiles cleanly in
`.cache/user_profiles.json`, and syncs lightweight cookie payloads back to browser.
"""

import os
import json
import uuid
import urllib.parse
import threading
from typing import Dict, List, Optional

_PROFILE_LOCK = threading.Lock()

ONBOARDING_GENRES: List[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Children's",
]


def _default_profile(uid: Optional[str] = None) -> Dict:
    return {
        "uid": uid or uuid.uuid4().hex[:12],
        "onboarded": False,
        "skipped": False,
        "genres": [],
        "watched": [],
    }


def _load_store(store_path: str) -> Dict[str, Dict]:
    if not os.path.exists(store_path):
        return {}
    try:
        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_store(store_path: str, store: Dict[str, Dict]):
    try:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump(store, f, ensure_ascii=False)
    except Exception:
        pass


def load_user_profile(cookies: Dict, cache_dir: str) -> Dict:
    """Resolve user profile from browser cookie + local disk cache."""
    store_path = os.path.join(cache_dir, "user_profiles.json")
    uid = str(cookies.get("cinerecsys_uid", "") or "").strip()
    raw_cookie = str(cookies.get("cinerecsys_profile", "") or "").strip()

    profile = None
    if uid:
        with _PROFILE_LOCK:
            store = _load_store(store_path)
            if uid in store and isinstance(store[uid], dict):
                profile = dict(store[uid])
                profile["uid"] = uid

    if profile is None and raw_cookie:
        try:
            decoded = json.loads(urllib.parse.unquote(raw_cookie))
            if isinstance(decoded, dict):
                profile = {
                    "uid": uid or str(decoded.get("uid") or uuid.uuid4().hex[:12]),
                    "onboarded": bool(decoded.get("onboarded", False)),
                    "skipped": bool(decoded.get("skipped", False)),
                    "genres": [str(g) for g in decoded.get("genres", []) if g],
                    "watched": [int(m) for m in decoded.get("watched", []) if str(m).isdigit()],
                }
        except Exception:
            profile = None

    if profile is None:
        profile = _default_profile(uid or None)

    profile.setdefault("onboarded", False)
    profile.setdefault("skipped", False)
    profile.setdefault("genres", [])
    profile.setdefault("watched", [])
    return profile


def save_user_profile(profile: Dict, cache_dir: str):
    """Persist profile to `.cache/user_profiles.json`."""
    store_path = os.path.join(cache_dir, "user_profiles.json")
    uid = str(profile.get("uid") or uuid.uuid4().hex[:12])
    clean = {
        "uid": uid,
        "onboarded": bool(profile.get("onboarded", False)),
        "skipped": bool(profile.get("skipped", False)),
        "genres": [str(g) for g in profile.get("genres", [])],
        "watched": [int(m) for m in profile.get("watched", [])][-100:],
    }
    with _PROFILE_LOCK:
        store = _load_store(store_path)
        store[uid] = clean
        _save_store(store_path, store)


def reset_user_profile(uid: Optional[str], cache_dir: str) -> Dict:
    """Clear existing profile from store and create a fresh New-User profile."""
    store_path = os.path.join(cache_dir, "user_profiles.json")
    if uid:
        with _PROFILE_LOCK:
            store = _load_store(store_path)
            if uid in store:
                store.pop(uid, None)
                _save_store(store_path, store)
    return _default_profile()


def build_cookie_sync_html(profile: Dict, clear: bool = False) -> str:
    """
    Generate zero-height HTML script to write (or clear) browser cookies on the parent document
    so Returning Users bypass onboarding immediately on reload.
    """
    if clear:
        return """
        <script>
        (function() {
            try {
                const targetDoc = (window.parent && window.parent.document) ? window.parent.document : document;
                targetDoc.cookie = "cinerecsys_uid=; path=/; max-age=0; SameSite=Lax";
                targetDoc.cookie = "cinerecsys_profile=; path=/; max-age=0; SameSite=Lax";
            } catch (e) {}
        })();
        </script>
        """

    compact = {
        "uid": str(profile.get("uid", "")),
        "onboarded": bool(profile.get("onboarded", False)),
        "skipped": bool(profile.get("skipped", False)),
        "genres": list(profile.get("genres", []))[:10],
        "watched": [int(x) for x in profile.get("watched", [])][-20:],
    }
    uid_val = urllib.parse.quote(compact["uid"], safe="")
    prof_val = urllib.parse.quote(json.dumps(compact, separators=(",", ":")), safe="")
    return f"""
    <script>
    (function() {{
        try {{
            const targetDoc = (window.parent && window.parent.document) ? window.parent.document : document;
            targetDoc.cookie = "cinerecsys_uid={uid_val}; path=/; max-age=31536000; SameSite=Lax";
            targetDoc.cookie = "cinerecsys_profile={prof_val}; path=/; max-age=31536000; SameSite=Lax";
        }} catch (e) {{}}
    }})();
    </script>
    """
