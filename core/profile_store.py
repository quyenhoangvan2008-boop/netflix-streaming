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


def parse_compact_profile_json(raw: str, fallback_uid: Optional[str] = None) -> Optional[Dict]:
    """Parse URL-encoded or plain JSON profile payload from cookies or localStorage."""
    text = str(raw or "").strip()
    if not text:
        return None
    for candidate in (text, urllib.parse.unquote(text)):
        try:
            decoded = json.loads(candidate)
            if isinstance(decoded, dict):
                return {
                    "uid": str(decoded.get("uid") or fallback_uid or uuid.uuid4().hex[:12]),
                    "onboarded": bool(decoded.get("onboarded", False)),
                    "skipped": bool(decoded.get("skipped", False)),
                    "genres": [str(g) for g in decoded.get("genres", []) if g],
                    "watched": [int(m) for m in decoded.get("watched", []) if str(m).isdigit()],
                }
        except Exception:
            continue
    return None


def compact_profile_json(profile: Dict) -> str:
    """Serialize lightweight profile payload suitable for cookies (<4KB) & localStorage."""
    compact = {
        "uid": str(profile.get("uid", "")),
        "onboarded": bool(profile.get("onboarded", False)),
        "skipped": bool(profile.get("skipped", False)),
        "genres": list(profile.get("genres", []))[:10],
        "watched": [int(x) for x in profile.get("watched", [])][-20:],
    }
    return json.dumps(compact, separators=(",", ":"), ensure_ascii=False)


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
        parsed = parse_compact_profile_json(raw_cookie, fallback_uid=uid or None)
        if parsed is not None:
            profile = parsed

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


STORAGE_V2_JS = """
export default function(component) {
  const { data, setTriggerValue } = component;
  if (!data || !data.action) return;

  const isHttps = typeof window !== "undefined" && window.location && window.location.protocol === "https:";
  const writeCookie = (name, val, maxAgeSec) => {
    const base = `${name}=${val}; path=/; max-age=${maxAgeSec}`;
    document.cookie = isHttps ? `${base}; SameSite=None; Secure` : `${base}; SameSite=Lax`;
  };
  const readCookie = (name) => {
    const prefix = name + "=";
    const parts = (document.cookie || "").split(";");
    for (let i = 0; i < parts.length; i++) {
      const c = parts[i].trim();
      if (c.startsWith(prefix)) {
        try { return decodeURIComponent(c.slice(prefix.length)); } catch (e) { return c.slice(prefix.length); }
      }
    }
    return "";
  };

  if (data.action === "clear") {
    if (window.__cineSeq === data.seq) return;
    window.__cineSeq = data.seq;
    writeCookie("cinerecsys_uid", "", 0);
    writeCookie("cinerecsys_profile", "", 0);
    try { window.localStorage.removeItem("cinerecsys_profile"); } catch (e) {}
    return;
  }

  if (data.action === "save") {
    if (window.__cineSeq === data.seq) return;
    window.__cineSeq = data.seq;
    const raw = data.profile_json || "";
    writeCookie("cinerecsys_uid", encodeURIComponent(data.uid || ""), 31536000);
    writeCookie("cinerecsys_profile", encodeURIComponent(raw), 31536000);
    try { window.localStorage.setItem("cinerecsys_profile", raw); } catch (e) {}
    return;
  }

  if (data.action === "hydrate") {
    if (window.__cineHydrated) return;
    window.__cineHydrated = true;
    let raw = "";
    try { raw = window.localStorage.getItem("cinerecsys_profile") || ""; } catch (e) {}
    if (!raw) {
      raw = readCookie("cinerecsys_profile");
    }
    if (raw) {
      try {
        const parsed = JSON.parse(raw);
        if (parsed && parsed.onboarded) {
          writeCookie("cinerecsys_uid", encodeURIComponent(parsed.uid || ""), 31536000);
          writeCookie("cinerecsys_profile", encodeURIComponent(raw), 31536000);
          setTriggerValue("restored", raw);
        }
      } catch (e) {}
    }
  }
}
"""


def build_cookie_sync_html(profile: Dict, clear: bool = False) -> str:
    """
    Generate HTML/JS script executing in the app host page context (`st.html(..., unsafe_allow_javascript=True)`)
    to sync cookies & localStorage directly on the host origin without iframe sandbox restrictions.
    """
    if clear:
        return """
        <script>
        (function() {
            try {
                const isHttps = window.location && window.location.protocol === "https:";
                const s = isHttps ? "; path=/; max-age=0; SameSite=None; Secure" : "; path=/; max-age=0; SameSite=Lax";
                document.cookie = "cinerecsys_uid=" + s;
                document.cookie = "cinerecsys_profile=" + s;
                try { window.localStorage.removeItem("cinerecsys_profile"); } catch (e2) {}
            } catch (e) {}
        })();
        </script>
        """

    raw_json = compact_profile_json(profile)
    uid_val = urllib.parse.quote(str(profile.get("uid", "")), safe="")
    prof_val = urllib.parse.quote(raw_json, safe="")
    raw_js = json.dumps(raw_json)
    return f"""
    <script>
    (function() {{
        try {{
            const isHttps = window.location && window.location.protocol === "https:";
            const s = isHttps ? "; path=/; max-age=31536000; SameSite=None; Secure" : "; path=/; max-age=31536000; SameSite=Lax";
            document.cookie = "cinerecsys_uid={uid_val}" + s;
            document.cookie = "cinerecsys_profile={prof_val}" + s;
            try {{ window.localStorage.setItem("cinerecsys_profile", {raw_js}); }} catch (e2) {{}}
        }} catch (e) {{}}
    }})();
    </script>
    """
