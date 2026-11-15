"""
CineRecSys — Modern Netflix / YouTube Real-Time Recommender
- Blurred Home + New User Onboarding Popup (Vietnamese genres, live title filter, click-to-toggle cards, Skip)
- Browser Cookie persistence (New User vs Returning User)
- Full-card clickable 16:9 banners + large movie titles (zero extra "Xem" buttons)
- Pure LightGCN vs LightGCN + Gemini 3.5 Flash-Lite live toggle
- Dedicated YouTube-style Watch Page with real-time recommendation updates
"""

import html
import os
from typing import Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

from core.engine import RecSysEngine, translate_genre_vi, translate_genres_vi
from core.tmdb_client import TMDBClient
from core.profile_store import (
    ONBOARDING_GENRES,
    build_cookie_sync_html,
    load_user_profile,
    reset_user_profile,
    save_user_profile,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache")
CACHE_PATH = os.path.join(CACHE_DIR, "tmdb_cache.json")

st.set_page_config(
    page_title="CineRecSys",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Modern Netflix / YouTube Dark Cinema CSS
# ------------------------------------------------------------------
st.markdown(
    """
<style>
  [data-testid="stSidebar"],
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"],
  #MainMenu,
  footer {
    display: none !important;
  }

  /* Keep Streamlit Cloud toolbar translucent & give header clear breathing room */
  [data-testid="stHeader"] {
    background: rgba(11, 13, 20, 0.72) !important;
    backdrop-filter: blur(10px) !important;
    height: 2.85rem !important;
  }

  .stApp {
    background: radial-gradient(circle at 18% 0%, #191325 0%, #0B0D14 48%, #07090E 100%);
    color: #F8FAFC;
  }

  .block-container {
    max-width: 1400px !important;
    padding-top: 3.85rem !important;
    padding-bottom: 3.5rem !important;
    padding-left: 2.2rem !important;
    padding-right: 2.2rem !important;
  }

  /* Dialog blurred backdrop over Home screen */
  [data-baseweb="modal"] > div:first-child,
  div[role="dialog"]::backdrop {
    backdrop-filter: blur(12px) brightness(0.45) !important;
    -webkit-backdrop-filter: blur(12px) brightness(0.45) !important;
    background: rgba(5, 7, 12, 0.74) !important;
  }
  [data-testid="stDialog"] div[role="dialog"] {
    background: #10141F !important;
    border: 1px solid rgba(229, 9, 20, 0.38) !important;
    border-radius: 18px !important;
    box-shadow: 0 28px 64px rgba(0, 0, 0, 0.75) !important;
  }

  /* Top Navigation Bar */
  .brand-btn-wrap .stButton button {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0.2rem !important;
    font-size: 1.65rem !important;
    font-weight: 900 !important;
    letter-spacing: -0.6px !important;
    color: #FFFFFF !important;
    box-shadow: none !important;
    white-space: nowrap !important;
  }
  .brand-btn-wrap .stButton button:hover {
    color: #E50914 !important;
  }
  .reset-btn-wrap .stButton button {
    white-space: nowrap !important;
  }

  .section-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0.75rem 0 1rem 0;
  }
  .section-heading {
    font-size: 1.38rem;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.3px;
  }
  .latency-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #CBD5E1;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
  }
  .latency-pill b {
    color: #34D399;
  }

  /* Clickable Full-Card Overlay System (Zero visible buttons) */
  div[class*="st-key-cardwrap_"] {
    position: relative !important;
    display: block !important;
    width: 100% !important;
    min-height: 80px !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    cursor: pointer !important;
    transition: transform 0.22s cubic-bezier(.22,1,.36,1), box-shadow 0.22s ease !important;
  }
  div[class*="st-key-cardwrap_"]:hover {
    transform: translateY(-4px) scale(1.015) !important;
    box-shadow: 0 16px 34px rgba(0, 0, 0, 0.55), 0 0 0 1.5px rgba(229, 9, 20, 0.78) !important;
  }
  div[class*="st-key-cardwrap_"] > div[data-testid="stElementContainer"]:last-child,
  div[class*="st-key-cardwrap_"] > div.element-container:last-child {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    z-index: 25 !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  div[class*="st-key-cardwrap_"] .stButton,
  div[class*="st-key-cardwrap_"] .stButton > button {
    position: absolute !important;
    inset: 0 !important;
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    opacity: 0 !important;
    cursor: pointer !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
  }

  /* Movie Card Visuals (16:9 Banner + Big Bold Title) */
  .nf-card {
    background: linear-gradient(180deg, #161B28 0%, #10141F 100%);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    overflow: hidden;
    transition: border-color 0.2s ease, background 0.2s ease;
  }
  .nf-card.nf-selected {
    background: linear-gradient(180deg, rgba(229, 9, 20, 0.22) 0%, #18131E 100%);
    border: 2px solid #E50914;
    box-shadow: 0 0 22px rgba(229, 9, 20, 0.32);
  }
  .nf-thumb-box {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #1A2030;
    overflow: hidden;
  }
  .nf-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.3s ease;
  }
  div[class*="st-key-cardwrap_"]:hover .nf-thumb-box img {
    transform: scale(1.05);
  }
  .nf-thumb-grad {
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(0,0,0,0) 48%, rgba(16,20,31,0.92) 100%);
  }
  .nf-thumb-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.2rem;
    text-align: center;
    background: linear-gradient(135deg, #251B2E 0%, #111827 100%);
    color: #E2E8F0;
    font-weight: 800;
    font-size: 1.12rem;
  }
  .nf-star-badge {
    position: absolute;
    right: 10px;
    bottom: 10px;
    background: rgba(11, 13, 20, 0.86);
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #FACC15;
    font-size: 0.82rem;
    font-weight: 700;
    padding: 3px 9px;
    border-radius: 999px;
    backdrop-filter: blur(4px);
  }
  .nf-check-badge {
    position: absolute;
    left: 10px;
    top: 10px;
    background: #E50914;
    color: #FFFFFF;
    font-size: 0.78rem;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 999px;
    box-shadow: 0 4px 12px rgba(229, 9, 20, 0.45);
  }
  .nf-card-body {
    padding: 0.9rem 1.05rem 1.05rem 1.05rem;
  }
  .nf-title {
    font-size: 1.14rem;
    font-weight: 800;
    color: #FFFFFF;
    line-height: 1.32;
    min-height: 2.64em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    letter-spacing: -0.2px;
  }
  .nf-why {
    font-size: 0.84rem;
    color: #94A3B8;
    margin-top: 0.4rem;
    line-height: 1.35;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .nf-genre {
    font-size: 0.86rem;
    font-weight: 500;
    color: #94A3B8;
    margin-top: 0.35rem;
    line-height: 1.3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Watch Page Horizontal Right-Side Recommendation Row (Zero button, full-row clickable) */
  .side-row-card {
    display: flex;
    align-items: center;
    gap: 14px;
    background: #131722;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 8px;
  }
  .side-row-thumb {
    position: relative;
    width: 172px;
    min-width: 172px;
    aspect-ratio: 16 / 9;
    border-radius: 8px;
    overflow: hidden;
    background: #1A2030;
  }
  .side-row-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .side-row-info {
    flex: 1;
    min-width: 0;
    padding-right: 4px;
  }
  .side-row-title {
    font-size: 1.02rem;
    font-weight: 800;
    color: #F8FAFC;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .side-row-star {
    font-size: 0.82rem;
    color: #FACC15;
    font-weight: 700;
    margin-top: 0.25rem;
  }
  .side-row-genre {
    font-size: 0.80rem;
    color: #94A3B8;
    margin-top: 0.28rem;
    line-height: 1.25;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Watch Page Main Info */
  .watch-title {
    font-size: 1.85rem;
    font-weight: 900;
    color: #FFFFFF;
    margin: 0.9rem 0 0.4rem 0;
    line-height: 1.22;
    letter-spacing: -0.4px;
  }
  .watch-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 0.9rem;
  }
  .meta-chip {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #E2E8F0;
    font-size: 0.86rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 999px;
  }
  .meta-chip.genre {
    background: rgba(229, 9, 20, 0.16);
    border-color: rgba(229, 9, 20, 0.38);
    color: #FECDD3;
  }
  .watch-desc {
    background: #131722;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.15rem 1.3rem;
    color: #E2E8F0;
    font-size: 0.98rem;
    line-height: 1.65;
  }
  .agent-banner {
    background: rgba(20, 24, 36, 0.95);
    border-left: 3px solid #E50914;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 1.15rem;
    font-size: 0.98rem;
    color: #F8FAFC;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Secrets & Cached Singletons
# ------------------------------------------------------------------
def _load_secrets() -> Dict:
    tmdb_token = os.getenv("TMDB_READ_TOKEN", "")
    tmdb_key = os.getenv("TMDB_API_KEY", "")
    gemini_keys: List[str] = []

    try:
        if "TMDB_READ_TOKEN" in st.secrets:
            tmdb_token = str(st.secrets["TMDB_READ_TOKEN"])
        if "TMDB_API_KEY" in st.secrets:
            tmdb_key = str(st.secrets["TMDB_API_KEY"])
        if "GEMINI_API_KEYS" in st.secrets:
            val = st.secrets["GEMINI_API_KEYS"]
            gemini_keys = [str(x).strip() for x in val] if isinstance(val, (list, tuple)) else [str(val).strip()]
        elif "GEMINI_API_KEY" in st.secrets:
            gemini_keys = [str(st.secrets["GEMINI_API_KEY"]).strip()]
    except Exception:
        pass

    if not gemini_keys and os.getenv("GEMINI_API_KEY"):
        gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]

    return {
        "tmdb_read_token": tmdb_token,
        "tmdb_api_key": tmdb_key,
        "gemini_keys": [k for k in gemini_keys if k],
    }


@st.cache_resource(show_spinner="Đang khởi tạo CineRecSys...")
def get_engine() -> RecSysEngine:
    return RecSysEngine(BASE_DIR)


@st.cache_resource
def get_tmdb_client(read_token: str, api_key: str) -> TMDBClient:
    return TMDBClient(read_token=read_token, api_key=api_key, cache_path=CACHE_PATH)


secrets = _load_secrets()
engine = get_engine()
tmdb = get_tmdb_client(secrets["tmdb_read_token"], secrets["tmdb_api_key"])
gemini_keys = secrets["gemini_keys"]


# ------------------------------------------------------------------
# Session, Profile & Navigation Helpers
# ------------------------------------------------------------------
def _init_session_profile():
    if "profile" not in st.session_state:
        cookies_dict = dict(st.context.cookies) if hasattr(st, "context") and hasattr(st.context, "cookies") else {}
        st.session_state.profile = load_user_profile(cookies_dict, CACHE_DIR)
        st.session_state._sync_cookie = True
    st.session_state.setdefault("view", "home")
    st.session_state.setdefault("watch_movie_id", None)
    st.session_state.setdefault("search_query", "")
    st.session_state.setdefault("search_result", None)
    st.session_state.setdefault("rec_mode", "pure")
    st.session_state.setdefault("ob_selected_movies", [])
    st.session_state.setdefault("ob_selected_genres", [])
    st.session_state.setdefault("ob_movie_dropdown", list(st.session_state.ob_selected_movies))


def _persist_profile(profile: Dict):
    st.session_state.profile = profile
    save_user_profile(profile, CACHE_DIR)
    st.session_state._sync_cookie = True
    st.session_state._clear_cookie = False


def _on_dismiss_onboarding():
    profile = dict(st.session_state.profile)
    if not profile.get("onboarded"):
        profile["onboarded"] = True
        profile["skipped"] = True
        profile["genres"] = []
        profile["watched"] = []
        _persist_profile(profile)


def _open_watch_page(movie_id: int):
    profile = dict(st.session_state.profile)
    watched = [int(x) for x in profile.get("watched", [])]
    if int(movie_id) in watched:
        watched.remove(int(movie_id))
    watched.append(int(movie_id))
    profile["watched"] = watched[-100:]
    profile["onboarded"] = True
    _persist_profile(profile)
    st.session_state.watch_movie_id = int(movie_id)
    st.session_state.view = "watch"
    st.rerun()


def _run_unified_search(query: str):
    q = (query or "").strip()
    if not q:
        return
    profile = dict(st.session_state.profile)
    if not profile.get("onboarded"):
        profile["onboarded"] = True
        _persist_profile(profile)
    res = engine.hybrid_search_agent(q, gemini_keys=gemini_keys, top_n=9)
    st.session_state.search_query = q
    st.session_state.search_result = res
    st.session_state.view = "home"
    st.rerun()


def _thumb_url(item: Dict, tmdb_map: Dict[int, Dict]) -> Optional[str]:
    mid = int(item["movie_id"])
    tinfo = tmdb_map.get(mid, {})
    return tinfo.get("backdrop_url") or tinfo.get("poster_url")


def _card_html(
    item: Dict,
    tmdb_map: Dict[int, Dict],
    selected: bool = False,
    show_why: bool = False,
    compact_ob: bool = False,
) -> str:
    title = html.escape(item.get("display_title", item.get("title", "")))
    rating = item.get("ml_avg_rating", 0.0)
    thumb = _thumb_url(item, tmdb_map)
    genres_vi = translate_genres_vi(item.get("genres", [])[: (2 if compact_ob else 3)])

    if thumb:
        media_html = f'<img src="{html.escape(thumb)}" alt="{title}"><div class="nf-thumb-grad"></div>'
    else:
        media_html = f'<div class="nf-thumb-placeholder">{title}</div>'

    check_badge = '<div class="nf-check-badge">✓ Đã xem</div>' if selected else ""
    sel_cls = " nf-selected" if selected else ""
    genre_html = f'<div class="nf-genre">{html.escape(genres_vi)}</div>' if genres_vi else ""
    why_html = f'<div class="nf-why">{html.escape(item["why_match"])}</div>' if (show_why and item.get("why_match")) else ""
    title_style = ' style="font-size:1.02rem;"' if compact_ob else ""

    return (
        f'<div class="nf-card{sel_cls}">'
        f'<div class="nf-thumb-box">{media_html}{check_badge}<div class="nf-star-badge">★ {rating:.1f}</div></div>'
        f'<div class="nf-card-body"><div class="nf-title"{title_style}>{title}</div>{genre_html}{why_html}</div>'
        f"</div>"
    )


def _render_clickable_grid(
    items: List[Dict],
    tmdb_map: Dict[int, Dict],
    key_prefix: str,
    cols_per_row: int = 3,
    show_why: bool = False,
    interactive: bool = True,
):
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="large")
        for idx, it in enumerate(row_items):
            mid = int(it["movie_id"])
            title_txt = it.get("display_title", it.get("title", f"Movie {mid}"))
            with cols[idx]:
                if interactive:
                    with st.container(key=f"cardwrap_{key_prefix}_{mid}_{row_start}_{idx}"):
                        st.markdown(_card_html(it, tmdb_map, show_why=show_why), unsafe_allow_html=True)
                        if st.button(title_txt, key=f"btn_{key_prefix}_{mid}_{row_start}_{idx}"):
                            _open_watch_page(mid)
                else:
                    st.markdown(_card_html(it, tmdb_map, show_why=show_why), unsafe_allow_html=True)
        st.markdown("<div style='height:0.85rem'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Top Header & Engine Mode Switcher
# ------------------------------------------------------------------
def _render_top_bar():
    c_logo, c_search, c_reset = st.columns([1.85, 4.75, 1.4], vertical_alignment="center")

    with c_logo:
        st.markdown("<div class='brand-btn-wrap'>", unsafe_allow_html=True)
        if st.button("🎬 CineRecSys", key="nav_home_logo"):
            st.session_state.view = "home"
            st.session_state.search_query = ""
            st.session_state.search_result = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    with c_search:
        with st.form("top_search_form", clear_on_submit=False, border=False):
            s_col, b_col = st.columns([5.1, 1.0], vertical_alignment="center")
            q_val = s_col.text_input(
                "Tìm kiếm",
                value=st.session_state.get("search_query", ""),
                placeholder="Tìm tên phim hoặc mô tả gu phim (vd: hôm nay tôi muốn xem phim cuốn hút, kịch tính)...",
                label_visibility="collapsed",
            )
            submitted = b_col.form_submit_button("Tìm", use_container_width=True, type="primary")
        if submitted and q_val.strip():
            _run_unified_search(q_val)

    with c_reset:
        st.markdown("<div class='reset-btn-wrap'>", unsafe_allow_html=True)
        if st.button("Khảo sát lại", key="btn_reset_survey", use_container_width=True):
            fresh = reset_user_profile(st.session_state.profile.get("uid"), CACHE_DIR)
            st.session_state.profile = fresh
            st.session_state.view = "home"
            st.session_state.watch_movie_id = None
            st.session_state.search_query = ""
            st.session_state.search_result = None
            st.session_state.ob_selected_movies = []
            st.session_state.ob_selected_genres = []
            st.session_state.ob_movie_dropdown = []
            st.session_state._sync_cookie = False
            st.session_state._clear_cookie = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


def _render_mode_selector(key_suffix: str):
    """Render LightGCN vs LightGCN+LLM switcher BEFORE inference to avoid extra reruns."""
    mode_options = ["pure", "llm"]
    mode_labels = {
        "pure": "⚡ LightGCN Thuần",
        "llm": "✨ LightGCN + LLM Reranker",
    }
    cur_mode = st.session_state.get("rec_mode", "pure")
    c_seg, c_stat = st.columns([3.8, 2.2], vertical_alignment="center")
    with c_seg:
        picked = st.segmented_control(
            "Chế độ gợi ý",
            options=mode_options,
            default=cur_mode,
            format_func=lambda x: mode_labels[x],
            selection_mode="single",
            label_visibility="collapsed",
            key=f"seg_mode_{key_suffix}",
        )
    active_mode = picked if picked in mode_options else cur_mode
    st.session_state.rec_mode = active_mode
    stat_box = c_stat.empty()
    return active_mode == "llm", stat_box


def _update_latency_pill(stat_box, latency_ms: float, engine_name: str):
    stat_box.markdown(
        f"<div style='text-align:right;'><span class='latency-pill'>{html.escape(engine_name)} • <b>{latency_ms:.0f} ms</b></span></div>",
        unsafe_allow_html=True,
    )


def _on_ob_dropdown_change():
    picked = st.session_state.get("ob_movie_dropdown", [])
    st.session_state.ob_selected_movies = list(dict.fromkeys(int(x) for x in picked))


# ------------------------------------------------------------------
# New User Onboarding Popup (Renders cleanly over blurred Home screen)
# ------------------------------------------------------------------
@st.dialog("🎬 Khảo sát người dùng mới", width="large", on_dismiss=_on_dismiss_onboarding)
def _show_onboarding_popup():
    # 1. Thể loại yêu thích (Vietnamese)
    picked_genres = st.pills(
        "Thể loại yêu thích",
        options=ONBOARDING_GENRES,
        default=st.session_state.ob_selected_genres,
        selection_mode="multi",
        format_func=translate_genre_vi,
        key="ob_pills_genres",
    )
    if picked_genres is not None:
        st.session_state.ob_selected_genres = list(picked_genres)

    # 2. Searchable dropdown ("thanh thả xuống") — keeps the showcase movie grid intact below
    if st.session_state.pop("_sync_ob_dropdown", False) or "ob_movie_dropdown" not in st.session_state:
        st.session_state.ob_movie_dropdown = list(st.session_state.ob_selected_movies)

    base_dropdown = list(getattr(engine, "_popular_dropdown_ids", engine._search_mids[:400]))
    dropdown_options = list(dict.fromkeys(list(st.session_state.ob_selected_movies) + base_dropdown))
    st.multiselect(
        "Tìm phim bạn từng xem",
        options=dropdown_options,
        format_func=lambda mid: (
            f"{engine.metadata[mid]['display_title']} • {translate_genres_vi(engine.metadata[mid]['genres'][:2])}"
            if mid in engine.metadata
            else f"Movie {mid}"
        ),
        placeholder="Gõ tên phim để lọc danh sách thả xuống (vd: Matrix, Titanic, Star Wars)...",
        on_change=_on_ob_dropdown_change,
        key="ob_movie_dropdown",
    )

    candidates = engine.get_onboarding_showcase()[:9]
    ob_tmdb = tmdb.prefetch_movies(candidates, max_workers=12)
    selected_set = set(st.session_state.ob_selected_movies)

    st.caption("Hoặc ấn trực tiếp vào hình / tên phim phổ biến bên dưới để đánh dấu đã xem (ấn lại lần nữa để bỏ chọn):")

    for row_start in range(0, len(candidates), 3):
        row_items = candidates[row_start : row_start + 3]
        cols = st.columns(3, gap="medium")
        for idx, it in enumerate(row_items):
            mid = int(it["movie_id"])
            is_sel = mid in selected_set
            title_txt = it.get("display_title", it.get("title", f"Movie {mid}"))
            with cols[idx]:
                with st.container(key=f"cardwrap_ob_{mid}_{row_start}_{idx}"):
                    st.markdown(
                        _card_html(it, ob_tmdb, selected=is_sel, compact_ob=True),
                        unsafe_allow_html=True,
                    )
                    if st.button(title_txt, key=f"btn_ob_{mid}_{row_start}_{idx}"):
                        cur = list(st.session_state.ob_selected_movies)
                        if mid in cur:
                            cur.remove(mid)
                        else:
                            cur.append(mid)
                        st.session_state.ob_selected_movies = cur
                        st.session_state._sync_ob_dropdown = True
                        st.rerun(scope="fragment")
        st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)

    st.divider()
    sel_cnt = len(st.session_state.ob_selected_movies)
    gen_cnt = len(st.session_state.ob_selected_genres)
    c_info, c_done, c_skip = st.columns([2.6, 1.8, 1.5], vertical_alignment="center")
    c_info.markdown(f"**Đã chọn:** `{sel_cnt}` phim • `{gen_cnt}` thể loại")

    with c_done:
        if st.button("Hoàn tất & Xem gợi ý", type="primary", use_container_width=True, key="btn_ob_finish"):
            profile = dict(st.session_state.profile)
            profile["onboarded"] = True
            profile["skipped"] = False
            profile["genres"] = list(st.session_state.ob_selected_genres)
            profile["watched"] = list(st.session_state.ob_selected_movies)
            _persist_profile(profile)
            st.rerun()

    with c_skip:
        if st.button("Bỏ qua (Skip)", use_container_width=True, key="btn_ob_skip"):
            profile = dict(st.session_state.profile)
            profile["onboarded"] = True
            profile["skipped"] = True
            profile["genres"] = []
            profile["watched"] = []
            _persist_profile(profile)
            st.rerun()


# ------------------------------------------------------------------
# Home Interface (Background for popup + Landing view for returning users)
# ------------------------------------------------------------------
def _render_home(interactive: bool = True):
    _render_top_bar()
    s_res = st.session_state.get("search_result")

    if s_res is not None:
        h1, h2 = st.columns([5.6, 1.3], vertical_alignment="center")
        with h1:
            clean_reply = str(s_res.get("assistant_reply", "")).replace("**", "")
            st.markdown(
                f"<div class='agent-banner'>{html.escape(clean_reply)}</div>",
                unsafe_allow_html=True,
            )
        with h2:
            if st.button("← Trang chủ", key="btn_clear_search", use_container_width=True):
                st.session_state.search_query = ""
                st.session_state.search_result = None
                st.rerun()

        items = s_res.get("items", [])
        if items:
            tmdb_map = tmdb.prefetch_movies(items, max_workers=8)
            _render_clickable_grid(items, tmdb_map, key_prefix="search", cols_per_row=3, show_why=True, interactive=interactive)
        else:
            st.info("Không tìm thấy phim phù hợp.")
        return

    profile = st.session_state.profile
    has_profile_signals = bool(profile.get("watched") or profile.get("genres"))
    heading = "Đề xuất cho bạn" if has_profile_signals else "Phim phổ biến"

    st.markdown(f"<div class='section-heading' style='margin-bottom:0.4rem'>{heading}</div>", unsafe_allow_html=True)
    stat_box = st.empty() if (has_profile_signals and interactive) else None

    res = engine.recommend_realtime_stream(
        watched_movie_ids=profile.get("watched", []),
        preferred_genres=profile.get("genres", []),
        current_movie_id=None,
        gemini_keys=None,
        top_k=12 if interactive else 6,
        use_llm=False,
    )
    items = res["items"]
    if stat_box is not None:
        _update_latency_pill(stat_box, res["latency_ms"], res["engine"])

    tmdb_map = tmdb.prefetch_movies(items, max_workers=12)
    _render_clickable_grid(items, tmdb_map, key_prefix="home", cols_per_row=3, interactive=interactive)


# ------------------------------------------------------------------
# Watch Page (YouTube Layout + Real-Time Clickable Right Rail)
# ------------------------------------------------------------------
def _render_watch():
    _render_top_bar()
    mid = st.session_state.get("watch_movie_id")
    if not mid or int(mid) not in engine.metadata:
        st.session_state.view = "home"
        st.rerun()
        return

    mid = int(mid)
    meta = engine.metadata[mid]
    tinfo = tmdb.get_movie_details(
        mid,
        tmdb_id=meta.get("tmdb_id"),
        title=meta.get("title", ""),
        fetch_videos=True,
    )
    tmdb._save_cache()

    profile = st.session_state.profile
    col_main, col_side = st.columns([1.85, 1.0], gap="large")

    with col_side:
        st.markdown("<div class='section-heading' style='margin-bottom:0.4rem'>Đề xuất tiếp theo</div>", unsafe_allow_html=True)
        stat_box = st.empty()
        st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    rt_res = engine.recommend_realtime_stream(
        watched_movie_ids=profile.get("watched", []),
        preferred_genres=profile.get("genres", []),
        current_movie_id=mid,
        gemini_keys=None,
        top_k=10,
        use_llm=False,
    )
    up_next = rt_res["items"]
    side_tmdb = tmdb.prefetch_movies(up_next, max_workers=8)

    with col_main:
        if st.button("← Trang chủ", key="btn_watch_back"):
            st.session_state.view = "home"
            st.rerun()

        yt_url = tinfo.get("youtube_url")
        if yt_url:
            st.video(yt_url)
        elif tinfo.get("backdrop_url") or tinfo.get("poster_url"):
            img = tinfo.get("backdrop_url") or tinfo.get("poster_url")
            st.markdown(
                f'<div class="nf-thumb-box" style="border-radius:14px;"><img src="{html.escape(img)}" alt="Backdrop"></div>',
                unsafe_allow_html=True,
            )

        title_esc = html.escape(meta.get("display_title", ""))
        st.markdown(f"<div class='watch-title'>{title_esc}</div>", unsafe_allow_html=True)

        chips = [f"<span class='meta-chip'>MovieLens ★ {meta.get('ml_avg_rating', 0):.1f}</span>"]
        if tinfo.get("vote_average"):
            chips.append(f"<span class='meta-chip'>TMDB ★ {float(tinfo['vote_average']):.1f}</span>")
        if tinfo.get("runtime"):
            chips.append(f"<span class='meta-chip'>{int(tinfo['runtime'])} phút</span>")
        for g in meta.get("genres", []):
            chips.append(f"<span class='meta-chip genre'>{html.escape(translate_genre_vi(g))}</span>")
        st.markdown(f"<div class='watch-meta'>{''.join(chips)}</div>", unsafe_allow_html=True)

        overview = tinfo.get("tmdb_overview") or meta.get("description") or ""
        director = tinfo.get("director") or meta.get("director") or ""
        cast = ", ".join(tinfo.get("cast", []))
        extra_lines = []
        if director:
            extra_lines.append(f"<b>Đạo diễn:</b> {html.escape(director)}")
        if cast:
            extra_lines.append(f"<b>Diễn viên:</b> {html.escape(cast)}")
        if overview:
            extra_lines.append(html.escape(overview))
        if extra_lines:
            st.markdown(f"<div class='watch-desc'>{'<br><br>'.join(extra_lines)}</div>", unsafe_allow_html=True)

    with col_side:
        _update_latency_pill(stat_box, rt_res["latency_ms"], rt_res["engine"])

        for idx, it in enumerate(up_next):
            n_mid = int(it["movie_id"])
            n_title = html.escape(it.get("display_title", it.get("title", "")))
            n_genres = translate_genres_vi(it.get("genres", [])[:2])
            n_rating = it.get("ml_avg_rating", 0.0)
            n_thumb = _thumb_url(it, side_tmdb)

            if n_thumb:
                thumb_inner = f'<img src="{html.escape(n_thumb)}" alt="{n_title}">'
            else:
                thumb_inner = f'<div class="nf-thumb-placeholder" style="font-size:0.82rem">{n_title}</div>'

            genre_line = f'<div class="side-row-genre">{html.escape(n_genres)}</div>' if n_genres else ""

            with st.container(key=f"cardwrap_next_{mid}_{n_mid}_{idx}"):
                row_html = (
                    f'<div class="side-row-card">'
                    f'<div class="side-row-thumb">{thumb_inner}</div>'
                    f'<div class="side-row-info">'
                    f'<div class="side-row-title">{n_title}</div>'
                    f'{genre_line}'
                    f'<div class="side-row-star">★ {n_rating:.1f}</div>'
                    f'</div></div>'
                )
                st.markdown(row_html, unsafe_allow_html=True)
                if st.button(it.get("display_title", f"Movie {n_mid}"), key=f"btn_next_{mid}_{n_mid}_{idx}"):
                    _open_watch_page(n_mid)
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main Dispatch & Browser Cookie Sync
# ------------------------------------------------------------------
_init_session_profile()

if not st.session_state.profile.get("onboarded"):
    _render_home(interactive=False)
    _show_onboarding_popup()
elif st.session_state.get("view") == "watch" and st.session_state.get("watch_movie_id"):
    _render_watch()
else:
    _render_home(interactive=True)

if st.session_state.pop("_clear_cookie", False):
    components.html(build_cookie_sync_html(st.session_state.profile, clear=True), height=0)
elif st.session_state.pop("_sync_cookie", False):
    components.html(build_cookie_sync_html(st.session_state.profile, clear=False), height=0)
