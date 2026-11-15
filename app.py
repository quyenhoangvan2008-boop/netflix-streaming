"""
CineRecSys — Clean Netflix / YouTube Style Real-Time Recommender
- Cookie-backed New User Onboarding (Hybrid Search, Genre Checklist, Popular Movies Poster Picker, Skip)
- Landscape 16:9 YouTube-style Home Grid
- Real-Time Watch Page: Pretrained LightGCN propagation + Fast Gemini 3.5 Flash-Lite rerank (auto fallback)
"""

import html
import os
from typing import Dict, List, Optional
import streamlit as st
import streamlit.components.v1 as components

from core.engine import RecSysEngine
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
# Clean Dark CSS — Zero Sidebar, Spacious 16:9 Thumbnails & Crisp Type
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

  .stApp {
    background-color: #0B0D14;
    color: #F8FAFC;
  }

  .block-container {
    max-width: 1360px !important;
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
  }

  .cine-logo {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.6px;
    color: #FFFFFF;
    line-height: 1.1;
  }
  .cine-logo span {
    color: #E50914;
  }

  .section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #F8FAFC;
    margin: 0.5rem 0 1rem 0;
  }

  .yt-card {
    background: #131722;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    overflow: hidden;
    margin-bottom: 0.45rem;
    transition: transform 0.16s ease, border-color 0.16s ease;
  }
  .yt-card:hover {
    border-color: rgba(229, 9, 20, 0.55);
  }
  .yt-thumb-box {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #1A2030;
    overflow: hidden;
  }
  .yt-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    transition: transform 0.25s ease;
  }
  .yt-card:hover .yt-thumb-box img {
    transform: scale(1.03);
  }
  .yt-thumb-placeholder {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
    text-align: center;
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    color: #CBD5E1;
    font-weight: 700;
    font-size: 0.95rem;
  }
  .yt-rating-chip {
    position: absolute;
    right: 8px;
    bottom: 8px;
    background: rgba(11, 13, 20, 0.86);
    color: #FACC15;
    font-size: 0.78rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 6px;
  }
  .yt-card-body {
    padding: 0.85rem 1rem 0.9rem 1rem;
  }
  .yt-title {
    font-size: 1.02rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.35;
    min-height: 2.7em;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .yt-meta {
    font-size: 0.84rem;
    color: #94A3B8;
    margin-top: 0.35rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* Watch Page Right Sidebar Mini Cards */
  .side-thumb-box {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: #1A2030;
    border-radius: 10px;
    overflow: hidden;
  }
  .side-thumb-box img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
  }
  .side-title {
    font-size: 0.94rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    margin-bottom: 0.25rem;
  }
  .side-meta {
    font-size: 0.80rem;
    color: #94A3B8;
  }

  .watch-title {
    font-size: 1.65rem;
    font-weight: 800;
    color: #FFFFFF;
    margin: 0.9rem 0 0.35rem 0;
    line-height: 1.25;
  }
  .watch-meta {
    font-size: 0.94rem;
    color: #CBD5E1;
    margin-bottom: 0.8rem;
  }
  .watch-desc {
    background: #131722;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #E2E8F0;
    font-size: 0.96rem;
    line-height: 1.6;
  }

  .onboarding-box {
    background: #131722;
    border: 1px solid rgba(229, 9, 20, 0.32);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
  }
  .agent-banner {
    background: rgba(20, 24, 36, 0.95);
    border-left: 3px solid #E50914;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 1.1rem;
    font-size: 0.96rem;
    color: #F1F5F9;
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
# Profile & Navigation Helpers
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


def _persist_profile(profile: Dict):
    st.session_state.profile = profile
    save_user_profile(profile, CACHE_DIR)
    st.session_state._sync_cookie = True
    st.session_state._clear_cookie = False


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


def _card_html(item: Dict, tmdb_map: Dict[int, Dict]) -> str:
    title = html.escape(item.get("display_title", item.get("title", "")))
    genres = html.escape(", ".join(item.get("genres", [])[:2]) or "Movie")
    rating = item.get("ml_avg_rating", 0.0)
    thumb = _thumb_url(item, tmdb_map)
    if thumb:
        media_html = f'<img src="{html.escape(thumb)}" alt="{title}">'
    else:
        media_html = f'<div class="yt-thumb-placeholder">{title}</div>'

    return f"""
    <div class="yt-card">
      <div class="yt-thumb-box">
        {media_html}
        <div class="yt-rating-chip">★ {rating:.1f}</div>
      </div>
      <div class="yt-card-body">
        <div class="yt-title">{title}</div>
        <div class="yt-meta">{genres}</div>
      </div>
    </div>
    """


def _render_grid_cards(items: List[Dict], tmdb_map: Dict[int, Dict], key_prefix: str, cols_per_row: int = 3):
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="large")
        for idx, it in enumerate(row_items):
            mid = int(it["movie_id"])
            with cols[idx]:
                st.markdown(_card_html(it, tmdb_map), unsafe_allow_html=True)
                why = it.get("why_match")
                if why:
                    st.caption(why)
                if st.button("▶ Xem", key=f"{key_prefix}_{mid}_{row_start}_{idx}", use_container_width=True):
                    _open_watch_page(mid)
        st.markdown("<div style='height:0.65rem'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Top Bar
# ------------------------------------------------------------------
def _render_top_bar(show_search: bool = True):
    if show_search:
        c_logo, c_search, c_btn = st.columns([1.4, 5.2, 1.3], vertical_alignment="center")
    else:
        c_logo, c_btn = st.columns([6.6, 1.3], vertical_alignment="center")
        c_search = None

    with c_logo:
        if st.button("🎬 CineRecSys", key="nav_home_logo", type="tertiary"):
            st.session_state.view = "home"
            st.session_state.search_query = ""
            st.session_state.search_result = None
            st.rerun()

    if show_search and c_search is not None:
        with c_search:
            with st.form("top_search_form", clear_on_submit=False, border=False):
                s_col, b_col = st.columns([5.2, 1.0], vertical_alignment="center")
                q_val = s_col.text_input(
                    "Tìm kiếm",
                    value=st.session_state.get("search_query", ""),
                    placeholder="Tìm tên phim hoặc mô tả gu phim (vd: hôm nay tôi muốn xem phim cuốn hút, kịch tính)...",
                    label_visibility="collapsed",
                )
                submitted = b_col.form_submit_button("Tìm", use_container_width=True, type="primary")
            if submitted and q_val.strip():
                _run_unified_search(q_val)

    with c_btn:
        if st.session_state.profile.get("onboarded"):
            if st.button("Làm lại khảo sát", key="btn_reset_survey", use_container_width=True):
                fresh = reset_user_profile(st.session_state.profile.get("uid"), CACHE_DIR)
                st.session_state.profile = fresh
                st.session_state.view = "home"
                st.session_state.watch_movie_id = None
                st.session_state.search_query = ""
                st.session_state.search_result = None
                st.session_state._sync_cookie = False
                st.session_state._clear_cookie = True
                st.rerun()


# ------------------------------------------------------------------
# View 1: New User Onboarding
# ------------------------------------------------------------------
def _render_onboarding():
    _render_top_bar(show_search=False)
    st.markdown(
        """
        <div class="onboarding-box">
          <div class="cine-logo">Cine<span>RecSys</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Unified Search + Agent Bar right inside onboarding
    with st.form("onboarding_search_form", border=False):
        q_col, q_btn = st.columns([5.5, 1.2], vertical_alignment="center")
        ob_query = q_col.text_input(
            "Tìm phim hoặc hỏi gợi ý",
            placeholder="Tìm tên phim hoặc hỏi AI (vd: hôm nay tôi muốn xem phim cuốn hút, kịch tính)...",
            label_visibility="collapsed",
        )
        ob_search_go = q_btn.form_submit_button("Tìm / Gợi ý", use_container_width=True, type="primary")
    if ob_search_go and ob_query.strip():
        _run_unified_search(ob_query)

    st.markdown("<div class='section-heading'>Thể loại yêu thích</div>", unsafe_allow_html=True)
    genre_cols = st.columns(7)
    picked_genres: List[str] = []
    for idx, g in enumerate(ONBOARDING_GENRES):
        if genre_cols[idx % 7].checkbox(g, key=f"ob_genre_{g}"):
            picked_genres.append(g)

    st.markdown("<div class='section-heading'>Phim phổ biến bạn đã xem</div>", unsafe_allow_html=True)
    showcase = engine.get_onboarding_showcase()
    tmdb_map = tmdb.prefetch_movies(showcase, max_workers=8)

    picked_movies: List[int] = []
    for row_start in range(0, len(showcase), 3):
        row_items = showcase[row_start : row_start + 3]
        cols = st.columns(3, gap="large")
        for idx, it in enumerate(row_items):
            mid = int(it["movie_id"])
            with cols[idx]:
                st.markdown(_card_html(it, tmdb_map), unsafe_allow_html=True)
                if st.checkbox("Đã xem", key=f"ob_movie_{mid}"):
                    picked_movies.append(mid)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.divider()
    act1, act2, _ = st.columns([1.8, 1.8, 4.4])
    with act1:
        if st.button("Bắt đầu xem phim", type="primary", use_container_width=True, key="btn_ob_finish"):
            profile = dict(st.session_state.profile)
            profile["onboarded"] = True
            profile["skipped"] = False
            profile["genres"] = picked_genres
            profile["watched"] = picked_movies
            _persist_profile(profile)
            st.rerun()
    with act2:
        if st.button("Bỏ qua (Skip)", use_container_width=True, key="btn_ob_skip"):
            profile = dict(st.session_state.profile)
            profile["onboarded"] = True
            profile["skipped"] = True
            profile["genres"] = []
            profile["watched"] = []
            _persist_profile(profile)
            st.rerun()


# ------------------------------------------------------------------
# View 2: Home Feed (Returning Users & Skipped Cold-Start)
# ------------------------------------------------------------------
def _render_home():
    _render_top_bar(show_search=True)
    s_res = st.session_state.get("search_result")

    if s_res is not None:
        h1, h2 = st.columns([5.5, 1.3], vertical_alignment="center")
        with h1:
            st.markdown(
                f"<div class='agent-banner'>{html.escape(s_res.get('assistant_reply', ''))}</div>",
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
            _render_grid_cards(items, tmdb_map, key_prefix="search_card", cols_per_row=3)
        else:
            st.info("Không tìm thấy phim phù hợp.")
        return

    profile = st.session_state.profile
    res = engine.recommend_realtime_stream(
        watched_movie_ids=profile.get("watched", []),
        preferred_genres=profile.get("genres", []),
        current_movie_id=None,
        gemini_keys=gemini_keys,
        top_k=12,
    )
    items = res["items"]
    heading = "Phim phổ biến" if res["engine"] == "Popularity" else "Đề xuất cho bạn"
    st.markdown(f"<div class='section-heading'>{heading}</div>", unsafe_allow_html=True)

    tmdb_map = tmdb.prefetch_movies(items, max_workers=8)
    _render_grid_cards(items, tmdb_map, key_prefix="home_card", cols_per_row=3)


# ------------------------------------------------------------------
# View 3: Watch Page (YouTube Layout + Real-Time Right Rail)
# ------------------------------------------------------------------
def _render_watch():
    _render_top_bar(show_search=True)
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
    rt_res = engine.recommend_realtime_stream(
        watched_movie_ids=profile.get("watched", []),
        preferred_genres=profile.get("genres", []),
        current_movie_id=mid,
        gemini_keys=gemini_keys,
        top_k=10,
    )
    up_next = rt_res["items"]
    side_tmdb = tmdb.prefetch_movies(up_next, max_workers=8)

    col_main, col_side = st.columns([1.85, 1.0], gap="large")

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
                f'<div class="yt-thumb-box" style="border-radius:14px;"><img src="{html.escape(img)}" alt="Backdrop"></div>',
                unsafe_allow_html=True,
            )

        title_esc = html.escape(meta.get("display_title", ""))
        st.markdown(f"<div class='watch-title'>{title_esc}</div>", unsafe_allow_html=True)

        genres_str = ", ".join(meta.get("genres", []))
        tmdb_vote = tinfo.get("vote_average")
        rating_bits = [f"MovieLens ★ {meta.get('ml_avg_rating', 0):.1f}"]
        if tmdb_vote:
            rating_bits.append(f"TMDB ★ {float(tmdb_vote):.1f}")
        if tinfo.get("runtime"):
            rating_bits.append(f"{int(tinfo['runtime'])} phút")
        if genres_str:
            rating_bits.append(genres_str)

        st.markdown(
            f"<div class='watch-meta'>{' &nbsp;•&nbsp; '.join(html.escape(x) for x in rating_bits)}</div>",
            unsafe_allow_html=True,
        )

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
        st.markdown("<div class='section-heading' style='margin-top:0'>Đề xuất tiếp theo</div>", unsafe_allow_html=True)
        for idx, it in enumerate(up_next):
            n_mid = int(it["movie_id"])
            n_title = html.escape(it.get("display_title", it.get("title", "")))
            n_genres = html.escape(", ".join(it.get("genres", [])[:2]) or "Movie")
            n_rating = it.get("ml_avg_rating", 0.0)
            n_thumb = _thumb_url(it, side_tmdb)

            t_col, info_col = st.columns([1.05, 1.35], gap="small", vertical_alignment="top")
            with t_col:
                if n_thumb:
                    st.markdown(
                        f'<div class="side-thumb-box"><img src="{html.escape(n_thumb)}" alt="{n_title}"></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="side-thumb-box"><div class="yt-thumb-placeholder" style="font-size:0.78rem">{n_title}</div></div>',
                        unsafe_allow_html=True,
                    )
            with info_col:
                st.markdown(
                    f"<div class='side-title'>{n_title}</div>"
                    f"<div class='side-meta'>{n_genres} • ★ {n_rating:.1f}</div>",
                    unsafe_allow_html=True,
                )
                if st.button("Xem", key=f"upnext_{mid}_{n_mid}_{idx}", use_container_width=True):
                    _open_watch_page(n_mid)
            st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main Dispatch + Browser Cookie Sync
# ------------------------------------------------------------------
_init_session_profile()

if not st.session_state.profile.get("onboarded"):
    _render_onboarding()
elif st.session_state.get("view") == "watch" and st.session_state.get("watch_movie_id"):
    _render_watch()
else:
    _render_home()

if st.session_state.pop("_clear_cookie", False):
    components.html(build_cookie_sync_html(st.session_state.profile, clear=True), height=0)
elif st.session_state.pop("_sync_cookie", False):
    components.html(build_cookie_sync_html(st.session_state.profile, clear=False), height=0)
