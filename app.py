"""
Netflix-Style MovieLens-1M RecSys Streaming & LLM Reranker Demo (Streamlit)
Features:
1. User Inspector (10 Most Recent Watched w/ Real Ratings vs Top-10 Recs w/ Ground-Truth Hits).
2. Model Switcher (Baseline Popularity -> Classical/MF -> Deep Retrieval -> LightGCN + Gemini 3.5 Flash-Lite).
3. Interactive Cold-Start Sandbox (Pick 3-5 movies -> Real-time personalized recommendations).
4. Hybrid Search + Mood Discovery AI Agent (Title search + Vietnamese/English mood queries).
5. Click-to-Watch Movie Detail Dialog with TMDB Metadata & Embedded YouTube Trailer (`st.video`).
6. Full Verified 9-Model Benchmark Dashboard across 1,188 Test Users & 100,021 Ratings.
"""

import os
import json
from typing import Dict, List, Optional
import pandas as pd
import streamlit as st

from core.engine import RecSysEngine, MODEL_CATALOG
from core.tmdb_client import TMDBClient

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(BASE_DIR, ".cache", "tmdb_cache.json")
BENCHMARK_JSON = os.path.join(BASE_DIR, "artifacts", "unified_full_benchmark.json")
CHARTS_DIR = os.path.join(BASE_DIR, "artifacts", "charts")

st.set_page_config(
    page_title="CineRecSys • Netflix AI Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Netflix Dark Cinema CSS
# ------------------------------------------------------------------
st.markdown(
    """
<style>
  .stApp {
    background-color: #0B0D14;
    color: #F8FAFC;
  }
  .netflix-hero {
    background: linear-gradient(120deg, rgba(229,9,20,0.22) 0%, rgba(20,24,36,0.95) 55%, rgba(11,13,20,0.98) 100%);
    border: 1px solid rgba(229,9,20,0.35);
    border-radius: 16px;
    padding: 1.25rem 1.6rem;
    margin-bottom: 1rem;
  }
  .netflix-brand {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #FFFFFF;
  }
  .netflix-brand span {
    color: #E50914;
  }
  .movie-card {
    background: #141824;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 8px;
    transition: transform 0.15s ease, border-color 0.15s ease;
    height: 100%;
  }
  .movie-card-hit {
    background: linear-gradient(180deg, rgba(16,185,129,0.16) 0%, #141824 55%);
    border: 1.5px solid #10B981;
    border-radius: 12px;
    padding: 8px;
    box-shadow: 0 0 18px rgba(16,185,129,0.22);
    height: 100%;
  }
  .badge-hit {
    display: inline-block;
    background: #10B981;
    color: #042F2E;
    font-weight: 800;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 999px;
    margin-bottom: 4px;
    letter-spacing: 0.3px;
  }
  .badge-rank {
    display: inline-block;
    background: #E50914;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 999px;
    margin-bottom: 4px;
  }
  .badge-rating {
    display: inline-block;
    background: #F59E0B;
    color: #1E1B18;
    font-weight: 800;
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 999px;
    margin-bottom: 4px;
  }
  .card-title {
    font-size: 13px;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.25;
    margin-top: 6px;
    min-height: 32px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .card-genres {
    font-size: 11px;
    color: #94A3B8;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 6px;
  }
  .poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    border-radius: 8px;
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px dashed rgba(255,255,255,0.18);
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 10px;
    color: #CBD5E1;
    font-weight: 700;
    font-size: 13px;
  }
  .metric-box {
    background: #141824;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 12px 14px;
  }
  .metric-val {
    font-size: 1.35rem;
    font-weight: 800;
    color: #F8FAFC;
  }
  .metric-lbl {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: #94A3B8;
  }
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Cached Singletons & Secret Loading
# ------------------------------------------------------------------
def _load_default_secrets() -> Dict:
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


@st.cache_resource(show_spinner="Đang khởi tạo MovieLens-1M & nạp 9 mô hình RecSys...")
def get_engine() -> RecSysEngine:
    return RecSysEngine(BASE_DIR)


@st.cache_resource
def get_tmdb_client(read_token: str, api_key: str) -> TMDBClient:
    return TMDBClient(read_token=read_token, api_key=api_key, cache_path=CACHE_PATH)


@st.cache_data
def load_benchmark_json() -> Dict:
    if os.path.exists(BENCHMARK_JSON):
        with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


engine = get_engine()
defaults = _load_default_secrets()

# ------------------------------------------------------------------
# Sidebar: Keys & System Info
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 CineRecSys Controls")
    st.caption("MovieLens-1M • 2-Stage RecSys + TMDB + YouTube + Gemini 3.5 Flash-Lite")

    with st.expander("🔑 Cấu hình API (TMDB & Gemini)", expanded=False):
        tmdb_tok_input = st.text_input(
            "TMDB Read Access Token",
            value="" if defaults["tmdb_read_token"] else "",
            placeholder="Đã nạp từ secrets.toml" if defaults["tmdb_read_token"] else "Paste TMDB v4 Bearer Token...",
            type="password",
        )
        tmdb_key_input = st.text_input(
            "TMDB API Key (v3)",
            value="",
            placeholder="Đã nạp từ secrets.toml" if defaults["tmdb_api_key"] else "Paste TMDB v3 API Key...",
            type="password",
        )
        gem_key_input = st.text_input(
            "Gemini API Key(s) (phân tách bằng dấu phẩy)",
            value="",
            placeholder=f"Đã nạp {len(defaults['gemini_keys'])} keys từ secrets.toml" if defaults["gemini_keys"] else "Paste Gemini API Key...",
            type="password",
        )

    eff_tmdb_tok = tmdb_tok_input.strip() or defaults["tmdb_read_token"]
    eff_tmdb_key = tmdb_key_input.strip() or defaults["tmdb_api_key"]
    eff_gem_keys = (
        [k.strip() for k in gem_key_input.split(",") if k.strip()]
        if gem_key_input.strip()
        else defaults["gemini_keys"]
    )

    tmdb = get_tmdb_client(eff_tmdb_tok, eff_tmdb_key)
    tmdb.update_credentials(eff_tmdb_tok, eff_tmdb_key)

    st.markdown("---")
    st.markdown("**Trạng thái hệ thống:**")
    st.markdown(f"- Thiết bị tính toán: `{str(engine.device).upper()}`")
    st.markdown(f"- Kho phim: `{len(engine.metadata):,}` phim (`links.csv` TMDB)")
    st.markdown(f"- Tập Test chuẩn: `{len(engine.test_pos):,}` users")
    st.markdown(f"- TMDB Poster & Trailer: `{'Sẵn sàng' if tmdb.is_configured else 'Chưa có key'}`")
    st.markdown(f"- Gemini 3.5 Flash-Lite: `{len(eff_gem_keys)} key`")


# ------------------------------------------------------------------
# Movie Detail & YouTube Trailer Modal Dialog
# ------------------------------------------------------------------
@st.dialog("🎬 Chi Tiết Phim & Trailer (TMDB + YouTube)", width="large")
def show_movie_modal(movie_id: int):
    m = engine.metadata.get(movie_id, {})
    title = m.get("display_title", f"Movie {movie_id}")
    with st.spinner("Đang tải trailer & thông tin từ TMDB..."):
        td = tmdb.get_movie_details(
            movie_id=movie_id,
            tmdb_id=m.get("tmdb_id"),
            title=m.get("title", title),
            fetch_videos=True,
        )

    st.markdown(f"## {title}")
    if td.get("tagline"):
        st.caption(f"*{td['tagline']}*")

    # YouTube Trailer Embed right inside Streamlit
    yt_url = td.get("youtube_url")
    if yt_url:
        st.markdown(f"**▶️ Trailer ({td.get('video_name', 'Official Trailer')})**")
        st.video(yt_url)
    else:
        st.info("Phim này chưa có video trailer YouTube trên TMDB; đang hiển thị ảnh & thông tin chi tiết từ TMDB / MovieLens.")
        if td.get("backdrop_url"):
            st.image(td["backdrop_url"], use_container_width=True)

    c_poster, c_info = st.columns([1, 2.2])
    with c_poster:
        if td.get("poster_url"):
            st.image(td["poster_url"], use_container_width=True)
        else:
            st.markdown(f"<div class='poster-placeholder'>{title}</div>", unsafe_allow_html=True)

    with c_info:
        genres_str = " • ".join(m.get("categories", [])) or "Cinema"
        tmdb_score = f"★ {td['vote_average']:.1f}/10 ({td.get('vote_count', 0):,} votes)" if td.get("vote_average") else "N/A"
        runtime_str = f"{td['runtime']} phút" if td.get("runtime") else "N/A"
        rel_date = td.get("release_date") or "N/A"
        director = td.get("director") or m.get("director") or "N/A"
        cast_str = ", ".join(td.get("cast", [])) or "N/A"

        st.markdown(f"**Thể loại:** `{genres_str}`")
        st.markdown(
            f"**TMDB:** {tmdb_score} &nbsp;|&nbsp; **MovieLens:** ★ {m.get('ml_avg_rating', 0):.2f}/5 "
            f"({m.get('popularity_count', 0):,} ratings)"
        )
        st.markdown(f"**Phát hành:** {rel_date} &nbsp;|&nbsp; **Thời lượng:** {runtime_str}")
        st.markdown(f"**Đạo diễn:** {director}")
        st.markdown(f"**Diễn viên chính:** {cast_str}")
        st.markdown("---")
        overview = td.get("tmdb_overview") or m.get("description") or "Chưa có tóm tắt nội dung."
        st.markdown(f"**Nội dung:** {overview}")
        if td.get("tmdb_id"):
            st.caption(f"TMDB ID: `{td['tmdb_id']}` • IMDB: `{m.get('imdb_id', 'N/A')}` • YouTube: {yt_url or 'N/A'}")


def open_movie(movie_id: int):
    show_movie_modal(int(movie_id))


# ------------------------------------------------------------------
# Helper: Render 5-Column Netflix Movie Grid
# ------------------------------------------------------------------
def render_movie_grid(
    items: List[Dict],
    tmdb_details_map: Dict[int, Dict],
    key_prefix: str,
    mode: str = "rec",
    cols_per_row: int = 5,
):
    for row_start in range(0, len(items), cols_per_row):
        row_items = items[row_start : row_start + cols_per_row]
        cols = st.columns(cols_per_row, gap="small")
        for col, it in zip(cols, row_items):
            mid = int(it["movie_id"])
            td = tmdb_details_map.get(mid, {})
            poster = td.get("poster_url")
            title = it.get("display_title", f"Movie {mid}")
            genres = ", ".join(it.get("categories", [])[:2])
            is_hit = bool(it.get("is_hit"))

            with col:
                card_cls = "movie-card-hit" if is_hit else "movie-card"
                st.markdown(f"<div class='{card_cls}'>", unsafe_allow_html=True)
                if mode == "history":
                    st.markdown(f"<span class='badge-rating'>★ {it.get('user_rating', 0):.0f}/5</span>", unsafe_allow_html=True)
                elif mode == "rec":
                    if is_hit:
                        st.markdown(f"<span class='badge-hit'>✔ #{it.get('rank')} GROUND TRUTH</span>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span class='badge-rank'>#{it.get('rank')} • {genres}</span>", unsafe_allow_html=True)
                else:
                    yt_badge = " 🎬" if td.get("youtube_key") else ""
                    st.markdown(f"<span class='badge-rank'>#{it.get('rank', '?')}{yt_badge}</span>", unsafe_allow_html=True)

                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown(f"<div class='poster-placeholder'>{title}</div>", unsafe_allow_html=True)

                st.markdown(f"<div class='card-title' title='{title}'>{title}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='card-genres'>{genres} &bull; ML ★{it.get('ml_avg_rating', 0):.1f}</div>", unsafe_allow_html=True)
                if it.get("why_match"):
                    st.caption(it["why_match"])

                if st.button("▶ Xem Trailer", key=f"{key_prefix}_{mid}_{row_start}", use_container_width=True):
                    open_movie(mid)
                st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Top Hero Banner
# ------------------------------------------------------------------
st.markdown(
    """
<div class="netflix-hero">
  <div class="netflix-brand">CINE<span>RECSYS</span> &bull; 2-Stage Streaming AI Recommender</div>
  <div style="color:#CBD5E1; font-size:0.92rem; margin-top:4px;">
    Kiểm chứng trực tiếp <b>9 mô hình Recommendation</b> (Popularity → CF/SVD → LightGCN / SASRec / BERT4Rec → Gemini 3.5 Flash-Lite),
    trải nghiệm <b>Cold-Start tương tác</b>, <b>AI Search &amp; Mood Agent</b>, cùng <b>Trailer YouTube từ TMDB</b>.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_inspector, tab_coldstart, tab_agent, tab_benchmark = st.tabs(
    [
        "🔍 User Inspector & Model Switcher",
        "✨ Cold-Start Tương Tác (3-5 Phim)",
        "🤖 Agent Tìm Kiếm & Tâm Trạng",
        "📊 Benchmark 1,188 Test Users",
    ]
)

# ==================================================================
# TAB 1: USER INSPECTOR & MODEL SWITCHER (Image 1 Features #1 & #3)
# ==================================================================
with tab_inspector:
    # Build informative selector list for positive test users (having both history and test GT)
    test_users_with_hist = sorted(
        [u for u in engine.test_pos.keys() if len(engine.rated_history.get(u, [])) >= 10]
    )
    if not test_users_with_hist:
        test_users_with_hist = sorted(engine.test_pos.keys())

    # Default highlight user with great diversity & hits
    default_uid = 24 if 24 in test_users_with_hist else test_users_with_hist[0]

    def _fmt_user(u: int) -> str:
        prof = engine.user_info.get(u, {})
        n_hist = len(engine.rated_history.get(u, []))
        n_gt = len(engine.test_pos.get(u, set()))
        desc = f"{prof.get('gender', '')}, {prof.get('age_group', '')}, {prof.get('occupation', '')}".strip(", ")
        return f"User #{u} ({desc} • {n_hist} đã xem, {n_gt} Ground Truth)"

    col_u1, col_u2 = st.columns([2.2, 1.3])
    with col_u1:
        selected_uid = st.selectbox(
            "1️⃣ Chọn User từ Tập Test (1,188 positive test users):",
            options=test_users_with_hist,
            index=test_users_with_hist.index(default_uid),
            format_func=_fmt_user,
        )
    with col_u2:
        direct_uid = st.number_input(
            "Hoặc nhập nhanh UserID:",
            min_value=1,
            max_value=6040,
            value=int(selected_uid),
            step=1,
        )
        if int(direct_uid) in engine.test_pos:
            active_uid = int(direct_uid)
        else:
            active_uid = int(selected_uid)
            st.caption(f"User #{int(direct_uid)} không có Ground Truth dương trong test.csv; đang hiển thị User #{active_uid}.")

    # Quick Model Switcher (Image 1 Feature #3)
    model_keys = [m[0] for m in MODEL_CATALOG]
    model_labels = {m[0]: m[1] for m in MODEL_CATALOG}

    st.markdown("##### 2️⃣ Thanh Chuyển Đổi Mô Hình Nhanh (Model Switcher)")
    selected_model = st.radio(
        "Chuyển đổi qua lại ngay trên cùng 1 User để kiểm chứng độ cá nhân hóa & Ground Truth Hits:",
        options=model_keys,
        index=model_keys.index("lightgcn"),
        format_func=lambda k: model_labels[k],
        horizontal=True,
    )

    # Fetch user watch history (10 most recent with actual ratings) + Recommendations
    history_items = engine.get_user_watch_history(active_uid, limit=10)
    rec_res = engine.recommend_for_user(active_uid, selected_model, top_k=10, gemini_keys=eff_gem_keys)

    if rec_res.get("warning"):
        st.warning(rec_res["warning"])

    # Prefetch posters & YouTube trailers cleanly via TMDB
    all_visible = history_items + rec_res["items"]
    tmdb_map = tmdb.prefetch_movies(all_visible, max_workers=8)

    # Live Single-User Metrics Strip + Baseline vs Retrieval vs Full Pipeline comparison!
    mcols = st.columns(5)
    mcols[0].markdown(
        f"<div class='metric-box'><div class='metric-lbl'>Ground Truth Hits@10</div>"
        f"<div class='metric-val' style='color:#10B981'>{rec_res['hits']} / {min(10, rec_res['total_gt'])}</div></div>",
        unsafe_allow_html=True,
    )
    mcols[1].markdown(
        f"<div class='metric-box'><div class='metric-lbl'>Precision@10</div>"
        f"<div class='metric-val'>{rec_res['precision_at_10']*100:.1f}%</div></div>",
        unsafe_allow_html=True,
    )
    mcols[2].markdown(
        f"<div class='metric-box'><div class='metric-lbl'>NDCG@10 (User này)</div>"
        f"<div class='metric-val' style='color:#38BDF8'>{rec_res['ndcg_at_10']:.4f}</div></div>",
        unsafe_allow_html=True,
    )
    mcols[3].markdown(
        f"<div class='metric-box'><div class='metric-lbl'>Độ Đa Dạng (ILD@10)</div>"
        f"<div class='metric-val' style='color:#F59E0B'>{rec_res['ild_at_10']:.4f}</div></div>",
        unsafe_allow_html=True,
    )
    mcols[4].markdown(
        f"<div class='metric-box'><div class='metric-lbl'>Độ Trễ Phản Hồi</div>"
        f"<div class='metric-val'>{rec_res['latency_ms']:.1f} ms</div></div>",
        unsafe_allow_html=True,
    )

    with st.expander("⚡ So sánh nhanh 3 tầng trên chính User này: Baseline (Popularity) → Retrieval (LightGCN) → SASRec → BERT4Rec"):
        cmp_rows = []
        for mk, mname, mgrp in [
            ("popularity", "Baseline: Popularity", "Baseline"),
            ("svd", "Low-Rank: Netflix SVD", "Classical / MF"),
            ("lightgcn", "Retrieval: LightGCN", "Deep Graph"),
            ("sasrec", "Retrieval: SASRec", "Deep Seq"),
            ("bert4rec", "Retrieval: BERT4Rec", "Deep Seq"),
        ]:
            r_cmp = engine.recommend_for_user(active_uid, mk, top_k=10)
            cmp_rows.append(
                {
                    "Mô Hình": mname,
                    "Hits@10": f"{r_cmp['hits']}/{min(10, r_cmp['total_gt'])}",
                    "Precision@10": f"{r_cmp['precision_at_10']*100:.1f}%",
                    "NDCG@10": round(r_cmp["ndcg_at_10"], 4),
                    "Diversity ILD@10": round(r_cmp["ild_at_10"], 4),
                    "Latency (ms)": r_cmp["latency_ms"],
                }
            )
        st.dataframe(pd.DataFrame(cmp_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown(f"#### 🎞️ Hàng Trên: 10 Bộ Phim User #{active_uid} Đã Xem Gần Nhất (Kèm Điểm Rating Thực Tế)")
    render_movie_grid(history_items, tmdb_map, key_prefix=f"hist_{active_uid}", mode="history")

    st.markdown("---")
    st.markdown(f"#### 🎯 Hàng Dưới: Top 10 Phim Đề Xuất Từ `{model_labels[selected_model]}` (Viền Xanh = Trùng Ground Truth Test)")
    if rec_res.get("user_intent"):
        st.info(f"**Gemini 3.5 Flash-Lite phân tích gu xem phim:** {rec_res['user_intent']}")
    render_movie_grid(rec_res["items"], tmdb_map, key_prefix=f"rec_{active_uid}_{selected_model}", mode="rec")


# ==================================================================
# TAB 2: INTERACTIVE COLD-START SANDBOX (Image 1 Feature #2)
# ==================================================================
with tab_coldstart:
    st.markdown("### ✨ Chế Độ Cold-Start Tương Tác Trực Tiếp (Người Dùng Hoàn Toàn Mới)")
    st.caption(
        "Nhập vai 1 user mới tinh chưa từng có trong dữ liệu MovieLens: hãy tự tích chọn **3–5 bộ phim** bạn thích dưới đây "
        "rồi chọn mô hình bất kỳ để hệ thống suy luận gợi ý Top-10 theo thời gian thực!"
    )

    # Quick curated presets
    presets = {
        "Tự chọn thủ công": [],
        "🚀 Sci-Fi & Hack Não (Matrix, Blade Runner, Terminator 2, Star Wars V)": [2571, 541, 589, 1196],
        "🧸 Hoạt Hình Gia Đình (Toy Story, Lion King, Aladdin, Beauty & Beast)": [1, 364, 588, 595],
        "💔 Tâm Lý Sâu Lắng (Shawshank, Forrest Gump, Good Will Hunting, American Beauty)": [318, 356, 1704, 2858],
        "🕵️ Hình Sự Kinh Điển (Godfather, Pulp Fiction, Silence of the Lambs, Usual Suspects)": [858, 296, 593, 50],
    }

    p_col1, p_col2 = st.columns([1.4, 2.2])
    with p_col1:
        preset_choice = st.selectbox("Hoặc chọn nhanh 1 bộ gu phim mẫu:", options=list(presets.keys()))
    with p_col2:
        cs_model = st.selectbox(
            "Chọn mô hình gợi ý Cold-Start:",
            options=[m[0] for m in MODEL_CATALOG],
            index=[m[0] for m in MODEL_CATALOG].index("lightgcn"),
            format_func=lambda k: model_labels[k],
            key="cs_model_select",
        )

    # Searchable top-600 popular movies list for smooth UX
    showcase_ids = engine.popular_movie_ids[:600]
    for pid in presets[preset_choice]:
        if pid not in showcase_ids and pid in engine.metadata:
            showcase_ids.append(pid)

    default_seeds = presets[preset_choice] if presets[preset_choice] else [1, 2571, 318]
    selected_seeds = st.multiselect(
        "Chọn 3 đến 5 bộ phim bạn yêu thích (có thể gõ tên để tìm):",
        options=showcase_ids,
        default=default_seeds,
        max_selections=5,
        format_func=lambda mid: f"{engine.metadata[mid]['display_title']}  [{', '.join(engine.metadata[mid]['categories'][:2])}]",
    )

    if len(selected_seeds) < 3:
        st.info("Vui lòng chọn ít nhất **3 bộ phim** (tối đa 5 phim) để kích hoạt suy luận Cold-Start.")
    else:
        # Show chosen seed chips
        seed_names = [engine.metadata[m]["display_title"] for m in selected_seeds]
        st.success(f"Hồ sơ Cold-Start tạm thời ({len(selected_seeds)} phim): **{' → '.join(seed_names)}**")

        run_cs = st.button("🚀 Nhận Gợi Ý Theo Thời Gian Thực", type="primary")
        if run_cs or "cs_last_result" not in st.session_state or st.session_state.get("cs_last_key") != (tuple(selected_seeds), cs_model):
            cs_res = engine.recommend_for_cold_seeds(selected_seeds, cs_model, top_k=10, gemini_keys=eff_gem_keys)
            st.session_state["cs_last_result"] = cs_res
            st.session_state["cs_last_key"] = (tuple(selected_seeds), cs_model)

        cs_res = st.session_state["cs_last_result"]
        if cs_res.get("warning"):
            st.warning(cs_res["warning"])
        if cs_res.get("user_intent"):
            st.info(f"**Gemini 3.5 Flash-Lite phân tích gu Cold-Start:** {cs_res['user_intent']}")

        scols = st.columns(3)
        scols[0].metric("Mô Hình Đang Chạy", model_labels[cs_model].split("•")[-1].strip())
        scols[1].metric("Độ Đa Dạng (ILD@10)", f"{cs_res['ild_at_10']:.4f}")
        scols[2].metric("Thời Gian Suy Luận", f"{cs_res['latency_ms']:.1f} ms")

        cs_tmdb = tmdb.prefetch_movies(cs_res["items"], max_workers=8)
        render_movie_grid(cs_res["items"], cs_tmdb, key_prefix=f"cs_{cs_model}", mode="cold")


# ==================================================================
# TAB 3: HYBRID SEARCH + MOOD AGENT BOT (Image 2 Feature)
# ==================================================================
with tab_agent:
    st.markdown("### 🤖 AI Agent Lai: Tìm Kiếm Tên Phim & Gợi Ý Theo Tâm Trạng")
    st.caption(
        "Gõ **tên phim cụ thể** (VD: `Toy Story`, `The Matrix`) để tìm kiếm trực tiếp, hoặc gõ **tâm trạng / mô tả tự nhiên** "
        "(VD: `hôm nay tôi muốn xem phim thể loại buồn, nhẹ nhàng, sâu lắng, ý nghĩa`) — Agent sẽ hiểu ngữ cảnh, lọc MovieLens-1M + Gemini 3.5 Flash-Lite và trả về phim kèm Trailer YouTube."
    )

    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []

    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    sample_prompt = None
    if qcol1.button("😢 Buồn, nhẹ nhàng, sâu lắng, ý nghĩa", use_container_width=True):
        sample_prompt = "Hôm nay tôi muốn xem phim thể loại buồn, nhẹ nhàng, sâu lắng, ý nghĩa"
    if qcol2.button("🧠 Sci-Fi hack não kiểu Matrix", use_container_width=True):
        sample_prompt = "Tìm phim viễn tưởng hack não, chiều sâu tâm lý giống The Matrix"
    if qcol3.button("🔎 Tìm phim: Toy Story", use_container_width=True):
        sample_prompt = "Toy Story"
    if qcol4.button("😂 Hài hước lãng mạn nhẹ nhàng", use_container_width=True):
        sample_prompt = "Phim hài hước tình cảm nhẹ nhàng cuối tuần"

    user_chat_input = st.chat_input("Nhập tên phim hoặc tâm trạng bạn muốn xem hôm nay...")
    active_query = sample_prompt or user_chat_input

    if active_query:
        with st.spinner("Agent đang phân tích ý định, tìm kiếm & suy luận gợi ý..."):
            ans = engine.hybrid_search_agent(active_query, gemini_keys=eff_gem_keys, top_n=8)
        st.session_state["agent_history"].append({"query": active_query, "response": ans})

    # Render latest interaction prominently + recent turns cleanly
    if not st.session_state["agent_history"]:
        st.info("Hãy thử bấm một gợi ý nhanh ở trên hoặc nhập câu hỏi của bạn vào ô chat bên dưới!")
    else:
        for turn_idx, turn in enumerate(reversed(st.session_state["agent_history"][-3:])):
            resp = turn["response"]
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                mode_lbl = (
                    "🎯 Chế độ Tìm Kiếm Tên Phim"
                    if resp.get("intent_mode") == "title_search"
                    else f"✨ Agent Gợi Ý Tâm Trạng ({'Gemini 3.5 Flash-Lite' if resp.get('mode') == 'gemini_agent' else 'Hybrid Semantic'})"
                )
                st.caption(f"{mode_lbl} • {resp.get('latency_ms', 0)} ms")
                st.markdown(resp.get("assistant_reply", ""))
                items = resp.get("items", [])
                if items:
                    ag_tmdb = tmdb.prefetch_movies(items, max_workers=8)
                    render_movie_grid(items, ag_tmdb, key_prefix=f"ag_{turn_idx}", mode="agent", cols_per_row=4)


# ==================================================================
# TAB 4: UNIFIED 9-MODEL BENCHMARK (1,188 TEST USERS)
# ==================================================================
with tab_benchmark:
    st.markdown("### 📊 Kết Quả Thực Nghiệm Chuẩn Trên Toàn Bộ Tập Test MovieLens-1M (`1,188` Test Users & `100,021` Ratings)")
    bench_data = load_benchmark_json()
    results = bench_data.get("results", {})
    if results:
        rows = []
        for name, info in results.items():
            m = info.get("metrics", {})
            rmse = info.get("rmse")
            rows.append(
                {
                    "Mô Hình": name,
                    "Nhóm": info.get("group", ""),
                    "Test Users": info.get("num_eval_users", 1188),
                    "Prec@10": f"{m.get('Precision@10', 0)*100:.2f}%",
                    "Recall@10": f"{m.get('Recall@10', 0)*100:.2f}%",
                    "HR@10": f"{m.get('HitRate@10', 0)*100:.2f}%",
                    "NDCG@10": round(m.get("NDCG@10", 0), 4),
                    "MRR@10": round(m.get("MRR@10", 0), 4),
                    "RMSE": round(rmse["RMSE"], 4) if rmse else "N/A",
                    "ILD@10": round(m.get("Diversity_ILD@10", 0), 4),
                    "ARP@10": round(m.get("PopBias_ARP@10", 0), 1),
                    "Novelty": round(m.get("Novelty@10", 0), 2),
                    "Cov@10": f"{m.get('Coverage@10', 0)*100:.2f}%",
                    "Latency (ms)": round(info.get("latency_ms", 0), 2),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        p1 = os.path.join(CHARTS_DIR, "accuracy_comparison.png")
        p2 = os.path.join(CHARTS_DIR, "rmse_comparison.png")
        p3 = os.path.join(CHARTS_DIR, "diversity_bias_comparison.png")
        p4 = os.path.join(CHARTS_DIR, "hybrid_accuracy_comparison.png")
        p5 = os.path.join(CHARTS_DIR, "deep_convergence_curves.png")
        if os.path.exists(p1):
            c1.image(p1, caption="So sánh Độ Chính Xác Top-10 (9 Mô Hình)", use_container_width=True)
        if os.path.exists(p2):
            c2.image(p2, caption="Sai Số Dự Báo Điểm Rating (100,021 Test Ratings)", use_container_width=True)
        if os.path.exists(p3):
            st.image(p3, caption="Độ Đa Dạng (ILD@10), Thiên Vị Phổ Biến (ARP@10) & Novelty", use_container_width=True)
        c3, c4 = st.columns(2)
        if os.path.exists(p4):
            c3.image(p4, caption="Stage-1 Deep Retrieval vs LightGCN + Gemini 3.5 Flash-Lite", use_container_width=True)
        if os.path.exists(p5):
            c4.image(p5, caption="Quá Trình Hội Tụ Loss (LightGCN, SASRec, BERT4Rec)", use_container_width=True)
    else:
        st.info("Chưa tìm thấy file `artifacts/unified_full_benchmark.json`.")
