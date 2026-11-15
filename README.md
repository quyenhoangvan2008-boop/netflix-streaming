# CineRecSys — Netflix-Style 2-Stage Movie Recommender & LLM Reranker

A portfolio-grade, Netflix-inspired interactive Movie Streaming & Recommender Verification System built on **MovieLens-1M** (`900,188` train+validation interactions, `100,021` test ratings, `1,188` positive test users, `3,883` enriched films), **PyTorch** (GPU-accelerated Graph & Transformer models), **TMDB API** (posters, cast, overviews & embedded YouTube trailers), and **Google Gemini 3.5 Flash-Lite**.

---

## Key Interactive Demo Features

1. **User Inspector (`Lịch sử vs. Gợi ý`)**
   - Pick any of the **1,188 positive test users** from `test.csv`.
   - **Top Row**: The user's **10 most recently watched films** along with their actual star ratings (`★ 1–5`).
   - **Bottom Row**: The **Top-10 recommended films**, highlighting exact **Ground Truth Hits** from the held-out test split (`✔ MATCH GROUND TRUTH`).
2. **Instant Model Switcher**
   - Compare models live on the exact same user across all stages:
     - **Baseline**: `Popularity`
     - **Classical / Low-Rank MF**: `Content-Based (Genre)`, `User-Based CF`, `Item-Based CF`, `Netflix SVD (PureSVD + Biased MF)`
     - **Deep Retrieval**: `LightGCN` (Graph Neural Network), `SASRec` (Causal Self-Attention), `BERT4Rec` (Bidirectional Masked Transformer)
     - **Full 2-Stage LLM Pipeline**: `LightGCN + Gemini 3.5 Flash-Lite` (Calibrated Rank & Z-Score Fusion)
3. **Interactive Cold-Start Sandbox (3–5 Seed Movies)**
   - Act as a brand-new user with zero database history: pick **3–5 favorite films** (or choose curated taste presets) and get real-time recommendations using **Inductive Graph Seed Folding** (`LightGCN`), **Fold-in SVD / Item-CF / User-CF**, **SASRec / BERT4Rec** sequence inference, and **Gemini 3.5 Flash-Lite**.
4. **Hybrid Search & Mood Discovery AI Agent**
   - Understands both **exact/fuzzy title queries** (e.g. `"Toy Story"`, `"The Matrix"`) and **natural-language Vietnamese/English mood prompts** (e.g. `"hôm nay tôi muốn xem phim thể loại buồn, nhẹ nhàng, sâu lắng, ý nghĩa"`), combining TF-IDF + bilingual mood-genre expansion over enriched Wikipedia/MovieLens metadata with **Gemini 3.5 Flash-Lite** curation.
5. **Click-to-Watch Movie Details & Embedded YouTube Trailers**
   - Clicking any film card opens a theater dialog pulling live metadata, posters, cast/director, and **embedded YouTube trailers (`st.video`)** via TMDB v3 API (`links.csv` `tmdbId` + `/search/movie` fallback).

---

## Full-Test Benchmark Results (`1,188` Positive Test Users & `100,021` Ratings)

| Model | Group | Prec@10 | Recall@10 | HR@10 | NDCG@10 | MRR@10 | RMSE | ILD@10 | ARP@10 | Novelty | Cov@10 | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Popularity | Classical / MF | 19.18% | 5.25% | 63.05% | 0.2070 | 0.3567 | 0.9535 | 0.6595 | 2307.9 | 8.63 | 4.00% | 0.00 ms |
| Content-Based (Genre) | Classical / MF | 7.35% | 2.36% | 39.39% | 0.0837 | 0.1898 | 1.1093 | 0.0897 | **574.8** | **11.61** | 20.28% | 0.12 ms |
| User-Based CF | Classical / MF | **21.46%** | 6.98% | 66.58% | **0.2357** | **0.4010** | 0.9413 | 0.6799 | 1702.2 | 9.18 | 14.33% | 6.80 ms |
| Item-Based CF | Classical / MF | 20.29% | 6.80% | 66.08% | 0.2247 | 0.3883 | 0.9075 | 0.6474 | 1548.4 | 9.36 | 18.81% | 1.15 ms |
| Netflix SVD (Low-Rank) | Classical / MF | 19.23% | 6.64% | 66.50% | 0.2132 | 0.3771 | **0.8933** | **0.6872** | 1236.6 | 9.67 | 18.98% | 0.96 ms |
| LightGCN | Deep Stage-1 | 20.92% | 7.30% | 68.10% | 0.2303 | 0.3876 | N/A | 0.6224 | 1309.9 | 9.74 | 32.16% | 0.53 ms |
| SASRec | Deep Stage-1 | 12.71% | 4.99% | 51.85% | 0.1404 | 0.2620 | N/A | 0.5378 | 887.8 | 10.47 | **44.05%** | 3.29 ms |
| BERT4Rec | Deep Stage-1 | 13.40% | 5.42% | 54.80% | 0.1522 | 0.2892 | N/A | 0.5410 | 920.0 | 10.31 | 34.91% | 3.07 ms |
| **LightGCN + Gemini-3.5-Flash-Lite** | Hybrid LLM | **21.25%** | **7.45%** | **68.35%** | **0.2346** | **0.3981** | N/A | 0.6179 | 1313.0 | 9.73 | 31.87% | 2299.47 ms |

---

## Project Structure

```text
netflix-streaming/
├── app.py                        # Main Streamlit Netflix-style UI
├── core/
│   ├── data_loader.py            # MovieLens-1M clean split & rated history loader
│   ├── engine.py                 # 9-model RecSys engine, Cold-Start & Hybrid Agent
│   ├── tmdb_client.py            # TMDB v3 posters, details & YouTube trailers
│   └── metrics.py                # Ranking, Diversity, Novelty, Popularity Bias metrics
├── models/                       # Classical/CF/SVD, LightGCN, SASRec, BERT4Rec
├── rerankers/gemini_reranker.py  # Gemini 3.5 Flash-Lite key-rotating reranker & agent
├── data/movielens/               # Clean train/val/test split, links.csv, enriched metadata
├── checkpoints/                  # Pretrained PyTorch checkpoints
├── artifacts/                    # Unified 1,188-user benchmark JSON & Matplotlib charts
└── tests/test_engine.py          # Automated verification suite
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API keys (never commit secrets.toml!)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your TMDB & Google AI Studio Gemini keys
# (Or enter keys directly in the Streamlit Sidebar at runtime)

# 3. Run unit tests
python -m unittest discover -s tests -v

# 4. Launch the Netflix-style Streamlit Web App
streamlit run app.py
```
