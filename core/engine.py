"""
Unified 2-Stage Recommender Engine for the Netflix-Style Streaming Demo.
Supports:
- All 9 benchmarked models across 1,188 test users (User Inspector + Model Switcher).
- Real-time Interactive Cold-Start from arbitrary 3-5 user-picked seed movies across ALL 9 models.
- Bilingual Hybrid Search & Mood Discovery Agent (TF-IDF + Moody Lexicon + Gemini 3.5 Flash-Lite).
"""

import os
import re
import time
import difflib
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from core.data_loader import load_movielens_bundle, build_movielens_graph_adj
from models.classical import (
    PopularityRecommender,
    ContentBasedGenreRecommender,
    UserBasedCFRecommender,
    ItemBasedCFRecommender,
    NetflixSVDRecommender,
)
from models.lightgcn import LightGCN
from models.sasrec import SASRec
from models.bert4rec import BERT4Rec
from rerankers.gemini_reranker import rerank_with_gemini_calibrated, agent_curate_movies

MODEL_CATALOG = [
    ("popularity", "Baseline • Popularity", "Baseline"),
    ("content_based", "Classical • Content-Based (Genre)", "Classical / MF"),
    ("user_cf", "Collaborative • User-Based CF", "Classical / MF"),
    ("item_cf", "Collaborative • Item-Based CF", "Classical / MF"),
    ("svd", "Low-Rank • Netflix SVD", "Classical / MF"),
    ("lightgcn", "Retrieval • LightGCN (Graph)", "Deep Retrieval"),
    ("sasrec", "Retrieval • SASRec (Transformer)", "Deep Retrieval"),
    ("bert4rec", "Retrieval • BERT4Rec (Bidirectional)", "Deep Retrieval"),
    ("lightgcn_gemini", "Full Pipeline • LightGCN + Gemini 3.5 Flash-Lite", "Retrieval + LLM Reranker"),
]

# Bilingual Vietnamese <-> English cinema mood & genre bridge for accurate local retrieval
VN_MOOD_BRIDGE = {
    "buồn": ("tragic heartbreaking tearjerker melancholy grief sorrow poignant emotional drama", ["Drama"]),
    "sâu lắng": ("poignant thoughtful introspective touching bittersweet soulful intimate cinema drama", ["Drama"]),
    "nhẹ nhàng": ("gentle heartwarming tender warm touching quiet calm uplifting romance drama", ["Drama", "Romance"]),
    "ý nghĩa": ("inspiring profound humanitarian classic masterpiece academy award moving friendship hope", ["Drama"]),
    "cảm động": ("touching heartfelt emotional moving tears compassion love devotion", ["Drama", "Romance"]),
    "tâm lý": ("psychological character study suspense moral dilemma mind drama", ["Drama", "Thriller"]),
    "lãng mạn": ("romance love couple relationship passionate longing", ["Romance"]),
    "tình cảm": ("romance drama love relationship emotional tender", ["Romance", "Drama"]),
    "hành động": ("action hero chase fight explosion martial arts rescue", ["Action"]),
    "viễn tưởng": ("sci-fi futuristic space alien time travel dystopia cyber technology", ["Sci-Fi"]),
    "hack não": ("mind-bending twist mystery surreal dream sci-fi psychological labyrinth", ["Sci-Fi", "Mystery", "Thriller"]),
    "kinh dị": ("horror scary ghost killer haunted terror nightmare supernatural", ["Horror"]),
    "hài": ("comedy funny hilarious humor laugh satire witty", ["Comedy"]),
    "hoạt hình": ("animation animated pixar disney cartoon", ["Animation", "Children's"]),
    "gia đình": ("family children kids heartwarming wholesome adventure", ["Children's", "Animation", "Comedy"]),
    "hình sự": ("crime detective mafia heist murder investigation", ["Crime", "Thriller"]),
    "trinh thám": ("mystery detective investigation clue suspense", ["Mystery", "Crime"]),
    "chiến tranh": ("war soldier military battle history wwii", ["War", "Drama"]),
    "phiêu lưu": ("adventure journey quest expedition fantasy", ["Adventure", "Fantasy"]),
    "cuốn hút": ("gripping captivating riveting thrilling intense engaging compelling masterpiece", ["Thriller", "Drama", "Sci-Fi", "Action"]),
    "kịch tính": ("dramatic suspenseful intense tension gripping plot twist thriller", ["Thriller", "Drama", "Crime"]),
    "gay cấn": ("thrilling intense suspenseful action chase thriller", ["Thriller", "Action", "Crime"]),
    "hồi hộp": ("suspenseful tense mystery thriller gripping twist", ["Thriller", "Mystery"]),
    "hấp dẫn": ("engaging exciting fascinating entertaining blockbuster thriller adventure", ["Action", "Adventure", "Thriller"]),
    "kinh điển": ("classic masterpiece acclaim iconic top rated cinema", ["Drama", "Crime", "Sci-Fi"]),
}

ONBOARDING_SHOWCASE_IDS: List[int] = [
    2571,  # The Matrix (Action|Sci-Fi|Thriller)
    318,   # The Shawshank Redemption (Drama)
    1,     # Toy Story (Animation|Children's|Comedy)
    593,   # The Silence of the Lambs (Drama|Thriller)
    858,   # The Godfather (Action|Crime|Drama)
    2028,  # Saving Private Ryan (Action|Drama|War)
    480,   # Jurassic Park (Action|Adventure|Sci-Fi)
    356,   # Forrest Gump (Comedy|Romance|War)
    2762,  # The Sixth Sense (Thriller)
    1265,  # Groundhog Day (Comedy|Romance)
    1214,  # Alien (Action|Horror|Sci-Fi|Thriller)
    2396,  # Shakespeare in Love (Comedy|Romance)
]

GENRE_VI_MAP: Dict[str, str] = {
    "Action": "Hành động",
    "Adventure": "Phiêu lưu",
    "Animation": "Hoạt hình",
    "Children's": "Gia đình & Thiếu nhi",
    "Comedy": "Hài",
    "Crime": "Hình sự",
    "Documentary": "Tài liệu",
    "Drama": "Chính kịch",
    "Fantasy": "Giả tưởng",
    "Film-Noir": "Phim Noir",
    "Horror": "Kinh dị",
    "Musical": "Nhạc kịch",
    "Mystery": "Bí ẩn",
    "Romance": "Lãng mạn",
    "Sci-Fi": "Khoa học viễn tưởng",
    "Thriller": "Giật gân",
    "War": "Chiến tranh",
    "Western": "Viễn Tây",
}


def translate_genre_vi(genre: str) -> str:
    return GENRE_VI_MAP.get(str(genre), str(genre))


def translate_genres_vi(genres: List[str]) -> str:
    return ", ".join(translate_genre_vi(g) for g in genres if g)


def _cosine_genre_ild(item_ids: List[int], genre_matrix: np.ndarray) -> float:
    if len(item_ids) < 2:
        return 0.0
    vecs = genre_matrix[np.array(item_ids, dtype=np.int64)]
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vn = vecs / norms
    sim = vn @ vn.T
    k = len(item_ids)
    tri_idx = np.triu_indices(k, k=1)
    return float(np.mean(1.0 - sim[tri_idx]))


class RecSysEngine:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data", "movielens")
        self.ckpt_dir = os.path.join(base_dir, "checkpoints")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_users = 6041
        self.num_items = 3953

        bundle = load_movielens_bundle(self.data_dir)
        self.train_val_df = bundle["train_val_df"]
        self.test_df = bundle["test_df"]
        self.movies_df = bundle["movies_df"]
        self.metadata = bundle["metadata"]
        self.user_info = bundle["user_info"]
        self.rated_history = bundle["rated_history"]
        self.train_seqs = bundle["train_seqs"]
        self.seen_map = bundle["seen_map"]
        self.test_pos = bundle["test_pos"]
        self.valid_catalog = bundle["valid_catalog"]
        self.popular_movie_ids = bundle["popular_movie_ids"]
        self.item_pop_counts = bundle["item_pop_counts"]
        self.total_interactions = bundle["total_interactions"]

        self.valid_mask_np = np.zeros(self.num_items + 1, dtype=bool)
        self.valid_mask_np[self.valid_catalog] = True
        self.valid_mask_t = torch.zeros(self.num_items, dtype=torch.bool, device=self.device)
        idx_t = torch.tensor([int(m) - 1 for m in self.valid_catalog if 1 <= int(m) <= self.num_items], dtype=torch.long, device=self.device)
        self.valid_mask_t[idx_t] = True

        self._fit_classical_models()
        self._load_deep_checkpoints()
        self._build_search_index()
        self._rt_cache: Dict[Tuple, Dict] = {}

    def _fit_classical_models(self):
        self.popularity = PopularityRecommender(num_items=self.num_items)
        self.popularity.fit(self.train_val_df)

        self.genre_matrix, self.genre_names = ContentBasedGenreRecommender.build_genre_matrix(self.movies_df, self.num_items)
        self.genre_to_idx = {str(g): i for i, g in enumerate(self.genre_names)}
        norms = np.linalg.norm(self.genre_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.genre_norm_matrix = (self.genre_matrix / norms).astype(np.float32)

        self.content_based = ContentBasedGenreRecommender(num_users=self.num_users, num_items=self.num_items)
        self.content_based.fit(self.train_val_df, self.genre_matrix)

        self.user_cf = UserBasedCFRecommender(num_users=self.num_users, num_items=self.num_items, k_neighbors=40)
        self.user_cf.fit(self.train_val_df)

        self.item_cf = ItemBasedCFRecommender(num_users=self.num_users, num_items=self.num_items, k_neighbors=40)
        self.item_cf.fit(self.train_val_df)

        svd = NetflixSVDRecommender(num_users=self.num_users, num_items=self.num_items, rank=64)
        svd_ckpt = os.path.join(self.ckpt_dir, "netflix_svd.pt")
        if os.path.exists(svd_ckpt):
            ckpt = torch.load(svd_ckpt, map_location=self.device, weights_only=False)
            svd.load_state_dict(ckpt["state_dict"], strict=False)
            svd.mu = ckpt["mu"]
        svd.attach_implicit_matrix(self.train_val_df)
        self.svd = svd.to(self.device)
        self.svd_Vt_np = self.svd.Vt_pure.cpu().numpy()

    def _load_deep_checkpoints(self):
        gcn_adj = build_movielens_graph_adj(self.train_seqs, self.num_users, self.num_items).to(self.device)
        gcn = LightGCN(self.num_users, self.num_items, embedding_dim=64, num_layers=2).to(self.device)
        gcn_ckpt = os.path.join(self.ckpt_dir, "lightgcn.pt")
        if os.path.exists(gcn_ckpt):
            gcn.load_state_dict(torch.load(gcn_ckpt, map_location=self.device, weights_only=True))
        gcn.eval()
        with torch.no_grad():
            self.gcn_u_emb, self.gcn_i_emb = gcn(gcn_adj)
        self.lightgcn = gcn

        sas = SASRec(self.num_items, max_seq_len=50, hidden_dim=64).to(self.device)
        sas_ckpt = os.path.join(self.ckpt_dir, "sasrec.pt")
        if os.path.exists(sas_ckpt):
            sas.load_state_dict(torch.load(sas_ckpt, map_location=self.device, weights_only=True))
        sas.eval()
        self.sasrec = sas

        bert = BERT4Rec(self.num_items, max_seq_len=50, hidden_dim=64).to(self.device)
        bert_ckpt = os.path.join(self.ckpt_dir, "bert4rec.pt")
        if os.path.exists(bert_ckpt):
            bert.load_state_dict(torch.load(bert_ckpt, map_location=self.device, weights_only=True))
        bert.eval()
        self.bert4rec = bert

    def _build_search_index(self):
        self._search_mids = sorted(self.metadata.keys())
        corpus = []
        self._clean_titles = {}
        self._genres_arr = []
        self._quality_prior_arr = np.zeros(len(self._search_mids), dtype=np.float32)

        for idx, mid in enumerate(self._search_mids):
            m = self.metadata[mid]
            disp = m["display_title"]
            no_yr = re.sub(r"\s*\(\d{4}\)\s*$", "", disp).lower().strip()
            self._clean_titles[mid] = no_yr
            self._genres_arr.append(frozenset(m["genres"]))

            votes = float(m.get("popularity_count", 0))
            rating = float(m.get("ml_avg_rating", 0.0))
            # Bayesian acclaimed prior: rewards well-rated, widely loved films; penalizes weak < 3.4 fillers
            bayes_r = (votes * rating + 80.0 * 3.3) / (votes + 80.0)
            pop_part = 0.045 * np.log1p(votes)
            qual_part = 0.28 * (bayes_r - 3.3)
            self._quality_prior_arr[idx] = pop_part + qual_part

            doc = f"{disp} {m['title']} {' '.join(m['genres'])} {m.get('director', '')} {m.get('description', '')}"
            corpus.append(doc)

        self._tfidf = TfidfVectorizer(stop_words="english", max_features=12000, ngram_range=(1, 2))
        self._tfidf_mat = self._tfidf.fit_transform(corpus)
        self._search_cache: Dict[Tuple, Dict] = {}
        self._popular_dropdown_ids: List[int] = [int(m) for m in self.popular_movie_ids]

    # ------------------------------------------------------------------
    # Scoring Helper -> Returns ordered (movie_id, score) pairs cleanly
    # ------------------------------------------------------------------
    def _rank_from_array(self, scores_np: np.ndarray, seen_set: Set[int], top_k: int) -> List[Tuple[int, float]]:
        sc = scores_np.astype(np.float32, copy=True)
        sc[~self.valid_mask_np] = -1e9
        sc[0] = -1e9
        if seen_set:
            arr = [i for i in seen_set if 1 <= i <= self.num_items]
            if arr:
                sc[arr] = -1e9
        k = min(top_k, len(self.valid_catalog))
        idx = np.argpartition(-sc, k)[:k]
        idx = idx[np.argsort(-sc[idx])]
        return [(int(i), float(sc[i])) for i in idx if sc[i] > -1e8]

    def _score_stage1_user(self, user_id: int, model_key: str, top_k: int = 20) -> List[Tuple[int, float]]:
        history = self.train_seqs.get(user_id, [])
        seen_set = self.seen_map.get(user_id, set())

        if model_key == "popularity" or (model_key not in ("popularity",) and not history and model_key in ("lightgcn", "sasrec", "bert4rec")):
            scores = self.popularity.item_counts.copy()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "content_based":
            u_prof = self.content_based.user_profiles[user_id]
            if np.linalg.norm(u_prof) == 0:
                scores = self.content_based.item_counts.copy()
            else:
                scores = self.genre_norm_matrix @ u_prof
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "user_cf":
            u_vec = self.user_cf.R_norm_imp[user_id]
            if u_vec.nnz == 0:
                scores = self.user_cf.item_counts.copy()
            else:
                sims = (u_vec @ self.user_cf.R_norm_imp.T).toarray().ravel()
                sims[user_id] = 0.0
                top_u = np.argpartition(-sims, self.user_cf.k_neighbors)[: self.user_cf.k_neighbors]
                scores = (sims[top_u] @ self.user_cf.R_imp[top_u].toarray()).ravel()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "item_cf":
            u_row = self.item_cf.R_imp[user_id]
            if u_row.nnz == 0:
                scores = self.item_cf.item_counts.copy()
            else:
                scores = (u_row @ self.item_cf.S_imp_topk.T).toarray().ravel()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "svd":
            if self.svd.R_imp is not None and self.svd.R_imp[user_id].nnz > 0:
                scores = (self.svd.R_imp[user_id] @ self.svd_Vt_np.T @ self.svd_Vt_np).ravel()
            else:
                scores = self.popularity.item_counts.copy()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "lightgcn":
            with torch.no_grad():
                scores_t = self.lightgcn.predict_for_user(user_id, self.gcn_u_emb, self.gcn_i_emb).clone()
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [it - 1 for it in seen_set if 1 <= it <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=top_k)
                return [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]

        if model_key == "sasrec":
            with torch.no_grad():
                seq = history[-50:]
                if len(seq) < 50:
                    seq = [0] * (50 - len(seq)) + seq
                inp = torch.tensor([seq], dtype=torch.long, device=self.device)
                scores_t = self.sasrec.predict_next_item(inp).clone()
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [it - 1 for it in seen_set if 1 <= it <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=top_k)
                return [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]

        if model_key == "bert4rec":
            with torch.no_grad():
                seq = history[-49:] + [self.bert4rec.mask_token]
                if len(seq) < 50:
                    seq = [0] * (50 - len(seq)) + seq
                inp = torch.tensor([seq], dtype=torch.long, device=self.device)
                scores_t = self.bert4rec.predict_next_item(inp).clone()
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [it - 1 for it in seen_set if 1 <= it <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=top_k)
                return [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]

        return self._rank_from_array(self.popularity.item_counts.copy(), seen_set, top_k)

    # ------------------------------------------------------------------
    # 1. User Inspector & Model Switcher
    # ------------------------------------------------------------------
    def get_user_watch_history(self, user_id: int, limit: int = 10) -> List[Dict]:
        hist = self.rated_history.get(user_id, [])
        recent = list(reversed(hist[-limit:]))  # Most recent first
        items = []
        for mid, rating, ts in recent:
            m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
            m["user_rating"] = rating
            m["timestamp"] = ts
            items.append(m)
        return items

    def recommend_for_user(
        self,
        user_id: int,
        model_key: str,
        top_k: int = 10,
        gemini_keys: Optional[List[str]] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        gt_set = self.test_pos.get(user_id, set())
        user_intent = ""
        reranked = False
        warning = ""

        if model_key == "lightgcn_gemini":
            s1_pairs = self._score_stage1_user(user_id, "lightgcn", top_k=20)
            cand_items = []
            for mid, sc in s1_pairs:
                m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
                m["score"] = sc
                cand_items.append(m)

            if gemini_keys:
                hist_titles = [
                    self.metadata[m]["display_title"]
                    for m in self.train_seqs.get(user_id, [])[-10:]
                    if m in self.metadata
                ]
                try:
                    res = rerank_with_gemini_calibrated(
                        api_keys=gemini_keys,
                        history_titles=hist_titles,
                        cand_items=cand_items,
                        retriever_name="LightGCN",
                        top_n=top_k,
                        alpha_fuse=0.30,
                    )
                    score_map = dict(s1_pairs)
                    ranked_pairs = [(mid, score_map.get(mid, 0.0)) for mid in res["ranked_ids"]]
                    user_intent = res.get("user_intent", "")
                    reranked = True
                except Exception as exc:
                    ranked_pairs = s1_pairs[:top_k]
                    warning = f"Gemini API fallback về LightGCN Stage-1: {exc}"
            else:
                ranked_pairs = s1_pairs[:top_k]
                warning = "Chưa nhập Gemini API Key — đang hiển thị kết quả LightGCN Stage-1."
        else:
            ranked_pairs = self._score_stage1_user(user_id, model_key, top_k=top_k)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        rec_items = []
        rec_ids = []
        hits = 0
        dcg = 0.0
        for rank, (mid, sc) in enumerate(ranked_pairs[:top_k], 1):
            is_hit = mid in gt_set
            if is_hit:
                hits += 1
                dcg += 1.0 / np.log2(rank + 1)
            m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
            m["rank"] = rank
            m["score"] = round(float(sc), 4)
            m["is_hit"] = is_hit
            rec_items.append(m)
            rec_ids.append(mid)

        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(top_k, len(gt_set)))) if gt_set else 0.0
        ndcg = float(dcg / idcg) if idcg > 0 else 0.0
        prec = float(hits / max(1, len(rec_ids)))
        rec = float(hits / len(gt_set)) if gt_set else 0.0
        ild = _cosine_genre_ild(rec_ids, self.genre_matrix)

        return {
            "items": rec_items,
            "latency_ms": round(latency_ms, 2),
            "hits": hits,
            "total_gt": len(gt_set),
            "precision_at_10": prec,
            "recall_at_10": rec,
            "ndcg_at_10": ndcg,
            "ild_at_10": ild,
            "user_intent": user_intent,
            "reranked": reranked,
            "warning": warning,
        }

    # ------------------------------------------------------------------
    # 2. Interactive Cold-Start from 3-5 Arbitrary Seed Movies
    # ------------------------------------------------------------------
    def _score_stage1_cold_seeds(self, seed_ids: List[int], model_key: str, top_k: int = 20) -> List[Tuple[int, float]]:
        seen_set = set(int(x) for x in seed_ids)
        seeds = [int(x) for x in seed_ids if 1 <= int(x) <= self.num_items]

        if not seeds or model_key == "popularity":
            return self._rank_from_array(self.popularity.item_counts.copy(), seen_set, top_k)

        if model_key == "content_based":
            u_prof = self.genre_norm_matrix[seeds].mean(axis=0)
            norm = float(np.linalg.norm(u_prof))
            if norm > 0:
                u_prof = u_prof / norm
            scores = self.genre_norm_matrix @ u_prof
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "item_cf":
            u_row = sp.csr_matrix(
                (np.ones(len(seeds), dtype=np.float32), (np.zeros(len(seeds), dtype=np.int64), np.array(seeds))),
                shape=(1, self.num_items + 1),
                dtype=np.float32,
            )
            scores = (u_row @ self.item_cf.S_imp_topk.T).toarray().ravel()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "user_cf":
            vec = sp.csr_matrix(
                (np.ones(len(seeds), dtype=np.float32) / np.sqrt(len(seeds)), (np.zeros(len(seeds), dtype=np.int64), np.array(seeds))),
                shape=(1, self.num_items + 1),
                dtype=np.float32,
            )
            sims = (vec @ self.user_cf.R_norm_imp.T).toarray().ravel()
            top_u = np.argpartition(-sims, self.user_cf.k_neighbors)[: self.user_cf.k_neighbors]
            scores = (sims[top_u] @ self.user_cf.R_imp[top_u].toarray()).ravel()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "svd":
            v_seed = sp.csr_matrix(
                (np.ones(len(seeds), dtype=np.float32), (np.zeros(len(seeds), dtype=np.int64), np.array(seeds))),
                shape=(1, self.num_items + 1),
                dtype=np.float32,
            )
            scores = (v_seed @ self.svd_Vt_np.T @ self.svd_Vt_np).ravel()
            return self._rank_from_array(scores, seen_set, top_k)

        if model_key == "lightgcn":
            with torch.no_grad():
                idx = torch.tensor([s - 1 for s in seeds], dtype=torch.long, device=self.device)
                u_pseudo = self.gcn_i_emb[idx].mean(dim=0)
                scores_t = torch.matmul(self.gcn_i_emb, u_pseudo).clone()
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [s - 1 for s in seen_set if 1 <= s <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=top_k)
                return [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]

        if model_key == "sasrec":
            with torch.no_grad():
                seq = seeds[-50:]
                if len(seq) < 50:
                    seq = [0] * (50 - len(seq)) + seq
                inp = torch.tensor([seq], dtype=torch.long, device=self.device)
                scores_t = self.sasrec.predict_next_item(inp).clone()
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [s - 1 for s in seen_set if 1 <= s <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=top_k)
                return [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]

        if model_key == "bert4rec":
            with torch.no_grad():
                seq = seeds[-49:] + [self.bert4rec.mask_token]
                if len(seq) < 50:
                    seq = [0] * (50 - len(seq)) + seq
                inp = torch.tensor([seq], dtype=torch.long, device=self.device)
                scores_t = self.bert4rec.predict_next_item(inp).clone()
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [s - 1 for s in seen_set if 1 <= s <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=top_k)
                return [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]

        return self._rank_from_array(self.popularity.item_counts.copy(), seen_set, top_k)

    def recommend_for_cold_seeds(
        self,
        seed_movie_ids: List[int],
        model_key: str,
        top_k: int = 10,
        gemini_keys: Optional[List[str]] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        user_intent = ""
        reranked = False
        warning = ""

        if model_key == "lightgcn_gemini":
            s1_pairs = self._score_stage1_cold_seeds(seed_movie_ids, "lightgcn", top_k=20)
            cand_items = []
            for mid, sc in s1_pairs:
                m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
                m["score"] = sc
                cand_items.append(m)
            if gemini_keys:
                hist_titles = [
                    self.metadata[m]["display_title"]
                    for m in seed_movie_ids
                    if m in self.metadata
                ]
                try:
                    res = rerank_with_gemini_calibrated(
                        api_keys=gemini_keys,
                        history_titles=hist_titles,
                        cand_items=cand_items,
                        retriever_name="LightGCN (Cold-Start Seed Folding)",
                        top_n=top_k,
                        alpha_fuse=0.35,
                    )
                    score_map = dict(s1_pairs)
                    ranked_pairs = [(mid, score_map.get(mid, 0.0)) for mid in res["ranked_ids"]]
                    user_intent = res.get("user_intent", "")
                    reranked = True
                except Exception as exc:
                    ranked_pairs = s1_pairs[:top_k]
                    warning = f"Gemini fallback về LightGCN Seed-Folding: {exc}"
            else:
                ranked_pairs = s1_pairs[:top_k]
                warning = "Chưa có Gemini API Key — đang hiển thị LightGCN Inductive Seed Folding."
        else:
            ranked_pairs = self._score_stage1_cold_seeds(seed_movie_ids, model_key, top_k=top_k)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        rec_items = []
        rec_ids = []
        for rank, (mid, sc) in enumerate(ranked_pairs[:top_k], 1):
            m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
            m["rank"] = rank
            m["score"] = round(float(sc), 4)
            rec_items.append(m)
            rec_ids.append(mid)

        return {
            "items": rec_items,
            "latency_ms": round(latency_ms, 2),
            "ild_at_10": _cosine_genre_ild(rec_ids, self.genre_matrix),
            "user_intent": user_intent,
            "reranked": reranked,
            "warning": warning,
        }

    # ------------------------------------------------------------------
    # 3. Hybrid Search + Mood Discovery Agent
    # ------------------------------------------------------------------
    def hybrid_search_agent(
        self,
        query: str,
        gemini_keys: Optional[List[str]] = None,
        top_n: int = 8,
    ) -> Dict:
        q = (query or "").strip()
        if not q:
            return {"mode": "empty", "assistant_reply": "Hãy nhập tên phim hoặc tâm trạng/thể loại bạn muốn xem!", "items": []}

        q_low = q.lower()
        cache_k = (q_low, int(top_n), bool(gemini_keys))
        if cache_k in self._search_cache:
            return self._search_cache[cache_k]

        t0 = time.perf_counter()

        # Check Vietnamese mood triggers & expand query with genre vote weights
        expanded_terms = [q_low]
        genre_weights: Dict[str, float] = {}
        matched_vn_moods = []
        for vn_kw, (en_exp, genres) in VN_MOOD_BRIDGE.items():
            if vn_kw in q_low:
                expanded_terms.append(en_exp)
                for g in genres:
                    genre_weights[g] = genre_weights.get(g, 0.0) + 1.0
                matched_vn_moods.append(vn_kw)

        # Fast vectorized-style title match (avoids slow 3,883x difflib loop)
        title_scores = np.zeros(len(self._search_mids), dtype=np.float32)
        sub_hits = 0
        max_exact_score = 0.0
        for idx, mid in enumerate(self._search_mids):
            t_clean = self._clean_titles[mid]
            if q_low == t_clean:
                s = 1.0
                sub_hits += 1
            elif q_low in t_clean:
                s = 0.86 + 0.12 * (len(q_low) / max(1, len(t_clean)))
                sub_hits += 1
            else:
                s = 0.0
            title_scores[idx] = s
            if s > max_exact_score:
                max_exact_score = s

        # Only run fuzzy matching when substring search yields very few hits & not a VN mood query
        if sub_hits < 5 and len(q_low) >= 3 and not matched_vn_moods:
            for idx, mid in enumerate(self._search_mids):
                if title_scores[idx] == 0.0:
                    r = difflib.SequenceMatcher(None, q_low, self._clean_titles[mid]).ratio()
                    if r >= 0.55:
                        title_scores[idx] = r * 0.65
                        if title_scores[idx] > max_exact_score:
                            max_exact_score = title_scores[idx]

        # TF-IDF semantic match over enriched overview + genres + director
        q_vec = self._tfidf.transform([" ".join(expanded_terms)])
        sem_scores = cosine_similarity(q_vec, self._tfidf_mat).ravel()

        # Weighted genre alignment & family/horror guard for emotional adult prompts
        genre_bonus = np.zeros(len(self._search_mids), dtype=np.float32)
        wants_kids = any(g in genre_weights for g in ("Animation", "Children's"))
        wants_horror = "Horror" in genre_weights
        max_gw = max(1.0, max(genre_weights.values())) if genre_weights else 1.0
        for idx in range(len(self._search_mids)):
            gs = self._genres_arr[idx]
            if genre_weights:
                matched_w = sum(genre_weights[g] for g in gs if g in genre_weights)
                if matched_w > 0:
                    genre_bonus[idx] = 0.34 * (matched_w / max_gw)
                if not wants_kids and ("Animation" in gs or "Children's" in gs):
                    genre_bonus[idx] -= 0.45
                if not wants_horror and "Horror" in gs:
                    genre_bonus[idx] -= 0.35

        is_title_search = (max_exact_score >= 0.72) and not matched_vn_moods
        intent_mode = "title_search" if is_title_search else "mood_recommend"

        if is_title_search:
            total = 2.4 * title_scores + 0.35 * sem_scores + 0.35 * self._quality_prior_arr
        else:
            total = 0.25 * title_scores + 1.15 * sem_scores + genre_bonus + 1.15 * self._quality_prior_arr

        pool_k = 32
        top_pool_idx = np.argpartition(-total, pool_k)[:pool_k]
        top_pool_idx = top_pool_idx[np.argsort(-total[top_pool_idx])]
        pool_items = []
        for idx in top_pool_idx:
            mid = self._search_mids[idx]
            m = dict(self.metadata[mid])
            m["match_score"] = round(float(total[idx]), 4)
            pool_items.append(m)

        # Use Gemini 3.5 Flash-Lite curation for mood/semantic queries (fast cacheable)
        if gemini_keys and (not is_title_search or max_exact_score < 0.85):
            try:
                res = agent_curate_movies(
                    api_keys=gemini_keys,
                    user_query=q,
                    local_candidates=pool_items,
                    intent_mode=intent_mode,
                    top_n=top_n,
                )
                items_out = []
                for rank, p in enumerate(res["picks"], 1):
                    mid = p["movie_id"]
                    m = dict(self.metadata[mid])
                    m["rank"] = rank
                    m["why_match"] = p["reason"]
                    items_out.append(m)
                out = {
                    "mode": "gemini_agent",
                    "intent_mode": intent_mode,
                    "assistant_reply": res["assistant_reply"],
                    "items": items_out,
                    "latency_ms": res["latency_ms"],
                }
                self._search_cache[cache_k] = out
                return out
            except Exception:
                pass

        # Clean instant local fallback / direct title search result
        items_out = []
        for rank, it in enumerate(pool_items[:top_n], 1):
            it_copy = dict(it)
            it_copy["rank"] = rank
            genres_vi = translate_genres_vi(it["genres"])
            if is_title_search:
                it_copy["why_match"] = f"Thể loại: {genres_vi} • ★{it['ml_avg_rating']}"
            else:
                it_copy["why_match"] = f"{genres_vi} • ★{it['ml_avg_rating']} ({it['popularity_count']} đánh giá)"
            items_out.append(it_copy)

        if is_title_search:
            reply = f"Tìm thấy {len(items_out)} phim khớp nhất với \"{q}\":"
        else:
            mood_note = f" ({', '.join(matched_vn_moods)})" if matched_vn_moods else ""
            reply = f"Gợi ý phim phù hợp theo tâm trạng{mood_note} dành cho bạn:"

        out = {
            "mode": "local_hybrid_search",
            "intent_mode": intent_mode,
            "assistant_reply": reply,
            "items": items_out,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        if len(self._search_cache) > 64:
            self._search_cache.clear()
        self._search_cache[cache_k] = out
        return out

    # ------------------------------------------------------------------
    # 4. Streamlined Cookie + Real-Time Watch Page (LightGCN -> Fast LLM)
    # ------------------------------------------------------------------
    def get_onboarding_showcase(self) -> List[Dict]:
        items = []
        for mid in ONBOARDING_SHOWCASE_IDS:
            if mid in self.metadata:
                items.append(dict(self.metadata[mid]))
        return items

    def search_titles_live(self, query: str, limit: int = 9) -> List[Dict]:
        """Instant live title filter for the onboarding movie picker."""
        q = (query or "").strip().lower()
        if not q:
            return self.get_onboarding_showcase()[:limit]

        scored = []
        for mid in self._search_mids:
            m = self.metadata[mid]
            disp_low = m["display_title"].lower()
            clean_low = self._clean_titles[mid]
            if clean_low.startswith(q) or disp_low.startswith(q):
                tier = 3.0
            elif q in disp_low:
                tier = 2.0
            else:
                ratio = difflib.SequenceMatcher(None, q, clean_low).ratio()
                if ratio < 0.45:
                    continue
                tier = ratio
            pop_score = float(np.log1p(m.get("popularity_count", 0)))
            scored.append((tier, pop_score, mid))

        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [dict(self.metadata[mid]) for _, _, mid in scored[:limit]]

    def recommend_realtime_stream(
        self,
        watched_movie_ids: List[int],
        preferred_genres: Optional[List[str]] = None,
        current_movie_id: Optional[int] = None,
        gemini_keys: Optional[List[str]] = None,
        top_k: int = 12,
        use_llm: bool = True,
    ) -> Dict:
        """
        Real-time recommender for Home & Watch views:
        - Cold-start (no watched, no genres, no current movie) -> Popularity
        - Genre-only cold-start -> Genre match weighted by Popularity (+ optional LLM rerank)
        - Active / Returning user (history or currently watching) -> Pretrained LightGCN
          inductive graph propagation (zero retraining) + optional fast Gemini 3.5 Flash-Lite
          rerank (cleanly skips back to LightGCN if disabled, slow, or unavailable).
        """
        t0 = time.perf_counter()
        genres = [str(g) for g in (preferred_genres or []) if g in self.genre_to_idx]
        clean_watched = [int(m) for m in watched_movie_ids if 1 <= int(m) <= self.num_items]

        seen_set: Set[int] = set(clean_watched)
        if current_movie_id and 1 <= int(current_movie_id) <= self.num_items:
            seen_set.add(int(current_movie_id))

        cache_key = (
            tuple(clean_watched[-10:]),
            int(current_movie_id) if current_movie_id else None,
            tuple(sorted(genres)),
            int(top_k),
            bool(use_llm and gemini_keys),
        )
        if cache_key in self._rt_cache:
            return self._rt_cache[cache_key]

        # Genre preference boost vector across all catalog movies
        genre_boost_np = np.zeros(self.num_items, dtype=np.float32)
        if genres:
            g_vec = np.zeros(self.genre_matrix.shape[1], dtype=np.float32)
            for g in genres:
                g_vec[self.genre_to_idx[g]] = 1.0
            norm = float(np.linalg.norm(g_vec))
            if norm > 0:
                g_vec /= norm
            genre_boost_np = (self.genre_norm_matrix[1 : self.num_items + 1] @ g_vec).astype(np.float32)

        # Stage-1 retrieval
        has_seeds = bool(clean_watched) or (current_movie_id is not None and 1 <= int(current_movie_id) <= self.num_items)
        if not has_seeds and not genres:
            s1_pairs = self._rank_from_array(self.popularity.item_counts.copy(), seen_set, top_k=top_k)
            rec_items = []
            for rank, (mid, sc) in enumerate(s1_pairs, 1):
                m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
                m["rank"] = rank
                m["score"] = round(float(sc), 4)
                rec_items.append(m)
            out = {
                "items": rec_items,
                "engine": "Popularity",
                "reranked": False,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
            }
            self._rt_cache[cache_key] = out
            return out

        if not has_seeds and genres:
            pop_arr = self.popularity.item_counts[1 : self.num_items + 1].astype(np.float32)
            pop_norm = pop_arr / (float(pop_arr.max()) + 1e-6)
            combined = 2.0 * genre_boost_np + 0.65 * pop_norm
            score_full = np.zeros(self.num_items + 1, dtype=np.float32)
            score_full[1 : self.num_items + 1] = combined
            s1_pairs = self._rank_from_array(score_full, seen_set, top_k=24)
            stage1_name = "Genre + Popularity"
        else:
            # Inductive graph embedding propagation using pretrained LightGCN weights (no retraining)
            with torch.no_grad():
                emb_parts = []
                weights = []
                # Recent watch history (recency-weighted)
                recent_tail = [m for m in clean_watched[-12:] if m != current_movie_id]
                for idx_pos, mid in enumerate(recent_tail):
                    emb_parts.append(self.gcn_i_emb[mid - 1])
                    weights.append(0.65 + 0.08 * idx_pos)
                # Currently playing movie drives strong real-time context shift
                if current_movie_id and 1 <= int(current_movie_id) <= self.num_items:
                    emb_parts.append(self.gcn_i_emb[int(current_movie_id) - 1])
                    weights.append(2.2)

                w_t = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)
                stacked = torch.stack(emb_parts, dim=0)
                u_vec = (stacked * w_t).sum(dim=0) / w_t.sum()

                scores_t = torch.matmul(self.gcn_i_emb, u_vec).clone()
                if genres:
                    scores_t = scores_t + 0.20 * torch.from_numpy(genre_boost_np).to(self.device)
                scores_t[~self.valid_mask_t] = -1e9
                seen_idx = [s - 1 for s in seen_set if 1 <= s <= self.num_items]
                if seen_idx:
                    scores_t[seen_idx] = -1e9
                top_vals, top_idx = torch.topk(scores_t, k=24)
                s1_pairs = [(int(i + 1), float(v)) for i, v in zip(top_idx.cpu().tolist(), top_vals.cpu().tolist())]
            stage1_name = "LightGCN"

        # Fast Stage-2 Gemini 3.5 Flash-Lite Rerank -> strict fast fallback to Stage-1 if slow/error
        ranked_pairs = s1_pairs[:top_k]
        engine_label = stage1_name
        reranked = False

        if use_llm and gemini_keys and len(s1_pairs) >= top_k:
            cand_items = []
            for mid, sc in s1_pairs:
                m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
                m["score"] = sc
                cand_items.append(m)

            hist_titles = []
            if current_movie_id and current_movie_id in self.metadata:
                hist_titles.append(f"[Đang xem] {self.metadata[current_movie_id]['display_title']}")
            for m in clean_watched[-8:]:
                if m != current_movie_id and m in self.metadata:
                    hist_titles.append(self.metadata[m]["display_title"])
            if not hist_titles and genres:
                hist_titles = [f"Yêu thích thể loại: {', '.join(genres)}"]

            try:
                res = rerank_with_gemini_calibrated(
                    api_keys=gemini_keys,
                    history_titles=hist_titles,
                    cand_items=cand_items,
                    retriever_name=stage1_name,
                    top_n=top_k,
                    alpha_fuse=0.30,
                    timeout_ms=12000,
                    max_wall_seconds=5.5,
                )
                sc_map = dict(s1_pairs)
                ranked_pairs = [(mid, sc_map.get(mid, 0.0)) for mid in res["ranked_ids"]]
                engine_label = f"{stage1_name} + Gemini 3.5 Flash-Lite"
                reranked = True
            except Exception:
                ranked_pairs = s1_pairs[:top_k]
                engine_label = stage1_name
                reranked = False

        rec_items = []
        for rank, (mid, sc) in enumerate(ranked_pairs[:top_k], 1):
            m = dict(self.metadata.get(mid, {"movie_id": mid, "display_title": f"Movie {mid}", "categories": []}))
            m["rank"] = rank
            m["score"] = round(float(sc), 4)
            rec_items.append(m)

        out = {
            "items": rec_items,
            "engine": engine_label,
            "reranked": reranked,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 1),
        }
        if len(self._rt_cache) > 64:
            self._rt_cache.clear()
        self._rt_cache[cache_key] = out
        return out
