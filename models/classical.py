"""
Classical & Matrix Factorization Recommender Models (Audited & Fixed):
1. Popularity Baseline
2. Content-Based Filtering (Genre One-Hot / Multi-Hot Encoding)
3. User-Based Collaborative Filtering (Shrunk Explicit CF for RMSE + Implicit Cosine CF for Top-K)
4. Item-Based Collaborative Filtering (Shrunk Explicit CF for RMSE + Implicit Cosine CF for Top-K)
5. Low-Rank Approximation (Biased SVD for RMSE + PureSVD Rank-64 for Top-K, Cremonesi et al. 2010)
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds
import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# 1. Popularity Baseline
# ---------------------------------------------------------------------------
class PopularityRecommender:
    def __init__(self, num_items: int = 3953):
        self.num_items = num_items
        self.item_counts = np.zeros(num_items + 1, dtype=np.float32)
        self.item_means = np.zeros(num_items + 1, dtype=np.float32)
        self.global_mean = 3.5
        self.ranked_catalog = []

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        stats = train_df.groupby("movie_id")["rating"].agg(["count", "mean"])
        self.item_means[:] = self.global_mean
        for mid, row in stats.iterrows():
            if mid <= self.num_items:
                c = float(row["count"])
                self.item_counts[int(mid)] = c
                self.item_means[int(mid)] = (c * float(row["mean"]) + 10.0 * self.global_mean) / (c + 10.0)

        valid_ids = np.arange(1, self.num_items + 1)
        order = np.argsort(-self.item_counts[valid_ids])
        self.ranked_catalog = valid_ids[order].tolist()

    def predict_ratings(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        safe_items = np.clip(item_ids, 0, self.num_items)
        return self.item_means[safe_items]

    def recommend(self, user_id: int, seen_items: Set[int], top_k: int = 10) -> List[int]:
        recs = []
        for mid in self.ranked_catalog:
            if mid not in seen_items:
                recs.append(int(mid))
                if len(recs) >= top_k:
                    break
        return recs


# ---------------------------------------------------------------------------
# 2. Content-Based Filtering (Genre One-Hot / Multi-Hot Encoding)
# ---------------------------------------------------------------------------
class ContentBasedGenreRecommender:
    def __init__(self, num_users: int = 6041, num_items: int = 3953):
        self.num_users = num_users
        self.num_items = num_items
        self.genre_matrix = None  # (num_items + 1, num_genres), L2 normalized
        self.user_profiles = None # (num_users + 1, num_genres)
        self.user_means = None
        self.item_quality = None  # small tie-breaker among identical genre vectors
        self.global_mean = 3.5

    @staticmethod
    def build_genre_matrix(movies_df: pd.DataFrame, num_items: int = 3953) -> Tuple[np.ndarray, List[str]]:
        all_genres = sorted({
            g for gs in movies_df["genres"].dropna() for g in str(gs).split("|")
        })
        g2idx = {g: idx for idx, g in enumerate(all_genres)}
        mat = np.zeros((num_items + 1, len(all_genres)), dtype=np.float32)

        for _, row in movies_df.iterrows():
            mid = int(row["movie_id"])
            if mid <= num_items and pd.notna(row["genres"]):
                for g in str(row["genres"]).split("|"):
                    if g in g2idx:
                        mat[mid, g2idx[g]] = 1.0

        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        return mat, all_genres

    def fit(self, train_df: pd.DataFrame, genre_matrix: np.ndarray):
        self.genre_matrix = genre_matrix
        num_genres = genre_matrix.shape[1]
        self.global_mean = float(train_df["rating"].mean())
        self.user_means = np.full(self.num_users + 1, self.global_mean, dtype=np.float32)
        self.user_profiles = np.zeros((self.num_users + 1, num_genres), dtype=np.float32)

        u_means = train_df.groupby("user_id")["rating"].mean()
        for uid, m in u_means.items():
            if uid <= self.num_users:
                self.user_means[int(uid)] = float(m)

        # Build user genre profile from positive interactions (rating >= 4) weighted by (rating - 3)
        pos_df = train_df[train_df["rating"] >= 4]
        uids = pos_df["user_id"].values
        mids = pos_df["movie_id"].values
        weights = (pos_df["rating"].values.astype(np.float32) - 3.0)
        item_vecs = self.genre_matrix[mids] * weights[:, None]
        np.add.at(self.user_profiles, uids, item_vecs)

        norms = np.linalg.norm(self.user_profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_profiles = self.user_profiles / norms

        # Damped Bayesian quality solely to break ties among identical 18-dim genre vectors
        self.item_quality = np.full(self.num_items + 1, self.global_mean, dtype=np.float32)
        self.item_counts = np.zeros(self.num_items + 1, dtype=np.float32)
        stats = train_df.groupby("movie_id")["rating"].agg(["count", "mean"])
        for mid, row in stats.iterrows():
            if mid <= self.num_items:
                c = float(row["count"])
                self.item_counts[int(mid)] = c
                self.item_quality[int(mid)] = (c * float(row["mean"]) + 20.0 * self.global_mean) / (c + 20.0)

    def predict_ratings(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        u_vecs = self.user_profiles[user_ids]
        i_vecs = self.genre_matrix[item_ids]
        sims = np.sum(u_vecs * i_vecs, axis=1)
        return self.user_means[user_ids] + 0.8 * (sims - 0.3)

    def recommend(self, user_id: int, seen_items: Set[int], valid_catalog: np.ndarray, top_k: int = 10) -> List[int]:
        u_vec = self.user_profiles[user_id]
        if np.all(u_vec == 0):
            scores = self.item_counts.copy()
        else:
            genre_sim = self.genre_matrix @ u_vec
            # Pure genre similarity as primary score + tiny quality tie-breaker (1e-3 scale)
            scores = genre_sim + 0.001 * self.item_quality

        valid_mask = np.zeros(self.num_items + 1, dtype=bool)
        valid_mask[valid_catalog] = True
        scores[~valid_mask] = -1e9
        if seen_items:
            seen_arr = [i for i in seen_items if i <= self.num_items]
            scores[seen_arr] = -1e9

        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.tolist()


# ---------------------------------------------------------------------------
# 3. User-Based Collaborative Filtering (User-CF)
# ---------------------------------------------------------------------------
class UserBasedCFRecommender:
    def __init__(self, num_users: int = 6041, num_items: int = 3953, k_neighbors: int = 50, shrinkage: float = 25.0):
        self.num_users = num_users
        self.num_items = num_items
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.global_mean = 3.5
        self.user_means = None
        self.item_means = None
        # Explicit centered matrix for RMSE
        self.R_centered = None
        self.R_norm_exp = None
        # Implicit positive binary matrix (relevant==1) for Top-K Ranking
        self.R_imp = None
        self.R_norm_imp = None

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        self.user_means = np.full(self.num_users + 1, self.global_mean, dtype=np.float32)
        self.item_means = np.full(self.num_items + 1, self.global_mean, dtype=np.float32)
        self.item_counts = np.zeros(self.num_items + 1, dtype=np.float32)

        u_means = train_df.groupby("user_id")["rating"].mean()
        for uid, m in u_means.items():
            if uid <= self.num_users:
                self.user_means[int(uid)] = float(m)

        i_stats = train_df.groupby("movie_id")["rating"].agg(["count", "mean"])
        for mid, row in i_stats.iterrows():
            if mid <= self.num_items:
                c = float(row["count"])
                self.item_counts[int(mid)] = c
                self.item_means[int(mid)] = (c * float(row["mean"]) + 10.0 * self.global_mean) / (c + 10.0)

        # Explicit matrix for RMSE
        uids = train_df["user_id"].values
        mids = train_df["movie_id"].values
        ratings = train_df["rating"].values.astype(np.float32)
        centered = ratings - self.user_means[uids]

        self.R_centered = sp.csr_matrix(
            (centered, (uids, mids)), shape=(self.num_users + 1, self.num_items + 1), dtype=np.float32
        )
        row_norms = np.sqrt(self.R_centered.power(2).sum(axis=1)).A1
        row_norms[row_norms == 0] = 1.0
        self.R_norm_exp = (sp.diags(1.0 / row_norms) @ self.R_centered).tocsr()

        # Implicit positive matrix (rating >= 4) for Top-K ranking
        pos_df = train_df[train_df["rating"] >= 4]
        self.R_imp = sp.csr_matrix(
            (np.ones(len(pos_df), dtype=np.float32), (pos_df["user_id"].values, pos_df["movie_id"].values)),
            shape=(self.num_users + 1, self.num_items + 1),
            dtype=np.float32,
        )
        imp_norms = np.sqrt(self.R_imp.power(2).sum(axis=1)).A1
        imp_norms[imp_norms == 0] = 1.0
        self.R_norm_imp = (sp.diags(1.0 / imp_norms) @ self.R_imp).tocsr()

    def _predict_explicit_user(self, user_id: int) -> np.ndarray:
        u_vec = self.R_norm_exp[user_id]
        if u_vec.nnz == 0:
            return self.item_means.copy()

        sims = (u_vec @ self.R_norm_exp.T).toarray().ravel()
        sims[user_id] = -1.0
        pos_idx = np.where(sims > 0)[0]
        if len(pos_idx) == 0:
            return self.item_means.copy()

        if len(pos_idx) > self.k_neighbors:
            top_rel = np.argpartition(-sims[pos_idx], self.k_neighbors)[:self.k_neighbors]
            nbrs = pos_idx[top_rel]
        else:
            nbrs = pos_idx

        nbr_sims = sims[nbrs]
        nbr_ratings = self.R_centered[nbrs].toarray()
        rated_mask = (nbr_ratings != 0).astype(np.float32)

        num = nbr_sims @ nbr_ratings
        # Shrinkage denominator prevents 1-neighbor noise from exploding
        den = np.abs(nbr_sims) @ rated_mask + self.shrinkage * 0.1
        base = 0.5 * self.user_means[user_id] + 0.5 * self.item_means
        return base + (num / den)

    def predict_ratings_for_test_df(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(test_df), self.global_mean, dtype=np.float32)
        for uid, group in test_df.groupby("user_id"):
            u_preds = self._predict_explicit_user(int(uid))
            mids = group["movie_id"].values
            preds[group.index.values] = u_preds[mids]
        return preds

    def recommend(self, user_id: int, seen_items: Set[int], valid_catalog: np.ndarray, top_k: int = 10) -> List[int]:
        u_vec = self.R_norm_imp[user_id]
        if u_vec.nnz == 0:
            scores = self.item_counts.copy()
        else:
            sims = (u_vec @ self.R_norm_imp.T).toarray().ravel()
            sims[user_id] = 0.0
            top_u = np.argpartition(-sims, self.k_neighbors)[:self.k_neighbors]
            # Unnormalized similarity-weighted sum naturally accounts for neighbor support
            scores = (sims[top_u] @ self.R_imp[top_u].toarray()).ravel()

        valid_mask = np.zeros(self.num_items + 1, dtype=bool)
        valid_mask[valid_catalog] = True
        scores[~valid_mask] = -1e9
        if seen_items:
            seen_arr = [i for i in seen_items if i <= self.num_items]
            scores[seen_arr] = -1e9
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.tolist()


# ---------------------------------------------------------------------------
# 4. Item-Based Collaborative Filtering (Item-CF)
# ---------------------------------------------------------------------------
class ItemBasedCFRecommender:
    def __init__(self, num_users: int = 6041, num_items: int = 3953, k_neighbors: int = 50, shrinkage: float = 10.0):
        self.num_users = num_users
        self.num_items = num_items
        self.k_neighbors = k_neighbors
        self.shrinkage = shrinkage
        self.global_mean = 3.5
        self.item_means = None
        self.R_centered = None
        self.S_exp_topk = None
        self.R_imp = None
        self.S_imp_topk = None

    def _build_topk_sim(self, S_dense: np.ndarray) -> sp.csr_matrix:
        np.fill_diagonal(S_dense, 0.0)
        S_dense[S_dense < 0] = 0.0
        rows, cols, vals = [], [], []
        for i in range(1, self.num_items + 1):
            row = S_dense[i]
            nz = np.where(row > 0)[0]
            if len(nz) > self.k_neighbors:
                top_rel = np.argpartition(-row[nz], self.k_neighbors)[:self.k_neighbors]
                nz = nz[top_rel]
            if len(nz) > 0:
                rows.extend([i] * len(nz))
                cols.extend(nz.tolist())
                vals.extend(row[nz].tolist())
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.num_items + 1, self.num_items + 1), dtype=np.float32)

    def fit(self, train_df: pd.DataFrame):
        self.global_mean = float(train_df["rating"].mean())
        self.item_means = np.full(self.num_items + 1, self.global_mean, dtype=np.float32)
        self.item_counts = np.zeros(self.num_items + 1, dtype=np.float32)

        i_stats = train_df.groupby("movie_id")["rating"].agg(["count", "mean"])
        for mid, row in i_stats.iterrows():
            if mid <= self.num_items:
                c = float(row["count"])
                self.item_counts[int(mid)] = c
                self.item_means[int(mid)] = (c * float(row["mean"]) + 10.0 * self.global_mean) / (c + 10.0)

        # Explicit Item-CF for RMSE (with support shrinkage)
        uids = train_df["user_id"].values
        mids = train_df["movie_id"].values
        ratings = train_df["rating"].values.astype(np.float32)
        centered = ratings - self.item_means[mids]

        self.R_centered = sp.csr_matrix(
            (centered, (uids, mids)), shape=(self.num_users + 1, self.num_items + 1), dtype=np.float32
        )
        col_norms = np.sqrt(self.R_centered.power(2).sum(axis=0)).A1
        col_norms[col_norms == 0] = 1.0
        R_col_norm = self.R_centered @ sp.diags(1.0 / col_norms)
        S_exp = (R_col_norm.T @ R_col_norm).toarray()

        # Co-occurrence support shrinkage for explicit similarity
        R_bin = (self.R_centered != 0).astype(np.float32)
        co_counts = (R_bin.T @ R_bin).toarray()
        S_exp = S_exp * (co_counts / (co_counts + self.shrinkage))
        self.S_exp_topk = self._build_topk_sim(S_exp)

        # Implicit Item-CF (positive interactions relevant==1) for Top-K ranking
        pos_df = train_df[train_df["rating"] >= 4]
        self.R_imp = sp.csr_matrix(
            (np.ones(len(pos_df), dtype=np.float32), (pos_df["user_id"].values, pos_df["movie_id"].values)),
            shape=(self.num_users + 1, self.num_items + 1),
            dtype=np.float32,
        )
        imp_norms = np.sqrt(self.R_imp.power(2).sum(axis=0)).A1
        imp_norms[imp_norms == 0] = 1.0
        R_imp_norm = self.R_imp @ sp.diags(1.0 / imp_norms)
        S_imp = (R_imp_norm.T @ R_imp_norm).toarray()
        self.S_imp_topk = self._build_topk_sim(S_imp)

    def _predict_explicit_user(self, user_id: int) -> np.ndarray:
        u_row = self.R_centered[user_id]
        if u_row.nnz == 0:
            return self.item_means.copy()
        num = (u_row @ self.S_exp_topk.T).toarray().ravel()
        u_mask = (u_row != 0).astype(np.float32)
        den = (u_mask @ self.S_exp_topk.T).toarray().ravel() + 0.5
        return self.item_means + (num / den)

    def predict_ratings_for_test_df(self, test_df: pd.DataFrame) -> np.ndarray:
        preds = np.full(len(test_df), self.global_mean, dtype=np.float32)
        for uid, group in test_df.groupby("user_id"):
            u_preds = self._predict_explicit_user(int(uid))
            mids = group["movie_id"].values
            preds[group.index.values] = u_preds[mids]
        return preds

    def recommend(self, user_id: int, seen_items: Set[int], valid_catalog: np.ndarray, top_k: int = 10) -> List[int]:
        u_row = self.R_imp[user_id]
        if u_row.nnz == 0:
            scores = self.item_counts.copy()
        else:
            scores = (u_row @ self.S_imp_topk.T).toarray().ravel()

        valid_mask = np.zeros(self.num_items + 1, dtype=bool)
        valid_mask[valid_catalog] = True
        scores[~valid_mask] = -1e9
        if seen_items:
            seen_arr = [i for i in seen_items if i <= self.num_items]
            scores[seen_arr] = -1e9
        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.tolist()


# ---------------------------------------------------------------------------
# 5. Low-Rank Approximation: Biased SVD (RMSE) + PureSVD Rank-64 (Top-K)
# ---------------------------------------------------------------------------
class NetflixSVDRecommender(nn.Module):
    """
    Combines:
    - Regularized Biased Matrix Factorization (Koren et al. 2009) for explicit RMSE prediction.
    - PureSVD Rank-64 (Cremonesi et al., RecSys 2010) on the implicit positive matrix for Top-K ranking.
    """
    def __init__(self, num_users: int = 6041, num_items: int = 3953, rank: int = 64, reg: float = 0.02):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.rank = rank
        self.reg = reg

        self.mu = 3.5
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.item_bias = nn.Embedding(num_items + 1, 1)
        self.user_factors = nn.Embedding(num_users + 1, rank)
        self.item_factors = nn.Embedding(num_items + 1, rank)
        self.register_buffer("Vt_pure", torch.zeros(rank, num_items + 1))
        self.register_buffer("item_counts", torch.zeros(num_items + 1))
        self.R_imp = None

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_factors.weight, std=0.05)
        nn.init.normal_(self.item_factors.weight, std=0.05)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)
        pu = self.user_factors(user_ids)
        qi = self.item_factors(item_ids)
        dot = (pu * qi).sum(dim=-1)
        return self.mu + bu + bi + dot

    def fit(self, train_df: pd.DataFrame, epochs: int = 12, batch_size: int = 8192, lr: float = 0.005, device: str = "cpu"):
        # 1. Fit PureSVD (truncated SVD on implicit positive matrix) for Top-K ranking
        cnts = train_df.groupby("movie_id")["rating"].count()
        self.item_counts.zero_()
        for mid, c in cnts.items():
            if mid <= self.num_items:
                self.item_counts[int(mid)] = float(c)
        pos_df = train_df[train_df["rating"] >= 4]
        self.R_imp = sp.csr_matrix(
            (np.ones(len(pos_df), dtype=np.float32), (pos_df["user_id"].values, pos_df["movie_id"].values)),
            shape=(self.num_users + 1, self.num_items + 1),
            dtype=np.float32,
        )
        _, _, Vt = svds(self.R_imp, k=self.rank)
        self.Vt_pure.copy_(torch.from_numpy(Vt.copy()))

        # 2. Fit Biased SVD (FunkSVD) on explicit ratings for RMSE
        self.to(device)
        self.mu = float(train_df["rating"].mean())
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=self.reg * 1e-3)

        u_arr = torch.tensor(train_df["user_id"].values, dtype=torch.long)
        i_arr = torch.tensor(train_df["movie_id"].values, dtype=torch.long)
        r_arr = torch.tensor(train_df["rating"].values, dtype=torch.float32)
        n = len(u_arr)

        for epoch in range(1, epochs + 1):
            self.train()
            perm = torch.randperm(n)
            total_loss = 0.0
            steps = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                ub = u_arr[idx].to(device)
                ib = i_arr[idx].to(device)
                rb = r_arr[idx].to(device)

                optimizer.zero_grad()
                preds = self.forward(ub, ib)
                mse = torch.mean((preds - rb) ** 2)
                reg_loss = self.reg * (
                    self.user_bias(ub).pow(2).mean()
                    + self.item_bias(ib).pow(2).mean()
                    + self.user_factors(ub).pow(2).sum(dim=-1).mean()
                    + self.item_factors(ib).pow(2).sum(dim=-1).mean()
                )
                loss = mse + reg_loss
                loss.backward()
                optimizer.step()
                total_loss += mse.item()
                steps += 1

            if epoch % 4 == 0 or epoch == epochs:
                rmse = float(np.sqrt(total_loss / max(steps, 1)))
                print(f"  [Netflix SVD] Epoch {epoch:2d}/{epochs} | Train RMSE: {rmse:.4f}")

    def attach_implicit_matrix(self, train_df: pd.DataFrame):
        if self.R_imp is None:
            pos_df = train_df[train_df["rating"] >= 4]
            self.R_imp = sp.csr_matrix(
                (np.ones(len(pos_df), dtype=np.float32), (pos_df["user_id"].values, pos_df["movie_id"].values)),
                shape=(self.num_users + 1, self.num_items + 1),
                dtype=np.float32,
            )

    @torch.no_grad()
    def predict_ratings_for_test_df(self, test_df: pd.DataFrame, device: str = "cpu") -> np.ndarray:
        self.eval()
        ub = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
        ib = torch.tensor(test_df["movie_id"].values, dtype=torch.long, device=device)
        return self.forward(ub, ib).cpu().numpy()

    @torch.no_grad()
    def recommend(self, user_id: int, seen_items: Set[int], valid_catalog: np.ndarray, top_k: int = 10, device: str = "cpu") -> List[int]:
        Vt_np = self.Vt_pure.cpu().numpy()
        if self.R_imp is not None and self.R_imp[user_id].nnz > 0:
            scores = (self.R_imp[user_id] @ Vt_np.T @ Vt_np).ravel()
        elif not seen_items:
            scores = self.item_counts.cpu().numpy().copy()
        else:
            u_t = torch.tensor([user_id], dtype=torch.long, device=device)
            bu = self.user_bias(u_t).item()
            pu = self.user_factors(u_t).squeeze(0)
            all_bi = self.item_bias.weight.squeeze(-1)
            all_qi = self.item_factors.weight
            scores = (self.mu + bu + all_bi + torch.matmul(all_qi, pu)).cpu().numpy()

        valid_mask = np.zeros(self.num_items + 1, dtype=bool)
        valid_mask[valid_catalog] = True
        scores[~valid_mask] = -1e9
        if seen_items:
            seen_arr = [i for i in seen_items if i <= self.num_items]
            scores[seen_arr] = -1e9

        top_idx = np.argpartition(-scores, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return top_idx.tolist()
