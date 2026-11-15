"""
Data Loader for MovieLens-1M Clean Split Protocol (split_manifest.json).
Provides chronological rated history (with actual user ratings), positive sequence maps,
test ground-truth sets, enriched metadata (genres, synopsis, links.csv TMDB/IMDB IDs),
and movie popularity / rating statistics.
"""

import os
import json
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def clean_display_title(title: str) -> str:
    s = str(title).strip()
    import re
    m = re.match(r"^(.*?),\s*(The|A|An)\s*(\(\d{4}\))$", s)
    if m:
        return f"{m.group(2)} {m.group(1)} {m.group(3)}"
    return s


def load_movielens_bundle(data_dir: str) -> Dict:
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_sorted = train_val_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    test_sorted = test_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    movies_df = pd.read_csv(os.path.join(data_dir, "movies_clean.csv"))
    users_df = pd.read_csv(os.path.join(data_dir, "users_clean.csv"))

    # Load links.csv for tmdbId / imdbId
    tmdb_map: Dict[int, str] = {}
    imdb_map: Dict[int, str] = {}
    links_path = os.path.join(data_dir, "links.csv")
    if os.path.exists(links_path):
        ldf = pd.read_csv(links_path)
        for row in ldf.itertuples(index=False):
            mid = int(row.movie_id)
            tid = getattr(row, "tmdbId", None)
            if pd.notna(tid):
                tmdb_map[mid] = str(int(float(tid)))
            itt = getattr(row, "imdb_tt", None)
            if pd.notna(itt):
                imdb_map[mid] = str(itt)

    # Load Wikipedia / Wikidata enriched metadata
    enriched_map: Dict[str, Dict] = {}
    enr_path = os.path.join(data_dir, "movies_enriched.json")
    if os.path.exists(enr_path):
        with open(enr_path, "r", encoding="utf-8") as f:
            enriched_map = json.load(f)

    # Compute rating statistics on train+val
    pop_counts = train_val_df["movie_id"].value_counts().to_dict()
    mean_ratings = train_val_df.groupby("movie_id")["rating"].mean().to_dict()

    metadata: Dict[int, Dict] = {}
    for row in movies_df.itertuples(index=False):
        mid = int(row.movie_id)
        genres_list = str(row.genres).split("|") if pd.notna(row.genres) else []
        enr = enriched_map.get(str(mid), {})
        tmdb_id = tmdb_map.get(mid) or str(enr.get("tmdb_id", "")).strip()
        if tmdb_id.endswith(".0"):
            tmdb_id = tmdb_id[:-2]
        metadata[mid] = {
            "movie_id": mid,
            "title": str(row.title),
            "display_title": clean_display_title(str(row.title)),
            "genres": genres_list,
            "categories": genres_list,
            "tmdb_id": tmdb_id,
            "imdb_id": imdb_map.get(mid) or enr.get("imdb_id", ""),
            "director": enr.get("director", ""),
            "description": enr.get("overview", "") or enr.get("tagline", "") or f"{row.title} ({', '.join(genres_list)})",
            "tagline": enr.get("tagline", ""),
            "popularity_count": int(pop_counts.get(mid, 0)),
            "ml_avg_rating": round(float(mean_ratings.get(mid, 0.0)), 2),
        }

    # Build user demographics
    user_info: Dict[int, Dict] = {}
    for row in users_df.itertuples(index=False):
        user_info[int(row.user_id)] = {
            "gender": "Nam" if row.gender == "M" else "Nữ",
            "age_group": str(getattr(row, "age_label", getattr(row, "age", ""))),
            "occupation": str(getattr(row, "occupation_label", getattr(row, "occupation", ""))),
        }

    # Chronological rated history (movie_id, rating, timestamp) + positive sequences + seen set
    rated_history: Dict[int, List[Tuple[int, float, int]]] = {}
    train_seqs: Dict[int, List[int]] = {}
    seen_map: Dict[int, Set[int]] = {}

    for row in train_val_sorted.itertuples(index=False):
        uid = int(row.user_id)
        mid = int(row.movie_id)
        rat = float(row.rating)
        ts = int(row.timestamp)
        rated_history.setdefault(uid, []).append((mid, rat, ts))
        seen_map.setdefault(uid, set()).add(mid)
        if int(row.relevant) == 1:
            train_seqs.setdefault(uid, []).append(mid)

    # Test ground truth positives & ratings
    test_pos: Dict[int, Set[int]] = {}
    test_pos_ratings: Dict[int, Dict[int, float]] = {}
    for row in test_sorted.itertuples(index=False):
        uid = int(row.user_id)
        mid = int(row.movie_id)
        rat = float(row.rating)
        if int(row.relevant) == 1:
            test_pos.setdefault(uid, set()).add(mid)
            test_pos_ratings.setdefault(uid, {})[mid] = rat

    valid_catalog = np.array(sorted(train_val_df["movie_id"].unique()), dtype=np.int64)
    all_movie_ids = sorted(metadata.keys(), key=lambda m: metadata[m]["popularity_count"], reverse=True)

    return {
        "train_val_df": train_val_df,
        "test_df": test_sorted,
        "movies_df": movies_df,
        "metadata": metadata,
        "user_info": user_info,
        "rated_history": rated_history,
        "train_seqs": train_seqs,
        "seen_map": seen_map,
        "test_pos": test_pos,
        "test_pos_ratings": test_pos_ratings,
        "valid_catalog": valid_catalog,
        "popular_movie_ids": all_movie_ids,
        "item_pop_counts": {int(k): int(v) for k, v in pop_counts.items()},
        "total_interactions": int(len(train_val_df)),
    }


def build_movielens_graph_adj(
    train_sequences: Dict[int, List[int]],
    num_users: int = 6041,
    num_items: int = 3953,
) -> torch.Tensor:
    """Construct normalized bipartite sparse adjacency matrix for LightGCN on MovieLens."""
    u_nodes = []
    i_nodes = []
    for u, items in train_sequences.items():
        for i in items:
            u_nodes.append(int(u) - 1)
            i_nodes.append(int(i) - 1)

    u_arr = np.array(u_nodes, dtype=np.int64)
    i_arr = np.array(i_nodes, dtype=np.int64)
    ratings = np.ones_like(u_arr, dtype=np.float32)

    R = sp.coo_matrix((ratings, (u_arr, i_arr)), shape=(num_users, num_items))
    adj = sp.bmat([[None, R], [R.T, None]], format="csr")

    rowsum = np.array(adj.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5, where=rowsum > 0)
    d_inv_sqrt[rowsum <= 0] = 0.0
    d_mat = sp.diags(d_inv_sqrt)

    norm_adj = d_mat.dot(adj).dot(d_mat).tocoo()
    indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
    values = torch.from_numpy(norm_adj.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, torch.Size(norm_adj.shape)).coalesce()
