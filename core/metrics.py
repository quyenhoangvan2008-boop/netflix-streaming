"""
Comprehensive Evaluation Metrics for Recommendation Systems:
- Accuracy (Top-K): Precision@K, Recall@K, HitRate@K, NDCG@K, MRR@K
- Rating Prediction: RMSE, MAE
- Beyond-Accuracy: Genre ILD (Diversity), Novelty, ARP (Popularity Bias), Catalog Coverage
"""

import math
import numpy as np
from typing import List, Dict, Set, Union

def precision_at_k(ranked_items: List[int], ground_truth: Union[int, Set[int]], k: int) -> float:
    top_k = ranked_items[:k]
    if len(top_k) == 0:
        return 0.0
    gt_set = {ground_truth} if isinstance(ground_truth, (int, np.integer)) else set(ground_truth)
    hits = len(set(top_k) & gt_set)
    return float(hits) / float(k)

def recall_at_k(ranked_items: List[int], ground_truth: Union[int, Set[int]], k: int) -> float:
    top_k = ranked_items[:k]
    if isinstance(ground_truth, (int, np.integer)):
        return 1.0 if ground_truth in top_k else 0.0
    if len(ground_truth) == 0:
        return 0.0
    hits = len(set(top_k) & set(ground_truth))
    return float(hits) / float(len(ground_truth))

def hit_rate_at_k(ranked_items: List[int], ground_truth: Union[int, Set[int]], k: int) -> float:
    top_k = ranked_items[:k]
    if isinstance(ground_truth, (int, np.integer)):
        return 1.0 if ground_truth in top_k else 0.0
    hits = len(set(top_k) & set(ground_truth))
    return 1.0 if hits > 0 else 0.0

def ndcg_at_k(ranked_items: List[int], ground_truth: Union[int, Set[int]], k: int) -> float:
    top_k = ranked_items[:k]
    if isinstance(ground_truth, (int, np.integer)):
        if ground_truth not in top_k:
            return 0.0
        rank = top_k.index(ground_truth) + 1
        return 1.0 / math.log2(rank + 1)

    if len(ground_truth) == 0:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(top_k, 1):
        if item in ground_truth:
            dcg += 1.0 / math.log2(idx + 1)

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, len(ground_truth)) + 1))
    return dcg / idcg if idcg > 0 else 0.0

def mrr_at_k(ranked_items: List[int], ground_truth: Union[int, Set[int]], k: int) -> float:
    top_k = ranked_items[:k]
    gt_set = {ground_truth} if isinstance(ground_truth, (int, np.integer)) else set(ground_truth)
    for idx, item in enumerate(top_k, 1):
        if item in gt_set:
            return 1.0 / float(idx)
    return 0.0

def genre_ild(ranked_items: List[int], genre_matrix: np.ndarray, k: int = 10) -> float:
    """
    Intra-List Diversity: Mean pairwise cosine distance between genre vectors of top-K items.
    genre_matrix: shape (num_items + 1, num_genres), L2-normalized rows.
    """
    top_k = [i for i in ranked_items[:k] if i < genre_matrix.shape[0]]
    if len(top_k) < 2:
        return 0.0
    vecs = genre_matrix[top_k]  # (K, G)
    sim_mat = vecs @ vecs.T     # (K, K)
    # Upper triangle without diagonal
    iu = np.triu_indices(len(top_k), k=1)
    pair_sims = sim_mat[iu]
    pair_dists = 1.0 - np.clip(pair_sims, 0.0, 1.0)
    return float(np.mean(pair_dists))

def evaluate_rankings(
    predictions: Dict[int, List[int]],
    test_data: Dict[int, Union[int, Set[int]]],
    k_list: List[int] = [10],
    item_pop_counts: Dict[int, int] = None,
    total_interactions: int = None,
    genre_matrix: np.ndarray = None,
    catalog_size: int = None
) -> Dict[str, float]:
    metrics = {}
    for k in k_list:
        metrics[f"Precision@{k}"] = []
        metrics[f"Recall@{k}"] = []
        metrics[f"HitRate@{k}"] = []
        metrics[f"NDCG@{k}"] = []
        metrics[f"MRR@{k}"] = []

    ild_list = []
    arp_list = []
    novelty_list = []
    all_recommended = set()
    eval_k = k_list[0]

    for user_id, target in test_data.items():
        if user_id not in predictions:
            continue
        ranked = predictions[user_id]
        for k in k_list:
            metrics[f"Precision@{k}"].append(precision_at_k(ranked, target, k))
            metrics[f"Recall@{k}"].append(recall_at_k(ranked, target, k))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(ranked, target, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked, target, k))
            metrics[f"MRR@{k}"].append(mrr_at_k(ranked, target, k))

        top_k = ranked[:eval_k]
        all_recommended.update(top_k)

        if genre_matrix is not None:
            ild_list.append(genre_ild(top_k, genre_matrix, eval_k))

        if item_pop_counts is not None and total_interactions:
            pops = [item_pop_counts.get(it, 1) for it in top_k]
            arp_list.append(float(np.mean(pops)))
            novs = [-math.log2(max(p, 1) / total_interactions) for p in pops]
            novelty_list.append(float(np.mean(novs)))

    results = {m: float(np.mean(vals)) if len(vals) > 0 else 0.0 for m, vals in metrics.items()}
    if ild_list:
        results[f"Diversity_ILD@{eval_k}"] = float(np.mean(ild_list))
    if arp_list:
        results[f"PopBias_ARP@{eval_k}"] = float(np.mean(arp_list))
    if novelty_list:
        results[f"Novelty@{eval_k}"] = float(np.mean(novelty_list))
    if catalog_size and catalog_size > 0:
        results[f"Coverage@{eval_k}"] = float(len(all_recommended)) / float(catalog_size)

    return results

def compute_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_pred_clipped = np.clip(y_pred, 1.0, 5.0)
    err = y_true - y_pred_clipped
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    return {"RMSE": rmse, "MAE": mae}
