from typing import Dict, List, Set
import numpy as np


def hit_rate_at_k(recommendations: List[int], ground_truth: Set[int], k: int) -> float:
    """Hit Rate @ K."""
    top_k_recs = recommendations[:k]
    return 1.0 if any(item in ground_truth for item in top_k_recs) else 0.0


def recall_at_k(recommendations: List[int], ground_truth: Set[int], k: int) -> float:
    """Recall @ K."""
    if not ground_truth:
        return 0.0
    top_k_recs = recommendations[:k]
    hits = sum(1 for item in top_k_recs if item in ground_truth)
    return hits / len(ground_truth)


def mrr_at_k(recommendations: List[int], ground_truth: Set[int], k: int) -> float:
    """Mean Reciprocal Rank @ K."""
    top_k_recs = recommendations[:k]
    for rank, item in enumerate(top_k_recs, start=1):
        if item in ground_truth:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(recommendations: List[int], ground_truth: Set[int], k: int) -> float:
    """Normalized Discounted Cumulative Gain @ K with binary relevance."""
    if not ground_truth:
        return 0.0

    top_k_recs = recommendations[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k_recs, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(rank + 1)

    # Ideal DCG
    ideal_hits = min(len(ground_truth), k)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def compute_ranking_metrics(
    recommendations: Dict[int, List[int]],
    ground_truth: Dict[int, Set[int]],
    k_list: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """Compute mean ranking metrics across all users."""
    results: Dict[str, List[float]] = {}
    for k in k_list:
        results[f"NDCG@{k}"] = []
        results[f"Recall@{k}"] = []
        results[f"HitRate@{k}"] = []
        results[f"MRR@{k}"] = []

    for user_id, true_items in ground_truth.items():
        if not true_items:
            continue
        user_recs = recommendations.get(user_id, [])
        for k in k_list:
            results[f"NDCG@{k}"].append(ndcg_at_k(user_recs, true_items, k))
            results[f"Recall@{k}"].append(recall_at_k(user_recs, true_items, k))
            results[f"HitRate@{k}"].append(hit_rate_at_k(user_recs, true_items, k))
            results[f"MRR@{k}"].append(mrr_at_k(user_recs, true_items, k))

    return {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in results.items()}
