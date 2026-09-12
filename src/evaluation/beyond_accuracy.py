from typing import Dict, List, Set
import numpy as np


def catalog_coverage(recommendations: Dict[int, List[int]], all_items: Set[int], k: int = 10) -> float:
    """Proportion of all unique items recommended at least once in top-K."""
    if not all_items:
        return 0.0
    recommended_items = set()
    for recs in recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items.intersection(all_items)) / len(all_items)


def gini_index(recommendations: Dict[int, List[int]], all_items: Set[int], k: int = 10) -> float:
    """Gini coefficient of recommendation distribution across all items.
    0 = absolute equality, 1 = maximum inequality.
    """
    item_counts: Dict[int, int] = {item: 0 for item in all_items}
    for recs in recommendations.values():
        for item in recs[:k]:
            if item in item_counts:
                item_counts[item] += 1

    counts = np.sort(list(item_counts.values()))
    n = len(counts)
    if n == 0 or np.sum(counts) == 0:
        return 0.0

    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * counts)) / (n * np.sum(counts)) - (n + 1) / n)


def novelty_at_k(
    recommendations: Dict[int, List[int]],
    item_frequencies: Dict[int, int],
    total_interactions: int,
    k: int = 10,
) -> float:
    """Self-information based novelty. Higher value means less obvious recommendations."""
    if total_interactions == 0 or not recommendations:
        return 0.0

    user_novelties = []
    for recs in recommendations.values():
        top_k = recs[:k]
        if not top_k:
            continue
        scores = []
        for item in top_k:
            freq = item_frequencies.get(item, 1)
            prob = freq / total_interactions
            scores.append(-np.log2(prob))
        user_novelties.append(np.mean(scores))

    return float(np.mean(user_novelties)) if user_novelties else 0.0
