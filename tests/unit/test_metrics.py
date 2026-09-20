import pytest

from src.evaluation.ranking_metrics import (
    compute_ranking_metrics,
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_hit_rate_at_k():
    recs = [1, 2, 3, 4, 5]
    ground_truth = {3, 10}
    assert hit_rate_at_k(recs, ground_truth, k=3) == 1.0
    assert hit_rate_at_k(recs, ground_truth, k=2) == 0.0


def test_recall_at_k():
    recs = [1, 2, 3, 4, 5]
    ground_truth = {2, 4, 9}
    # 2 hits out of 3 total relevant items in top 5
    assert recall_at_k(recs, ground_truth, k=5) == pytest.approx(2 / 3)
    # 1 hit in top 2 (item 2)
    assert recall_at_k(recs, ground_truth, k=2) == pytest.approx(1 / 3)


def test_mrr_at_k():
    recs = [10, 20, 30]
    ground_truth = {20}
    # First hit at rank 2
    assert mrr_at_k(recs, ground_truth, k=3) == pytest.approx(0.5)
    assert mrr_at_k(recs, ground_truth, k=1) == 0.0


def test_ndcg_at_k():
    recs = [1, 2, 3, 4, 5]
    # Perfect ranking of 2 relevant items in top 2
    ground_truth = {1, 2}
    assert ndcg_at_k(recs, ground_truth, k=2) == pytest.approx(1.0)

    # Inverted ranking
    # ground truth is item 2 only. At rank 2, DCG = 1 / log2(3), IDCG = 1 / log2(2) = 1.0
    ground_truth_single = {2}
    import numpy as np

    expected_ndcg = (1.0 / np.log2(3)) / (1.0 / np.log2(2))
    assert ndcg_at_k(recs, ground_truth_single, k=5) == pytest.approx(expected_ndcg)


def test_compute_ranking_metrics():
    recommendations = {
        1: [10, 20, 30],
        2: [40, 50, 60],
    }
    ground_truth = {
        1: {10, 99},
        2: {50},
    }
    metrics = compute_ranking_metrics(recommendations, ground_truth, k_list=[2])
    assert "NDCG@2" in metrics
    assert "Recall@2" in metrics
    assert "HitRate@2" in metrics
    assert "MRR@2" in metrics
