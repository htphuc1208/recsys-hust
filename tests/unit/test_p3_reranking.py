"""Kiểm tra P3 độc lập với dữ liệu MINDlarge."""

import numpy as np
import pytest

from src.reranking.pipeline import RerankingPipeline
from src.reranking.popularity_diversity import PopularityDiversityReranker, RerankingWeights


def make_reranker(relevance=0.7, diversity=0.2, long_tail=0.1, top_k=3):
    """Dựng catalog nhỏ để kiểm tra trực tiếp hiệu ứng ba mục tiêu."""
    categories = ["sports", "sports", "technology", "finance", "", "sports"]
    counts = [100, 90, 1, 0, 0, 5]
    return PopularityDiversityReranker(
        categories=categories,
        train_click_counts=np.array(counts),
        weights=RerankingWeights(relevance, diversity, long_tail),
        top_k=top_k,
    )


def test_relevance_only_preserves_baseline_order():
    """P3 chỉ có relevance phải giống thứ hạng baseline, kể cả khi đồng điểm."""
    reranker = make_reranker(1.0, 0.0, 0.0)
    result = reranker.rerank([3, 1, 0, 2], [0.9, 0.9, 0.2, 0.8])
    assert result == [3, 1, 2, 0]


def test_diversity_promotes_new_category():
    """Danh mục mới được ưu tiên sau bài đầu tiên khi trọng số diversity lớn."""
    reranker = make_reranker(0.20, 0.80, 0.0)
    result = reranker.rerank([0, 1, 2, 3], [0.99, 0.95, 0.8, 0.7])
    assert result[0] == 0
    assert result[1] == 2
    assert result[2] == 3
    assert sorted(result) == [0, 1, 2, 3]


def test_zero_click_long_tail_gets_priority():
    """Bài không click trong train vẫn được chấm novelty cao."""
    reranker = make_reranker(0.0, 0.0, 1.0)
    result = reranker.rerank([0, 1, 3], [0.95, 0.9, 0.2], top_k=2)
    assert result[0] == 3
    assert reranker.train_click_counts[3] == 0


def test_unknown_category_does_not_get_fake_diversity():
    """Category trống không được xem là một danh mục mới."""
    reranker = make_reranker(0.0, 1.0, 0.0)
    result = reranker.rerank([4, 0, 2], [0.95, 0.8, 0.7])
    assert result[:2] == [0, 2]


def test_topk_only_changes_prefix_and_preserves_candidates():
    """Chỉ Top-K thay đổi, tất cả ứng viên đều được giữ lại."""
    reranker = make_reranker(0.2, 0.8, 0.0, top_k=2)
    baseline_items = [0, 1, 2, 3]
    result = reranker.rerank(baseline_items, [0.99, 0.95, 0.8, 0.7])
    assert result[:2] == [0, 2]
    assert result[2:] == [1, 3]
    assert set(result) == set(baseline_items)


def test_empty_and_constant_scores_are_handled():
    """Ứng viên rỗng và baseline đồng điểm được xử lý ổn định."""
    reranker = make_reranker(1.0, 0.0, 0.0)
    assert reranker.rerank([], []) == []
    assert reranker.rerank([2, 1, 0], [0.5, 0.5, 0.5]) == [2, 1, 0]


@pytest.mark.parametrize("weights", [(-0.1, 0.5, 0.6), (0.3, 0.3, 0.3), (float("nan"), 0.5, 0.5)])
def test_invalid_weights_rejected(weights):
    """Không chấp nhận trọng số âm, không chuẩn hóa, hoặc NaN."""
    with pytest.raises(ValueError):
        RerankingWeights(*weights)


@pytest.mark.parametrize(
    "items,scores", [([0, 1], [0.8]), ([0, 0], [0.8, 0.6]), ([100], [0.5]), ([0], [float("nan")])]
)
def test_invalid_candidates_rejected(items, scores):
    """P3 phát hiện ID, điểm số và số lượng phần tử không hợp lệ."""
    with pytest.raises(ValueError):
        make_reranker().rerank(items, scores)


def test_candidate_pool_constraint():
    """Không cho top_k vượt candidate pool."""
    with pytest.raises(ValueError):
        PopularityDiversityReranker(["news"] * 3, np.zeros(3), top_k=3, candidate_pool_size=2)


class StubModel:
    """Mô hình nền giả dùng để kiểm tra việc truyền history."""

    def __init__(self):
        self.history = None

    def predict(self, user_ids, item_ids, history_item_idxs=None):
        self.history = history_item_idxs
        return np.array([0.9, 0.85, 0.2], dtype=np.float32)


def test_pipeline_scores_and_passes_history():
    """P3 tái dùng điểm mô hình, không tính lại hay sử dụng click labels."""
    model = StubModel()
    pipeline = RerankingPipeline(make_reranker(0.2, 0.8, 0.0))
    baseline, reranked, scores = pipeline.score_and_rerank(model, 42, [0, 1, 2], [5, 6])
    assert model.history == [5, 6]
    assert baseline == [0, 1, 2]
    assert reranked == [0, 2, 1]
    assert scores.shape == (3,)
