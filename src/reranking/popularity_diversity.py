"""P3: tái xếp hạng theo độ liên quan, đa dạng và mức ít phổ biến."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RerankingWeights:
    """Lưu và kiểm tra trọng số của ba mục tiêu P3."""

    relevance: float = 0.70
    diversity: float = 0.20
    long_tail: float = 0.10

    def __post_init__(self) -> None:
        values = (self.relevance, self.diversity, self.long_tail)
        if not all(np.isfinite(value) and value >= 0 for value in values):
            raise ValueError("Trọng số P3 phải là số hữu hạn, không âm.")
        if not np.isclose(sum(values), 1.0, atol=1e-8):
            raise ValueError("Tổng ba trọng số P3 phải bằng 1.")


class PopularityDiversityReranker:
    """Chọn lần lượt Top-K từ một tập ứng viên đã được mô hình nền chấm điểm."""

    def __init__(
        self,
        categories: list[str],
        train_click_counts: np.ndarray,
        weights: RerankingWeights | None = None,
        top_k: int = 20,
        candidate_pool_size: int | None = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")
        if candidate_pool_size is not None and candidate_pool_size < top_k:
            raise ValueError("candidate_pool_size phải >= top_k, hoặc để null.")
        counts = np.asarray(train_click_counts, dtype=np.int64)
        if counts.ndim != 1 or len(categories) != counts.size:
            raise ValueError("categories và train_click_counts phải có cùng chiều dài.")
        if np.any(counts < 0):
            raise ValueError("Số lượt click không được âm.")

        self.categories = list(categories)
        self.train_click_counts = counts
        self.weights = weights if weights is not None else RerankingWeights()
        self.top_k = top_k
        self.candidate_pool_size = candidate_pool_size

        # Tính novelty bằng tần suất train; bài zero-click có giá trị 1.
        maximum = int(counts.max()) if counts.size else 0
        if maximum == 0:
            self.long_tail_scores = np.ones(counts.size, dtype=np.float32)
        else:
            self.long_tail_scores = (1.0 - np.log1p(counts) / np.log1p(maximum)).astype(np.float32)

    @staticmethod
    def _normalize_relevance(scores: np.ndarray) -> np.ndarray:
        """Đưa điểm mô hình nền về [0, 1] trong từng impression."""
        minimum = float(scores.min())
        maximum = float(scores.max())
        if maximum == minimum:
            return np.ones(scores.size, dtype=np.float64)
        return (scores - minimum) / (maximum - minimum)

    def rerank(
        self,
        candidate_item_idxs: list[int] | np.ndarray,
        relevance_scores: list[float] | np.ndarray,
        top_k: int | None = None,
    ) -> list[int]:
        """Trả về toàn bộ ứng viên, trong đó Top-K được P3 tái xếp hạng."""
        items = np.asarray(candidate_item_idxs, dtype=np.int64).reshape(-1)
        scores = np.asarray(relevance_scores, dtype=np.float64).reshape(-1)
        if items.size != scores.size:
            raise ValueError("Số ứng viên và điểm dự đoán không khớp.")
        if items.size == 0:
            return []
        if len(set(items.tolist())) != items.size:
            raise ValueError("Danh sách ứng viên chứa item_idx trùng lặp.")
        if np.any(items < 0) or np.any(items >= len(self.categories)):
            raise ValueError("Có item_idx nằm ngoài danh mục metadata.")
        if not np.all(np.isfinite(scores)):
            raise ValueError("Điểm relevance chứa NaN hoặc vô cực.")

        limit = self.top_k if top_k is None else int(top_k)
        if limit <= 0:
            raise ValueError("top_k phải lớn hơn 0.")
        if self.candidate_pool_size is not None and limit > self.candidate_pool_size:
            raise ValueError("top_k không được vượt candidate_pool_size.")

        # Stable sort giúp thứ tự gốc quyết định trường hợp đồng điểm.
        baseline_indices = np.argsort(-scores, kind="stable")
        pool_size = min(
            items.size,
            self.candidate_pool_size if self.candidate_pool_size is not None else items.size,
        )
        pool_indices = baseline_indices[:pool_size]
        pool_items = items[pool_indices]
        pool_scores = self._normalize_relevance(scores[pool_indices])
        pool_novelty = self.long_tail_scores[pool_items]

        selected_positions: list[int] = []
        selected_categories: set[str] = set()
        remaining_positions = list(range(pool_size))

        for _ in range(min(limit, pool_size)):
            best_position: int | None = None
            best_score = -float("inf")
            for position in remaining_positions:
                category = self.categories[int(pool_items[position])]
                diversity = 1.0 if category and category not in selected_categories else 0.0
                combined_score = (
                    self.weights.relevance * float(pool_scores[position])
                    + self.weights.diversity * diversity
                    + self.weights.long_tail * float(pool_novelty[position])
                )
                # Không thay thế nếu đồng điểm: giữ thứ tự baseline ổn định.
                if combined_score > best_score + 1e-12:
                    best_position = position
                    best_score = combined_score
            assert best_position is not None
            selected_positions.append(best_position)
            remaining_positions.remove(best_position)
            chosen_category = self.categories[int(pool_items[best_position])]
            if chosen_category:
                selected_categories.add(chosen_category)

        # Phần còn lại giữ thứ tự baseline; không bỏ hay thêm ứng viên.
        selected_items = [int(pool_items[position]) for position in selected_positions]
        selected_set = set(selected_items)
        trailing_items = [
            int(items[index]) for index in baseline_indices if int(items[index]) not in selected_set
        ]
        return selected_items + trailing_items
