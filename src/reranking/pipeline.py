"""Nạp metadata, thống kê train và kết nối P3 với mô hình nền."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.models.base import BaseRecommender
from src.reranking.popularity_diversity import PopularityDiversityReranker, RerankingWeights


def load_item_catalog(mapping_path: str | Path, news_paths: list[str | Path]) -> list[str]:
    """Đọc category theo item_idx; không đọc click labels từ dev/test."""
    import pyarrow.dataset as ds

    mapping = Path(mapping_path)
    if not mapping.exists():
        raise FileNotFoundError(f"Thiếu item mapping: {mapping}")
    mapping_ds = ds.dataset(str(mapping), format="parquet")
    if "item_idx" not in mapping_ds.schema.names:
        raise ValueError("Item mapping không có cột item_idx.")
    item_indices = mapping_ds.to_table(columns=["item_idx"])["item_idx"].to_numpy()
    if len(item_indices) == 0 or np.any(item_indices < 0):
        raise ValueError("Danh mục item rỗng hoặc có ID âm.")
    if len(np.unique(item_indices)) != len(item_indices):
        raise ValueError("Item mapping chứa item_idx trùng.")
    if int(item_indices.max()) + 1 != len(item_indices):
        raise ValueError("Item mapping phải liên tục từ 0 đến num_items - 1.")

    categories = [""] * len(item_indices)
    valid_news = 0
    for path_value in news_paths:
        path = Path(path_value)
        if not path.exists():
            continue  # Test metadata có thể không có trên máy.
        news_ds = ds.dataset(str(path), format="parquet")
        if not {"item_idx", "category"}.issubset(news_ds.schema.names):
            raise ValueError(f"News metadata thiếu item_idx/category: {path}")
        for batch in news_ds.to_batches(columns=["item_idx", "category"], batch_size=8192):
            rows = batch.to_pydict()
            for item_id, category in zip(rows["item_idx"], rows["category"], strict=True):
                index = int(item_id)
                if index < 0 or index >= len(categories):
                    raise ValueError(f"News metadata có item_idx không hợp lệ: {index}")
                text = str(category).strip().lower() if category is not None else ""
                if text and not categories[index]:
                    categories[index] = text
                valid_news += 1
    if valid_news == 0:
        raise FileNotFoundError("Không tìm thấy dữ liệu news metadata train/dev/test.")
    return categories


def load_train_click_counts(train_interactions: str | Path, num_items: int) -> np.ndarray:
    """Đếm click dương duy nhất từ tập train, không sử dụng dev/test."""
    import pyarrow.dataset as ds

    path = Path(train_interactions)
    if not path.exists():
        raise FileNotFoundError(f"Thiếu train interactions: {path}")
    dataset = ds.dataset(str(path), format="parquet")
    if "item_idx" not in dataset.schema.names:
        raise ValueError("Train interactions không có cột item_idx.")
    has_label = "label" in dataset.schema.names
    columns = ["item_idx", "label"] if has_label else ["item_idx"]
    counts = np.zeros(num_items, dtype=np.int64)
    for batch in dataset.to_batches(columns=columns, batch_size=65536):
        data = batch.to_pydict()
        ids = np.asarray(data["item_idx"], dtype=np.int64)
        if has_label:
            labels = np.asarray(data["label"])
            ids = ids[labels == 1]
        if np.any(ids < 0) or np.any(ids >= num_items):
            raise ValueError("Train interactions có item_idx không hợp lệ.")
        counts += np.bincount(ids, minlength=num_items)
    return counts


class RerankingPipeline:
    """Tái xếp hạng điểm dự đoán của bất kỳ BaseRecommender nào."""

    def __init__(self, reranker: PopularityDiversityReranker):
        self.reranker = reranker

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> RerankingPipeline:
        """Khởi tạo P3 từ cấu hình và metadata MIND đã chuẩn bị."""
        rerank_cfg = config.get("reranking", {})
        if rerank_cfg.get("method", "popularity_diversity") != "popularity_diversity":
            raise ValueError("Phương pháp P3 không được hỗ trợ.")
        dataset_cfg = config.get("dataset", {})
        data_root = Path(dataset_cfg.get("processed_root", "data/processed/mind_large"))
        mapping_path = dataset_cfg.get("items_mapping", str(data_root / "mappings/items"))
        news_paths = rerank_cfg.get(
            "news_paths", [str(data_root / split / "news") for split in ("train", "dev", "test")]
        )
        categories = load_item_catalog(mapping_path, news_paths)
        train_path = rerank_cfg.get(
            "popularity_train_path", str(data_root / "train/positive_interactions")
        )
        click_counts = load_train_click_counts(train_path, len(categories))
        weight_cfg = rerank_cfg.get("weights", {})
        weights = RerankingWeights(
            relevance=float(weight_cfg.get("relevance", 0.70)),
            diversity=float(weight_cfg.get("diversity", 0.20)),
            long_tail=float(weight_cfg.get("long_tail", 0.10)),
        )
        reranker = PopularityDiversityReranker(
            categories=categories,
            train_click_counts=click_counts,
            weights=weights,
            top_k=int(rerank_cfg.get("top_k", 20)),
            candidate_pool_size=rerank_cfg.get("candidate_pool_size"),
        )
        return cls(reranker)

    def rerank(
        self, candidate_item_idxs: list[int], scores: np.ndarray, top_k: int | None = None
    ) -> list[int]:
        """Chuyển điểm mô hình nền thành danh sách đã tái xếp hạng."""
        return self.reranker.rerank(candidate_item_idxs, scores, top_k=top_k)

    def score_and_rerank(
        self,
        model: BaseRecommender,
        user_idx: int,
        candidate_item_idxs: list[int],
        history_item_idxs: list[int] | None,
        top_k: int | None = None,
    ) -> tuple[list[int], list[int], np.ndarray]:
        """Chấm điểm một impression và trả về baseline, P3, điểm gốc."""
        scores = np.asarray(
            model.predict(
                np.array([user_idx], dtype=np.int64),
                np.asarray(candidate_item_idxs, dtype=np.int64),
                history_item_idxs=history_item_idxs,
            ),
            dtype=np.float64,
        ).reshape(-1)
        if scores.size != len(candidate_item_idxs):
            raise ValueError("Mô hình nền trả về số điểm không khớp số ứng viên.")
        baseline_indices = np.argsort(-scores, kind="stable")
        baseline = [int(candidate_item_idxs[index]) for index in baseline_indices]
        reranked = self.rerank(candidate_item_idxs, scores, top_k=top_k)
        return baseline, reranked, scores
