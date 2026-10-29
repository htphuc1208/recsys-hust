"""B4: gợi ý bài viết theo nội dung, dùng các feature đã tạo ở bước 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from src.models.base import BaseRecommender

FEATURE_FILES = {
    "title": "title_tfidf.npz",
    "abstract": "abstract_tfidf.npz",
    "category": "category_onehot.npz",
    "subcategory": "subcategory_onehot.npz",
    "title_entity": "title_entity_embeddings.npy",
}
# Tách optional để weight_sweep 5 chiều cũ không bị thay đổi.
OPTIONAL_FEATURE_FILES = {"abstract_entity": "abstract_entity_embeddings.npy"}
AVAILABLE_FEATURE_FILES = {**FEATURE_FILES, **OPTIONAL_FEATURE_FILES}
ENTITY_FEATURE_NAMES = {"title_entity", "abstract_entity"}

DEFAULT_WEIGHTS = {
    "title": 0.35,
    "abstract": 0.20,
    "category": 0.15,
    "subcategory": 0.10,
    "title_entity": 0.20,
    "abstract_entity": 0.00,
}


class ContentBasedRecommender(BaseRecommender):
    """Tạo hồ sơ nội dung từ history và chấm điểm cosine cho từng ứng viên."""

    def __init__(
        self,
        feature_dir: str = "data/processed/mind_large/features/b4",
        feature_weights: dict[str, float] | None = None,
        cold_user_strategy: str = "zero",
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        label_col: str = "label",
        recommendation_batch_size: int = 8192,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.feature_dir = Path(feature_dir)
        self.feature_weights = self._validate_weights(feature_weights)
        if cold_user_strategy not in ("popularity", "zero"):
            raise ValueError("cold_user_strategy phải là 'popularity' hoặc 'zero'.")
        if recommendation_batch_size <= 0:
            raise ValueError("recommendation_batch_size phải lớn hơn 0.")
        self.cold_user_strategy = cold_user_strategy
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.recommendation_batch_size = recommendation_batch_size

        self.num_items = 0
        self._features: dict[str, sparse.csr_matrix | np.ndarray] = {}
        self.user_item_matrix: sparse.csr_matrix | None = None
        self.popularity_scores = np.empty(0, dtype=np.float32)

    @staticmethod
    def _validate_weights(weights: dict[str, float] | None) -> dict[str, float]:
        """Kiểm tra trọng số và chuẩn hóa tổng về 1."""
        supplied = DEFAULT_WEIGHTS if weights is None else weights
        unknown = set(supplied) - set(AVAILABLE_FEATURE_FILES)
        if unknown:
            raise ValueError(f"Tên feature không hợp lệ: {sorted(unknown)}")
        values = {name: float(supplied.get(name, 0.0)) for name in AVAILABLE_FEATURE_FILES}
        if not all(np.isfinite(weight) and weight >= 0.0 for weight in values.values()):
            raise ValueError("Các trọng số phải hữu hạn và không âm.")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("Phải có ít nhất một trọng số dương.")
        return {name: weight / total for name, weight in values.items()}

    def _ensure_features_loaded(
        self, load_all: bool = False, include_optional: bool = False
    ) -> None:
        """Nạp feature cần thiết; sweep 6 chiều có thể yêu cầu cả optional có weight 0."""
        required_names = (
            set(FEATURE_FILES) | {
                name for name in OPTIONAL_FEATURE_FILES
                if include_optional or self.feature_weights.get(name, 0.0) > 0.0
            }
            if load_all
            else {name for name, weight in self.feature_weights.items() if weight > 0.0}
        )
        if required_names.issubset(self._features):
            return

        manifest_path = self.feature_dir / "manifest.json"
        item_index_path = self.feature_dir / "item_index.json"
        if not manifest_path.is_file() or not item_index_path.is_file():
            raise FileNotFoundError(
                f"Thiếu manifest.json hoặc item_index.json trong {self.feature_dir}. "
                "Hãy tạo đặc trưng bằng python -m src.features.news_content."
            )
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        with item_index_path.open("r", encoding="utf-8") as file:
            item_index = json.load(file)
        item_count = int(manifest["total_item_mapping"])
        if item_count <= 0 or len(item_index) != item_count:
            raise ValueError("Số item trong manifest và item_index.json không khớp.")
        if self.num_items not in (0, item_count):
            raise ValueError("Checkpoint không khớp số item với thư mục đặc trưng.")

        loaded: dict[str, sparse.csr_matrix | np.ndarray] = dict(self._features)
        for name, filename in AVAILABLE_FEATURE_FILES.items():
            if name not in required_names or name in loaded:
                continue
            path = self.feature_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Thiếu thành phần đặc trưng: {path}")
            if name in ENTITY_FEATURE_NAMES:
                matrix = np.load(path, mmap_mode="r", allow_pickle=False)
            else:
                matrix = sparse.load_npz(path).tocsr()
            expected_shape = tuple(manifest["feature_shapes"][filename.rsplit(".", 1)[0]])
            if matrix.shape != expected_shape or matrix.shape[0] != item_count:
                raise ValueError(f"Kích thước không hợp lệ của {filename}: {matrix.shape}")
            loaded[name] = matrix

        self._features = loaded
        self.num_items = item_count

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> ContentBasedRecommender:
        """Nạp đặc trưng, lưu lịch sử và thống kê popularity chỉ từ train."""
        self._ensure_features_loaded()
        required_columns = {self.user_col, self.item_col}
        missing = required_columns - set(train_df.columns)
        if missing:
            raise ValueError(f"Thiếu cột tương tác train: {sorted(missing)}")

        positive_df = train_df
        if self.label_col in train_df.columns:
            positive_df = train_df.loc[train_df[self.label_col] == 1]
        user_values = positive_df[self.user_col].to_numpy(dtype=np.int64)
        item_values = positive_df[self.item_col].to_numpy(dtype=np.int64)
        valid_items = (item_values >= 0) & (item_values < self.num_items)
        valid_users = user_values >= 0
        valid = valid_items & valid_users
        user_values = user_values[valid]
        item_values = item_values[valid]

        counts = np.bincount(item_values, minlength=self.num_items).astype(np.float32)
        log_counts = np.log1p(counts)
        maximum = float(log_counts.max(initial=0.0))
        self.popularity_scores = log_counts / maximum if maximum > 0 else log_counts

        # CSR lưu lịch sử train gọn hơn dictionary chứa hàng triệu Python int.
        num_users = int(user_values.max()) + 1 if user_values.size else 0
        values = np.ones(user_values.size, dtype=np.float32)
        self.user_item_matrix = sparse.csr_matrix(
            (values, (user_values, item_values)),
            shape=(num_users, self.num_items),
            dtype=np.float32,
        )
        self.user_item_matrix.sum_duplicates()
        self.is_fitted = True
        return self

    def _get_train_history(self, user_id: int) -> list[int]:
        """Lấy history train khi caller không truyền history tại impression."""
        if self.user_item_matrix is None or user_id < 0:
            return []
        if user_id >= self.user_item_matrix.shape[0]:
            return []
        row = self.user_item_matrix.getrow(user_id)
        return row.indices.tolist()

    def _clean_history(self, history: list[int] | np.ndarray | None) -> np.ndarray:
        """Bỏ ID không hợp lệ và loại trùng trong một hồ sơ người dùng."""
        if history is None:
            return np.empty(0, dtype=np.int64)
        item_ids = np.asarray(history, dtype=np.int64).reshape(-1)
        valid = (item_ids >= 0) & (item_ids < self.num_items)
        return np.unique(item_ids[valid])

    def _build_profiles(
        self, history: list[int] | np.ndarray | None
    ) -> dict[str, sparse.csr_matrix | np.ndarray]:
        """Lấy trung bình vector history và chuẩn hóa L2 theo từng feature."""
        history_ids = self._clean_history(history)
        if history_ids.size == 0:
            return {}
        self._ensure_features_loaded()
        profiles: dict[str, sparse.csr_matrix | np.ndarray] = {}

        for name, matrix in self._features.items():
            if name in ENTITY_FEATURE_NAMES:
                profile = np.asarray(matrix[history_ids], dtype=np.float32).mean(axis=0)
                norm = float(np.linalg.norm(profile))
                if norm > 0:
                    profiles[name] = profile / norm
                continue

            profile = sparse.csr_matrix(matrix[history_ids].mean(axis=0), dtype=np.float32)
            norm = float(np.sqrt(profile.multiply(profile).sum()))
            if norm > 0:
                profiles[name] = profile / norm

        return profiles

    def _score_candidates(
        self,
        candidates: np.ndarray,
        profiles: dict[str, sparse.csr_matrix | np.ndarray],
    ) -> np.ndarray:
        """Cộng cosine của các nhóm đặc trưng theo trọng số cấu hình."""
        feature_scores = self._score_candidates_by_feature(candidates, profiles)
        scores = np.zeros(candidates.size, dtype=np.float32)
        for name, similarity in feature_scores.items():
            scores += self.feature_weights[name] * similarity
        return scores

    def _score_candidates_by_feature(
        self,
        candidates: np.ndarray,
        profiles: dict[str, sparse.csr_matrix | np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Trả similarity riêng của từng feature để sweep trọng số không chấm lại."""
        feature_scores: dict[str, np.ndarray] = {}
        for name, profile in profiles.items():
            matrix = self._features[name]
            if name in ENTITY_FEATURE_NAMES:
                similarity = np.asarray(matrix[candidates] @ profile, dtype=np.float32)
            else:
                similarity = np.asarray(
                    (matrix[candidates] @ profile.T).toarray(), dtype=np.float32
                ).reshape(-1)
            feature_scores[name] = similarity
        return feature_scores

    def _score_one_user(
        self,
        user_id: int,
        candidates: np.ndarray,
        history: list[int] | np.ndarray | None,
    ) -> np.ndarray:
        """Dùng history tường minh; thiếu history thì dùng train hoặc fallback."""
        if history is None:
            history = self._get_train_history(user_id)
        profiles = self._build_profiles(history)
        if profiles:
            return self._score_candidates(candidates, profiles)
        if self.cold_user_strategy == "popularity":
            return self.popularity_scores[candidates].copy()
        return np.zeros(candidates.size, dtype=np.float32)

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        history_item_idxs: list[int] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Dự đoán theo danh sách ứng viên; hỗ trợ một user hoặc từng cặp user-item."""
        if not self.is_fitted:
            raise RuntimeError("Cần gọi fit() hoặc load checkpoint trước khi predict().")
        self._ensure_features_loaded()
        candidates = np.asarray(item_ids, dtype=np.int64).reshape(-1)
        users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
        if candidates.size == 0:
            return np.empty(0, dtype=np.float32)
        if users.size not in (1, candidates.size):
            raise ValueError("user_ids phải có 1 phần tử hoặc cùng chiều dài với item_ids.")
        if users.size == 0:
            raise ValueError("user_ids không được rỗng khi có ứng viên.")

        scores = np.zeros(candidates.size, dtype=np.float32)
        valid = (candidates >= 0) & (candidates < self.num_items)
        if not np.any(valid):
            return scores
        if users.size == 1:
            scores[valid] = self._score_one_user(
                int(users[0]), candidates[valid], history_item_idxs
            )
            return scores

        if history_item_idxs is not None and np.unique(users).size > 1:
            raise ValueError("history_item_idxs chỉ dùng cho một user trong một lần predict().")
        for user_id in np.unique(users[valid]):
            selected = valid & (users == user_id)
            scores[selected] = self._score_one_user(
                int(user_id), candidates[selected], history_item_idxs
            )
        return scores

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
        user_histories: dict[int, list[int]] | None = None,
        candidate_item_idxs: list[int] | np.ndarray | None = None,
    ) -> dict[int, list[int]]:
        """Tạo Top-K, chấm theo batch và loại bài trong lịch sử nếu được yêu cầu."""
        if not self.is_fitted:
            raise RuntimeError("Cần gọi fit() hoặc load checkpoint trước khi recommend().")
        self._ensure_features_loaded()
        if top_k < 0:
            raise ValueError("top_k không được âm.")
        if candidate_item_idxs is None:
            catalog = np.arange(self.num_items, dtype=np.int64)
        else:
            catalog = self._clean_history(candidate_item_idxs)
        recommendations: dict[int, list[int]] = {}
        for user_id in np.asarray(user_ids, dtype=np.int64).reshape(-1):
            user_id = int(user_id)
            history = (
                user_histories.get(user_id, [])
                if user_histories is not None
                else self._get_train_history(user_id)
            )
            valid_history = self._clean_history(history)
            available = catalog
            if filter_seen and valid_history.size:
                available = catalog[~np.isin(catalog, valid_history)]
            if top_k == 0 or available.size == 0:
                recommendations[user_id] = []
                continue
            # Hồ sơ chỉ tính một lần cho user, tránh tính lại ở từng batch.
            profiles = self._build_profiles(valid_history)
            scores_by_chunk = []
            for start in range(0, available.size, self.recommendation_batch_size):
                chunk = available[start : start + self.recommendation_batch_size]
                if profiles:
                    chunk_scores = self._score_candidates(chunk, profiles)
                elif self.cold_user_strategy == "popularity":
                    chunk_scores = self.popularity_scores[chunk].copy()
                else:
                    chunk_scores = np.zeros(chunk.size, dtype=np.float32)
                scores_by_chunk.append(chunk_scores)
            scores = np.concatenate(scores_by_chunk)
            ranking = np.argsort(-scores, kind="stable")[:top_k]
            recommendations[user_id] = available[ranking].tolist()
        return recommendations

    def __getstate__(self) -> dict[str, Any]:
        """Checkpoint chỉ lưu cấu hình, tương tác train và đường dẫn feature."""
        state = self.__dict__.copy()
        state["_features"] = {}
        return state
