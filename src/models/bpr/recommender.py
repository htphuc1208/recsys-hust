from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.common.logger import setup_logger
from src.common.seed import set_seed
from src.models.bpr.data import (
    build_bpr_pairs,
    convert_pairs_to_tensors,
    iter_bpr_batches,
    scan_bpr_training_data,
)
from src.models.bpr.model import BPR
from src.models.pytorch_base import PyTorchRecommenderBase


class BPRRecommender(PyTorchRecommenderBase):
    """Bayesian Personalized Ranking dùng pair positive-negative theo impression."""

    def __init__(
        self,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        label_col: str = "label",
        impression_col: str = "impression_id",
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 3,
        batch_size: int = 4096,
        use_item_bias: bool = True,
        seed: int = 42,
        device: str | None = None,
        max_train_samples: int | None = None,
        num_users: int | None = None,
        num_items: int | None = None,
        shuffle_buffer_size: int = 65536,
        read_batch_size: int = 65536,
        track_user_seen: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            user_col=user_col,
            item_col=item_col,
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            epochs=epochs,
            batch_size=batch_size,
            seed=seed,
            device=device,
            max_train_samples=max_train_samples,
            **kwargs,
        )
        if shuffle_buffer_size < batch_size:
            raise ValueError("shuffle_buffer_size phải lớn hơn hoặc bằng batch_size.")
        if read_batch_size <= 0:
            raise ValueError("read_batch_size phải lớn hơn 0.")

        self.label_col = label_col
        self.impression_col = impression_col
        self.use_item_bias = use_item_bias
        self.num_users = num_users or 0
        self.num_items = num_items or 0
        self.shuffle_buffer_size = shuffle_buffer_size
        self.read_batch_size = read_batch_size
        self.track_user_seen = track_user_seen
        self.torch_model: BPR | None = None

        # Mask ngăn embedding ngẫu nhiên của user/item chưa từng train được dùng khi infer.
        self.observed_user_mask = np.zeros(0, dtype=bool)
        self.observed_item_mask = np.zeros(0, dtype=bool)

    def _initialize_model(self) -> torch.optim.Optimizer:
        """Khởi tạo BPR-MF và optimizer theo kích thước đã xác định."""
        if self.num_users <= 0 or self.num_items <= 0:
            raise ValueError("num_users và num_items phải lớn hơn 0 trước khi train BPR.")

        self.torch_model = BPR(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            use_item_bias=self.use_item_bias,
        ).to(self.device)

        return torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _train_batches(
        self,
        batches: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
    ) -> tuple[int, int, float]:
        """Train một epoch từ iterator batch và trả thống kê loss."""
        if self.torch_model is None:
            raise RuntimeError("BPR model chưa được khởi tạo.")

        pair_count = 0
        batch_count = 0
        weighted_loss_sum = 0.0
        self.torch_model.train()

        for user_ids, positive_ids, negative_ids in batches:
            current_batch_size = len(user_ids)
            if current_batch_size == 0:
                continue

            user_batch = user_ids.to(self.device)
            positive_batch = positive_ids.to(self.device)
            negative_batch = negative_ids.to(self.device)

            optimizer.zero_grad()
            positive_scores = self.torch_model(user_batch, positive_batch)
            negative_scores = self.torch_model(user_batch, negative_batch)
            loss = -F.logsigmoid(positive_scores - negative_scores).mean()
            loss.backward()
            optimizer.step()

            pair_count += current_batch_size
            batch_count += 1
            weighted_loss_sum += float(loss.item()) * current_batch_size

        if pair_count == 0:
            raise ValueError("Không có BPR pair hợp lệ trong epoch.")

        mean_loss = weighted_loss_sum / pair_count
        return pair_count, batch_count, mean_loss

    def fit_from_path(
        self,
        train_path: str | Path,
        val_df: pd.DataFrame | None = None,
    ) -> BPRRecommender:
        """Train BPR streaming trực tiếp từ thư mục pointwise Parquet."""
        del val_df
        set_seed(self.seed)
        logger = setup_logger("bpr")
        data_dir = Path(train_path)

        if not data_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy dữ liệu BPR: {data_dir}")

        if self.max_train_samples is not None:
            logger.warning(
                "fit_from_path() dùng toàn bộ dữ liệu streaming; "
                "max_train_samples=%s sẽ bị bỏ qua.",
                self.max_train_samples,
            )

        logger.info("Scanning pointwise data before streaming training...")
        scan_start = time.perf_counter()
        summary = scan_bpr_training_data(
            data_dir=data_dir,
            read_batch_size=self.read_batch_size,
            track_user_seen=self.track_user_seen,
        )
        scan_time = time.perf_counter() - scan_start

        self.num_users = max(self.num_users, summary.max_user_idx + 1)
        self.num_items = max(self.num_items, summary.max_item_idx + 1)
        self.observed_user_mask = summary.observed_user_mask
        self.observed_item_mask = summary.observed_item_mask
        self.popular_items_fallback = summary.popular_items
        self.user_seen_items = summary.user_seen_items if self.track_user_seen else {}

        logger.info(
            "Scan completed in %.2fs: rows=%s, pairs=%s, positives=%s, negatives=%s, "
            "observed_users=%s, observed_items=%s, embedding_shape=(%s users, %s items)",
            scan_time,
            f"{summary.pointwise_rows:,}",
            f"{summary.pair_count:,}",
            f"{summary.positive_rows:,}",
            f"{summary.negative_rows:,}",
            f"{summary.num_observed_users:,}",
            f"{summary.num_observed_items:,}",
            f"{self.num_users:,}",
            f"{self.num_items:,}",
        )

        optimizer = self._initialize_model()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()
            batches = iter_bpr_batches(
                data_dir=data_dir,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed + epoch - 1,
                shuffle_buffer_size=self.shuffle_buffer_size,
                read_batch_size=self.read_batch_size,
            )
            pair_count, batch_count, mean_loss = self._train_batches(
                batches=batches,
                optimizer=optimizer,
            )
            epoch_time = time.perf_counter() - epoch_start

            logger.info(
                "Epoch %d/%d | pairs=%s | batches=%s | mean_bpr_loss=%.6f | time=%.2fs",
                epoch,
                self.epochs,
                f"{pair_count:,}",
                f"{batch_count:,}",
                mean_loss,
                epoch_time,
            )

        if self.torch_model is None:
            raise RuntimeError("BPR model chưa được khởi tạo sau training.")

        self._sync_factors_to_numpy(self.torch_model)
        self.is_fitted = True
        return self

    def _build_dataframe_pairs(
        self,
        df: pd.DataFrame,
    ) -> list[tuple[int, int, int]]:
        """Tạo BPR pair cho đường fit(DataFrame) tương thích code cũ."""
        if self.label_col not in df.columns:
            return []

        if self.impression_col in df.columns:
            pairs: list[tuple[int, int, int]] = []
            for _impression_id, group in df.groupby(self.impression_col, sort=False):
                if group.empty:
                    continue

                user_idx = int(group[self.user_col].iloc[0])
                pairs.extend(
                    build_bpr_pairs(
                        user_idx=user_idx,
                        item_ids=[int(value) for value in group[self.item_col].tolist()],
                        labels=[int(value) for value in group[self.label_col].tolist()],
                    )
                )
            return pairs

        # Fallback cho DataFrame cũ không có impression_id.
        positive_df = df[df[self.label_col] == 1]
        negative_df = df[df[self.label_col] == 0]
        pairs = []
        positives_by_user = positive_df.groupby(self.user_col)[self.item_col].apply(list).to_dict()
        negatives_by_user = negative_df.groupby(self.user_col)[self.item_col].apply(list).to_dict()

        rng = np.random.default_rng(self.seed)
        for user_idx, positive_items in positives_by_user.items():
            negative_items = negatives_by_user.get(user_idx, [])
            seen_u = self.user_seen_items.get(int(user_idx), set())
            if not negative_items:
                for positive_item in positive_items:
                    neg = int(rng.integers(0, self.num_items))
                    for _ in range(10):
                        if neg not in seen_u:
                            break
                        neg = int(rng.integers(0, self.num_items))
                    pairs.append((int(user_idx), int(positive_item), neg))
            else:
                for negative_index, negative_item in enumerate(negative_items):
                    positive_item = positive_items[negative_index % len(positive_items)]
                    pairs.append(
                        (
                            int(user_idx),
                            int(positive_item),
                            int(negative_item),
                        )
                    )

        return pairs

    def _iter_dataframe_batches(
        self,
        pairs: list[tuple[int, int, int]],
        rng: np.random.Generator,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Xáo và chia pair DataFrame thành batch nhỏ."""
        order = rng.permutation(len(pairs))

        for start in range(0, len(order), self.batch_size):
            batch_indices = order[start : start + self.batch_size]
            batch_pairs = [pairs[int(index)] for index in batch_indices]
            yield convert_pairs_to_tensors(batch_pairs)

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> BPRRecommender:
        """Fit từ DataFrame nhỏ; pipeline BPR chính dùng fit_from_path()."""
        del val_df
        set_seed(self.seed)

        df = train_df
        if self.max_train_samples and len(df) > self.max_train_samples:
            df = df.sample(n=self.max_train_samples, random_state=self.seed)

        if df.empty:
            raise ValueError("Dữ liệu train BPR rỗng.")

        valid_users = df[self.user_col].to_numpy(dtype=np.int64, copy=False)
        valid_items = df[self.item_col].to_numpy(dtype=np.int64, copy=False)
        valid_user_ids = valid_users[valid_users >= 0]
        valid_item_ids = valid_items[valid_items >= 0]

        if len(valid_user_ids) == 0 or len(valid_item_ids) == 0:
            raise ValueError("Không có user/item hợp lệ để train BPR.")

        self.num_users = max(self.num_users, int(valid_user_ids.max()) + 1)
        self.num_items = max(self.num_items, int(valid_item_ids.max()) + 1)
        self.observed_user_mask = np.zeros(self.num_users, dtype=bool)
        self.observed_item_mask = np.zeros(self.num_items, dtype=bool)
        self.observed_user_mask[valid_user_ids] = True
        self.observed_item_mask[valid_item_ids] = True

        if self.label_col in df.columns:
            positive_df = df[df[self.label_col] == 1]
        else:
            positive_df = df

        item_counts = positive_df[self.item_col].value_counts()
        self.popular_items_fallback = [
            int(item_id) for item_id in item_counts.index.tolist() if int(item_id) >= 0
        ]

        if self.track_user_seen:
            self.user_seen_items = (
                positive_df.groupby(self.user_col)[self.item_col]
                .apply(lambda values: {int(value) for value in values if int(value) >= 0})
                .to_dict()
            )
        else:
            self.user_seen_items = {}

        pairs = self._build_dataframe_pairs(df)
        if not pairs:
            raise ValueError("Không tìm thấy BPR pair hợp lệ trong DataFrame.")

        # Chỉ user/item thật sự xuất hiện trong pair mới được xem là đã train.
        self.observed_user_mask[:] = False
        self.observed_item_mask[:] = False
        for user_idx, positive_item, negative_item in pairs:
            self.observed_user_mask[user_idx] = True
            self.observed_item_mask[positive_item] = True
            self.observed_item_mask[negative_item] = True

        optimizer = self._initialize_model()
        rng = np.random.default_rng(self.seed)

        for _epoch in range(1, self.epochs + 1):
            self._train_batches(
                batches=self._iter_dataframe_batches(pairs, rng),
                optimizer=optimizer,
            )

        if self.torch_model is None:
            raise RuntimeError("BPR model chưa được khởi tạo sau training.")

        self._sync_factors_to_numpy(self.torch_model)
        self.is_fitted = True
        return self

    def _observed_item_mask_for(self, item_ids: np.ndarray) -> np.ndarray:
        """Trả mask item đã thực sự tham gia train BPR."""
        mask = getattr(self, "observed_item_mask", None)
        item_arr = np.asarray(item_ids, dtype=int)
        if mask is None or len(mask) == 0:
            return np.ones(item_arr.shape, dtype=bool)

        result = np.zeros(item_arr.shape, dtype=bool)
        valid_mask = (item_arr >= 0) & (item_arr < len(mask))

        if np.any(valid_mask):
            valid_positions = np.flatnonzero(valid_mask)
            result[valid_positions] = mask[item_arr[valid_positions]]

        return result

    def _replace_unobserved_users(self, user_ids: np.ndarray) -> np.ndarray:
        """Biến user chưa train thành cold user để không dùng embedding ngẫu nhiên."""
        mask = getattr(self, "observed_user_mask", None)
        user_arr = np.asarray(user_ids, dtype=int).copy()
        if mask is None or len(mask) == 0:
            return user_arr

        if user_arr.ndim == 0:
            user_idx = int(user_arr.item())
            if user_idx < 0 or user_idx >= len(mask) or not mask[user_idx]:
                return np.asarray(-1, dtype=int)
            return user_arr

        valid_mask = (user_arr >= 0) & (user_arr < len(mask))
        observed_mask = np.zeros(user_arr.shape, dtype=bool)

        if np.any(valid_mask):
            valid_positions = np.flatnonzero(valid_mask)
            observed_mask[valid_positions] = mask[user_arr[valid_positions]]

        user_arr[~observed_mask] = -1
        return user_arr

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Chấm điểm và không dùng embedding ngẫu nhiên của user/item chưa train."""
        safe_user_ids = self._replace_unobserved_users(user_ids)
        scores = super().predict(
            user_ids=safe_user_ids,
            item_ids=item_ids,
            **kwargs,
        )

        observed_items = self._observed_item_mask_for(item_ids)
        scores[~observed_items] = 0.0
        return scores

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
    ) -> dict[int, list[int]]:
        """Recommend chỉ trong tập item đã thực sự tham gia train BPR."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend().")

        item_mask = getattr(self, "observed_item_mask", None)
        user_mask = getattr(self, "observed_user_mask", None)
        observed_items = (
            np.flatnonzero(item_mask)
            if item_mask is not None and len(item_mask) > 0
            else np.arange(self.num_items)
        )
        recs: dict[int, list[int]] = {}

        for user_id in user_ids:
            user_idx = int(user_id)
            seen = self.user_seen_items.get(user_idx, set()) if filter_seen else set()

            is_warm_user = 0 <= user_idx < len(self.user_factors) and (
                user_mask is None or (user_idx < len(user_mask) and bool(user_mask[user_idx]))
            )

            if not is_warm_user:
                recs[user_idx] = [
                    item_id for item_id in self.popular_items_fallback if item_id not in seen
                ][:top_k]
                continue

            candidate_items = observed_items
            if filter_seen and seen:
                candidate_items = np.array(
                    [int(item_id) for item_id in observed_items if int(item_id) not in seen],
                    dtype=int,
                )

            if len(candidate_items) == 0:
                recs[user_idx] = []
                continue

            scores = np.dot(
                self.item_factors[candidate_items],
                self.user_factors[user_idx],
            )
            if self.item_biases is not None:
                scores = scores + self.item_biases[candidate_items]

            result_size = min(top_k, len(candidate_items))
            if result_size == len(candidate_items):
                top_positions = np.argsort(-scores)
            else:
                partition = np.argpartition(-scores, result_size - 1)[:result_size]
                top_positions = partition[np.argsort(-scores[partition])]

            recs[user_idx] = [
                int(item_id) for item_id in candidate_items[top_positions[:result_size]].tolist()
            ]

        return recs
