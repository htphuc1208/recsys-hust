import pickle
from abc import abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.models.base import BaseRecommender


class PyTorchRecommenderBase(BaseRecommender):
    """Base recommender class for PyTorch embedding-based models (MF, BPR, etc.).

    Provides high-performance vectorized NumPy scoring for evaluation and serving,
    robust cold-start handling, and checkpoint serialization.
    """

    def __init__(
        self,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 3,
        batch_size: int = 4096,
        seed: int = 42,
        device: str | None = None,
        max_train_samples: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.user_col = user_col
        self.item_col = item_col
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.max_train_samples = max_train_samples

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_users: int = 0
        self.num_items: int = 0
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_biases: np.ndarray | None = None
        self.item_biases: np.ndarray | None = None
        self.user_seen_items: dict[int, set[int]] = {}
        self.popular_items_fallback: list[int] = []

    def _sync_factors_to_numpy(self, torch_model: torch.nn.Module) -> None:
        """Cache model embeddings into NumPy arrays for fast CPU inference."""
        if hasattr(torch_model, "user_embedding"):
            self.user_factors = (
                torch_model.user_embedding.weight.detach().cpu().numpy().astype(np.float32)
            )
        if hasattr(torch_model, "item_embedding"):
            self.item_factors = (
                torch_model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
            )
        if hasattr(torch_model, "user_bias") and torch_model.user_bias is not None:
            self.user_biases = (
                torch_model.user_bias.weight.detach().cpu().numpy().flatten().astype(np.float32)
            )
        else:
            self.user_biases = None
        if hasattr(torch_model, "item_bias") and torch_model.item_bias is not None:
            self.item_biases = (
                torch_model.item_bias.weight.detach().cpu().numpy().flatten().astype(np.float32)
            )
        else:
            self.item_biases = None

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute recommendation score for user-item pairs using cached embeddings."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        user_arr = np.asarray(user_ids)
        item_arr = np.asarray(item_ids, dtype=int)
        n_items = len(item_arr)

        if n_items == 0:
            return np.array([], dtype=float)

        # Single user query (common case in ranking evaluation impressions)
        if len(user_arr) == 1 or user_arr.ndim == 0:
            u = int(user_arr.item()) if user_arr.ndim == 0 else int(user_arr[0])

            # Cold user fallback
            if u < 0 or u >= len(self.user_factors):
                scores = np.zeros(n_items, dtype=float)
                if self.item_biases is not None:
                    valid_mask = (item_arr >= 0) & (item_arr < len(self.item_biases))
                    scores[valid_mask] = self.item_biases[item_arr[valid_mask]]
                return scores

            u_vec = self.user_factors[u]
            u_bias = self.user_biases[u] if self.user_biases is not None else 0.0

            valid_item_mask = (item_arr >= 0) & (item_arr < len(self.item_factors))
            scores = np.zeros(n_items, dtype=float)

            if np.any(valid_item_mask):
                valid_items = item_arr[valid_item_mask]
                item_matrix = self.item_factors[valid_items]
                valid_scores = np.dot(item_matrix, u_vec) + u_bias

                if self.item_biases is not None:
                    valid_scores += self.item_biases[valid_items]

                scores[valid_item_mask] = valid_scores

            return scores

        # Pointwise 1-to-1 pairs: user_ids and item_ids same length
        scores = np.zeros(len(user_arr), dtype=float)
        valid_mask = (
            (user_arr >= 0)
            & (user_arr < len(self.user_factors))
            & (item_arr >= 0)
            & (item_arr < len(self.item_factors))
        )

        if np.any(valid_mask):
            u_idx = user_arr[valid_mask]
            i_idx = item_arr[valid_mask]
            pair_scores = np.sum(self.user_factors[u_idx] * self.item_factors[i_idx], axis=1)
            if self.user_biases is not None:
                pair_scores += self.user_biases[u_idx]
            if self.item_biases is not None:
                pair_scores += self.item_biases[i_idx]
            scores[valid_mask] = pair_scores

        return scores

    def recommend(
        self, user_ids: np.ndarray, top_k: int = 10, filter_seen: bool = True
    ) -> dict[int, list[int]]:
        """Generate top_k recommendations for specified users."""
        if not self.is_fitted or self.user_factors is None or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend().")

        recs: dict[int, list[int]] = {}
        total_items = len(self.item_factors)

        for u in user_ids:
            u_int = int(u)
            seen = self.user_seen_items.get(u_int, set()) if filter_seen else set()

            if u_int < 0 or u_int >= len(self.user_factors):
                # Cold user: recommend most popular items
                user_recs = [i for i in self.popular_items_fallback if i not in seen][:top_k]
                recs[u_int] = user_recs
                continue

            all_scores = np.dot(self.item_factors, self.user_factors[u_int])
            if self.item_biases is not None:
                all_scores += self.item_biases

            if filter_seen and seen:
                seen_indices = np.array([i for i in seen if 0 <= i < total_items], dtype=int)
                if len(seen_indices) > 0:
                    all_scores[seen_indices] = -np.inf

            top_indices = np.argpartition(-all_scores, min(top_k, total_items - 1))[:top_k]
            sorted_top = top_indices[np.argsort(-all_scores[top_indices])]
            recs[u_int] = [int(x) for x in sorted_top.tolist()]

        return recs

    def save(self, path: Any) -> None:
        """Serialize model state and cached factors to disk."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Any) -> "PyTorchRecommenderBase":
        """Deserialize model from disk."""
        file_path = Path(path)
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model

    @abstractmethod
    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "PyTorchRecommenderBase":
        """Fit recommender model on training dataframe."""
        pass
