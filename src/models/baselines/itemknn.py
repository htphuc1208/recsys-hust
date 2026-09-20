from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models.base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """Item-based Collaborative Filtering Recommender (Sarwar et al. 2001).

    Computes cosine similarity between items using implicit user feedback with shrinkage.
    Scores candidates for a user based on similarity to items in the user's reading history.
    """

    def __init__(
        self,
        top_k_neighbors: int = 100,
        shrinkage: float = 10.0,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.top_k_neighbors = top_k_neighbors
        self.shrinkage = float(shrinkage)
        self.user_col = user_col
        self.item_col = item_col

        # item_id -> {neighbor_item_id: similarity_weight}
        self.item_similarities: dict[int, dict[int, float]] = {}
        self.catalog_items: list[int] = []

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "ItemKNNRecommender":
        # 1. Map users and items to contiguous indices
        unique_users = train_df[self.user_col].unique()
        unique_items = train_df[self.item_col].unique()
        self.catalog_items = [int(x) for x in unique_items]

        user_map = {uid: idx for idx, uid in enumerate(unique_users)}
        item_map = {iid: idx for idx, iid in enumerate(unique_items)}
        rev_item_map = {idx: int(iid) for idx, iid in enumerate(unique_items)}

        n_users = len(user_map)
        n_items = len(item_map)

        row_indices = train_df[self.user_col].map(user_map).to_numpy()
        col_indices = train_df[self.item_col].map(item_map).to_numpy()

        # 2. Build binary user-item CSR matrix (R: n_users x n_items)
        data = np.ones(len(train_df), dtype=np.float32)
        R = sp.csr_matrix(
            (data, (row_indices, col_indices)), shape=(n_users, n_items), dtype=np.float32
        )

        # 3. Compute item-item co-occurrence matrix C = R^T * R (shape: n_items x n_items)
        C = R.T.dot(R).tocsr()

        # Item norms (L2 norm of item column vector = sqrt(frequency))
        item_frequencies = np.array(C.diagonal(), dtype=np.float32)
        item_norms = np.sqrt(item_frequencies)

        # 4. Compute cosine similarity with shrinkage and retain top-k neighbors per item
        self.item_similarities = {}

        for i in range(n_items):
            start_ptr = C.indptr[i]
            end_ptr = C.indptr[i + 1]

            if start_ptr == end_ptr:
                continue

            neighbor_cols = C.indices[start_ptr:end_ptr]
            co_counts = C.data[start_ptr:end_ptr]

            # Filter out self-similarity
            valid_mask = neighbor_cols != i
            if not np.any(valid_mask):
                continue

            neighbor_cols = neighbor_cols[valid_mask]
            co_counts = co_counts[valid_mask]

            # Cosine with shrinkage: S_ij = C_ij / (norm_i * norm_j + lambda)
            denoms = item_norms[i] * item_norms[neighbor_cols] + self.shrinkage
            sims = co_counts / denoms

            # Keep top-K neighbors
            if len(sims) > self.top_k_neighbors:
                top_idx = np.argpartition(-sims, self.top_k_neighbors)[: self.top_k_neighbors]
                neighbor_cols = neighbor_cols[top_idx]
                sims = sims[top_idx]

            item_id = rev_item_map[i]
            self.item_similarities[item_id] = {
                rev_item_map[col]: float(sim)
                for col, sim in zip(neighbor_cols, sims, strict=False)
                if sim > 0.0
            }

        self.is_fitted = True
        return self

    def predict(
        self,
        user_ids: np.ndarray | None,
        item_ids: np.ndarray,
        history_item_idxs: list[int] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Score candidate items by sum of similarities to items in user's history."""
        if not self.is_fitted or not history_item_idxs:
            return np.zeros(len(item_ids), dtype=float)

        history_set = {int(h) for h in history_item_idxs if h >= 0}

        if not history_set:
            return np.zeros(len(item_ids), dtype=float)

        scores = np.zeros(len(item_ids), dtype=float)
        for idx, item in enumerate(item_ids):
            item_int = int(item)
            neighbors = self.item_similarities.get(item_int)
            if not neighbors:
                continue

            # Sum similarities to history items
            score = 0.0
            for h in history_set:
                if h in neighbors:
                    score += neighbors[h]
            scores[idx] = score

        return scores

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
        user_histories: dict[int, list[int]] | None = None,
    ) -> dict[int, list[int]]:
        assert self.is_fitted, "Model must be fitted before recommend()"
        recs: dict[int, list[int]] = {}

        for user in user_ids:
            u_int = int(user)
            history = user_histories.get(u_int, []) if user_histories else []
            if not history:
                recs[u_int] = self.catalog_items[:top_k]
                continue

            candidate_scores: dict[int, float] = {}
            for h in history:
                neighbors = self.item_similarities.get(h, {})
                for neighbor_id, sim in neighbors.items():
                    candidate_scores[neighbor_id] = candidate_scores.get(neighbor_id, 0.0) + sim

            if filter_seen:
                for h in history:
                    candidate_scores.pop(h, None)

            sorted_items = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
            recs[u_int] = [item for item, _ in sorted_items[:top_k]]

        return recs
