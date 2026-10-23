from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Random baseline recommender (Sanity check lower bound).

    Assigns uniform random scores to candidates or suggests random catalog items.
    """

    def __init__(
        self,
        seed: int = 42,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.seed = seed
        self.user_col = user_col
        self.item_col = item_col
        self.catalog_items: list[int] = []
        self.user_seen_items: dict[int, set[int]] = {}
        self.rng = np.random.default_rng(self.seed)

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "RandomRecommender":
        self.catalog_items = [int(x) for x in train_df[self.item_col].unique().tolist()]
        if self.user_col in train_df.columns:
            self.user_seen_items = (
                train_df.groupby(self.user_col)[self.item_col]
                .apply(lambda s: {int(x) for x in s})
                .to_dict()
            )
        self.rng = np.random.default_rng(self.seed)
        self.is_fitted = True
        return self

    def predict(
        self, user_ids: np.ndarray | None, item_ids: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Assign uniform random scores in [0, 1) to candidates."""
        return self.rng.random(len(item_ids), dtype=float)

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
    ) -> dict[int, list[int]]:
        assert self.is_fitted, "Model must be fitted before recommend()"
        recs: dict[int, list[int]] = {}

        for user in user_ids:
            u_int = int(user)
            seen = self.user_seen_items.get(u_int, set()) if filter_seen else set()
            available = (
                [x for x in self.catalog_items if x not in seen]
                if filter_seen
                else self.catalog_items
            )
            items_arr = np.array(available if available else self.catalog_items)

            if len(items_arr) >= top_k:
                sampled = self.rng.choice(items_arr, size=top_k, replace=False)
            else:
                sampled = self.rng.choice(items_arr, size=top_k, replace=True)
            recs[u_int] = [int(x) for x in sampled]

        return recs
