from typing import Any

import numpy as np
import pandas as pd

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Most-popular items baseline recommender.

    Scores candidate items proportionally to their interaction frequency in the training data.
    Cold items not seen during training receive a score of 0.0.
    """

    def __init__(
        self,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        timestamp_col: str | None = None,
        decay_factor: float | None = None,
        track_user_seen: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.user_col = user_col
        self.item_col = item_col
        self.timestamp_col = timestamp_col
        self.decay_factor = decay_factor
        self.track_user_seen = track_user_seen
        self.popular_items: list[int] = []
        self.item_scores: dict[int, float] = {}
        self.user_seen_items: dict[int, set] = {}

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "PopularityRecommender":
        if (
            self.decay_factor is not None
            and self.timestamp_col
            and self.timestamp_col in train_df.columns
        ):
            ts = pd.to_datetime(train_df[self.timestamp_col])
            max_ts = ts.max()
            days_diff = (max_ts - ts).dt.total_seconds() / 86400.0
            weights = np.exp(-self.decay_factor * days_diff)
            item_counts = train_df.groupby(self.item_col).apply(
                lambda g: weights.loc[g.index].sum()
            )
        else:
            item_counts = train_df[self.item_col].value_counts()

        self.popular_items = [int(x) for x in item_counts.index.tolist()]
        self.item_scores = {int(item): float(count) for item, count in item_counts.items()}

        if self.track_user_seen and self.user_col in train_df.columns:
            self.user_seen_items = (
                train_df.groupby(self.user_col)[self.item_col].apply(set).to_dict()
            )
        self.is_fitted = True
        return self

    def predict(
        self, user_ids: np.ndarray | None, item_ids: np.ndarray, **kwargs: Any
    ) -> np.ndarray:
        """Score candidate items by precomputed popularity."""
        return np.array([self.item_scores.get(int(item), 0.0) for item in item_ids], dtype=float)

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
    ) -> dict[int, list[int]]:
        assert self.is_fitted, "Model must be fitted before recommend()"
        recs: dict[int, list[int]] = {}

        for user in user_ids:
            seen = (
                self.user_seen_items.get(int(user), set())
                if (filter_seen and self.track_user_seen)
                else set()
            )
            user_recs = []
            for item in self.popular_items:
                if item not in seen:
                    user_recs.append(item)
                if len(user_recs) == top_k:
                    break
            recs[int(user)] = user_recs

        return recs
