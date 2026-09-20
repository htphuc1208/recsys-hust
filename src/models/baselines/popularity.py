from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Most-popular items baseline recommender."""

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.user_col = user_col
        self.item_col = item_col
        self.popular_items: List[int] = []
        self.user_seen_items: Dict[int, set] = {}

    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "PopularityRecommender":
        # Compute item frequencies
        item_counts = train_df[self.item_col].value_counts()
        self.popular_items = item_counts.index.tolist()

        # Cache seen items per user
        self.user_seen_items = (
            train_df.groupby(self.user_col)[self.item_col]
            .apply(set)
            .to_dict()
        )
        self.is_fitted = True
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        # Returns item popularity rank or zero if unknown
        item_rank_map = {item: idx for idx, item in enumerate(self.popular_items)}
        return np.array([-item_rank_map.get(item, 1e6) for item in item_ids], dtype=float)

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
    ) -> Dict[int, List[int]]:
        assert self.is_fitted, "Model must be fitted before recommend()"
        recs: Dict[int, List[int]] = {}

        for user in user_ids:
            seen = self.user_seen_items.get(user, set()) if filter_seen else set()
            user_recs = []
            for item in self.popular_items:
                if item not in seen:
                    user_recs.append(item)
                if len(user_recs) == top_k:
                    break
            recs[user] = user_recs

        return recs
