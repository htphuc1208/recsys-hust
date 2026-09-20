from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Unified interface for all recommender models."""

    def __init__(self, **kwargs: Any):
        self.config = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BaseRecommender":
        """Fit model on training interactions."""
        pass

    @abstractmethod
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Pointwise score prediction for (user, item) pairs, with optional context kwargs."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
    ) -> dict[int, list[int]]:
        """Generate top-K recommended item IDs for given user IDs.

        Returns:
            Dictionary mapping user_id to list of recommended item_ids.
        """
        pass

    def save(self, path: Any) -> None:
        """Serialize model checkpoint to disk."""
        import pickle
        from pathlib import Path

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Any) -> "BaseRecommender":
        """Deserialize model checkpoint from disk."""
        import pickle
        from pathlib import Path

        file_path = Path(path)
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
