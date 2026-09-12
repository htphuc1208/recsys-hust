from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class BaseRecommender(ABC):
    """Unified interface for all recommender models."""

    def __init__(self, **kwargs: Any):
        self.config = kwargs
        self.is_fitted = False

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, val_df: Optional[pd.DataFrame] = None) -> "BaseRecommender":
        """Fit model on training interactions."""
        pass

    @abstractmethod
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        """Pointwise score prediction for (user, item) pairs."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
    ) -> Dict[int, List[int]]:
        """Generate top-K recommended item IDs for given user IDs.

        Returns:
            Dictionary mapping user_id to list of recommended item_ids.
        """
        pass
