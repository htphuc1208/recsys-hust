from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd


class BaseSplitter(ABC):
    """Abstract base class for dataset splitters."""

    @abstractmethod
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train, validation, and test sets.

        Args:
            df: Interaction dataframe containing user, item, and timestamp/rating info.

        Returns:
            Tuple of (train_df, val_df, test_df).
        """
        pass
