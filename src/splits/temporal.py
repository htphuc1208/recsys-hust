import pandas as pd

from src.splits.base import BaseSplitter


class GlobalTemporalSplitter(BaseSplitter):
    """Split interactions globally by timestamp without future leakage."""

    def __init__(
        self,
        timestamp_col: str = "timestamp",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"
        self.timestamp_col = timestamp_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sorted = df.sort_values(self.timestamp_col).reset_index(drop=True)
        n = len(df_sorted)

        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        return train_df, val_df, test_df
