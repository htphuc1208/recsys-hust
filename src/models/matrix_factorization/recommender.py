from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.common.seed import set_seed
from src.models.matrix_factorization.model import MatrixFactorization
from src.models.pytorch_base import PyTorchRecommenderBase


class MatrixFactorizationRecommender(PyTorchRecommenderBase):
    """Matrix Factorization Recommender compatible with BaseRecommender interface.

    Supports pointwise binary cross-entropy (BCE) or MSE loss, multi-epoch training,
    and fast vectorized inference for offline evaluation and live serving.
    """

    def __init__(
        self,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        label_col: str = "label",
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 3,
        batch_size: int = 4096,
        use_bias: bool = True,
        loss_type: str = "bce",
        seed: int = 42,
        device: str | None = None,
        max_train_samples: int | None = None,
        num_users: int | None = None,
        num_items: int | None = None,
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
        self.label_col = label_col
        self.use_bias = use_bias
        self.loss_type = loss_type.lower()
        self.num_users = num_users or 0
        self.num_items = num_items or 0
        self.torch_model: MatrixFactorization | None = None

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "MatrixFactorizationRecommender":
        """Fit Matrix Factorization model on training DataFrame."""
        set_seed(self.seed)

        df = train_df
        if self.max_train_samples and len(df) > self.max_train_samples:
            df = df.sample(n=self.max_train_samples, random_state=self.seed)

        max_u = int(df[self.user_col].max()) + 1 if len(df) > 0 else 0
        max_i = int(df[self.item_col].max()) + 1 if len(df) > 0 else 0
        self.num_users = max(self.num_users, max_u)
        self.num_items = max(self.num_items, max_i)

        # Track popular items for cold user fallback
        item_counts = df[self.item_col].value_counts()
        self.popular_items_fallback = [int(x) for x in item_counts.index.tolist()]

        # Track seen items for recommendation filtering
        self.user_seen_items = (
            df.groupby(self.user_col)[self.item_col]
            .apply(lambda s: {int(x) for x in s})
            .to_dict()
        )

        user_arr = df[self.user_col].to_numpy(dtype=np.int64, copy=True)
        item_arr = df[self.item_col].to_numpy(dtype=np.int64, copy=True)

        if self.label_col in df.columns:
            label_arr = df[self.label_col].to_numpy(dtype=np.float32, copy=True)
        else:
            label_arr = np.ones(len(df), dtype=np.float32)

        user_t = torch.from_numpy(user_arr)
        item_t = torch.from_numpy(item_arr)
        label_t = torch.from_numpy(label_arr)


        self.torch_model = MatrixFactorization(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            use_bias=self.use_bias,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        criterion = (
            nn.BCEWithLogitsLoss()
            if self.loss_type == "bce"
            else nn.MSELoss()
        )

        n_samples = len(user_t)
        indices = np.arange(n_samples)

        self.torch_model.train()
        for _epoch in range(1, self.epochs + 1):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]

                u_b = user_t[batch_idx].to(self.device)
                i_b = item_t[batch_idx].to(self.device)
                y_b = label_t[batch_idx].to(self.device)

                optimizer.zero_grad()
                preds = self.torch_model(u_b, i_b)
                loss = criterion(preds, y_b)
                loss.backward()
                optimizer.step()

        self._sync_factors_to_numpy(self.torch_model)
        self.is_fitted = True
        return self
