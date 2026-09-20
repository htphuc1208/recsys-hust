from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.common.seed import set_seed
from src.models.bpr.model import BPR
from src.models.pytorch_base import PyTorchRecommenderBase


class BPRRecommender(PyTorchRecommenderBase):
    """Bayesian Personalized Ranking (BPR-MF) Recommender compatible with BaseRecommender.

    Optimizes pairwise ranking loss: -log(sigmoid(score_pos - score_neg)).
    Automatically pairs positive and negative interactions from pointwise data
    or samples negatives uniformly from the catalog for implicit-only data.
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
        use_item_bias: bool = True,
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
        self.use_item_bias = use_item_bias
        self.num_users = num_users or 0
        self.num_items = num_items or 0
        self.torch_model: BPR | None = None

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "BPRRecommender":
        """Fit BPR model using pairwise positive-negative triplets."""
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

        # Track seen items for filtering in recommend()
        self.user_seen_items = (
            df.groupby(self.user_col)[self.item_col]
            .apply(lambda s: {int(x) for x in s})
            .to_dict()
        )

        # Prepare triplets (user, pos_item, neg_item)
        if self.label_col in df.columns:
            pos_df = df[df[self.label_col] == 1]
            neg_df = df[df[self.label_col] == 0]

            if len(neg_df) > 0 and len(pos_df) > 0:
                # Group by user to construct pairwise examples
                pos_by_user = pos_df.groupby(self.user_col)[self.item_col].apply(list).to_dict()
                neg_by_user = neg_df.groupby(self.user_col)[self.item_col].apply(list).to_dict()

                user_list = []
                pos_list = []
                neg_list = []

                for u, p_items in pos_by_user.items():
                    n_items = neg_by_user.get(u)
                    if not n_items:
                        # Fallback random negative
                        for p in p_items:
                            user_list.append(u)
                            pos_list.append(p)
                            neg_list.append(np.random.randint(0, self.num_items))
                    else:
                        for idx, n in enumerate(n_items):
                            p = p_items[idx % len(p_items)]
                            user_list.append(u)
                            pos_list.append(p)
                            neg_list.append(n)

                u_arr = np.array(user_list, dtype=np.int64)
                p_arr = np.array(pos_list, dtype=np.int64)
                n_arr = np.array(neg_list, dtype=np.int64)
            else:
                u_arr = pos_df[self.user_col].to_numpy(dtype=np.int64, copy=True)
                p_arr = pos_df[self.item_col].to_numpy(dtype=np.int64, copy=True)
                n_arr = np.random.randint(0, self.num_items, size=len(p_arr), dtype=np.int64)
        else:
            u_arr = df[self.user_col].to_numpy(dtype=np.int64, copy=True)
            p_arr = df[self.item_col].to_numpy(dtype=np.int64, copy=True)
            n_arr = np.random.randint(0, self.num_items, size=len(p_arr), dtype=np.int64)


        if len(u_arr) == 0:
            raise ValueError("No valid training pairs found for BPR.")

        u_tensor = torch.from_numpy(u_arr)
        p_tensor = torch.from_numpy(p_arr)
        n_tensor = torch.from_numpy(n_arr)

        self.torch_model = BPR(
            num_users=self.num_users,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            use_item_bias=self.use_item_bias,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        n_samples = len(u_tensor)
        indices = np.arange(n_samples)

        self.torch_model.train()
        for _epoch in range(1, self.epochs + 1):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]

                u_b = u_tensor[batch_idx].to(self.device)
                p_b = p_tensor[batch_idx].to(self.device)
                n_b = n_tensor[batch_idx].to(self.device)

                optimizer.zero_grad()
                pos_scores = self.torch_model(u_b, p_b)
                neg_scores = self.torch_model(u_b, n_b)
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()
                loss.backward()
                optimizer.step()

        self._sync_factors_to_numpy(self.torch_model)
        self.is_fitted = True
        return self
