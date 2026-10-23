"""SASRec Recommender wrapper compatible with BaseRecommender and PyTorchRecommenderBase."""

from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.common.seed import set_seed
from src.models.pytorch_base import PyTorchRecommenderBase
from src.models.sequential.model import SASRec


class SASRecRecommender(PyTorchRecommenderBase):
    """Self-Attentive Sequential Recommender (SASRec).

    Uses multi-head self-attention over user click sequences to generate contextual
    recommendations for next-item prediction.
    """

    def __init__(
        self,
        user_col: str = "user_idx",
        item_col: str = "item_idx",
        label_col: str = "label",
        history_col: str = "history_item_idxs",
        embedding_dim: int = 64,
        max_seq_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        epochs: int = 3,
        batch_size: int = 512,
        seed: int = 42,
        device: str | None = None,
        max_train_samples: int | None = None,
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
        self.history_col = history_col
        self.max_seq_len = max_seq_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.num_items = num_items or 0
        self.torch_model: SASRec | None = None

    def _pad_sequence(self, seq: list[int] | tuple[int, ...]) -> list[int]:
        """Right-pad or truncate sequence to max_seq_len using padding_idx (self.num_items)."""
        valid_items = [int(x) for x in seq if 0 <= x < self.num_items]
        if len(valid_items) > self.max_seq_len:
            valid_items = valid_items[-self.max_seq_len :]
        pad_len = self.max_seq_len - len(valid_items)
        return valid_items + [self.num_items] * pad_len

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "SASRecRecommender":
        """Fit SASRec model on interaction logs or sequence data."""
        set_seed(self.seed)

        df = train_df
        if self.max_train_samples and len(df) > self.max_train_samples:
            df = df.sample(n=self.max_train_samples, random_state=self.seed)

        # Track catalog size
        max_i = (
            int(df[self.item_col].max()) + 1 if len(df) > 0 and self.item_col in df.columns else 0
        )
        if self.history_col in df.columns:
            for hist in df[self.history_col].dropna():
                if hist:
                    max_i = max(max_i, max(hist) + 1)
        self.num_items = max(self.num_items, max_i)

        if self.user_col in df.columns:
            max_u = int(df[self.user_col].max()) + 1 if len(df) > 0 else 0
            self.num_users = max(self.num_users, max_u)

        # Track popular items for fallback
        if self.item_col in df.columns:
            item_counts = df[self.item_col].value_counts()
            self.popular_items_fallback = [int(x) for x in item_counts.index.tolist()]

        # Track user seen items
        if self.user_col in df.columns and self.item_col in df.columns:
            self.user_seen_items = (
                df.groupby(self.user_col)[self.item_col]
                .apply(lambda s: {int(x) for x in s if x >= 0})
                .to_dict()
            )

        # Construct training sequence triplets: (padded_history, pos_item, neg_item)
        sequences: list[list[int]] = []
        pos_items: list[int] = []
        neg_items: list[int] = []

        if self.history_col in df.columns:
            # Data already contains user click history (e.g. from impressions or behavior logs)
            pos_mask = (
                df[self.label_col] == 1
                if self.label_col in df.columns
                else pd.Series(True, index=df.index)
            )
            train_rows = df[pos_mask]

            for _, row in train_rows.iterrows():
                hist = row.get(self.history_col) or []
                pos_i = int(row[self.item_col])
                if pos_i < 0 or pos_i >= self.num_items:
                    continue
                padded = self._pad_sequence(hist)
                # Sample random negative
                neg_i = np.random.randint(0, self.num_items)
                while neg_i == pos_i and self.num_items > 1:
                    neg_i = np.random.randint(0, self.num_items)

                sequences.append(padded)
                pos_items.append(pos_i)
                neg_items.append(neg_i)
        elif self.user_col in df.columns and self.item_col in df.columns:
            # Construct sequences by grouping interactions per user
            for _u, group in df.groupby(self.user_col):
                items = [int(x) for x in group[self.item_col].tolist() if 0 <= x < self.num_items]
                if len(items) < 2:
                    continue
                seen_u = set(items)
                for t in range(1, len(items)):
                    hist = items[:t]
                    pos_i = items[t]
                    padded = self._pad_sequence(hist)

                    neg_i = np.random.randint(0, self.num_items)
                    for _ in range(5):
                        if neg_i not in seen_u:
                            break
                        neg_i = np.random.randint(0, self.num_items)

                    sequences.append(padded)
                    pos_items.append(pos_i)
                    neg_items.append(neg_i)

        if not sequences:
            # Fallback if no sequences could be built
            sequences = [[self.num_items] * self.max_seq_len]
            pos_items = [0]
            neg_items = [min(1, self.num_items - 1)]

        seq_tensor = torch.tensor(sequences, dtype=torch.long)
        pos_tensor = torch.tensor(pos_items, dtype=torch.long)
        neg_tensor = torch.tensor(neg_items, dtype=torch.long)

        self.torch_model = SASRec(
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            max_seq_len=self.max_seq_len,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.torch_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        n_samples = len(seq_tensor)
        indices = np.arange(n_samples)

        self.torch_model.train()
        for _epoch in range(1, self.epochs + 1):
            np.random.shuffle(indices)
            for start_idx in range(0, n_samples, self.batch_size):
                batch_idx = indices[start_idx : start_idx + self.batch_size]

                seq_b = seq_tensor[batch_idx].to(self.device)
                pos_b = pos_tensor[batch_idx].to(self.device)
                neg_b = neg_tensor[batch_idx].to(self.device)

                optimizer.zero_grad()
                pos_logits, neg_logits = self.torch_model(seq_b, pos_b, neg_b)

                # Pairwise ranking loss: -log(sigmoid(pos_logits - neg_logits))
                loss = -F.logsigmoid(pos_logits - neg_logits).mean()

                loss.backward()
                optimizer.step()

        # Cache item factors in NumPy for fast evaluation & serving
        self.item_factors = (
            self.torch_model.item_embedding.weight[: self.num_items]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        self.is_fitted = True
        return self

    def encode_history(self, history_item_idxs: list[int] | tuple[int, ...]) -> np.ndarray:
        """Encode a sequence of historical item indices into a user vector."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before encoding history.")

        if self.torch_model is None:
            if self.item_factors is not None and history_item_idxs:
                valid_h = [i for i in history_item_idxs if 0 <= i < len(self.item_factors)]
                if valid_h:
                    return np.mean(self.item_factors[valid_h], axis=0).astype(np.float32)
            return np.zeros(self.embedding_dim, dtype=np.float32)

        padded = self._pad_sequence(history_item_idxs)
        seq_t = torch.tensor([padded], dtype=torch.long, device=self.device)

        self.torch_model.eval()
        with torch.no_grad():
            user_repr = self.torch_model.encode_sequence(seq_t)
            return user_repr.cpu().numpy()[0].astype(np.float32)

    def predict(
        self,
        user_ids: np.ndarray | None,
        item_ids: np.ndarray,
        history_item_idxs: list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Score candidate items for a user, using click history when available."""
        if not self.is_fitted or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        candidates = np.asarray(item_ids, dtype=int)
        n_candidates = len(candidates)
        if n_candidates == 0:
            return np.array([], dtype=float)

        scores = np.zeros(n_candidates, dtype=float)
        valid_mask = (candidates >= 0) & (candidates < self.num_items)
        if not np.any(valid_mask):
            return scores

        # Encode user history if provided
        if history_item_idxs:
            u_vec = self.encode_history(history_item_idxs)
        else:
            # Cold sequence fallback: uniform or mean item factor
            u_vec = np.zeros(self.embedding_dim, dtype=np.float32)

        valid_items = candidates[valid_mask]
        valid_scores = np.dot(self.item_factors[valid_items], u_vec)
        scores[valid_mask] = valid_scores
        return scores

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
        user_histories: dict[int, list[int]] | None = None,
        **kwargs: Any,
    ) -> dict[int, list[int]]:
        """Generate top-k recommendations based on user history."""
        if not self.is_fitted or self.item_factors is None:
            raise RuntimeError("Model must be fitted before calling recommend().")

        recs: dict[int, list[int]] = {}
        for u in user_ids:
            u_int = int(u)
            seen = self.user_seen_items.get(u_int, set()) if filter_seen else set()
            hist = user_histories.get(u_int, []) if user_histories else []

            if not hist:
                # Cold user: recommend popular items
                user_recs = [i for i in self.popular_items_fallback if i not in seen][:top_k]
                recs[u_int] = user_recs
                continue

            u_vec = self.encode_history(hist)
            all_scores = np.dot(self.item_factors, u_vec)

            if filter_seen and seen:
                seen_idx = np.array([i for i in seen if 0 <= i < self.num_items], dtype=int)
                if len(seen_idx) > 0:
                    all_scores[seen_idx] = -np.inf

            top_idx = np.argpartition(-all_scores, min(top_k, self.num_items - 1))[:top_k]
            sorted_top = top_idx[np.argsort(-all_scores[top_idx])]
            recs[u_int] = [int(x) for x in sorted_top.tolist()]

        return recs
