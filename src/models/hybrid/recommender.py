"""Adaptive Gating Hybrid Recommender (P1 & P2 Core Contribution).

Fuses Collaborative Filtering (BPR/MF), Sequential (SASRec), and Content/Popularity
recommendation streams through an adaptive neural gating network or static ablation weights.
"""

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch

from src.common.seed import set_seed
from src.models.base import BaseRecommender
from src.models.baselines.popularity import PopularityRecommender
from src.models.bpr import BPRRecommender
from src.models.hybrid.gating import AdaptiveGatingNetwork
from src.models.sequential.recommender import SASRecRecommender

HybridMode = Literal["adaptive", "static", "no_cf", "no_seq", "no_content"]


def _standardize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize score distribution to standard scale (mean 0, std 1) for fair fusion."""
    if len(scores) <= 1:
        return scores
    std = float(np.std(scores))
    if std > 1e-6:
        return (scores - np.mean(scores)) / std
    spread = float(np.max(scores) - np.min(scores))
    if spread > 1e-6:
        return (scores - np.min(scores)) / spread
    return scores


class AdaptiveGatingHybrid(BaseRecommender):
    """Core Contribution Recommender: Adaptive Gating Multi-Branch Hybrid.

    Integrates:
      1. Collaborative Filtering (BPRRecommender) -> Long-term preferences
      2. Sequential Model (SASRecRecommender)      -> Short-term dynamic interest
      3. Content / Item Baseline (Popularity)      -> Cold-start / Broad popularity
    Gated dynamically by AdaptiveGatingNetwork with full support for ablation studies.
    """

    def __init__(
        self,
        mode: HybridMode = "adaptive",
        embedding_dim: int = 64,
        static_weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
        normalize_branch_scores: bool = True,
        cf_checkpoint: str | Path | None = None,
        seq_checkpoint: str | Path | None = None,
        content_checkpoint: str | Path | None = None,
        cf_params: dict[str, Any] | None = None,
        seq_params: dict[str, Any] | None = None,
        content_params: dict[str, Any] | None = None,
        seed: int = 42,
        device: str | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.mode: HybridMode = mode
        self.embedding_dim = embedding_dim
        self.static_weights = static_weights
        self.normalize_branch_scores = normalize_branch_scores
        self.seed = seed

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load or initialize branch sub-models
        self.cf_model: BaseRecommender = (
            BaseRecommender.load(cf_checkpoint)
            if cf_checkpoint
            else BPRRecommender(embedding_dim=embedding_dim, seed=seed, **(cf_params or {}))
        )
        self.seq_model: BaseRecommender = (
            BaseRecommender.load(seq_checkpoint)
            if seq_checkpoint
            else SASRecRecommender(embedding_dim=embedding_dim, seed=seed, **(seq_params or {}))
        )
        self.content_model: BaseRecommender = (
            BaseRecommender.load(content_checkpoint)
            if content_checkpoint
            else PopularityRecommender(**(content_params or {}))
        )

        # Gating network
        self.gating_network = AdaptiveGatingNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=32,
            dropout_rate=0.1,
        ).to(self.device)

        # Expose item factors for FAISS indexing compatibility
        self.item_factors: np.ndarray | None = None
        self.popular_items_fallback: list[int] = []

        # If seq_model is not pre-fitted but cf_model is loaded, align item factors
        if (
            getattr(self.cf_model, "is_fitted", False)
            and hasattr(self.cf_model, "item_factors")
            and self.cf_model.item_factors is not None
        ):
            if not getattr(self.seq_model, "is_fitted", False):
                self.seq_model.num_items = len(self.cf_model.item_factors)
                self.seq_model.item_factors = self.cf_model.item_factors
                self.seq_model.is_fitted = True

    @property
    def user_factors(self) -> np.ndarray | None:
        """Expose CF user factors for FAISS vector index queries."""
        if hasattr(self.cf_model, "user_factors") and self.cf_model.user_factors is not None:
            return self.cf_model.user_factors
        return None

    def _get_branch_mask(self) -> torch.Tensor:
        """Return boolean mask tensor for active branches: [CF, SEQ, CONTENT]."""
        if self.mode == "no_cf":
            return torch.tensor([False, True, True], dtype=torch.bool)
        if self.mode == "no_seq":
            return torch.tensor([True, False, True], dtype=torch.bool)
        if self.mode == "no_content":
            return torch.tensor([True, True, False], dtype=torch.bool)
        return torch.tensor([True, True, True], dtype=torch.bool)

    def fit(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None
    ) -> "AdaptiveGatingHybrid":
        """Fit individual branch recommenders and calibrate gating network."""
        set_seed(self.seed)

        # 1. Fit sub-models if not already loaded from checkpoint
        if not getattr(self.cf_model, "is_fitted", False):
            self.cf_model.fit(train_df, val_df)
        if not getattr(self.seq_model, "is_fitted", False):
            self.seq_model.fit(train_df, val_df)
        if not getattr(self.content_model, "is_fitted", False):
            self.content_model.fit(train_df, val_df)

        # 2. Sync popular items fallback from sub-models
        self.popular_items_fallback = getattr(self.content_model, "popular_items", []) or getattr(
            self.cf_model, "popular_items_fallback", []
        )

        # 3. Cache combined item factors from sequential / CF branches
        if hasattr(self.seq_model, "item_factors") and self.seq_model.item_factors is not None:
            self.item_factors = self.seq_model.item_factors
        elif hasattr(self.cf_model, "item_factors") and self.cf_model.item_factors is not None:
            self.item_factors = self.cf_model.item_factors

        self.is_fitted = True
        return self

    def predict(
        self,
        user_ids: np.ndarray | None,
        item_ids: np.ndarray,
        history_item_idxs: list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Compute hybrid recommendation score through adaptive gating or static ablation weights."""
        if not self.is_fitted:
            raise RuntimeError("AdaptiveGatingHybrid model must be fitted before predict().")

        candidates = np.asarray(item_ids, dtype=int)
        n_candidates = len(candidates)
        if n_candidates == 0:
            return np.array([], dtype=float)

        # 1. Predict scores across the 3 branches
        s_cf = self.cf_model.predict(
            user_ids, candidates, history_item_idxs=history_item_idxs, **kwargs
        )
        s_seq = self.seq_model.predict(
            user_ids, candidates, history_item_idxs=history_item_idxs, **kwargs
        )
        s_content = self.content_model.predict(
            user_ids, candidates, history_item_idxs=history_item_idxs, **kwargs
        )

        # 2. Standardize raw scores so weights operate on commensurate scales
        if self.normalize_branch_scores:
            s_cf = _standardize_scores(s_cf)
            s_seq = _standardize_scores(s_seq)
            s_content = _standardize_scores(s_content)

        # 3. Compute branch weights
        if self.mode == "static":
            w = np.array(self.static_weights, dtype=float)
            w = w / np.sum(w)
            return w[0] * s_cf + w[1] * s_seq + w[2] * s_content

        if self.mode in ("no_cf", "no_seq", "no_content"):
            mask = self._get_branch_mask().cpu().numpy()
            w = np.array(self.static_weights, dtype=float) * mask
            total = np.sum(w)
            w = w / total if total > 0 else np.array([0.33, 0.33, 0.33])
            return w[0] * s_cf + w[1] * s_seq + w[2] * s_content

        # 4. Adaptive Mode: compute gating weights via AdaptiveGatingNetwork
        user_repr = None
        if hasattr(self.seq_model, "encode_history") and history_item_idxs:
            user_repr = self.seq_model.encode_history(history_item_idxs)
        elif (
            hasattr(self.cf_model, "user_factors")
            and self.cf_model.user_factors is not None
            and user_ids is not None
            and len(user_ids) > 0
        ):
            u = int(user_ids[0]) if np.asarray(user_ids).ndim > 0 else int(user_ids)
            if 0 <= u < len(self.cf_model.user_factors):
                user_repr = self.cf_model.user_factors[u]

        if user_repr is None:
            user_repr = np.zeros(self.embedding_dim, dtype=np.float32)

        # Retrieve candidate representations
        item_repr = None
        if hasattr(self.seq_model, "item_factors") and self.seq_model.item_factors is not None:
            factors = self.seq_model.item_factors
            valid_mask = (candidates >= 0) & (candidates < len(factors))
            item_repr = np.zeros((n_candidates, self.embedding_dim), dtype=np.float32)
            item_repr[valid_mask] = factors[candidates[valid_mask]]
        elif hasattr(self.cf_model, "item_factors") and self.cf_model.item_factors is not None:
            factors = self.cf_model.item_factors
            valid_mask = (candidates >= 0) & (candidates < len(factors))
            item_repr = np.zeros((n_candidates, self.embedding_dim), dtype=np.float32)
            item_repr[valid_mask] = factors[candidates[valid_mask]]

        if item_repr is None:
            item_repr = np.zeros((n_candidates, self.embedding_dim), dtype=np.float32)

        # Inference through Gating Network
        self.gating_network.eval()
        with torch.no_grad():
            u_t = torch.from_numpy(user_repr).to(self.device).float()
            i_t = torch.from_numpy(item_repr).to(self.device).float()
            mask = self._get_branch_mask().to(self.device)
            weights = self.gating_network(u_t, i_t, branch_mask=mask).cpu().numpy()

        w_cf = weights[:, 0]
        w_seq = weights[:, 1]
        w_content = weights[:, 2]

        final_scores = w_cf * s_cf + w_seq * s_seq + w_content * s_content
        return final_scores

    def recommend(
        self,
        user_ids: np.ndarray,
        top_k: int = 10,
        filter_seen: bool = True,
        user_histories: dict[int, list[int]] | None = None,
        **kwargs: Any,
    ) -> dict[int, list[int]]:
        """Generate top-K recommendations for target users."""
        recs: dict[int, list[int]] = {}
        total_items = len(self.item_factors) if self.item_factors is not None else 1000
        all_candidates = np.arange(total_items, dtype=int)

        for u in user_ids:
            u_int = int(u)
            history = user_histories.get(u_int, []) if user_histories else []
            scores = self.predict(
                user_ids=np.array([u_int]),
                item_ids=all_candidates,
                history_item_idxs=history,
            )

            seen = (
                getattr(self.cf_model, "user_seen_items", {}).get(u_int, set())
                if filter_seen
                else set()
            )
            if filter_seen and seen:
                seen_idx = [i for i in seen if 0 <= i < total_items]
                scores[seen_idx] = -np.inf

            top_idx = np.argpartition(-scores, min(top_k, total_items - 1))[:top_k]
            sorted_top = top_idx[np.argsort(-scores[top_idx])]
            recs[u_int] = [int(x) for x in sorted_top.tolist()]

        return recs

    def save(self, path: Any) -> None:
        """Serialize hybrid recommender to disk."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Any) -> "AdaptiveGatingHybrid":
        """Deserialize hybrid recommender from disk."""
        file_path = Path(path)
        with open(file_path, "rb") as f:
            model = pickle.load(f)
        return model
