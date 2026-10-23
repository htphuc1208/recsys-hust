from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.hybrid import AdaptiveGatingHybrid, AdaptiveGatingNetwork
from src.models.registry import get_model


def test_adaptive_gating_network_weights_sum_to_one():
    gating = AdaptiveGatingNetwork(embedding_dim=16, hidden_dim=8)

    user_repr = torch.randn(1, 16)
    item_repr = torch.randn(5, 16)

    weights = gating(user_repr, item_repr)
    assert weights.shape == (5, 3)

    # Each row must sum to 1.0 (Softmax property)
    row_sums = weights.sum(dim=-1).detach().numpy()
    np.testing.assert_allclose(row_sums, np.ones(5), atol=1e-6)

    # Weights must be non-negative
    assert (weights >= 0.0).all()


def test_adaptive_gating_network_ablation_masking():
    gating = AdaptiveGatingNetwork(embedding_dim=16, hidden_dim=8)

    user_repr = torch.randn(1, 16)
    item_repr = torch.randn(4, 16)

    # 1. No CF branch
    mask_no_cf = torch.tensor([False, True, True])
    w_no_cf = gating(user_repr, item_repr, branch_mask=mask_no_cf)
    np.testing.assert_allclose(w_no_cf[:, 0].detach().numpy(), np.zeros(4), atol=1e-5)
    np.testing.assert_allclose(w_no_cf.sum(dim=-1).detach().numpy(), np.ones(4), atol=1e-5)

    # 2. No Seq branch
    mask_no_seq = torch.tensor([True, False, True])
    w_no_seq = gating(user_repr, item_repr, branch_mask=mask_no_seq)
    np.testing.assert_allclose(w_no_seq[:, 1].detach().numpy(), np.zeros(4), atol=1e-5)

    # 3. No Content branch
    mask_no_content = torch.tensor([True, True, False])
    w_no_content = gating(user_repr, item_repr, branch_mask=mask_no_content)
    np.testing.assert_allclose(w_no_content[:, 2].detach().numpy(), np.zeros(4), atol=1e-5)


def test_adaptive_hybrid_recommender_lifecycle(tmp_path: Path):
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "item_idx": [1, 2, 3, 2, 3, 4, 1, 3, 5],
            "label": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="h"),
        }
    )

    # Test Adaptive mode
    hybrid_adaptive = AdaptiveGatingHybrid(
        mode="adaptive",
        embedding_dim=8,
        seed=42,
        cf_params={"epochs": 1, "batch_size": 4},
        seq_params={"epochs": 1, "batch_size": 4, "max_seq_len": 4},
    )
    hybrid_adaptive.fit(train_df)

    assert hybrid_adaptive.is_fitted
    candidates = np.array([1, 2, 3, 4])
    scores_adaptive = hybrid_adaptive.predict(
        user_ids=np.array([0]),
        item_ids=candidates,
        history_item_idxs=[1, 2],
    )
    assert len(scores_adaptive) == 4
    assert np.isfinite(scores_adaptive).all()

    # Test Static mode
    hybrid_static = AdaptiveGatingHybrid(
        mode="static",
        embedding_dim=8,
        static_weights=(0.5, 0.3, 0.2),
        seed=42,
        cf_params={"epochs": 1, "batch_size": 4},
        seq_params={"epochs": 1, "batch_size": 4, "max_seq_len": 4},
    )
    hybrid_static.fit(train_df)
    scores_static = hybrid_static.predict(
        user_ids=np.array([0]),
        item_ids=candidates,
        history_item_idxs=[1, 2],
    )
    assert len(scores_static) == 4

    # Test Ablation modes
    for ab_mode in ("no_cf", "no_seq", "no_content"):
        ab_model = AdaptiveGatingHybrid(
            mode=ab_mode,
            embedding_dim=8,
            seed=42,
            cf_params={"epochs": 1, "batch_size": 4},
            seq_params={"epochs": 1, "batch_size": 4, "max_seq_len": 4},
        )
        ab_model.fit(train_df)
        ab_scores = ab_model.predict(
            user_ids=np.array([0]),
            item_ids=candidates,
            history_item_idxs=[1, 2],
        )
        assert len(ab_scores) == 4
        assert np.isfinite(ab_scores).all()

    # Serialization test
    ckpt_path = tmp_path / "hybrid.pkl"
    hybrid_adaptive.save(ckpt_path)
    assert ckpt_path.is_file()

    loaded = AdaptiveGatingHybrid.load(ckpt_path)
    assert loaded.is_fitted
    loaded_scores = loaded.predict(
        user_ids=np.array([0]),
        item_ids=candidates,
        history_item_idxs=[1, 2],
    )
    np.testing.assert_allclose(scores_adaptive, loaded_scores, rtol=1e-5)


def test_adaptive_hybrid_registry():
    model = get_model("hybrid", embedding_dim=16)
    assert isinstance(model, AdaptiveGatingHybrid)
