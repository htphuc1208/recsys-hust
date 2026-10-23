from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.models.registry import get_model
from src.models.sequential import SASRec, SASRecRecommender


def test_sasrec_forward_shape():
    model = SASRec(
        num_items=10,
        embedding_dim=16,
        max_seq_len=8,
        num_blocks=2,
        num_heads=2,
    )

    input_seqs = torch.tensor(
        [
            [1, 2, 3, 10, 10, 10, 10, 10],  # 10 is padding_idx
            [4, 5, 6, 7, 8, 10, 10, 10],
        ]
    )
    pos_items = torch.tensor([4, 9])
    neg_items = torch.tensor([0, 1])

    pos_logits, neg_logits = model(input_seqs, pos_items, neg_items)

    assert pos_logits.shape == (2,)
    assert neg_logits.shape == (2,)


def test_sasrec_predict_candidates():
    model = SASRec(
        num_items=10,
        embedding_dim=16,
        max_seq_len=6,
        num_blocks=1,
        num_heads=2,
    )
    input_seqs = torch.tensor([[1, 2, 3, 10, 10, 10]])
    candidates = torch.tensor([0, 1, 2, 3, 4, 5])

    scores = model.predict_candidates(input_seqs, candidates)
    assert scores.shape == (1, 6)


def test_sasrec_recommender_lifecycle(tmp_path: Path):
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "item_idx": [1, 2, 3, 2, 3, 4, 1, 3, 5],
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="h"),
        }
    )

    model = SASRecRecommender(
        embedding_dim=8,
        max_seq_len=6,
        num_blocks=1,
        num_heads=2,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        seed=42,
    )
    model.fit(train_df)

    assert model.is_fitted
    assert model.item_factors is not None
    assert len(model.item_factors) >= 6

    # Predict with history
    candidates = np.array([1, 2, 3, 4])
    scores = model.predict(
        user_ids=np.array([0]),
        item_ids=candidates,
        history_item_idxs=[1, 2],
    )
    assert len(scores) == 4
    assert np.isfinite(scores).all()

    # Predict without history (cold sequence fallback)
    cold_scores = model.predict(
        user_ids=np.array([99]),
        item_ids=candidates,
        history_item_idxs=[],
    )
    assert len(cold_scores) == 4

    # Recommend
    recs = model.recommend(
        user_ids=np.array([0, 1]),
        top_k=2,
        filter_seen=True,
        user_histories={0: [1, 2]},
    )
    assert 0 in recs
    assert len(recs[0]) <= 2

    # Save and Load
    ckpt_path = tmp_path / "sasrec.pkl"
    model.save(ckpt_path)
    assert ckpt_path.is_file()

    loaded = SASRecRecommender.load(ckpt_path)
    assert loaded.is_fitted
    loaded_scores = loaded.predict(
        user_ids=np.array([0]),
        item_ids=candidates,
        history_item_idxs=[1, 2],
    )
    np.testing.assert_allclose(scores, loaded_scores, rtol=1e-5)


def test_sasrec_registry():
    model = get_model("sasrec", embedding_dim=16)
    assert isinstance(model, SASRecRecommender)
    assert model.embedding_dim == 16
