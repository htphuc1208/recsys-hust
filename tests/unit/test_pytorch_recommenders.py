from pathlib import Path

import numpy as np
import pandas as pd

from src.models.bpr import BPRRecommender
from src.models.matrix_factorization import MatrixFactorizationRecommender
from src.models.registry import get_model


def test_mf_recommender_lifecycle(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [10, 20, 10, 30, 20, 30],
            "label": [1, 0, 1, 1, 0, 1],
        }
    )

    model = MatrixFactorizationRecommender(
        embedding_dim=8,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        seed=42,
    )
    model.fit(train_df)

    assert model.is_fitted
    assert model.user_factors is not None
    assert model.item_factors is not None

    # Predict warm user
    candidates = np.array([10, 20, 30, 99])  # 99 is cold item
    scores = model.predict(user_ids=np.array([0]), item_ids=candidates)
    assert len(scores) == 4
    # Cold item 99 should have score 0.0
    assert scores[3] == 0.0

    # Predict cold user
    cold_scores = model.predict(user_ids=np.array([999]), item_ids=candidates)
    assert len(cold_scores) == 4

    # Recommend
    recs = model.recommend(user_ids=np.array([0, 1]), top_k=2, filter_seen=True)
    assert 0 in recs
    assert 1 in recs
    assert len(recs[0]) <= 2

    # Serialization test
    ckpt_path = tmp_path / "mf_model.pkl"
    model.save(ckpt_path)
    assert ckpt_path.is_file()

    loaded = MatrixFactorizationRecommender.load(ckpt_path)
    assert loaded.is_fitted
    np.testing.assert_allclose(
        loaded.predict(np.array([0]), candidates),
        scores,
        rtol=1e-5,
    )


def test_bpr_recommender_lifecycle(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2, 2],
            "item_idx": [10, 20, 10, 30, 20, 30],
            "label": [1, 0, 1, 0, 1, 0],
        }
    )

    model = BPRRecommender(
        embedding_dim=8,
        epochs=2,
        batch_size=4,
        learning_rate=0.01,
        seed=42,
    )
    model.fit(train_df)

    assert model.is_fitted
    assert model.user_factors is not None
    assert model.item_factors is not None

    # Predict
    candidates = np.array([10, 20, 30, 99])
    scores = model.predict(user_ids=np.array([0]), item_ids=candidates)
    assert len(scores) == 4
    assert scores[3] == 0.0

    # Recommend
    recs = model.recommend(user_ids=np.array([0, 1]), top_k=2, filter_seen=False)
    assert len(recs[0]) <= 2
    assert len(recs[1]) <= 2

    # Serialization
    ckpt_path = tmp_path / "bpr_model.pkl"
    model.save(ckpt_path)
    assert ckpt_path.is_file()

    loaded = BPRRecommender.load(ckpt_path)
    assert loaded.is_fitted
    np.testing.assert_allclose(
        loaded.predict(np.array([0]), candidates),
        scores,
        rtol=1e-5,
    )


def test_registry_get_model_mf_and_bpr() -> None:
    mf = get_model("mf", embedding_dim=16)
    assert isinstance(mf, MatrixFactorizationRecommender)
    assert mf.embedding_dim == 16

    bpr = get_model("bpr", embedding_dim=32)
    assert isinstance(bpr, BPRRecommender)
    assert bpr.embedding_dim == 32
