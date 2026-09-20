from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.baselines.itemknn import ItemKNNRecommender
from src.models.baselines.random import RandomRecommender


def test_random_recommender(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 1, 2],
            "item_idx": [10, 20, 30],
        }
    )
    model = RandomRecommender(seed=42)
    model.fit(train_df)
    assert model.is_fitted
    assert set(model.catalog_items) == {10, 20, 30}

    candidates = np.array([10, 20, 99])
    scores1 = model.predict(np.array([0]), candidates)
    assert len(scores1) == 3
    assert np.all((scores1 >= 0.0) & (scores1 <= 1.0))

    recs = model.recommend(np.array([0, 1]), top_k=2)
    assert len(recs[0]) == 2
    assert len(recs[1]) == 2

    # Serialization test
    ckpt_path = tmp_path / "random.pkl"
    model.save(ckpt_path)
    loaded = RandomRecommender.load(ckpt_path)
    assert loaded.is_fitted
    assert loaded.catalog_items == model.catalog_items


def test_itemknn_recommender(tmp_path: Path) -> None:
    # Users 1, 2, 3 read items 10 and 20 together.
    # User 4 only reads item 30.
    train_df = pd.DataFrame(
        {
            "user_idx": [1, 1, 2, 2, 3, 3, 4],
            "item_idx": [10, 20, 10, 20, 10, 20, 30],
        }
    )
    model = ItemKNNRecommender(top_k_neighbors=10, shrinkage=0.0)
    model.fit(train_df)
    assert model.is_fitted

    # Item 10 and 20 should have similarity > 0
    assert 20 in model.item_similarities.get(10, {})
    assert 10 in model.item_similarities.get(20, {})
    # Item 30 has no co-occurrences with 10 or 20
    assert 30 not in model.item_similarities.get(10, {})

    # Predict: user has read item 10 in history
    # Candidates: [20 (neighbor), 30 (not neighbor), 99 (cold item)]
    candidates = np.array([20, 30, 99])
    scores = model.predict(
        user_ids=np.array([1]),
        item_ids=candidates,
        history_item_idxs=[10],
    )
    assert scores[0] > 0.0  # Item 20 is neighbor of 10
    assert scores[1] == 0.0  # Item 30 is not neighbor
    assert scores[2] == 0.0  # Item 99 is cold item

    # Empty history should score 0.0
    empty_scores = model.predict(
        user_ids=np.array([1]),
        item_ids=candidates,
        history_item_idxs=[],
    )
    assert np.all(empty_scores == 0.0)

    # Checkpoint serialization
    ckpt_path = tmp_path / "itemknn.pkl"
    model.save(ckpt_path)
    loaded = ItemKNNRecommender.load(ckpt_path)
    assert loaded.is_fitted
    loaded_scores = loaded.predict(
        user_ids=np.array([1]),
        item_ids=candidates,
        history_item_idxs=[10],
    )
    np.testing.assert_array_equal(scores, loaded_scores)
