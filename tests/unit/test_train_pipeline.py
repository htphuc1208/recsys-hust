from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.baselines.popularity import PopularityRecommender


def test_popularity_recommender_fit_predict_save_load(tmp_path: Path) -> None:
    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 1, 1, 2],
            "item_idx": [10, 20, 10, 30, 10],
        }
    )

    model = PopularityRecommender(user_col="user_idx", item_col="item_idx")
    model.fit(train_df)

    assert model.is_fitted
    assert model.popular_items == [10, 20, 30]
    assert model.item_scores[10] == 3.0
    assert model.item_scores[20] == 1.0
    assert model.item_scores[30] == 1.0

    # Cold item (item 99) should score 0.0
    candidates = np.array([30, 99, 10])
    scores = model.predict(np.array([0]), candidates)
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] == 3.0

    # Test serialization
    ckpt_path = tmp_path / "model.pkl"
    model.save(ckpt_path)
    assert ckpt_path.is_file()

    loaded_model = PopularityRecommender.load(ckpt_path)
    assert loaded_model.is_fitted
    assert loaded_model.popular_items == [10, 20, 30]

    loaded_scores = loaded_model.predict(np.array([0]), candidates)
    np.testing.assert_array_equal(scores, loaded_scores)
