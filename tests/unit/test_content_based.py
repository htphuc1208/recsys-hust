"""Kiểm thử B4 trên ma trận nội dung giả lập nhỏ, không cần dữ liệu MINDlarge."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.models.base import BaseRecommender
from src.models.content_based import ContentBasedRecommender
from src.models.registry import get_model


def create_features(tmp_path):
    """Tạo 5 nhóm feature với item 3 là bài cold nhưng giống item 0."""
    feature_dir = tmp_path / "b4"
    feature_dir.mkdir()
    title = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ], dtype=np.float32,
    )
    category = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32,
    )
    groups = {
        "title_tfidf": title,
        "abstract_tfidf": title,
        "category_onehot": category,
        "subcategory_onehot": category,
    }
    for name, values in groups.items():
        sparse.save_npz(feature_dir / f"{name}.npz", sparse.csr_matrix(values))
    entity = np.zeros((5, 100), dtype=np.float32)
    entity[[0, 3], 0] = 1.0
    entity[[1, 2], 1] = 1.0
    np.save(feature_dir / "title_entity_embeddings.npy", entity)
    manifest = {
        "total_item_mapping": 5,
        "feature_shapes": {
            **{name: list(values.shape) for name, values in groups.items()},
            "title_entity_embeddings": [5, 100],
        },
    }
    (feature_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (feature_dir / "item_index.json").write_text(
        json.dumps(["N0", "N1", "N2", "N3", "N4"]), encoding="utf-8"
    )
    return feature_dir


def create_train_data():
    """Item 3 không có lượt click train; item 0 là bài phổ biến."""
    return pd.DataFrame({
        "user_idx": [0, 0, 1, 2],
        "item_idx": [0, 0, 1, 2],
    })


def test_registry_and_weights(tmp_path):
    """B4 có trong registry và kiểm tra trọng số hợp lệ."""
    feature_dir = create_features(tmp_path)
    model = get_model("content_based", feature_dir=str(feature_dir))
    assert isinstance(model, ContentBasedRecommender)
    assert sum(model.feature_weights.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="trọng số"):
        ContentBasedRecommender(feature_weights={"title": -1})
    with pytest.raises(ValueError, match="không hợp lệ"):
        ContentBasedRecommender(feature_weights={"unknown": 1})


def test_cold_item_scores_without_clicks(tmp_path):
    """Item cold có nội dung giống history được chấm điểm dương."""
    model = ContentBasedRecommender(feature_dir=str(create_features(tmp_path)))
    model.fit(create_train_data())
    candidates = np.array([3, 1, 2, 4])
    scores = model.predict(np.array([0]), candidates, history_item_idxs=[0])
    assert scores.shape == (4,)
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]
    assert scores[0] > scores[3]
    assert model.popularity_scores[3] == 0.0
    assert model.predict(np.array([0]), np.array([-1, 100]), history_item_idxs=[0]).tolist() == [0, 0]


def test_explicit_history_overrides_training_history(tmp_path):
    """History của impression được dùng thay vì history lưu từ train."""
    model = ContentBasedRecommender(feature_dir=str(create_features(tmp_path)))
    model.fit(create_train_data())
    score_from_train = model.predict(np.array([0]), np.array([0, 1]))
    score_from_impression = model.predict(
        np.array([0]), np.array([0, 1]), history_item_idxs=[1]
    )
    assert score_from_train[0] > score_from_train[1]
    assert score_from_impression[1] > score_from_impression[0]


def test_empty_history_default_is_pure_content_zero(tmp_path):
    """User không có history không được chuyển sang một recommender popularity khác."""
    model = ContentBasedRecommender(feature_dir=str(create_features(tmp_path)))
    model.fit(create_train_data())
    scores = model.predict(np.array([-1]), np.array([0, 3]), history_item_idxs=[])
    assert scores.tolist() == [0, 0]

    popularity_model = ContentBasedRecommender(
        feature_dir=str(model.feature_dir), cold_user_strategy="popularity"
    ).fit(create_train_data())
    popularity_scores = popularity_model.predict(
        np.array([-1]), np.array([0, 3]), history_item_idxs=[]
    )
    assert popularity_scores[0] > popularity_scores[1]
    assert popularity_scores[1] == 0.0


def test_predict_pointwise_multiple_users(tmp_path):
    """Pointwise batch với mỗi cặp user-item vẫn được hỗ trợ."""
    model = ContentBasedRecommender(feature_dir=str(create_features(tmp_path)))
    model.fit(create_train_data())
    scores = model.predict(np.array([0, 1]), np.array([0, 1]))
    assert scores.shape == (2,)
    assert scores[0] > 0
    assert scores[1] > 0
    with pytest.raises(ValueError, match="chỉ dùng cho một user"):
        model.predict(np.array([0, 1]), np.array([0, 1]), history_item_idxs=[0])


def test_recommend_filter_seen_and_candidate_list(tmp_path):
    """Recommend loại bài đã đọc và có thể hạn chế tập ứng viên."""
    model = ContentBasedRecommender(
        feature_dir=str(create_features(tmp_path)), recommendation_batch_size=2
    )
    model.fit(create_train_data())
    recommendations = model.recommend(
        np.array([0]), top_k=2, candidate_item_idxs=[0, 1, 3], filter_seen=True
    )
    assert recommendations[0] == [3, 1]
    empty_user = model.recommend(
        np.array([999]), top_k=1, candidate_item_idxs=[0, 3]
    )
    assert empty_user[999] == [0]


def test_checkpoint_reload_without_embedded_feature_matrices(tmp_path):
    """Checkpoint có thể nạp lại và đọc feature từ đường dẫn đã cấu hình."""
    model = ContentBasedRecommender(feature_dir=str(create_features(tmp_path)))
    model.fit(create_train_data())
    original = model.predict(np.array([0]), np.array([0, 1, 3]), history_item_idxs=[0])
    checkpoint = tmp_path / "b4.pkl"
    model.save(checkpoint)
    loaded = BaseRecommender.load(checkpoint)
    assert loaded._features == {}
    actual = loaded.predict(np.array([0]), np.array([0, 1, 3]), history_item_idxs=[0])
    np.testing.assert_allclose(actual, original)
    assert loaded._features


def test_missing_or_mismatched_feature_files(tmp_path):
    """Dừng có thông báo khi thiếu đặc trưng hoặc sai số lượng item."""
    feature_dir = create_features(tmp_path)
    model = ContentBasedRecommender(feature_dir=str(feature_dir))
    model.num_items = 6
    with pytest.raises(ValueError, match="Checkpoint"):
        model.fit(create_train_data())

    (feature_dir / "title_tfidf.npz").unlink()
    with pytest.raises(FileNotFoundError, match="Thiếu thành phần"):
        ContentBasedRecommender(feature_dir=str(feature_dir)).fit(create_train_data())
