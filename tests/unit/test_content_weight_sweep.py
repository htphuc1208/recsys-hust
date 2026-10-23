"""Test cho công cụ tuning 5 trọng số Content-based."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.models.content_based.recommender import ContentBasedRecommender
from src.models.content_based.weight_sweep import (
    FEATURE_NAMES,
    CachedContentImpression,
    combine_feature_scores,
    evaluate_cached_weights,
    generate_weight_grid,
)


def _create_feature_dir(root: Path) -> Path:
    """Tạo 5 feature nhỏ để kiểm tra load_all trong unit test."""
    feature_dir = root / "features"
    feature_dir.mkdir(parents=True)

    matrices = {
        "title_tfidf": sparse.csr_matrix(np.eye(3, dtype=np.float32)),
        "abstract_tfidf": sparse.csr_matrix(np.eye(3, dtype=np.float32)),
        "category_onehot": sparse.csr_matrix(np.eye(3, dtype=np.float32)),
        "subcategory_onehot": sparse.csr_matrix(np.eye(3, dtype=np.float32)),
    }
    for name, matrix in matrices.items():
        sparse.save_npz(feature_dir / f"{name}.npz", matrix)

    np.save(
        feature_dir / "title_entity_embeddings.npy",
        np.eye(3, dtype=np.float32),
        allow_pickle=False,
    )

    manifest = {
        "total_item_mapping": 3,
        "feature_shapes": {
            "title_tfidf": [3, 3],
            "abstract_tfidf": [3, 3],
            "category_onehot": [3, 3],
            "subcategory_onehot": [3, 3],
            "title_entity_embeddings": [3, 3],
        },
    }
    (feature_dir / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    (feature_dir / "item_index.json").write_text(
        json.dumps([0, 1, 2]),
        encoding="utf-8",
    )
    return feature_dir


def test_generate_weight_grid_step_point_two_has_126_configs():
    """5 trọng số với step 0.2 phải có C(9,4)=126 tổ hợp."""
    grid = generate_weight_grid(0.2)
    assert len(grid) == 126
    for weights in grid:
        assert tuple(weights) == FEATURE_NAMES
        assert sum(weights.values()) == pytest.approx(1.0)
        assert all(weight >= 0.0 for weight in weights.values())




def test_generate_weight_grid_with_bounds_has_expected_configs():
    """Fine sweep phải chỉ sinh cấu hình nằm trong vùng bounds đã chọn."""
    bounds = {
        "title": (0.40, 0.70),
        "abstract": (0.00, 0.25),
        "category": (0.15, 0.25),
        "subcategory": (0.15, 0.30),
        "title_entity": (0.00, 0.05),
    }
    grid = generate_weight_grid(0.05, bounds=bounds)

    assert len(grid) == 95
    for weights in grid:
        assert sum(weights.values()) == pytest.approx(1.0)
        for name, (lower, upper) in bounds.items():
            assert lower <= weights[name] <= upper


def test_generate_weight_grid_rejects_bound_not_on_step():
    """Bounds phải nằm đúng trên lưới của step."""
    bounds = {
        "title": (0.43, 0.70),
    }
    with pytest.raises(ValueError, match="không nằm trên lưới"):
        generate_weight_grid(0.05, bounds=bounds)


def test_generate_weight_grid_rejects_impossible_bounds():
    """Không cho phép bounds có tổng min lớn hơn 1."""
    bounds = {
        "title": (0.50, 0.70),
        "abstract": (0.30, 0.40),
        "category": (0.20, 0.30),
        "subcategory": (0.10, 0.20),
        "title_entity": (0.00, 0.05),
    }
    with pytest.raises(ValueError, match="không thể tạo được"):
        generate_weight_grid(0.05, bounds=bounds)

def test_generate_weight_grid_rejects_invalid_step():
    """Step không chia hết 1 phải bị từ chối."""
    with pytest.raises(ValueError, match="chia hết 1"):
        generate_weight_grid(0.3)


def test_load_all_features_even_when_current_weight_is_zero(tmp_path):
    """Tuning vẫn nạp đủ 5 feature dù checkpoint đang tắt một feature."""
    feature_dir = _create_feature_dir(tmp_path)
    model = ContentBasedRecommender(
        feature_dir=str(feature_dir),
        feature_weights={"title": 1.0},
    )
    model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))
    assert set(model._features) == {"title"}

    model._ensure_features_loaded(load_all=True)
    assert set(model._features) == set(FEATURE_NAMES)


def test_combine_feature_scores_uses_supplied_weights():
    """Combined score phải là tổng component theo đúng trọng số."""
    impression = CachedContentImpression(
        impression_id=1,
        candidate_item_idxs=np.array([10, 20]),
        true_items={20},
        feature_scores={
            "title": np.array([1.0, 0.0], dtype=np.float32),
            "abstract": np.array([0.0, 1.0], dtype=np.float32),
            "category": np.zeros(2, dtype=np.float32),
            "subcategory": np.zeros(2, dtype=np.float32),
            "title_entity": np.zeros(2, dtype=np.float32),
        },
        fallback_scores=None,
        has_cold_candidate=False,
        is_cold_user=False,
    )
    title_only = {name: 0.0 for name in FEATURE_NAMES}
    title_only["title"] = 1.0
    abstract_only = {name: 0.0 for name in FEATURE_NAMES}
    abstract_only["abstract"] = 1.0

    np.testing.assert_allclose(
        combine_feature_scores(impression, title_only),
        [1.0, 0.0],
    )
    np.testing.assert_allclose(
        combine_feature_scores(impression, abstract_only),
        [0.0, 1.0],
    )


def test_evaluate_cached_weights_changes_ranking_and_slices():
    """Đổi feature weight phải đổi ranking và metric đúng slice."""
    impression = CachedContentImpression(
        impression_id=1,
        candidate_item_idxs=np.array([10, 20]),
        true_items={20},
        feature_scores={
            "title": np.array([1.0, 0.0], dtype=np.float32),
            "abstract": np.array([0.0, 1.0], dtype=np.float32),
            "category": np.zeros(2, dtype=np.float32),
            "subcategory": np.zeros(2, dtype=np.float32),
            "title_entity": np.zeros(2, dtype=np.float32),
        },
        fallback_scores=None,
        has_cold_candidate=True,
        is_cold_user=True,
    )
    title_only = {name: 0.0 for name in FEATURE_NAMES}
    title_only["title"] = 1.0
    abstract_only = {name: 0.0 for name in FEATURE_NAMES}
    abstract_only["abstract"] = 1.0

    title_metrics = evaluate_cached_weights([impression], title_only, top_k=1)
    abstract_metrics = evaluate_cached_weights([impression], abstract_only, top_k=1)

    assert title_metrics["NDCG@1"] == pytest.approx(0.0)
    assert abstract_metrics["NDCG@1"] == pytest.approx(1.0)
    assert abstract_metrics["cold_item_NDCG@1"] == pytest.approx(1.0)
    assert abstract_metrics["cold_user_NDCG@1"] == pytest.approx(1.0)
