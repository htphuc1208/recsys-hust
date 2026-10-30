"""Test mode 6 đặc trưng và tương thích với mode 5 đặc trưng."""

import numpy as np
import pandas as pd
import pytest

from src.models.content_based.recommender import ContentBasedRecommender
from src.models.content_based.weight_sweep import (
    FEATURE_NAMES,
    FEATURE_NAMES_WITH_ABSTRACT_ENTITY,
    CachedContentImpression,
    cache_feature_scores,
    combine_feature_scores,
    evaluate_cached_weights,
    generate_weight_grid,
)
from tests.unit.test_feature_store_tools import create_store


def test_six_feature_grid_local_bounds():
    bounds = {
        "title": (0.65, 0.70),
        "abstract": (0.0, 0.0),
        "category": (0.15, 0.15),
        "subcategory": (0.15, 0.15),
        "title_entity": (0.0, 0.0),
        "abstract_entity": (0.0, 0.05),
    }
    result = generate_weight_grid(0.01, bounds, feature_names=FEATURE_NAMES_WITH_ABSTRACT_ENTITY)
    assert len(result) == 6
    assert result[0]["abstract_entity"] == pytest.approx(0.05)
    assert result[-1]["abstract_entity"] == pytest.approx(0.0)
    for weights in result:
        assert sum(weights.values()) == pytest.approx(1.0)


def test_six_feature_score_changes_ranking():
    names = FEATURE_NAMES_WITH_ABSTRACT_ENTITY
    data = {name: np.zeros(2, dtype=np.float32) for name in names}
    data["title"] = np.array([1, 0], dtype=np.float32)
    data["abstract_entity"] = np.array([0, 1], dtype=np.float32)
    impression = CachedContentImpression(
        impression_id=1,
        candidate_item_idxs=np.array([10, 20]),
        true_items={20},
        feature_scores=data,
        fallback_scores=None,
        has_cold_candidate=True,
        is_cold_user=False,
    )
    title_only = dict.fromkeys(names, 0.0)
    title_only["title"] = 1.0
    abstract_only = dict.fromkeys(names, 0.0)
    abstract_only["abstract_entity"] = 1.0
    assert combine_feature_scores(impression, title_only, names).tolist() == [1.0, 0.0]
    assert combine_feature_scores(impression, abstract_only, names).tolist() == [0.0, 1.0]
    assert evaluate_cached_weights([impression], title_only, 1, names)["NDCG@1"] == 0
    assert evaluate_cached_weights([impression], abstract_only, 1, names)[
        "NDCG@1"
    ] == pytest.approx(1)


def test_six_feature_cache_loads_optional_with_zero_weight(tmp_path):
    store = create_store(tmp_path)
    model = ContentBasedRecommender(feature_dir=str(store), feature_weights={"title": 1.0})
    model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))
    assert "abstract_entity" not in model._features
    rows = [
        {
            "impression_id": 1,
            "user_idx": 0,
            "history_item_idxs": [0],
            "candidate_item_idxs": [0, 1],
            "labels": [0, 1],
            "candidate_seen_in_train_mask": [True, False],
        }
    ]
    cached = cache_feature_scores(model, rows, feature_names=FEATURE_NAMES_WITH_ABSTRACT_ENTITY)
    assert "abstract_entity" in model._features
    assert cached[0].feature_scores["abstract_entity"].shape == (2,)
    assert set(cached[0].feature_scores) == set(FEATURE_NAMES_WITH_ABSTRACT_ENTITY)
    assert len(generate_weight_grid(0.2, feature_names=FEATURE_NAMES)) == 126


def test_large_grid_protected():
    with pytest.raises(ValueError, match="Lưới vượt"):
        generate_weight_grid(0.01, feature_names=FEATURE_NAMES_WITH_ABSTRACT_ENTITY)


def test_run_sweep_generates_six_feature_candidate_yaml(tmp_path, monkeypatch):
    """Tích hợp mini không cần dữ liệu Parquet: xuất cả CSV, JSON và YAML."""
    import json
    from pathlib import Path

    import yaml

    from src.models.content_based import weight_sweep

    store = create_store(tmp_path)
    model = ContentBasedRecommender(
        feature_dir=str(store),
        feature_weights={"title": 0.7, "category": 0.15, "subcategory": 0.15},
    )
    model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))
    checkpoint = tmp_path / "baseline.pkl"
    model.save(checkpoint)
    config = {
        "experiment_name": "test_sweep",
        "dataset": {"dev_impressions": str(tmp_path / "dev")},
        "model": {
            "name": "content_based",
            "feature_dir": str(store),
            "feature_weights": {"title": 0.7, "category": 0.15, "subcategory": 0.15},
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(
        weight_sweep,
        "sample_dev_rows",
        lambda **kwargs: (
            [
                {
                    "impression_id": 1,
                    "user_idx": 0,
                    "history_item_idxs": [0],
                    "candidate_item_idxs": [0, 1],
                    "labels": [0, 1],
                    "candidate_seen_in_train_mask": [True, False],
                }
            ],
            1,
        ),
    )
    monkeypatch.chdir(tmp_path)
    bounds = {
        "title": (0.6, 0.7),
        "abstract": (0.0, 0.0),
        "category": (0.1, 0.2),
        "subcategory": (0.1, 0.2),
        "title_entity": (0.0, 0.0),
        "abstract_entity": (0.0, 0.1),
    }
    settings = weight_sweep.SweepSettings(
        step=0.1,
        sample_size=1,
        include_abstract_entity=True,
        weight_bounds=bounds,
        output_tag="local",
    )
    result = weight_sweep.run_weight_sweep(config_path, settings, checkpoint)
    assert "abstract_entity_weight" in result.columns
    output = tmp_path / "reports" / "content_tuning" / "test_sweep" / "local"
    candidate = yaml.safe_load((output / "best_config_candidate.yaml").read_text(encoding="utf-8"))
    assert candidate["model"]["feature_weights"]["abstract_entity"] == float(
        result.iloc[0]["abstract_entity_weight"]
    )
    assert sum(candidate["model"]["feature_weights"].values()) == pytest.approx(1.0)
    summary = json.loads((output / "weight_sweep_summary.json").read_text(encoding="utf-8"))
    assert summary["include_abstract_entity"] is True
    assert Path(summary["candidate_config"]).exists()
