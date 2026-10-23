"""Kiểm thử tích hợp P3 với Parquet và evaluation khi có pyarrow."""

import json

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.evaluate import evaluate_impressions
from src.models.baselines.popularity import PopularityRecommender
from src.reranking.pipeline import RerankingPipeline, load_item_catalog, load_train_click_counts
from src.reranking.run_pipeline import run_preview


def write_dataset(directory, rows):
    """Tạo một Parquet nhỏ cho kiểm thử từ đầu đến cuối."""
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-00000.parquet")


def prepare_paths(tmp_path):
    """Tạo catalog và train có click âm/dương, item mới chỉ có ở dev."""
    data_root = tmp_path / "processed"
    write_dataset(data_root / "mappings/items", [{"item_idx": x} for x in range(4)])
    write_dataset(data_root / "train/news", [
        {"item_idx": 0, "category": "sports"},
        {"item_idx": 1, "category": "sports"},
        {"item_idx": 2, "category": "technology"},
    ])
    write_dataset(data_root / "dev/news", [{"item_idx": 3, "category": "finance"}])
    write_dataset(data_root / "train/positive_interactions", [
        {"item_idx": 0, "label": 1},
        {"item_idx": 0, "label": 1},
        {"item_idx": 1, "label": 1},
        {"item_idx": 2, "label": 0},
    ])
    write_dataset(data_root / "dev/impressions", [
        {
            "impression_id": 1,
            "user_idx": 1,
            "history_item_idxs": [0, 1],
            "candidate_item_idxs": [0, 1, 2, 3],
            "labels": [0, 0, 1, 0],
            "candidate_seen_in_train_mask": [True, True, True, False],
        }
    ])
    return data_root


def test_train_only_popularity_and_dev_metadata(tmp_path):
    """Không dùng label dev; negative label train không tính là click."""
    root = prepare_paths(tmp_path)
    categories = load_item_catalog(root / "mappings/items", [root / "train/news", root / "dev/news"])
    counts = load_train_click_counts(root / "train/positive_interactions", len(categories))
    assert categories == ["sports", "sports", "technology", "finance"]
    assert counts.tolist() == [2, 1, 0, 0]


def test_preview_and_evaluate_integration(tmp_path):
    """Cùng thứ tự P3 được sử dụng ở preview và evaluate, không đổi baseline."""
    root = prepare_paths(tmp_path)
    model = PopularityRecommender()
    model.fit(pd.DataFrame({"item_idx": [0, 0, 1], "user_idx": [1, 1, 1]}))
    checkpoint = tmp_path / "model.pkl"
    model.save(checkpoint)
    config = {
        "experiment_name": "p3_test",
        "dataset": {
            "processed_root": str(root),
            "items_mapping": str(root / "mappings/items"),
            "dev_impressions": str(root / "dev/impressions"),
        },
        "reranking": {
            "enabled": True,
            "base_checkpoint": str(checkpoint),
            "top_k": 3,
            "weights": {"relevance": 0.2, "diversity": 0.8, "long_tail": 0.0},
        },
    }
    import yaml
    config_path = tmp_path / "p3.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    preview_path = run_preview(config_path, sample_size=1, output=tmp_path / "preview.jsonl")
    preview = json.loads(preview_path.read_text(encoding="utf-8").strip())
    assert preview["baseline_ranking"] == [0, 1, 2, 3]
    assert preview["reranked_ranking"][:3] == [0, 2, 3]
    assert "labels" not in preview

    pipeline = RerankingPipeline.from_config(config)
    baseline_result = evaluate_impressions(model, root / "dev/impressions", k_list=[1, 10])
    reranked_result = evaluate_impressions(model, root / "dev/impressions", k_list=[1, 10], reranking_pipeline=pipeline)
    assert baseline_result["slices"]["overall"]["metrics"]["NDCG@10"] < reranked_result["slices"]["overall"]["metrics"]["NDCG@10"]
