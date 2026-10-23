"""Kiểm thử slicing, beyond-accuracy và paired bootstrap cho P3."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.evaluation.bootstrap import BootstrapSettings, paired_bootstrap_comparison
from src.evaluation.item_slices import (
    HEAD,
    LONG_TAIL,
    ZERO_CLICK,
    build_item_popularity_groups,
)
from src.evaluation.p3_evaluation import evaluate_p3_impressions
from src.models.baselines.popularity import PopularityRecommender
from src.reranking.pipeline import RerankingPipeline


def _write_dataset(directory: Path, rows: list[dict]) -> None:
    """Ghi một dataset Parquet nhỏ cho unit test."""
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-00000.parquet")


def _prepare_p3_fixture(tmp_path: Path) -> tuple[Path, PopularityRecommender, RerankingPipeline]:
    """Tạo dữ liệu nhỏ có Head, Long-tail, zero-click và catalog-cold item."""
    root = tmp_path / "processed"
    _write_dataset(root / "mappings/items", [{"item_idx": item} for item in range(5)])
    _write_dataset(
        root / "train/news",
        [
            {"item_idx": 0, "category": "sports"},
            {"item_idx": 1, "category": "sports"},
            {"item_idx": 2, "category": "tech"},
            {"item_idx": 3, "category": "finance"},
        ],
    )
    _write_dataset(root / "dev/news", [{"item_idx": 4, "category": "health"}])
    _write_dataset(
        root / "train/positive_interactions",
        [
            {"item_idx": 0, "label": 1},
            {"item_idx": 0, "label": 1},
            {"item_idx": 0, "label": 1},
            {"item_idx": 1, "label": 1},
            {"item_idx": 2, "label": 1},
            {"item_idx": 3, "label": 0},
        ],
    )
    _write_dataset(
        root / "dev/impressions",
        [
            {
                "impression_id": 1,
                "user_idx": 10,
                "history_item_idxs": [0, 1],
                "candidate_item_idxs": [0, 1, 2, 4],
                "labels": [0, 0, 1, 0],
                "candidate_seen_in_train_mask": [True, True, True, False],
            },
            {
                "impression_id": 2,
                "user_idx": 11,
                "history_item_idxs": [0, 1, 2, 3, 4, 0],
                "candidate_item_idxs": [0, 1, 2, 4],
                "labels": [0, 0, 0, 1],
                "candidate_seen_in_train_mask": [True, True, True, False],
            },
        ],
    )

    model = PopularityRecommender()
    model.fit(
        pd.DataFrame(
            {
                "user_idx": [1, 1, 1, 1, 1],
                "item_idx": [0, 0, 0, 1, 2],
            }
        )
    )
    config = {
        "dataset": {
            "processed_root": str(root),
            "items_mapping": str(root / "mappings/items"),
        },
        "reranking": {
            "enabled": True,
            "method": "popularity_diversity",
            "popularity_train_path": str(root / "train/positive_interactions"),
            "news_paths": [str(root / "train/news"), str(root / "dev/news")],
            "top_k": 3,
            "weights": {
                "relevance": 0.2,
                "diversity": 0.7,
                "long_tail": 0.1,
            },
        },
    }
    pipeline = RerankingPipeline.from_config(config)
    return root, model, pipeline


def test_item_popularity_groups_keep_zero_click_separate() -> None:
    """Head/Long-tail chỉ xét item có click; zero-click không bị gộp vào tail."""
    groups = build_item_popularity_groups(np.array([10, 5, 1, 0, 0]), head_fraction=0.4)
    assert groups.group_codes.tolist() == [HEAD, HEAD, LONG_TAIL, ZERO_CLICK, ZERO_CLICK]
    assert groups.head_items == 2
    assert groups.long_tail_items == 1
    assert groups.zero_click_items == 2


def test_paired_bootstrap_zero_delta_has_zero_interval() -> None:
    """Nếu baseline = P3 thì paired delta và CI phải đúng bằng 0."""
    users = np.array([1, 1, 2, 3], dtype=np.int64)
    values = np.array([[1.0], [0.5], [0.0], [1.0]], dtype=np.float64)
    settings = BootstrapSettings(num_resamples=100, unit="user", seed=7, batch_size=10)
    result = paired_bootstrap_comparison(users, values, values, ["NDCG@10"], settings)
    assert result["NDCG@10"]["delta"] == 0.0
    assert result["NDCG@10"]["delta_ci"] == [0.0, 0.0]


def test_p3_evaluation_outputs_before_after_slices_and_bootstrap(tmp_path: Path) -> None:
    """Một lần evaluation phải trả baseline, P3, item slices và paired CI."""
    root, model, pipeline = _prepare_p3_fixture(tmp_path)
    settings = BootstrapSettings(
        enabled=True,
        num_resamples=100,
        confidence_level=0.95,
        unit="user",
        seed=42,
        batch_size=10,
    )
    results = evaluate_p3_impressions(
        model=model,
        impressions_path=root / "dev/impressions",
        reranking_pipeline=pipeline,
        k_list=[1, 3, 10],
        all_catalog_items=set(range(5)),
        head_fraction=0.34,
        bootstrap_settings=settings,
    )

    assert results["evaluation_mode"] == "paired_baseline_vs_p3"
    assert results["total_evaluated_impressions"] == 2
    assert results["baseline"]["slices"]["overall"]["count"] == 2
    assert results["reranked"]["slices"]["overall"]["count"] == 2

    # Impression 1 relevant item 2 thuộc long-tail; impression 2 relevant item 4 zero-click + catalog-cold.
    assert results["baseline"]["slices"]["long_tail_item"]["count"] == 1
    assert results["baseline"]["slices"]["zero_click_item"]["count"] == 1
    assert results["baseline"]["slices"]["catalog_cold_item"]["count"] == 1

    beyond = results["reranked"]["beyond_accuracy"]
    assert "category_diversity@10" in beyond
    assert "long_tail_exposure@10" in beyond
    assert "zero_click_exposure@10" in beyond

    bootstrap = results["comparison"]["bootstrap"]
    assert bootstrap["num_resamples"] == 100
    assert "NDCG@10" in bootstrap["metrics"]
    assert "Diversity@10" in bootstrap["metrics"]
    assert "LongTailExposure@10" in bootstrap["metrics"]


def test_head_fraction_validation() -> None:
    """Ngưỡng Head không được bằng 0 hoặc 1."""
    with pytest.raises(ValueError):
        build_item_popularity_groups(np.array([1, 2]), head_fraction=0.0)
