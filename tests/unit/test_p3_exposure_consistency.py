"""Kiểm tra aggregate Long-tail Exposure khớp với per-impression mean dùng cho CI."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from src.evaluation.bootstrap import BootstrapSettings
from src.evaluation.p3_evaluation import evaluate_p3_impressions
from src.models.baselines.popularity import PopularityRecommender
from src.reranking.pipeline import RerankingPipeline


def _write_dataset(directory: Path, rows: list[dict]) -> None:
    """Ghi Parquet nhỏ cho test."""
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "part-00000.parquet")


def test_long_tail_aggregate_matches_bootstrap_mean(tmp_path: Path) -> None:
    """Aggregate và bootstrap phải dùng cùng mean theo impression."""
    root = tmp_path / "processed"
    _write_dataset(root / "mappings/items", [{"item_idx": item} for item in range(4)])
    _write_dataset(
        root / "train/news",
        [
            {"item_idx": 0, "category": "sports"},
            {"item_idx": 1, "category": "news"},
            {"item_idx": 2, "category": "tech"},
            {"item_idx": 3, "category": "finance"},
        ],
    )
    _write_dataset(
        root / "train/positive_interactions",
        [
            {"item_idx": 0, "label": 1},
            {"item_idx": 0, "label": 1},
            {"item_idx": 1, "label": 1},
        ],
    )
    # Hai impression có số candidate khác nhau để bắt được lỗi weighted-by-position trước đây.
    _write_dataset(
        root / "dev/impressions",
        [
            {
                "impression_id": 1,
                "user_idx": 10,
                "history_item_idxs": [0],
                "candidate_item_idxs": [0, 1],
                "labels": [1, 0],
                "candidate_seen_in_train_mask": [True, True],
            },
            {
                "impression_id": 2,
                "user_idx": 11,
                "history_item_idxs": [0],
                "candidate_item_idxs": [0, 1, 2, 3],
                "labels": [0, 1, 0, 0],
                "candidate_seen_in_train_mask": [True, True, True, True],
            },
        ],
    )

    model = PopularityRecommender()
    model.fit(pd.DataFrame({"user_idx": [1, 1, 1], "item_idx": [0, 0, 1]}))
    config = {
        "dataset": {
            "processed_root": str(root),
            "items_mapping": str(root / "mappings/items"),
        },
        "reranking": {
            "enabled": True,
            "method": "popularity_diversity",
            "popularity_train_path": str(root / "train/positive_interactions"),
            "news_paths": [str(root / "train/news")],
            "top_k": 10,
            "weights": {"relevance": 0.7, "diversity": 0.2, "long_tail": 0.1},
        },
    }
    pipeline = RerankingPipeline.from_config(config)
    settings = BootstrapSettings(num_resamples=50, unit="impression", seed=42, batch_size=10)

    results = evaluate_p3_impressions(
        model=model,
        impressions_path=root / "dev/impressions",
        reranking_pipeline=pipeline,
        k_list=[10],
        all_catalog_items=set(range(4)),
        head_fraction=0.5,
        bootstrap_settings=settings,
    )

    baseline_aggregate = results["baseline"]["beyond_accuracy"]["long_tail_exposure@10"]
    baseline_boot_mean = results["comparison"]["bootstrap"]["metrics"]["LongTailExposure@10"][
        "baseline_mean"
    ]
    reranked_aggregate = results["reranked"]["beyond_accuracy"]["long_tail_exposure@10"]
    reranked_boot_mean = results["comparison"]["bootstrap"]["metrics"]["LongTailExposure@10"][
        "reranked_mean"
    ]

    assert np.isclose(baseline_aggregate, baseline_boot_mean, atol=1e-5)
    assert np.isclose(reranked_aggregate, reranked_boot_mean, atol=1e-5)
