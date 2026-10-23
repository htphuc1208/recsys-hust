from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml

from src.evaluate import evaluate_impressions
from src.evaluation.bootstrap import BootstrapSettings
from src.evaluation.data_minimization import (
    DatasetPaths,
    build_layout,
    build_model_config,
    build_summary,
    prepare_retention_datasets,
)
from src.models.base import BaseRecommender


class HistoryCaptureRecommender(BaseRecommender):
    """Model test chỉ ghi lại history evaluator truyền vào predict()."""

    def __init__(self) -> None:
        super().__init__()
        self.histories: list[object] = []
        self.is_fitted = True

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        del train_df, val_df
        return self

    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, **kwargs):
        del user_ids
        self.histories.append(kwargs.get("history_item_idxs"))
        return np.arange(len(item_ids), 0, -1, dtype=float)

    def recommend(self, user_ids: np.ndarray, top_k: int = 10, filter_seen: bool = True):
        del top_k, filter_seen
        return {int(user_id): [] for user_id in user_ids}


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path / "part-00000.parquet")


def _make_source_data(project_root: Path) -> None:
    dates = [
        datetime(2019, 11, 9, 12),
        datetime(2019, 11, 10, 12),
        datetime(2019, 11, 11, 12),
        datetime(2019, 11, 12, 12),
        datetime(2019, 11, 13, 12),
        datetime(2019, 11, 14, 12),
    ]
    positive_rows = [
        {
            "user_idx": index,
            "item_idx": index,
            "label": 1,
            "timestamp": timestamp,
        }
        for index, timestamp in enumerate(dates)
    ]
    pointwise_rows = []
    for index, timestamp in enumerate(dates):
        pointwise_rows.extend(
            [
                {
                    "impression_id": index,
                    "user_idx": index,
                    "item_idx": index,
                    "label": 1,
                    "timestamp": timestamp,
                },
                {
                    "impression_id": index,
                    "user_idx": index,
                    "item_idx": index + 100,
                    "label": 0,
                    "timestamp": timestamp,
                },
            ]
        )

    _write_parquet(
        project_root / "data/processed/mind_large/train/positive_interactions",
        positive_rows,
    )
    _write_parquet(
        project_root / "data/processed/mind_large/train/pointwise",
        pointwise_rows,
    )


def _master_config() -> dict:
    return {
        "source": {
            "pointwise": "data/processed/mind_large/train/pointwise",
            "positive_interactions": "data/processed/mind_large/train/positive_interactions",
        },
        "retention": {"timestamp_col": "timestamp", "cutoff_timestamp": "auto"},
        "windows": [
            {"name": "1d", "days": 1},
            {"name": "3d", "days": 3},
            {"name": "5d", "days": 5},
            {"name": "all", "days": None},
        ],
        "prepare": {"rows_per_file": 2, "compression": "snappy"},
        "paths": {
            "data_dir": "data/experiments/data_minimization",
            "reports_dir": "reports/data_minimization",
            "artifacts_dir": "artifacts/data_minimization",
        },
    }


def test_prepare_retention_windows_keeps_expected_recent_days(tmp_path: Path) -> None:
    _make_source_data(tmp_path)
    config = _master_config()
    layout = build_layout(tmp_path, config)

    paths, stats, cutoff = prepare_retention_datasets(
        project_root=tmp_path,
        config=config,
        layout=layout,
        selected_windows=None,
        overwrite=False,
    )

    assert cutoff == datetime(2019, 11, 14, 12, 0, 1)
    counts = stats.set_index("window")

    # Cutoff tự động là max timestamp + 1 giây, vì vậy 1d chỉ giữ ngày 14.
    assert int(counts.loc["1d", "positive_rows"]) == 1
    assert int(counts.loc["3d", "positive_rows"]) == 3
    assert int(counts.loc["5d", "positive_rows"]) == 5
    assert int(counts.loc["all", "positive_rows"]) == 6

    assert int(ds.dataset(str(paths["1d"].pointwise), format="parquet").count_rows()) == 2
    assert paths["all"].pointwise == (tmp_path / "data/experiments/data_minimization/all/pointwise")
    assert paths["all"].pointwise != (tmp_path / "data/processed/mind_large/train/pointwise")
    assert int(ds.dataset(str(paths["all"].pointwise), format="parquet").count_rows()) == 12
    assert float(counts.loc["all", "pointwise_storage_reduction_pct"]) == 0.0
    assert float(counts.loc["all", "positive_storage_reduction_pct"]) == 0.0
    assert bool(counts.loc["all", "is_materialized_copy"]) is True
    assert counts.loc["all", "storage_baseline"] == "normalized_all"


def test_build_content_config_disables_full_dev_history(tmp_path: Path) -> None:
    base_config_dir = tmp_path / "configs/experiment"
    base_config_dir.mkdir(parents=True, exist_ok=True)

    bpr_base = {
        "experiment_name": "exp04",
        "dataset": {
            "train_interactions": "old",
            "dev_impressions": "data/dev",
            "items_mapping": "data/items",
        },
        "model": {"name": "bpr"},
        "evaluation": {},
    }
    content_base = {
        "experiment_name": "exp05",
        "dataset": {
            "train_interactions": "old",
            "dev_impressions": "data/dev",
            "items_mapping": "data/items",
        },
        "model": {"name": "content_based", "cold_user_strategy": "zero"},
        "evaluation": {},
    }
    with (base_config_dir / "exp04.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(bpr_base, file)
    with (base_config_dir / "exp05.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(content_base, file)

    master = _master_config()
    master["models"] = {
        "bpr": {"base_config": "configs/experiment/exp04.yaml"},
        "content_based": {
            "base_config": "configs/experiment/exp05.yaml",
            "use_impression_history": False,
        },
    }
    layout = build_layout(tmp_path, master)
    dataset_paths = DatasetPaths(
        pointwise=tmp_path / "dm/pointwise",
        positive_interactions=tmp_path / "dm/positive",
    )

    content_config, _ = build_model_config(
        project_root=tmp_path,
        master_config=master,
        layout=layout,
        model_key="content_based",
        window_name="1d",
        dataset_paths=dataset_paths,
    )

    assert content_config["evaluation"]["use_impression_history"] is False
    assert content_config["dataset"]["train_interactions"] == "dm/positive"
    assert content_config["experiment_name"] == "dm_content_1d"


def test_evaluator_can_hide_impression_history(tmp_path: Path) -> None:
    impressions_path = tmp_path / "impressions"
    rows = [
        {
            "impression_id": 1,
            "user_idx": 7,
            "history_item_idxs": [2, 3, 4],
            "candidate_item_idxs": [10, 11],
            "labels": [1, 0],
            "candidate_seen_in_train_mask": [True, True],
        }
    ]
    _write_parquet(impressions_path, rows)

    model = HistoryCaptureRecommender()
    evaluate_impressions(
        model=model,
        impressions_path=impressions_path,
        k_list=[10],
        bootstrap_settings=BootstrapSettings(enabled=False),
        use_impression_history=False,
    )
    assert model.histories == [None]

    model_with_history = HistoryCaptureRecommender()
    evaluate_impressions(
        model=model_with_history,
        impressions_path=impressions_path,
        k_list=[10],
        bootstrap_settings=BootstrapSettings(enabled=False),
        use_impression_history=True,
    )
    assert model_with_history.histories == [[2, 3, 4]]


def test_build_summary_uses_model_specific_storage_and_rows(tmp_path: Path) -> None:
    config = _master_config()
    layout = build_layout(tmp_path, config)
    stats = pd.DataFrame(
        [
            {
                "window": "1d",
                "days": 1,
                "pointwise_rows": 100,
                "positive_rows": 20,
                "pointwise_retention_pct": 25.0,
                "positive_retention_pct": 20.0,
                "pointwise_storage_mb": 10.0,
                "positive_storage_mb": 2.0,
                "pointwise_storage_reduction_pct": 75.0,
                "positive_storage_reduction_pct": 80.0,
            }
        ]
    )
    runs = [
        {
            "model": "bpr",
            "window": "1d",
            "train_wall_seconds": 10.0,
            "eval_wall_seconds": 3.0,
            "NDCG@10": 0.3,
            "Recall@10": 0.5,
            "MRR@10": 0.2,
            "checkpoint_path": "bpr.pkl",
            "metrics_path": "bpr.json",
        },
        {
            "model": "content_based",
            "window": "1d",
            "train_wall_seconds": 1.0,
            "eval_wall_seconds": 4.0,
            "NDCG@10": 0.4,
            "Recall@10": 0.6,
            "MRR@10": 0.3,
            "checkpoint_path": "content.pkl",
            "metrics_path": "content.json",
        },
    ]

    summary = build_summary(stats, runs, layout).set_index("model")
    assert int(summary.loc["bpr", "train_rows"]) == 100
    assert float(summary.loc["bpr", "training_storage_mb"]) == 10.0
    assert int(summary.loc["content_based", "train_rows"]) == 20
    assert float(summary.loc["content_based", "training_storage_mb"]) == 2.0
