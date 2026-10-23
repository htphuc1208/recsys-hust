from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.models.bpr import BPRRecommender
from src.models.bpr.data import scan_bpr_training_data
from src.train import train


def _write_pointwise(path: Path, rows: list[dict]) -> None:
    """Ghi pointwise nhỏ phục vụ test streaming BPR."""
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                ("impression_id", pa.int64()),
                ("user_idx", pa.int64()),
                ("item_idx", pa.int64()),
                ("label", pa.int8()),
            ]
        ),
    )
    pq.write_table(table, path)


def _build_tiny_pointwise(data_dir: Path) -> None:
    """Tạo hai part có một impression đi qua ranh giới file."""
    data_dir.mkdir(parents=True, exist_ok=True)

    _write_pointwise(
        data_dir / "part-00000.parquet",
        [
            {"impression_id": 100, "user_idx": 0, "item_idx": 10, "label": 1},
            {"impression_id": 100, "user_idx": 0, "item_idx": 11, "label": 0},
        ],
    )
    _write_pointwise(
        data_dir / "part-00001.parquet",
        [
            {"impression_id": 100, "user_idx": 0, "item_idx": 12, "label": 0},
            {"impression_id": 101, "user_idx": 1, "item_idx": 20, "label": 1},
            {"impression_id": 101, "user_idx": 1, "item_idx": 21, "label": 0},
        ],
    )


def test_scan_bpr_training_data_uses_positive_only_for_fallback_and_seen(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "pointwise"
    _build_tiny_pointwise(data_dir)

    summary = scan_bpr_training_data(
        data_dir,
        read_batch_size=2,
        track_user_seen=True,
    )

    assert summary.pointwise_rows == 5
    assert summary.pair_count == 3
    assert summary.positive_rows == 2
    assert summary.negative_rows == 3
    assert summary.num_observed_users == 2
    assert summary.num_observed_items == 5
    assert summary.user_seen_items == {
        0: {10},
        1: {20},
    }
    assert set(summary.popular_items) == {10, 20}


def test_bpr_fit_from_path_trains_all_pairs_and_masks_unseen_items(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "pointwise"
    _build_tiny_pointwise(data_dir)

    model = BPRRecommender(
        embedding_dim=4,
        epochs=1,
        batch_size=2,
        learning_rate=0.01,
        seed=42,
        device="cpu",
        shuffle_buffer_size=2,
        read_batch_size=2,
    )
    model.fit_from_path(data_dir)

    assert model.is_fitted
    assert model.user_seen_items == {
        0: {10},
        1: {20},
    }
    assert model.observed_item_mask[10]
    assert model.observed_item_mask[11]
    assert model.observed_item_mask[12]
    assert model.observed_item_mask[20]
    assert model.observed_item_mask[21]

    scores = model.predict(
        user_ids=np.array([0]),
        item_ids=np.array([10, 11, 19, 99]),
    )

    assert len(scores) == 4
    assert scores[2] == 0.0
    assert scores[3] == 0.0


def test_generic_train_uses_bpr_streaming_path(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "pointwise"
    _build_tiny_pointwise(data_dir)

    artifacts_dir = tmp_path / "artifacts"
    config_path = tmp_path / "bpr.yaml"
    config = {
        "experiment_name": "tiny_bpr",
        "seed": 42,
        "dataset": {
            "train_interactions": str(data_dir),
        },
        "model": {
            "name": "bpr",
            "embedding_dim": 4,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "epochs": 1,
            "batch_size": 2,
            "use_item_bias": True,
            "max_train_samples": None,
            "shuffle_buffer_size": 2,
            "read_batch_size": 2,
            "track_user_seen": True,
            "device": "cpu",
        },
        "paths": {
            "artifacts_dir": str(artifacts_dir),
        },
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    checkpoint_path = train(config_path)

    assert checkpoint_path.is_file()
    loaded = BPRRecommender.load(checkpoint_path)
    assert loaded.is_fitted
    assert loaded.user_seen_items[0] == {10}
    assert loaded.user_seen_items[1] == {20}
