from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.features.prepare_mind import UNKNOWN_INDEX, prepare_dataset


def _write_split(
    root: Path,
    split: str,
    news_rows: list[dict],
    behavior_rows: list[dict],
) -> None:
    news_dir = root / split / "news"
    behaviors_dir = root / split / "behaviors"
    news_dir.mkdir(parents=True)
    behaviors_dir.mkdir()
    pq.write_table(pa.Table.from_pylist(news_rows), news_dir / "part-00000.parquet")
    pq.write_table(
        pa.Table.from_pylist(behavior_rows),
        behaviors_dir / "part-00000.parquet",
    )


def _news(news_id: str) -> dict:
    return {
        "news_id": news_id,
        "category": "news",
        "subcategory": "local",
        "title": f"Title {news_id}",
        "abstract": f"Abstract {news_id}",
        "url": f"https://example.com/{news_id}",
        "title_entity_ids": [],
        "abstract_entity_ids": [],
    }


def _behavior(
    impression_id: int,
    user_id: str,
    history: list[str],
    candidates: list[str],
    labels: list[int] | None,
) -> dict:
    return {
        "impression_id": impression_id,
        "user_id": user_id,
        "timestamp": datetime(2019, 11, 9, 12),
        "history": history,
        "candidate_news_ids": candidates,
        "labels": labels,
    }


def test_prepare_dataset_maps_all_news_but_marks_train_items(tmp_path: Path) -> None:
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    _write_split(
        interim,
        "train",
        [_news("N1"), _news("N2")],
        [_behavior(1, "U1", ["N1"], ["N1", "N2"], [1, 0])],
    )
    _write_split(
        interim,
        "dev",
        [_news("N2"), _news("N3")],
        [_behavior(2, "U2", ["N1", "N3"], ["N2", "N3"], [0, 1])],
    )
    _write_split(
        interim,
        "test",
        [_news("N3")],
        [_behavior(3, "U3", ["N1"], ["N3"], None)],
    )

    manifest = prepare_dataset(
        interim,
        processed,
        batch_size=2,
        chunk_size=2,
        negative_ratio=1,
    )

    assert manifest["num_train_users"] == 1
    assert manifest["num_train_items"] == 2
    assert manifest["num_items"] == 3
    assert manifest["splits"]["dev"]["unknown_user_rows"] == 1
    assert manifest["splits"]["dev"]["unknown_candidate_items"] == 0
    assert manifest["splits"]["dev"]["unseen_train_candidate_items"] == 1

    dev = pq.read_table(processed / "dev" / "impressions").to_pylist()[0]
    assert dev["user_idx"] == UNKNOWN_INDEX
    assert dev["history_item_idxs"][0] != UNKNOWN_INDEX
    assert dev["history_item_idxs"][1] != UNKNOWN_INDEX
    assert dev["candidate_known_mask"] == [True, True]
    assert dev["candidate_seen_in_train_mask"] == [True, False]

    item_mapping = pq.read_table(processed / "mappings" / "items").to_pylist()
    new_item = next(row for row in item_mapping if row["news_id"] == "N3")
    assert new_item["item_idx"] != UNKNOWN_INDEX
    assert new_item["seen_in_train"] is False
    assert new_item["first_seen_split"] == "dev"

    pointwise = pq.read_table(processed / "train" / "pointwise").to_pylist()
    assert sorted(row["label"] for row in pointwise) == [0, 1]
    assert not (processed / "test" / "positive_interactions").exists()


def test_prepare_dataset_refuses_to_replace_output_without_flag(tmp_path: Path) -> None:
    input_root = tmp_path / "interim"
    output_root = tmp_path / "processed"
    output_root.mkdir()

    try:
        prepare_dataset(input_root, output_root)
    except FileExistsError as exc:
        assert "--overwrite" in str(exc)
    else:
        raise AssertionError("Expected an existing output directory to be rejected")
