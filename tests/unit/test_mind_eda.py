from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.analysis.mind_eda import Reservoir, analyze_split


def test_reservoir_is_bounded_and_deterministic() -> None:
    first = Reservoir(size=10, seed=7)
    second = Reservoir(size=10, seed=7)

    first.add_many(range(100))
    second.add_many(range(100))

    assert first.values == second.values
    assert len(first.values) == 10
    assert first.seen == 100


def test_analyze_split_summarizes_news_and_behaviors(tmp_path: Path) -> None:
    split_root = tmp_path / "train"
    news_root = split_root / "news"
    behavior_root = split_root / "behaviors"
    news_root.mkdir(parents=True)
    behavior_root.mkdir()

    news = pa.Table.from_pylist(
        [
            {
                "news_id": "N1",
                "category": "sports",
                "title": "First title",
                "abstract": "First abstract",
                "title_entity_ids": ["Q1"],
                "abstract_entity_ids": [],
            },
            {
                "news_id": "N2",
                "category": "news",
                "title": "Second title",
                "abstract": "",
                "title_entity_ids": [],
                "abstract_entity_ids": ["Q2", "Q3"],
            },
        ]
    )
    behaviors = pa.Table.from_pylist(
        [
            {
                "user_id": "U1",
                "timestamp": datetime(2019, 11, 9, 12),
                "labels": [1, 0],
                "history_length": 0,
                "candidate_count": 2,
            },
            {
                "user_id": "U1",
                "timestamp": datetime(2019, 11, 10, 12),
                "labels": [0, 0, 1],
                "history_length": 4,
                "candidate_count": 3,
            },
        ]
    )
    pq.write_table(news, news_root / "part-00000.parquet")
    pq.write_table(behaviors, behavior_root / "part-00000.parquet")

    report, _ = analyze_split(tmp_path, "train", sample_size=100)

    assert report["news"]["rows"] == 2
    assert report["news"]["missing_abstract_pct"] == 50.0
    assert report["behaviors"]["unique_users"] == 1
    assert report["behaviors"]["history_length"]["mean"] == 2.0
    assert report["behaviors"]["candidate_count"]["max"] == 3
    assert report["behaviors"]["click_through_rate"] == 0.4
    assert report["behaviors"]["quantiles_approximate"] is False
