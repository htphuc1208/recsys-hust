"""Kiem thu entity coverage voi dataset gia lap nho, khong can MIND Large."""

from pathlib import Path

import numpy as np

from src.features import entity_coverage


def test_audit_coverage_uses_first_seen_and_train_priority(tmp_path, monkeypatch):
    processed_root = tmp_path / "processed"
    for split in ("train", "dev", "test"):
        (processed_root / split / "news").mkdir(parents=True)

    rows_mapping = [
        {"news_id": "N-A", "item_idx": 0, "first_seen_split": "train"},
        {"news_id": "N-B", "item_idx": 1, "first_seen_split": "dev"},
        {"news_id": "N-C", "item_idx": 2, "first_seen_split": "test"},
        {"news_id": "N-D", "item_idx": 3, "first_seen_split": "train"},
    ]

    def article(news_id, item_idx, title, abstract):
        return {
            "news_id": news_id,
            "item_idx": item_idx,
            "title_entity_ids": title,
            "abstract_entity_ids": abstract,
        }

    data = {
        "train": [article("N-A", 0, ["Q1", "Q1"], ["Q2"]), article("N-D", 3, [], [])],
        "dev": [article("N-B", 1, ["Q3"], ["Q1"]), article("N-A", 0, ["Q3"], [])],
        "test": [article("N-C", 2, ["Q2"], ["Q4"])],
    }

    def fake_iter(directory: Path, columns):
        if directory.name == "items":
            yield from rows_mapping
        else:
            yield from data[directory.parent.name]

    monkeypatch.setattr(entity_coverage, "iter_parquet_rows", fake_iter)
    # load_item_mapping duoc import tu news_content, nen patch tai module goc.
    from src.features import news_content

    monkeypatch.setattr(news_content, "iter_parquet_rows", fake_iter)

    vec_path = tmp_path / "entity_embedding.vec"
    vec_path.write_text(
        "Q1 "
        + " ".join(["1"] + ["0"] * 99)
        + "\n"
        + "Q2 "
        + " ".join(["0", "1"] + ["0"] * 98)
        + "\n",
        encoding="utf-8",
    )

    result = entity_coverage.analyze_coverage(processed_root, vec_path)
    assert result["metadata"]["total_mapping_items"] == 4
    assert result["metadata"]["items_without_metadata"] == 0
    assert result["metadata"]["unique_entity_ids_needed"] == 4
    assert result["metadata"]["unique_entity_ids_found"] == 2

    rows = {(row["split"], row["field"]): row for row in result["rows"]}
    assert rows[("overall", "title")]["articles"] == 4
    assert rows[("overall", "title")]["articles_with_entity_ids"] == 3
    assert rows[("overall", "title")]["articles_with_valid_embedding"] == 2
    assert rows[("overall", "abstract")]["articles_with_valid_embedding"] == 2
    assert rows[("train", "title")]["articles_with_valid_embedding"] == 1
    assert rows[("dev", "title")]["articles_with_valid_embedding"] == 0
    assert rows[("test", "title")]["articles_with_valid_embedding"] == 1
    complementarity = {row["split"]: row for row in result["complementarity_rows"]}
    assert complementarity["overall"]["both"] == 1
    assert complementarity["overall"]["title_only"] == 1
    assert complementarity["overall"]["abstract_only"] == 1
    assert complementarity["overall"]["neither"] == 1
    assert complementarity["overall"]["combined_coverage_pct"] == 75.0
    assert complementarity["overall"]["gain_over_title_pp"] == 25.0
    # Q1 duoc lap trong bai N-A nhung chi tinh mot mention.
    assert rows[("overall", "title")]["entity_mentions"] == 3

    csv_path = tmp_path / "report.csv"
    json_path = tmp_path / "report.json"
    entity_coverage.save_report(result, csv_path, json_path)
    assert csv_path.exists() and json_path.exists()
    assert "abstract" in csv_path.read_text(encoding="utf-8-sig")
    complementarity_path = csv_path.parent / "news_entity_complementarity.csv"
    assert complementarity_path.exists()
    assert "abstract_only" in complementarity_path.read_text(encoding="utf-8-sig")


def test_opposite_entity_vectors_produce_zero_mean():
    vectors = {
        "Q1": np.asarray([1.0, 0.0], dtype=np.float32),
        "Q2": np.asarray([-1.0, 0.0], dtype=np.float32),
    }
    assert not entity_coverage.is_valid_mean_vector(("Q1", "Q2"), vectors)
    assert entity_coverage.is_valid_mean_vector(("Q1",), vectors)
    assert not entity_coverage.is_valid_mean_vector((), vectors)
