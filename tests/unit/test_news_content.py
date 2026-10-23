"""Kiểm thử các thành phần tạo đặc trưng B4 bằng dữ liệu nhỏ."""

import numpy as np
from scipy import sparse

from src.features.news_content import (
    NewsRecord,
    build_category_feature,
    build_text_feature,
    build_title_entity_feature,
    load_train_entity_embeddings,
)


def make_record(item_idx, title, category, entity_ids=()):
    """Tạo một bài viết tối giản để kiểm thử."""

    return NewsRecord(
        item_idx=item_idx,
        title=title,
        abstract=title,
        category=category,
        subcategory=category,
        title_entity_ids=tuple(entity_ids),
    )


def test_train_only_tfidf_and_new_item():
    """Từ chỉ xuất hiện ở dev không được bổ sung vào từ vựng train."""

    records = [
        make_record(0, "football team news", "sports"),
        make_record(1, "football match news", "sports"),
        make_record(2, "quasar football", "science"),
    ]
    matrix, vectorizer = build_text_feature(
        records, records[:2], "title", 100, 1, 2
    )
    assert matrix.shape[0] == 3
    assert sparse.isspmatrix_csr(matrix)
    assert "quasar" not in vectorizer.vocabulary_
    assert matrix[2].nnz > 0  # dev vẫn dùng từ football có trong train.


def test_unknown_category_is_zero():
    """Danh mục chỉ xuất hiện ở dev không tạo cột mới."""

    records = [
        make_record(0, "football team", "sports"),
        make_record(1, "sports update", "sports"),
        make_record(2, "science update", "science"),
    ]
    matrix, _ = build_category_feature(records, records[:2], "category")
    assert matrix.shape == (3, 1)
    assert matrix[2].nnz == 0


def test_entity_coverage_and_zero_fallback(tmp_path):
    """Thực thể thiếu vector phải cho phần đặc trưng bằng 0."""

    entity_path = tmp_path / "entity_embedding.vec"
    entity_path.write_text(
        "Q1 " + " ".join(["1"] + ["0"] * 99) + "\n",
        encoding="utf-8",
    )
    vectors = load_train_entity_embeddings(entity_path, {"Q1", "Q2"})
    records = [
        make_record(0, "first", "sports", ("Q1",)),
        make_record(1, "second", "sports", ("Q2",)),
        make_record(2, "third", "sports", ()),
    ]
    matrix, statistics = build_title_entity_feature(records, vectors)
    assert matrix.shape == (3, 100)
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0)
    assert not np.any(matrix[1])
    assert not np.any(matrix[2])
    assert statistics["articles_with_matched_entities"] == 1


def test_build_features_preserves_mapping_and_train_vocabulary(tmp_path, monkeypatch):
    """Kiểm tra luồng đầu-cuối với mapping gốc và bài dev/test mới."""

    from src.features import news_content

    mapping_rows = [
        {"news_id": "N-A", "item_idx": 0, "first_seen_split": "train"},
        {"news_id": "N-B", "item_idx": 1, "first_seen_split": "train"},
        {"news_id": "N-C", "item_idx": 2, "first_seen_split": "dev"},
        {"news_id": "N-D", "item_idx": 3, "first_seen_split": "test"},
    ]

    def news_row(news_id, item_idx, title, category, entity_ids=()):
        return {
            "news_id": news_id,
            "item_idx": item_idx,
            "title": title,
            "abstract": title,
            "category": category,
            "subcategory": category,
            "title_entity_ids": list(entity_ids),
        }

    rows_by_split = {
        "train": [
            news_row("N-A", 0, "football match update", "sports", ("Q1",)),
            news_row("N-B", 1, "football team update", "sports", ("Q1",)),
        ],
        "dev": [
            news_row("N-A", 0, "football match update", "sports", ("Q1",)),
            news_row("N-C", 2, "quasar football", "science", ("Q2",)),
        ],
        "test": [news_row("N-D", 3, "another football game", "sports")],
    }

    def fake_iter_parquet_rows(directory, columns):
        if directory.name == "items":
            yield from mapping_rows
        else:
            yield from rows_by_split[directory.parent.name]

    monkeypatch.setattr(news_content, "iter_parquet_rows", fake_iter_parquet_rows)
    processed_root = tmp_path / "processed"
    for split in ("train", "dev", "test"):
        (processed_root / split / "news").mkdir(parents=True)

    entity_path = tmp_path / "entity_embedding.vec"
    entity_path.write_text(
        "Q1 " + " ".join(["1"] + ["0"] * 99) + "\n",
        encoding="utf-8",
    )
    output_directory = tmp_path / "output"
    news_content.build_features(
        processed_root, entity_path, output_directory,
        title_features=100, abstract_features=100,
        min_document_frequency=1, batch_size=2, overwrite=False,
    )

    import json

    index = json.loads((output_directory / "item_index.json").read_text(encoding="utf-8"))
    report = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    title_matrix = sparse.load_npz(output_directory / "title_tfidf.npz")
    entity_matrix = np.load(output_directory / "title_entity_embeddings.npy")
    vectorizer = news_content.joblib.load(output_directory / "title_tfidf_transformer.joblib")

    assert index == ["N-A", "N-B", "N-C", "N-D"]
    assert title_matrix.shape[0] == 4
    assert "quasar" not in vectorizer.vocabulary_
    assert title_matrix[2].nnz > 0
    assert report["train_items"] == 2
    assert report["dev_catalog_cold_items"] == 1
    assert np.any(entity_matrix[0])
    assert not np.any(entity_matrix[2])
