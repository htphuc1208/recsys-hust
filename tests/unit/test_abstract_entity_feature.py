"""Kiểm thử abstract entity và tương thích ngược với Feature Store B4."""

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features import news_content
from src.models.content_based.recommender import FEATURE_FILES, ContentBasedRecommender


def make_record(idx, title=(), abstract=()):
    return news_content.NewsRecord(
        item_idx=idx,
        title="sports football match",
        abstract="sports football match",
        category="sports",
        subcategory="football",
        title_entity_ids=tuple(title),
        abstract_entity_ids=tuple(abstract),
    )


def test_entity_complementarity_and_normalization():
    v1 = np.zeros(100, dtype=np.float32)
    v2 = np.zeros(100, dtype=np.float32)
    v1[0] = 1.0
    v2[1] = 1.0
    records = [make_record(0, title=("Q1",)), make_record(1, abstract=("Q2",)), make_record(2)]
    vectors = {"Q1": v1, "Q2": v2}
    title, _ = news_content.build_title_entity_feature(records, vectors)
    abstract, stats = news_content.build_entity_feature(records, vectors, "abstract_entity_ids")
    assert np.any(title[0]) and not np.any(title[1])
    assert np.any(abstract[1]) and not np.any(abstract[0])
    assert not np.any(abstract[2])
    assert np.linalg.norm(abstract[1]) == pytest.approx(1.0)
    assert stats["articles_with_entities"] == 1
    with pytest.raises(ValueError):
        news_content.build_entity_feature(records, vectors, "unknown")


def create_store(tmp_path, include_abstract):
    folder = tmp_path / "features"
    folder.mkdir()
    item_count = 3
    arr = sparse.csr_matrix(np.eye(3, dtype=np.float32))
    shape = {}
    for name in ("title_tfidf", "abstract_tfidf", "category_onehot", "subcategory_onehot"):
        sparse.save_npz(folder / f"{name}.npz", arr)
        shape[name] = list(arr.shape)
    entity = np.zeros((item_count, 100), dtype=np.float32)
    entity[0, 0] = 1.0
    np.save(folder / "title_entity_embeddings.npy", entity)
    shape["title_entity_embeddings"] = list(entity.shape)
    if include_abstract:
        abs_entity = np.zeros_like(entity)
        abs_entity[[0, 2], 0] = 1.0
        np.save(folder / "abstract_entity_embeddings.npy", abs_entity)
        shape["abstract_entity_embeddings"] = list(abs_entity.shape)
    (folder / "manifest.json").write_text(
        json.dumps({"total_item_mapping": item_count, "feature_shapes": shape}), encoding="utf8"
    )
    (folder / "item_index.json").write_text(json.dumps(["N0", "N1", "N2"]), encoding="utf8")
    return folder


def test_old_feature_store_still_loads(tmp_path):
    folder = create_store(tmp_path, include_abstract=False)
    model = ContentBasedRecommender(feature_dir=str(folder))
    model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))
    assert "abstract_entity" not in model._features
    assert len(FEATURE_FILES) == 5
    # Giả lập checkpoint B4 được lưu trước khi có khóa abstract_entity.
    model.feature_weights.pop("abstract_entity")
    model._ensure_features_loaded(load_all=True)
    assert "abstract_entity" not in model._features


def test_abstract_only_weight_predicts_candidate(tmp_path):
    folder = create_store(tmp_path, include_abstract=True)
    model = ContentBasedRecommender(
        feature_dir=str(folder), feature_weights={"abstract_entity": 1.0}
    )
    model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))
    scores = model.predict(np.array([0]), np.array([2, 1]), history_item_idxs=[0])
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert model._features["abstract_entity"].shape == (3, 100)


def test_missing_abstract_fails_only_if_weight_positive(tmp_path):
    folder = create_store(tmp_path, include_abstract=False)
    model = ContentBasedRecommender(
        feature_dir=str(folder), feature_weights={"title": 0.7, "abstract_entity": 0.3}
    )
    with pytest.raises(FileNotFoundError, match="abstract_entity_embeddings.npy"):
        model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))


def test_build_feature_store_includes_abstract_when_enabled(tmp_path, monkeypatch):
    """Kiểm tra end-to-end: đúng shape, index và báo cáo, không cần MIND thật."""
    mapping_rows = [
        {"news_id": "N0", "item_idx": 0, "first_seen_split": "train"},
        {"news_id": "N1", "item_idx": 1, "first_seen_split": "dev"},
    ]
    rows = {
        "train": [
            {
                "news_id": "N0",
                "item_idx": 0,
                "title": "football team news",
                "abstract": "match result news",
                "category": "sports",
                "subcategory": "football",
                "title_entity_ids": ["Q1"],
                "abstract_entity_ids": ["Q2"],
            }
        ],
        "dev": [
            {
                "news_id": "N1",
                "item_idx": 1,
                "title": "match football news",
                "abstract": "result team news",
                "category": "sports",
                "subcategory": "football",
                "title_entity_ids": [],
                "abstract_entity_ids": ["Q2"],
            }
        ],
    }

    def rows_stub(directory, columns):
        current = mapping_rows if directory.name == "items" else rows.get(directory.parent.name, [])
        for row in current:
            yield {key: row[key] for key in columns}

    monkeypatch.setattr(news_content, "iter_parquet_rows", rows_stub)
    processed_root = tmp_path / "processed"
    for split in ("train", "dev"):
        (processed_root / split / "news").mkdir(parents=True)
    entity_path = tmp_path / "entity.vec"
    v1 = ["1"] + ["0"] * 99
    v2 = ["0", "1"] + ["0"] * 98
    entity_path.write_text("Q1 " + " ".join(v1) + "\nQ2 " + " ".join(v2) + "\n")
    folder = tmp_path / "store"
    news_content.build_features(
        processed_root=processed_root,
        entity_path=entity_path,
        output_directory=folder,
        title_features=20,
        abstract_features=20,
        min_document_frequency=1,
        batch_size=2,
        overwrite=False,
        include_abstract_entity=True,
    )
    report = json.loads((folder / "manifest.json").read_text(encoding="utf8"))
    abstract_entity = np.load(folder / "abstract_entity_embeddings.npy")
    title_entity = np.load(folder / "title_entity_embeddings.npy")
    assert abstract_entity.shape == (2, 100)
    assert title_entity.shape == (2, 100)
    assert np.any(abstract_entity[1]) and not np.any(title_entity[1])
    assert report["abstract_entity_statistics"]["articles_with_matched_entities"] == 2
    assert report["feature_shapes"]["abstract_entity_embeddings"] == [2, 100]
    assert json.loads((folder / "item_index.json").read_text()) == ["N0", "N1"]
