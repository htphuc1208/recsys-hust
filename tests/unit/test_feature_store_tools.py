"""Kiểm tra conversion không đổi giá trị, mapping hay checkpoint."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src.features.feature_store_tools import benchmark_feature_loading, convert_sparse_storage
from src.models.content_based.recommender import ContentBasedRecommender


def create_store(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    matrices = {
        "title_tfidf": sparse.csr_matrix(np.array([[1., 0.], [0., 1.]], dtype=np.float32)),
        "abstract_tfidf": sparse.csr_matrix(np.eye(2, dtype=np.float32)),
        "category_onehot": sparse.csr_matrix(np.eye(2, dtype=np.float32)),
        "subcategory_onehot": sparse.csr_matrix(np.eye(2, dtype=np.float32)),
    }
    for name, matrix in matrices.items():
        sparse.save_npz(source / f"{name}.npz", matrix, compressed=True)
    np.save(source / "title_entity_embeddings.npy", np.eye(2, dtype=np.float32))
    np.save(source / "abstract_entity_embeddings.npy", np.eye(2, dtype=np.float32))
    (source / "item_index.json").write_text(json.dumps(["N1", "N2"]), encoding="utf-8")
    (source / "manifest.json").write_text(json.dumps({
        "total_item_mapping": 2,
        "feature_shapes": {
            **{k: list(m.shape) for k,m in matrices.items()},
            "title_entity_embeddings": [2,2],
            "abstract_entity_embeddings": [2,2],
        },
    }), encoding="utf-8")
    return source


def test_convert_preserves_every_feature(tmp_path):
    source = create_store(tmp_path)
    destination = tmp_path / "target"
    report = convert_sparse_storage(source, destination)
    assert report["sparse_compressed"] is False
    assert json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["sparse_storage"]["compressed"] is False
    for name in ("title_tfidf", "abstract_tfidf", "category_onehot", "subcategory_onehot"):
        assert (sparse.load_npz(source / f"{name}.npz") != sparse.load_npz(destination / f"{name}.npz")).nnz == 0
    np.testing.assert_array_equal(
        np.load(source / "abstract_entity_embeddings.npy"),
        np.load(destination / "abstract_entity_embeddings.npy"),
    )
    assert (source / "item_index.json").read_bytes() == (destination / "item_index.json").read_bytes()
    with pytest.raises(FileExistsError):
        convert_sparse_storage(source, destination)


def test_checkpoint_predictions_equal_after_conversion(tmp_path):
    source = create_store(tmp_path)
    destination = tmp_path / "target"
    convert_sparse_storage(source, destination)
    model = ContentBasedRecommender(feature_dir=str(source), feature_weights={"title": 0.7, "category": 0.3})
    model.fit(pd.DataFrame({"user_idx": [0], "item_idx": [0]}))
    checkpoint = tmp_path / "model.pkl"
    model.save(checkpoint)
    benchmark = benchmark_feature_loading(checkpoint, [source, destination], repeats=2)
    assert len(benchmark["results"]) == 2
    assert all(row["prediction_identical_to_first"] for row in benchmark["results"])


def test_save_feature_supports_uncompressed_npz(tmp_path):
    import zipfile
    from sklearn.feature_extraction.text import TfidfVectorizer
    from src.features.news_content import save_feature

    vectorizer = TfidfVectorizer().fit(["football match", "science news"])
    matrix = vectorizer.transform(["football match", "science news"]).tocsr()
    save_feature(tmp_path, "title_tfidf", matrix, vectorizer, compress_sparse=False)
    with zipfile.ZipFile(tmp_path / "title_tfidf.npz") as archive:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
    np.testing.assert_array_equal(sparse.load_npz(tmp_path / "title_tfidf.npz").toarray(), matrix.toarray())
