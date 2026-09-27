import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from src.indexing.faiss_index import FaissVectorIndex  # noqa: E402


def test_faiss_cosine_search_and_round_trip(tmp_path):
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=np.float32)
    item_ids = np.array([10, 20, 30], dtype=np.int64)
    index = FaissVectorIndex.build(
        vectors,
        item_ids,
        metric="cosine",
        source_model="test",
        model_version="v1",
    )

    result = index.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    assert result.item_ids.tolist() == [[10, 30]]
    assert result.scores[0, 0] == pytest.approx(1.0)

    path = tmp_path / "items.faiss"
    index.save(path)
    loaded = FaissVectorIndex.load(path)
    loaded_result = loaded.search(np.array([1.0, 0.0], dtype=np.float32), top_k=2)
    np.testing.assert_array_equal(loaded_result.item_ids, result.item_ids)
    np.testing.assert_allclose(loaded_result.scores, result.scores)


def test_faiss_rejects_invalid_vectors():
    with pytest.raises(ValueError, match="unique"):
        FaissVectorIndex.build(
            np.ones((2, 3), dtype=np.float32),
            np.array([1, 1]),
            source_model="test",
            model_version="v1",
        )
