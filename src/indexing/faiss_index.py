"""Small, versioned FAISS index wrapper used by offline builders and serving."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

Metric = Literal["inner_product", "cosine"]


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise RuntimeError(
            "FAISS is required for vector retrieval. Install the serving extra with "
            '`pip install -e ".[serving]"`.'
        ) from exc
    return faiss


@dataclass(frozen=True, slots=True)
class FaissIndexMetadata:
    schema_version: str
    index_type: str
    dimension: int
    num_items: int
    metric: Metric
    normalized: bool
    source_model: str
    model_version: str

    @classmethod
    def from_json(cls, path: str | Path) -> "FaissIndexMetadata":
        with Path(path).open(encoding="utf-8") as file:
            return cls(**json.load(file))

    def to_json(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(asdict(self), file, indent=2, sort_keys=True)


@dataclass(frozen=True, slots=True)
class SearchResult:
    item_ids: np.ndarray
    scores: np.ndarray


class FaissVectorIndex:
    """ID-mapped exact inner-product index with optional cosine normalization."""

    def __init__(self, index: Any, metadata: FaissIndexMetadata):
        self._index = index
        self.metadata = metadata

    @classmethod
    def build(
        cls,
        vectors: np.ndarray,
        item_ids: np.ndarray,
        *,
        metric: Metric = "inner_product",
        source_model: str,
        model_version: str,
    ) -> "FaissVectorIndex":
        faiss = _import_faiss()
        matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        ids = np.ascontiguousarray(item_ids, dtype=np.int64)

        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("vectors must be a non-empty 2D matrix")
        if ids.ndim != 1 or len(ids) != len(matrix):
            raise ValueError("item_ids must be a 1D array aligned with vectors")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("item_ids must be unique")
        if np.any(ids < 0):
            raise ValueError("item_ids must be non-negative")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must contain only finite values")
        if metric not in ("inner_product", "cosine"):
            raise ValueError(f"unsupported metric: {metric}")

        normalized = metric == "cosine"
        if normalized:
            norms = np.linalg.norm(matrix, axis=1)
            if np.any(norms == 0):
                raise ValueError("cosine index cannot contain zero vectors")
            faiss.normalize_L2(matrix)

        flat_index = faiss.IndexFlatIP(matrix.shape[1])
        index = faiss.IndexIDMap2(flat_index)
        index.add_with_ids(matrix, ids)
        metadata = FaissIndexMetadata(
            schema_version="1",
            index_type="IndexIDMap2(IndexFlatIP)",
            dimension=matrix.shape[1],
            num_items=matrix.shape[0],
            metric=metric,
            normalized=normalized,
            source_model=source_model,
            model_version=model_version,
        )
        return cls(index=index, metadata=metadata)

    def search(self, query_vectors: np.ndarray, top_k: int) -> SearchResult:
        faiss = _import_faiss()
        queries = np.asarray(query_vectors, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        queries = np.ascontiguousarray(queries)
        if queries.ndim != 2 or queries.shape[1] != self.metadata.dimension:
            raise ValueError(f"query vectors must have dimension {self.metadata.dimension}")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not np.isfinite(queries).all():
            raise ValueError("query vectors must contain only finite values")
        if self.metadata.normalized:
            norms = np.linalg.norm(queries, axis=1)
            if np.any(norms == 0):
                raise ValueError("cosine query cannot contain zero vectors")
            faiss.normalize_L2(queries)

        limit = min(top_k, self.metadata.num_items)
        scores, item_ids = self._index.search(queries, limit)
        return SearchResult(item_ids=item_ids, scores=scores)

    def save(self, index_path: str | Path, metadata_path: str | Path | None = None) -> None:
        faiss = _import_faiss()
        output_path = Path(index_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(output_path))
        self.metadata.to_json(metadata_path or output_path.with_suffix(".meta.json"))

    @classmethod
    def load(
        cls, index_path: str | Path, metadata_path: str | Path | None = None
    ) -> "FaissVectorIndex":
        faiss = _import_faiss()
        input_path = Path(index_path)
        metadata = FaissIndexMetadata.from_json(
            metadata_path or input_path.with_suffix(".meta.json")
        )
        index = faiss.read_index(str(input_path))
        if index.d != metadata.dimension:
            raise ValueError("FAISS index dimension does not match metadata")
        if index.ntotal != metadata.num_items:
            raise ValueError("FAISS item count does not match metadata")
        return cls(index=index, metadata=metadata)
