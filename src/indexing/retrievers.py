"""Candidate-source adapters used by the multi-source retrieval pipeline."""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from src.indexing.faiss_index import FaissVectorIndex
from src.indexing.types import RetrievalQuery, RetrievalSource, SourceHits


class Retriever(Protocol):
    source: RetrievalSource

    def retrieve(self, query: RetrievalQuery, limit: int) -> SourceHits: ...


class PopularRetriever:
    source = RetrievalSource.POPULAR

    def __init__(self, ranked_items: Sequence[int], scores: Sequence[float] | None = None):
        if scores is not None and len(scores) != len(ranked_items):
            raise ValueError("popularity scores must align with ranked_items")
        self._ranked_items = np.asarray(ranked_items, dtype=np.int64)
        self._scores = (
            np.asarray(scores, dtype=np.float32)
            if scores is not None
            else 1.0 / np.arange(1, len(self._ranked_items) + 1, dtype=np.float32)
        )

    def retrieve(self, query: RetrievalQuery, limit: int) -> SourceHits:
        # Over-fetch so filtering seen items still leaves ``limit`` results.
        end = limit + len(query.filter_item_idxs)
        return SourceHits.build(self._ranked_items[:end], self._scores[:end], query, limit)


class VectorRetriever:
    """FAISS retriever whose query vector is supplied by a model adapter."""

    def __init__(
        self,
        source: RetrievalSource,
        index: FaissVectorIndex,
        query_encoder: Callable[[RetrievalQuery], np.ndarray | None],
    ):
        if source == RetrievalSource.POPULAR:
            raise ValueError("VectorRetriever does not support the popular source")
        self.source = source
        self._index = index
        self._query_encoder = query_encoder

    def retrieve(self, query: RetrievalQuery, limit: int) -> SourceHits:
        vector = self._query_encoder(query)
        if vector is None:
            return SourceHits()

        # Ask for additional neighbors because filtering happens after ANN search.
        search_limit = min(
            self._index.metadata.num_items,
            limit + len(query.filter_item_idxs),
        )
        result = self._index.search(vector, search_limit)
        return SourceHits.build(result.item_ids[0], result.scores[0], query, limit)


class CallableRetriever:
    """Adapter for sequential or future retrieval models returning item-score pairs."""

    def __init__(
        self,
        source: RetrievalSource,
        retrieve_fn: Callable[[RetrievalQuery, int], Sequence[tuple[int, float]]],
    ):
        self.source = source
        self._retrieve_fn = retrieve_fn

    def retrieve(self, query: RetrievalQuery, limit: int) -> SourceHits:
        pairs = list(self._retrieve_fn(query, limit))
        if not pairs:
            return SourceHits()
        item_ids, scores = zip(*pairs, strict=True)
        return SourceHits.build(item_ids, scores, query, limit)
