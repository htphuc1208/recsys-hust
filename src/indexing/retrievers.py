"""Candidate-source adapters used by the multi-source retrieval pipeline."""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from src.indexing.faiss_index import FaissVectorIndex
from src.indexing.types import Candidate, RetrievalQuery, RetrievalSource


class Retriever(Protocol):
    source: RetrievalSource

    def retrieve(self, query: RetrievalQuery, limit: int) -> list[Candidate]: ...


class PopularRetriever:
    source = RetrievalSource.POPULAR

    def __init__(self, ranked_items: Sequence[int], scores: Sequence[float] | None = None):
        if scores is not None and len(scores) != len(ranked_items):
            raise ValueError("popularity scores must align with ranked_items")
        self._ranked_items = tuple(int(item) for item in ranked_items)
        self._scores = tuple(float(score) for score in scores) if scores is not None else None

    def retrieve(self, query: RetrievalQuery, limit: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        for original_rank, item_idx in enumerate(self._ranked_items, start=1):
            if item_idx in query.filter_item_idxs:
                continue
            rank = len(candidates) + 1
            raw_score = (
                self._scores[original_rank - 1] if self._scores is not None else 1.0 / original_rank
            )
            candidate = Candidate(item_idx=item_idx)
            candidate.add_source(self.source, raw_score, rank)
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates


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

    def retrieve(self, query: RetrievalQuery, limit: int) -> list[Candidate]:
        vector = self._query_encoder(query)
        if vector is None:
            return []

        # Ask for additional neighbors because filtering happens after ANN search.
        search_limit = min(
            self._index.metadata.num_items,
            limit + len(query.filter_item_idxs),
        )
        result = self._index.search(vector, search_limit)
        candidates: list[Candidate] = []
        for item_idx, score in zip(result.item_ids[0], result.scores[0], strict=True):
            item = int(item_idx)
            if item < 0 or item in query.filter_item_idxs:
                continue
            rank = len(candidates) + 1
            candidate = Candidate(item_idx=item)
            candidate.add_source(self.source, float(score), rank)
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates


class CallableRetriever:
    """Adapter for sequential or future retrieval models returning item-score pairs."""

    def __init__(
        self,
        source: RetrievalSource,
        retrieve_fn: Callable[[RetrievalQuery, int], Sequence[tuple[int, float]]],
    ):
        self.source = source
        self._retrieve_fn = retrieve_fn

    def retrieve(self, query: RetrievalQuery, limit: int) -> list[Candidate]:
        candidates: list[Candidate] = []
        for item_idx, score in self._retrieve_fn(query, limit):
            item = int(item_idx)
            if item < 0 or item in query.filter_item_idxs:
                continue
            rank = len(candidates) + 1
            candidate = Candidate(item_idx=item)
            candidate.add_source(self.source, float(score), rank)
            candidates.append(candidate)
            if len(candidates) >= limit:
                break
        return candidates
