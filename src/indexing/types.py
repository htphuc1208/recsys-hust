"""Shared contracts for candidate retrieval.

Retrievers operate exclusively on internal integer indices. Translation from
public user/news identifiers belongs to the serving layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class RetrievalSource(str, Enum):
    POPULAR = "popular"
    COLLABORATIVE = "collaborative"
    SEQUENTIAL = "sequential"
    CONTENT = "content"


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Internal representation of one candidate retrieval request."""

    user_idx: int | None
    history_item_idxs: tuple[int, ...] = ()
    context: dict[str, Any] = field(default_factory=dict)
    filter_item_idxs: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if self.user_idx is not None and self.user_idx < 0:
            raise ValueError("user_idx must be non-negative or None")
        if any(item_idx < 0 for item_idx in self.history_item_idxs):
            raise ValueError("history_item_idxs must contain only non-negative indices")
        if any(item_idx < 0 for item_idx in self.filter_item_idxs):
            raise ValueError("filter_item_idxs must contain only non-negative indices")


@dataclass(slots=True)
class Candidate:
    """Candidate plus provenance required for fusion and explanations."""

    item_idx: int
    sources: set[RetrievalSource] = field(default_factory=set)
    source_scores: dict[RetrievalSource, float] = field(default_factory=dict)
    source_ranks: dict[RetrievalSource, int] = field(default_factory=dict)
    retrieval_score: float = 0.0

    def __post_init__(self) -> None:
        if self.item_idx < 0:
            raise ValueError("item_idx must be non-negative")

    def add_source(self, source: RetrievalSource, score: float, rank: int) -> None:
        """Attach source provenance, retaining its best (lowest) rank."""
        if rank < 1:
            raise ValueError("source rank must start at 1")
        previous_rank = self.source_ranks.get(source)
        if previous_rank is None or rank < previous_rank:
            self.source_ranks[source] = rank
            self.source_scores[source] = float(score)
        self.sources.add(source)


_EMPTY_IDS = np.empty(0, dtype=np.int64)
_EMPTY_SCORES = np.empty(0, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class SourceHits:
    """One source's results; ``item_ids[i]`` has source rank ``i + 1``."""

    item_ids: np.ndarray = field(default_factory=lambda: _EMPTY_IDS)
    scores: np.ndarray = field(default_factory=lambda: _EMPTY_SCORES)

    @classmethod
    def build(cls, item_ids: Any, scores: Any, query: RetrievalQuery, limit: int) -> "SourceHits":
        """Drop invalid, filtered and duplicate items (keeping the best rank), then truncate."""
        ids = np.asarray(item_ids, dtype=np.int64).ravel()
        values = np.asarray(scores, dtype=np.float32).ravel()
        keep = ids >= 0
        if query.filter_item_idxs:
            filtered = np.fromiter(
                query.filter_item_idxs, dtype=np.int64, count=len(query.filter_item_idxs)
            )
            keep &= ~np.isin(ids, filtered)
        ids, values = ids[keep], values[keep]
        _, first = np.unique(ids, return_index=True)
        if len(first) != len(ids):
            first.sort()
            ids, values = ids[first], values[first]
        return cls(ids[:limit], values[:limit])


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Fused candidates as aligned arrays, sorted by ``scores`` descending.

    ``source_ranks[s][i]`` is the rank of ``item_ids[i]`` within source ``s`` (0 if absent).
    """

    item_ids: np.ndarray = field(default_factory=lambda: _EMPTY_IDS)
    scores: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    source_ranks: dict[RetrievalSource, np.ndarray] = field(default_factory=dict)
    source_scores: dict[RetrievalSource, np.ndarray] = field(default_factory=dict)
    source_latencies_ms: dict[RetrievalSource, float] = field(default_factory=dict)
    failed_sources: dict[RetrievalSource, str] = field(default_factory=dict)
    used_popularity_fill: bool = False

    def __len__(self) -> int:
        return len(self.item_ids)

    def candidate(self, position: int) -> Candidate:
        candidate = Candidate(
            int(self.item_ids[position]), retrieval_score=float(self.scores[position])
        )
        for source, ranks in self.source_ranks.items():
            rank = int(ranks[position])
            if rank:
                candidate.sources.add(source)
                candidate.source_ranks[source] = rank
                candidate.source_scores[source] = float(self.source_scores[source][position])
        return candidate

    @property
    def candidates(self) -> tuple[Candidate, ...]:
        """Materialize every candidate; avoid on the request path."""
        return tuple(self.candidate(position) for position in range(len(self)))
