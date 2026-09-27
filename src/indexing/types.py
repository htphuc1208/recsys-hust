"""Shared contracts for candidate retrieval.

Retrievers operate exclusively on internal integer indices. Translation from
public user/news identifiers belongs to the serving layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    candidates: tuple[Candidate, ...]
    source_latencies_ms: dict[RetrievalSource, float] = field(default_factory=dict)
    failed_sources: dict[RetrievalSource, str] = field(default_factory=dict)
    used_popularity_fill: bool = False
