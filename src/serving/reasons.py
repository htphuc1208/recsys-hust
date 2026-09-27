"""Deterministic explanation rules based on retrieval provenance."""

from src.indexing.types import Candidate, RetrievalSource
from src.serving.schemas import ReasonCode


def reason_for(candidate: Candidate) -> tuple[ReasonCode, dict[str, str | int | float]]:
    if RetrievalSource.CONTENT in candidate.sources:
        source = RetrievalSource.CONTENT
        code = ReasonCode.SIMILAR_TO_RECENT_HISTORY
    elif RetrievalSource.SEQUENTIAL in candidate.sources:
        source = RetrievalSource.SEQUENTIAL
        code = ReasonCode.CONTINUES_RECENT_INTEREST
    elif RetrievalSource.COLLABORATIVE in candidate.sources:
        source = RetrievalSource.COLLABORATIVE
        code = ReasonCode.MATCHES_YOUR_TASTE
    else:
        source = RetrievalSource.POPULAR
        code = ReasonCode.POPULAR_NOW

    evidence: dict[str, str | int | float] = {"source": source.value}
    if source in candidate.source_ranks:
        evidence["source_rank"] = candidate.source_ranks[source]
    return code, evidence
