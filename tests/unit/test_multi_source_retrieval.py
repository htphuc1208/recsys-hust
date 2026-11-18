import time

import pytest

from src.indexing.multi_source import MultiSourceRetriever
from src.indexing.retrievers import CallableRetriever, PopularRetriever
from src.indexing.types import RetrievalQuery, RetrievalSource
from src.serving.config import RetrievalConfig


def _config(timeout_ms: int = 50) -> RetrievalConfig:
    return RetrievalConfig.model_validate(
        {
            "min_candidates": 3,
            "target_candidates": 4,
            "max_candidates": 5,
            "rrf_constant": 10,
            "sources": {
                "collaborative": {"limit": 3, "weight": 1.0, "timeout_ms": timeout_ms},
                "content": {"limit": 3, "weight": 0.5, "timeout_ms": timeout_ms},
                "popular": {"limit": 5, "weight": 0.1, "timeout_ms": timeout_ms},
            },
        }
    )


def test_multi_source_deduplicates_filters_and_preserves_provenance():
    collaborative = CallableRetriever(
        RetrievalSource.COLLABORATIVE, lambda _query, _limit: [(1, 0.9), (2, 0.8), (3, 0.7)]
    )
    content = CallableRetriever(
        RetrievalSource.CONTENT, lambda _query, _limit: [(2, 0.95), (4, 0.8)]
    )
    retriever = MultiSourceRetriever(
        [collaborative, content, PopularRetriever([5, 2, 6, 7, 8])], _config()
    )
    try:
        result = retriever.retrieve(RetrievalQuery(user_idx=10, filter_item_idxs=frozenset({3, 5})))
    finally:
        retriever.close()

    item_ids = [candidate.item_idx for candidate in result.candidates]
    assert 3 not in item_ids
    assert 5 not in item_ids
    assert len(item_ids) == len(set(item_ids)) == 4
    item_two = next(candidate for candidate in result.candidates if candidate.item_idx == 2)
    assert item_two.sources == {
        RetrievalSource.COLLABORATIVE,
        RetrievalSource.CONTENT,
        RetrievalSource.POPULAR,
    }


def test_multi_source_rrf_scores_and_order_match_reference():
    collaborative = CallableRetriever(
        RetrievalSource.COLLABORATIVE, lambda _query, _limit: [(1, 0.9), (2, 0.8), (3, 0.7)]
    )
    content = CallableRetriever(
        RetrievalSource.CONTENT, lambda _query, _limit: [(3, 0.95), (2, 0.9), (3, 0.1)]
    )
    retriever = MultiSourceRetriever([collaborative, content, PopularRetriever([4, 1])], _config())
    try:
        result = retriever.retrieve(RetrievalQuery(user_idx=10))
    finally:
        retriever.close()

    # weight / (rrf_constant=10 + rank); the duplicate item 3 in content keeps rank 1 only.
    expected = {
        1: 1.0 / 11 + 0.1 / 12,
        2: 1.0 / 12 + 0.5 / 12,
        3: 1.0 / 13 + 0.5 / 11,
        4: 0.1 / 11,
    }
    ordered = sorted(expected, key=lambda item: (-expected[item], item))
    assert result.item_ids.tolist() == ordered
    assert result.scores.tolist() == pytest.approx([expected[item] for item in ordered])
    item_three = result.candidate(ordered.index(3))
    assert item_three.source_ranks == {
        RetrievalSource.COLLABORATIVE: 3,
        RetrievalSource.CONTENT: 1,
    }
    assert item_three.source_scores[RetrievalSource.CONTENT] == pytest.approx(0.95)


def test_source_latency_is_not_inflated_by_slower_earlier_source():
    def slow_source(_query, _limit):
        time.sleep(0.05)
        return [(1, 1.0)]

    collaborative = CallableRetriever(RetrievalSource.COLLABORATIVE, slow_source)
    content = CallableRetriever(RetrievalSource.CONTENT, lambda _query, _limit: [(2, 1.0)])
    retriever = MultiSourceRetriever(
        [collaborative, content, PopularRetriever([10])], _config(timeout_ms=500)
    )
    try:
        result = retriever.retrieve(RetrievalQuery(user_idx=10))
    finally:
        retriever.close()

    assert result.source_latencies_ms[RetrievalSource.COLLABORATIVE] >= 45
    assert result.source_latencies_ms[RetrievalSource.CONTENT] < 25


def test_multi_source_isolates_timeout_and_uses_popularity():
    def slow_source(_query, _limit):
        time.sleep(0.05)
        return [(1, 1.0)]

    collaborative = CallableRetriever(RetrievalSource.COLLABORATIVE, slow_source)
    retriever = MultiSourceRetriever(
        [collaborative, PopularRetriever([10, 11, 12, 13])], _config(timeout_ms=5)
    )
    try:
        result = retriever.retrieve(RetrievalQuery(user_idx=10))
    finally:
        retriever.close()

    assert result.failed_sources[RetrievalSource.COLLABORATIVE] == "timeout"
    assert [candidate.item_idx for candidate in result.candidates] == [10, 11, 12, 13]
    assert result.used_popularity_fill is True
