"""Concurrent multi-source retrieval, RRF fusion, and popularity fallback."""

import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError

import numpy as np

from src.indexing.retrievers import Retriever
from src.indexing.types import RetrievalQuery, RetrievalResult, RetrievalSource, SourceHits
from src.serving.config import RetrievalConfig


def _run_source(
    retriever: Retriever, query: RetrievalQuery, limit: int
) -> tuple[SourceHits, float]:
    """Return the hits with their completion time, so latency isn't skewed by read order."""
    hits = retriever.retrieve(query, limit)
    return hits, time.perf_counter()


class MultiSourceRetriever:
    def __init__(self, retrievers: list[Retriever], config: RetrievalConfig):
        self._retrievers = {retriever.source: retriever for retriever in retrievers}
        self._config = config
        if RetrievalSource.POPULAR not in self._retrievers:
            raise ValueError("MultiSourceRetriever requires a popularity retriever")
        self._executor = ThreadPoolExecutor(
            max_workers=max(len(self._retrievers), config.max_workers),
            thread_name_prefix="retrieval",
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        started_at: dict[RetrievalSource, float] = {}
        futures: dict[RetrievalSource, Future[tuple[SourceHits, float]]] = {}
        for source, source_config in self._config.sources.items():
            retriever = self._retrievers.get(source)
            if not source_config.enabled or retriever is None:
                continue
            started_at[source] = time.perf_counter()
            futures[source] = self._executor.submit(
                _run_source, retriever, query, source_config.limit
            )

        source_hits: dict[RetrievalSource, SourceHits] = {}
        latencies: dict[RetrievalSource, float] = {}
        failures: dict[RetrievalSource, str] = {}

        for source, future in futures.items():
            deadline = started_at[source] + self._config.sources[source].timeout_ms / 1000.0
            try:
                hits, finished_at = future.result(timeout=max(0.0, deadline - time.perf_counter()))
                source_hits[source] = hits
            except TimeoutError:
                future.cancel()
                failures[source] = "timeout"
                finished_at = time.perf_counter()
            except Exception as exc:  # noqa: BLE001 - isolate one optional source
                failures[source] = type(exc).__name__
                finished_at = time.perf_counter()
            latencies[source] = round((finished_at - started_at[source]) * 1000.0, 3)

        return self._fuse(source_hits, latencies, failures)

    def _fuse(
        self,
        source_hits: dict[RetrievalSource, SourceHits],
        latencies: dict[RetrievalSource, float],
        failures: dict[RetrievalSource, str],
    ) -> RetrievalResult:
        source_hits = {source: hits for source, hits in source_hits.items() if len(hits.item_ids)}
        if not source_hits:
            return RetrievalResult(source_latencies_ms=latencies, failed_sources=failures)

        item_ids, inverse = np.unique(
            np.concatenate([hits.item_ids for hits in source_hits.values()]),
            return_inverse=True,
        )
        fused = np.zeros(len(item_ids), dtype=np.float64)
        personalized = np.zeros(len(item_ids), dtype=bool)
        source_ranks: dict[RetrievalSource, np.ndarray] = {}
        source_scores: dict[RetrievalSource, np.ndarray] = {}
        offset = 0
        for source, hits in source_hits.items():
            count = len(hits.item_ids)
            # SourceHits are deduplicated, so positions are unique within one source.
            positions = inverse[offset : offset + count]
            offset += count
            ranks = np.arange(1, count + 1, dtype=np.int32)
            fused[positions] += self._config.sources[source].weight / (
                self._config.rrf_constant + ranks
            )
            source_ranks[source] = np.zeros(len(item_ids), dtype=np.int32)
            source_ranks[source][positions] = ranks
            source_scores[source] = np.zeros(len(item_ids), dtype=np.float32)
            source_scores[source][positions] = hits.scores
            if source != RetrievalSource.POPULAR:
                personalized[positions] = True

        order = np.lexsort((item_ids, -fused))[: self._config.target_candidates]
        used_popularity_fill = bool(
            personalized.sum() < self._config.min_candidates and not personalized[order].all()
        )
        return RetrievalResult(
            item_ids=item_ids[order],
            scores=fused[order],
            source_ranks={source: ranks[order] for source, ranks in source_ranks.items()},
            source_scores={source: scores[order] for source, scores in source_scores.items()},
            source_latencies_ms=latencies,
            failed_sources=failures,
            used_popularity_fill=used_popularity_fill,
        )
