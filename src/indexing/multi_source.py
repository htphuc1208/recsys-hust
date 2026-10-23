"""Concurrent multi-source retrieval, RRF fusion, and popularity fallback."""

import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError

from src.indexing.retrievers import Retriever
from src.indexing.types import Candidate, RetrievalQuery, RetrievalResult, RetrievalSource
from src.serving.config import RetrievalConfig


class MultiSourceRetriever:
    def __init__(self, retrievers: list[Retriever], config: RetrievalConfig):
        self._retrievers = {retriever.source: retriever for retriever in retrievers}
        self._config = config
        if RetrievalSource.POPULAR not in self._retrievers:
            raise ValueError("MultiSourceRetriever requires a popularity retriever")
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, len(self._retrievers)), thread_name_prefix="retrieval"
        )

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        started_at: dict[RetrievalSource, float] = {}
        futures: dict[RetrievalSource, Future[list[Candidate]]] = {}
        for source, source_config in self._config.sources.items():
            retriever = self._retrievers.get(source)
            if not source_config.enabled or retriever is None:
                continue
            started_at[source] = time.perf_counter()
            futures[source] = self._executor.submit(retriever.retrieve, query, source_config.limit)

        source_candidates: dict[RetrievalSource, list[Candidate]] = {}
        latencies: dict[RetrievalSource, float] = {}
        failures: dict[RetrievalSource, str] = {}

        deadlines = {
            source: started_at[source] + self._config.sources[source].timeout_ms / 1000.0
            for source in futures
        }
        for source, future in futures.items():
            remaining = max(0.0, deadlines[source] - time.perf_counter())
            try:
                source_candidates[source] = future.result(timeout=remaining)
            except TimeoutError:
                future.cancel()
                failures[source] = "timeout"
            except Exception as exc:  # noqa: BLE001 - isolate one optional source
                failures[source] = type(exc).__name__
            finally:
                latencies[source] = round((time.perf_counter() - started_at[source]) * 1000.0, 3)

        merged: dict[int, Candidate] = {}
        for source, candidates in source_candidates.items():
            weight = self._config.sources[source].weight
            for source_rank, incoming in enumerate(candidates, start=1):
                if incoming.item_idx in query.filter_item_idxs:
                    continue
                candidate = merged.setdefault(incoming.item_idx, Candidate(incoming.item_idx))
                raw_score = incoming.source_scores.get(source, 0.0)
                rank = incoming.source_ranks.get(source, source_rank)
                candidate.add_source(source, raw_score, rank)
                candidate.retrieval_score += weight / (self._config.rrf_constant + rank)

        personalized_count = sum(
            any(source != RetrievalSource.POPULAR for source in candidate.sources)
            for candidate in merged.values()
        )
        candidates = sorted(
            merged.values(), key=lambda item: (-item.retrieval_score, item.item_idx)
        )
        candidates = candidates[: self._config.target_candidates]
        used_popularity_fill = personalized_count < self._config.min_candidates and any(
            candidate.sources == {RetrievalSource.POPULAR} for candidate in candidates
        )

        return RetrievalResult(
            candidates=tuple(candidates),
            source_latencies_ms=latencies,
            failed_sources=failures,
            used_popularity_fill=used_popularity_fill,
        )
