"""Two-stage online recommendation pipeline."""

import time
from concurrent.futures import TimeoutError

import numpy as np

from src.indexing.types import RetrievalQuery, RetrievalSource
from src.serving.reasons import reason_for
from src.serving.resources import ServingResources
from src.serving.schemas import (
    FallbackReason,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)


class RecommendationPipeline:
    def __init__(self, resources: ServingResources):
        self.resources = resources

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        start = time.perf_counter()
        mappings = self.resources.mappings
        user_idx = mappings.user_to_idx.get(request.user_id) if request.user_id else None
        history = tuple(
            mappings.news_to_idx[news_id]
            for news_id in request.history_news_ids
            if news_id in mappings.news_to_idx
        )
        filter_items = frozenset(history) if request.filter_seen else frozenset()
        query = RetrievalQuery(
            user_idx=user_idx,
            history_item_idxs=history,
            filter_item_idxs=filter_items,
            context=request.context.model_dump(exclude_none=True),
        )
        retrieval = self.resources.retriever.retrieve(query)
        scores = retrieval.scores.copy()
        ranking_positions = np.arange(len(retrieval))
        ranker_failed = False
        ranker_timed_out = False

        rerank_limit = (
            self.resources.config.rerank_limit if self.resources.config is not None else 200
        )
        rank_timeout_s = (
            self.resources.config.rank_timeout_ms / 1000.0
            if self.resources.config is not None
            else 0.1
        )

        if self.resources.ranker is not None and user_idx is not None and len(retrieval):
            # Retrieval output is already sorted by fused score; rerank only its head.
            item_ids = retrieval.item_ids[: max(rerank_limit, request.top_k)]

            assert self.resources.ranker_executor is not None
            future = self.resources.ranker_executor.submit(
                self.resources.ranker.predict,
                np.array([user_idx], dtype=np.int64),
                item_ids,
                history_item_idxs=list(history),
                context=query.context,
            )
            try:
                ranker_scores = np.asarray(future.result(timeout=rank_timeout_s), dtype=np.float64)
                if len(ranker_scores) != len(item_ids) or not np.isfinite(ranker_scores).all():
                    raise ValueError("ranker returned invalid scores")
                scores[: len(item_ids)] = ranker_scores
                # Model scores and RRF scores have unrelated scales. Once ranking
                # succeeds, select only among candidates scored by the model.
                ranking_positions = np.arange(len(item_ids))
            except TimeoutError:
                # Drop it if still queued so an overloaded pool doesn't build a backlog.
                future.cancel()
                ranker_timed_out = True
            except Exception:  # noqa: BLE001 - fused retrieval score is the safe fallback
                ranker_failed = True

        order = np.lexsort((retrieval.item_ids[ranking_positions], -scores[ranking_positions]))
        top_positions = ranking_positions[order[: request.top_k]]
        ranked = [retrieval.candidate(int(position)) for position in top_positions]
        recommendations: list[RecommendedItem] = []
        for position, candidate in zip(top_positions, ranked, strict=True):
            news_id = mappings.idx_to_news.get(candidate.item_idx)
            if news_id is None:
                continue
            reason_code, evidence = reason_for(candidate)
            recommendations.append(
                RecommendedItem(
                    news_id=news_id,
                    score=float(scores[position]),
                    rank=len(recommendations) + 1,
                    reason_code=reason_code,
                    reason_evidence=evidence,
                )
            )

        fallback_reason = None
        if ranker_timed_out:
            fallback_reason = FallbackReason.RANKER_TIMEOUT
        elif ranker_failed:
            fallback_reason = FallbackReason.RANKER_ERROR
        elif any(reason == "timeout" for reason in retrieval.failed_sources.values()):
            fallback_reason = FallbackReason.RETRIEVER_TIMEOUT
        elif retrieval.failed_sources:
            fallback_reason = FallbackReason.RETRIEVER_ERROR
        elif user_idx is None:
            fallback_reason = FallbackReason.COLD_USER

        has_personalized_source = any(
            any(source != RetrievalSource.POPULAR for source in candidate.sources)
            for candidate in ranked
        )
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version=self.resources.manifest.model_version,
            strategy="hybrid" if has_personalized_source else "popular",
            degraded=fallback_reason is not None,
            fallback_reason=fallback_reason,
            latency_ms=round((time.perf_counter() - start) * 1000.0, 3),
        )
