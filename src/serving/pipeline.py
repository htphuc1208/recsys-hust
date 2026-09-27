"""Two-stage online recommendation pipeline."""

import time

import numpy as np

from src.indexing.types import RetrievalQuery, RetrievalSource
from src.reranking.rules import stable_top_k
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
        candidates = list(retrieval.candidates)

        scores = {candidate.item_idx: candidate.retrieval_score for candidate in candidates}
        ranker_failed = False
        if self.resources.ranker is not None and user_idx is not None and candidates:
            item_ids = np.array([candidate.item_idx for candidate in candidates], dtype=np.int64)
            try:
                ranker_scores = self.resources.ranker.predict(
                    np.array([user_idx], dtype=np.int64),
                    item_ids,
                    history_item_idxs=list(history),
                    retrieval_candidates=candidates,
                    context=query.context,
                )
                if len(ranker_scores) != len(item_ids) or not np.isfinite(ranker_scores).all():
                    raise ValueError("ranker returned invalid scores")
                scores = {
                    int(item_idx): float(score)
                    for item_idx, score in zip(item_ids, ranker_scores, strict=True)
                }
            except Exception:  # noqa: BLE001 - fused retrieval score is the safe fallback
                ranker_failed = True

        ranked = stable_top_k(candidates, scores, request.top_k)
        recommendations: list[RecommendedItem] = []
        for candidate in ranked:
            news_id = mappings.idx_to_news.get(candidate.item_idx)
            if news_id is None:
                continue
            reason_code, evidence = reason_for(candidate)
            recommendations.append(
                RecommendedItem(
                    news_id=news_id,
                    score=scores[candidate.item_idx],
                    rank=len(recommendations) + 1,
                    reason_code=reason_code,
                    reason_evidence=evidence,
                )
            )

        fallback_reason = None
        if ranker_failed:
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
