import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.indexing.multi_source import MultiSourceRetriever
from src.indexing.retrievers import CallableRetriever, PopularRetriever
from src.indexing.types import RetrievalSource
from src.serving.app import create_app
from src.serving.artifacts import ServingManifest
from src.serving.config import RetrievalConfig, ServingConfig
from src.serving.pipeline import RecommendationPipeline
from src.serving.resources import IdMappings, ServingResources
from src.serving.schemas import FallbackReason, ReasonCode, RecommendationRequest


def _pipeline() -> RecommendationPipeline:
    config = RetrievalConfig.model_validate(
        {
            "min_candidates": 2,
            "target_candidates": 3,
            "max_candidates": 4,
            "sources": {
                "collaborative": {"limit": 2, "weight": 1.0, "timeout_ms": 50},
                "popular": {"limit": 4, "weight": 0.1, "timeout_ms": 50},
            },
        }
    )
    collaborative = CallableRetriever(
        RetrievalSource.COLLABORATIVE,
        lambda query, _limit: [(1, 1.0), (2, 0.5)] if query.user_idx is not None else [],
    )
    resources = ServingResources(
        manifest=ServingManifest(
            model_version="test-v1",
            dataset="test",
            git_commit="abc",
            item_mapping_version="v1",
            num_users=1,
            num_items=4,
            artifacts={},
        ),
        mappings=IdMappings(
            user_to_idx={"U1": 0},
            news_to_idx={"N0": 0, "N1": 1, "N2": 2, "N3": 3},
            idx_to_news={0: "N0", 1: "N1", 2: "N2", 3: "N3"},
        ),
        retriever=MultiSourceRetriever([collaborative, PopularRetriever([0, 1, 2, 3])], config),
    )
    return RecommendationPipeline(resources)


def test_known_user_gets_personalized_reasons_and_seen_items_are_filtered():
    pipeline = _pipeline()
    try:
        response = pipeline.recommend(
            RecommendationRequest(user_id="U1", history_news_ids=["N1"], top_k=3)
        )
    finally:
        pipeline.resources.close()

    assert all(item.news_id != "N1" for item in response.recommendations)
    personalized = next(item for item in response.recommendations if item.news_id == "N2")
    assert personalized.reason_code == ReasonCode.MATCHES_YOUR_TASTE
    assert response.strategy == "hybrid"
    assert response.degraded is False


def test_unknown_user_falls_back_to_popularity():
    pipeline = _pipeline()
    try:
        response = pipeline.recommend(RecommendationRequest(user_id="UNKNOWN", top_k=2))
    finally:
        pipeline.resources.close()

    assert [item.news_id for item in response.recommendations] == ["N0", "N1"]
    assert all(item.reason_code == ReasonCode.POPULAR_NOW for item in response.recommendations)
    assert response.fallback_reason == FallbackReason.COLD_USER
    assert response.degraded is True


@pytest.mark.parametrize("top_k", [1, 3])
def test_reranking_never_compares_model_scores_with_unranked_rrf_scores(top_k):
    class NegativeRanker:
        def predict(self, user_ids, item_ids, **kwargs):
            # Every score is below the positive RRF scores of unranked candidates.
            return -np.arange(len(item_ids), 0, -1, dtype=float)

    pipeline = _pipeline()
    resources = pipeline.resources
    resources.ranker = NegativeRanker()
    resources.config = ServingConfig(retrieval=resources.retriever._config, rerank_limit=2)
    try:
        response = pipeline.recommend(RecommendationRequest(user_id="U1", top_k=top_k))
    finally:
        resources.close()

    # Retrieval is N1, N2, N0. Rank only the head (expanded to cover top_k).
    expected = ["N2"] if top_k == 1 else ["N0", "N2", "N1"]
    assert [item.news_id for item in response.recommendations] == expected
    assert all(item.score < 0 for item in response.recommendations)


def test_ranker_error_preserves_retrieval_order():
    class BrokenRanker:
        def predict(self, *args, **kwargs):
            raise ValueError("ranking unavailable")

    pipeline = _pipeline()
    pipeline.resources.ranker = BrokenRanker()
    try:
        response = pipeline.recommend(RecommendationRequest(user_id="U1", top_k=3))
    finally:
        pipeline.resources.close()

    assert [item.news_id for item in response.recommendations] == ["N1", "N2", "N0"]
    assert response.fallback_reason == FallbackReason.RANKER_ERROR


def test_api_readiness_and_recommendation_contract():
    pipeline = _pipeline()
    try:
        with TestClient(create_app(pipeline)) as client:
            ready = client.get("/ready")
            response = client.post("/recommend", json={"user_id": "U1", "top_k": 2})
    finally:
        pipeline.resources.close()

    assert ready.status_code == 200
    assert ready.json()["model_version"] == "test-v1"
    assert response.status_code == 200
    assert response.json()["recommendations"][0]["reason_code"] in {
        "MATCHES_YOUR_TASTE",
        "POPULAR_NOW",
    }


def test_unconfigured_app_is_not_ready(monkeypatch):
    monkeypatch.setenv("MODEL_BUNDLE_DIR", "missing-bundle")
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/recommend", json={"top_k": 2}).status_code == 503
