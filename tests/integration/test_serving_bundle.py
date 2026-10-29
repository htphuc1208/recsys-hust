import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.indexing.build_bundle import build_serving_bundle
from src.models.baselines.popularity import PopularityRecommender
from src.serving.config import RetrievalConfig, ServingConfig
from src.serving.pipeline import RecommendationPipeline
from src.serving.resources import ServingResources
from src.serving.schemas import RecommendationRequest


def test_built_bundle_loads_and_serves_external_ids(tmp_path):
    model = PopularityRecommender()
    model.fit(pd.DataFrame({"user_idx": [0, 0, 1], "item_idx": [2, 2, 1]}))
    checkpoint = tmp_path / "model.pkl"
    model.save(checkpoint)

    users = tmp_path / "source" / "users"
    items = tmp_path / "source" / "items"
    users.mkdir(parents=True)
    items.mkdir(parents=True)
    pq.write_table(pa.table({"user_id": ["U1", "U2"], "user_idx": [0, 1]}), users / "part.parquet")
    pq.write_table(
        pa.table({"news_id": ["N0", "N1", "N2"], "item_idx": [0, 1, 2]}),
        items / "part.parquet",
    )

    bundle = tmp_path / "bundle"
    build_serving_bundle(
        checkpoint=checkpoint,
        users_mapping=users,
        items_mapping=items,
        output_dir=bundle,
        model_version="integration-v1",
        git_commit="abc123",
    )
    config = ServingConfig(
        bundle_dir=bundle,
        retrieval=RetrievalConfig.model_validate(
            {
                "min_candidates": 1,
                "target_candidates": 3,
                "max_candidates": 3,
                "sources": {"popular": {"limit": 3}},
            }
        ),
    )
    resources = ServingResources.load(config)
    try:
        response = RecommendationPipeline(resources).recommend(
            RecommendationRequest(user_id="UNKNOWN", top_k=2)
        )
    finally:
        resources.close()

    assert [item.news_id for item in response.recommendations] == ["N2", "N1"]
    assert response.model_version == "integration-v1"


def _write_mappings(tmp_path, num_users: int, num_items: int):
    users = tmp_path / "source" / "users"
    items = tmp_path / "source" / "items"
    users.mkdir(parents=True)
    items.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {"user_id": [f"U{i}" for i in range(num_users)], "user_idx": list(range(num_users))}
        ),
        users / "part.parquet",
    )
    pq.write_table(
        pa.table(
            {"news_id": [f"N{i}" for i in range(num_items)], "item_idx": list(range(num_items))}
        ),
        items / "part.parquet",
    )
    return users, items


def test_hybrid_bundle_serves_all_retrieval_sources(tmp_path):
    import json

    from fastapi.testclient import TestClient

    from src.indexing.types import RetrievalQuery, RetrievalSource
    from src.models.content_based import ContentBasedRecommender
    from src.models.hybrid import AdaptiveGatingHybrid
    from src.serving.app import create_app
    from tests.unit.test_content_based import create_features

    train_df = pd.DataFrame(
        {
            "user_idx": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "item_idx": [0, 1, 2, 1, 2, 3, 0, 2, 3],
            "label": [1] * 9,
            "timestamp": pd.date_range("2026-01-01", periods=9, freq="h"),
        }
    )
    content = ContentBasedRecommender(
        feature_dir=str(create_features(tmp_path)),
        feature_weights={"title": 0.7, "category": 0.15, "subcategory": 0.15},
    )
    content.fit(train_df)
    content_checkpoint = tmp_path / "content.pkl"
    content.save(content_checkpoint)

    hybrid = AdaptiveGatingHybrid(
        mode="adaptive",
        embedding_dim=8,
        seed=42,
        device="cpu",
        content_checkpoint=content_checkpoint,
        cf_params={"epochs": 1, "batch_size": 4},
        seq_params={"epochs": 1, "batch_size": 4, "max_seq_len": 4},
    )
    hybrid.fit(train_df)
    checkpoint = tmp_path / "hybrid.pkl"
    hybrid.save(checkpoint)

    users, items = _write_mappings(tmp_path, num_users=3, num_items=5)
    bundle = tmp_path / "bundle"
    build_serving_bundle(
        checkpoint=checkpoint,
        users_mapping=users,
        items_mapping=items,
        output_dir=bundle,
        model_version="hybrid-v1",
    )

    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_type"] == "AdaptiveGatingHybrid"
    assert manifest["retrieval_sources"] == ["popular", "collaborative", "sequential", "content"]
    # Item 4 has no content features, so it must not enter the cosine index.
    content_meta = json.loads((bundle / "indexes" / "content.meta.json").read_text("utf-8"))
    assert content_meta["num_items"] == 4
    # The ranker must unpickle on another OS: no WindowsPath/PosixPath class references.
    assert b"WindowsPath" not in (bundle / "models" / "ranker.pkl").read_bytes()

    sources = {
        source: {"limit": 5, "timeout_ms": 5_000}
        for source in ("collaborative", "sequential", "content", "popular")
    }
    config = ServingConfig(
        bundle_dir=bundle,
        retrieval=RetrievalConfig.model_validate(
            {"min_candidates": 1, "target_candidates": 5, "max_candidates": 5, "sources": sources}
        ),
        rank_timeout_ms=5_000,
    )
    resources = ServingResources.load(config)
    try:
        assert resources.ranker.content_model.feature_dir == (
            bundle.resolve() / "features" / "content"
        )
        retrieval = resources.retriever.retrieve(
            RetrievalQuery(user_idx=0, history_item_idxs=(0, 1))
        )
        assert retrieval.failed_sources == {}
        assert set(retrieval.source_latencies_ms) == set(RetrievalSource)

        client = TestClient(create_app(RecommendationPipeline(resources)))
        with client:
            assert client.get("/ready").json()["model_version"] == "hybrid-v1"
            response = client.post(
                "/recommend",
                json={"user_id": "U0", "history_news_ids": ["N0", "N1"], "top_k": 3},
            )
    finally:
        resources.close()

    assert response.status_code == 200
    payload = response.json()
    assert payload["degraded"] is False
    assert payload["strategy"] == "hybrid"
    assert {item["news_id"] for item in payload["recommendations"]}.isdisjoint({"N0", "N1"})
