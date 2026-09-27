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
