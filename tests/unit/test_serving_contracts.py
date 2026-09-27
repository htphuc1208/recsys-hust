import hashlib
import json

import pytest
from pydantic import ValidationError

from src.indexing.types import Candidate, RetrievalQuery, RetrievalSource
from src.serving.artifacts import ServingManifest
from src.serving.config import load_serving_config
from src.serving.schemas import ReasonCode, RecommendationRequest, RecommendedItem


def test_candidate_tracks_provenance_and_best_rank():
    candidate = Candidate(item_idx=7)
    candidate.add_source(RetrievalSource.CONTENT, score=0.7, rank=4)
    candidate.add_source(RetrievalSource.CONTENT, score=0.9, rank=2)
    candidate.add_source(RetrievalSource.COLLABORATIVE, score=2.1, rank=8)

    assert candidate.sources == {
        RetrievalSource.CONTENT,
        RetrievalSource.COLLABORATIVE,
    }
    assert candidate.source_ranks[RetrievalSource.CONTENT] == 2
    assert candidate.source_scores[RetrievalSource.CONTENT] == pytest.approx(0.9)


def test_retrieval_query_rejects_unknown_internal_indices():
    with pytest.raises(ValueError, match="non-negative"):
        RetrievalQuery(user_idx=-1)


def test_public_schema_uses_external_ids_and_reason_enum():
    request = RecommendationRequest(
        user_id="U87243", history_news_ids=["N8668"], context={"category": "news"}
    )
    item = RecommendedItem(
        news_id="N94157",
        rank=1,
        reason_code=ReasonCode.SIMILAR_TO_RECENT_HISTORY,
        reason_evidence={"history_news_id": "N8668"},
    )

    assert request.user_id == "U87243"
    assert request.context.category == "news"
    assert item.reason_code == ReasonCode.SIMILAR_TO_RECENT_HISTORY


def test_serving_config_has_valid_candidate_limits():
    config = load_serving_config("configs/serving/mind_production.yaml")
    assert config.retrieval.min_candidates == 500
    assert config.retrieval.target_candidates == 800
    assert config.retrieval.max_candidates == 1000
    assert RetrievalSource.POPULAR in config.retrieval.sources


def test_serving_config_rejects_invalid_limits(tmp_path):
    path = tmp_path / "serving.yaml"
    path.write_text(
        """
retrieval:
  min_candidates: 900
  target_candidates: 800
  max_candidates: 1000
  sources:
    popular: {limit: 1000}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError, match="candidate limits"):
        load_serving_config(path)


def test_manifest_validates_required_file_and_checksum(tmp_path):
    artifact = tmp_path / "popularity.json"
    artifact.write_text("[]", encoding="utf-8")
    checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
    payload = {
        "model_version": "test-v1",
        "dataset": "MINDlarge",
        "git_commit": "abc123",
        "item_mapping_version": "items-v1",
        "num_users": 2,
        "num_items": 3,
        "artifacts": {"popularity": {"path": artifact.name, "sha256": checksum, "required": True}},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    manifest = ServingManifest.load(tmp_path)
    manifest.validate_files(tmp_path)
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        manifest.validate_files(tmp_path)
