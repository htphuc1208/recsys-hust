"""Typed serving configuration loaded from YAML and environment variables."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from src.indexing.types import RetrievalSource


class SourceConfig(BaseModel):
    enabled: bool = True
    limit: int = Field(ge=1, le=10_000)
    weight: float = Field(default=1.0, gt=0)
    timeout_ms: int = Field(default=50, ge=1, le=60_000)


class RetrievalConfig(BaseModel):
    target_candidates: int = Field(default=800, ge=1)
    min_candidates: int = Field(default=500, ge=1)
    max_candidates: int = Field(default=1000, ge=1)
    rrf_constant: int = Field(default=60, ge=1)
    sources: dict[RetrievalSource, SourceConfig]

    @model_validator(mode="after")
    def validate_candidate_limits(self) -> "RetrievalConfig":
        if not self.min_candidates <= self.target_candidates <= self.max_candidates:
            raise ValueError(
                "candidate limits must satisfy min_candidates <= target_candidates "
                "<= max_candidates"
            )
        if RetrievalSource.POPULAR not in self.sources:
            raise ValueError("a popular source is required for deterministic fallback")
        return self


class ServingConfig(BaseModel):
    bundle_dir: Path = Path("artifacts/serving/current")
    retrieval: RetrievalConfig
    rank_timeout_ms: int = Field(default=100, ge=1, le=60_000)
    rerank_limit: int = Field(default=200, ge=1, le=10_000)
    warmup: bool = True


def load_serving_config(path: str | Path | None = None) -> ServingConfig:
    """Load serving settings, allowing deployment paths to come from env vars."""
    config_path = Path(path or os.getenv("SERVING_CONFIG", "configs/serving/mind_production.yaml"))
    with config_path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}

    bundle_override = os.getenv("MODEL_BUNDLE_DIR")
    if bundle_override:
        payload["bundle_dir"] = bundle_override
    return ServingConfig.model_validate(payload)
