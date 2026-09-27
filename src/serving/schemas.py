from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReasonCode(str, Enum):
    SIMILAR_TO_RECENT_HISTORY = "SIMILAR_TO_RECENT_HISTORY"
    CONTINUES_RECENT_INTEREST = "CONTINUES_RECENT_INTEREST"
    MATCHES_YOUR_TASTE = "MATCHES_YOUR_TASTE"
    TRENDING_IN_CATEGORY = "TRENDING_IN_CATEGORY"
    POPULAR_NOW = "POPULAR_NOW"


class FallbackReason(str, Enum):
    COLD_USER = "COLD_USER"
    EMPTY_HISTORY = "EMPTY_HISTORY"
    RETRIEVER_TIMEOUT = "RETRIEVER_TIMEOUT"
    RETRIEVER_ERROR = "RETRIEVER_ERROR"
    RANKER_TIMEOUT = "RANKER_TIMEOUT"
    RANKER_ERROR = "RANKER_ERROR"


class RequestContext(BaseModel):
    category: str | None = None
    subcategory: str | None = None
    device: str | None = None
    locale: str | None = None


class RecommendationRequest(BaseModel):
    user_id: str | None = Field(default=None, description="External user ID, for example U87243")
    history_news_ids: list[str] = Field(
        default_factory=list,
        max_length=500,
        description="External news IDs ordered from oldest to newest",
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of items to recommend")
    filter_seen: bool = Field(
        default=True, description="Whether to filter out previously interacted items"
    )
    context: RequestContext = Field(default_factory=RequestContext)


class RecommendedItem(BaseModel):
    news_id: str
    score: float | None = None
    rank: int = Field(ge=1)
    reason_code: ReasonCode
    reason_evidence: dict[str, Any] = Field(default_factory=dict)


class RecommendationResponse(BaseModel):
    user_id: str | None
    recommendations: list[RecommendedItem]
    model_version: str
    strategy: str
    degraded: bool = False
    fallback_reason: FallbackReason | None = None
    latency_ms: float = Field(ge=0)


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadinessResponse(BaseModel):
    status: str
    model_version: str | None = None
