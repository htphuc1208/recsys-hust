from pydantic import BaseModel, Field


class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="Target user identifier")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of items to recommend")
    filter_seen: bool = Field(
        default=True, description="Whether to filter out previously interacted items"
    )


class RecommendedItem(BaseModel):
    item_id: int
    score: float | None = None
    rank: int


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: list[RecommendedItem]
    model_name: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    version: str
