import time

import numpy as np
from fastapi import FastAPI, HTTPException

from src.models.registry import get_model
from src.serving.schemas import (
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    RecommendedItem,
)

app = FastAPI(
    title="RecSys HUST Recommendation API",
    description="High-performance recommendation inference service",
    version="0.1.0",
)

# Placeholder in-memory model (popularity default)
# In production, this loads serialized checkpoints/FAISS index from artifacts/
_model = get_model("popularity")


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="healthy", version="0.1.0")


@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendationRequest):
    start_time = time.perf_counter()
    try:
        # If model is not yet fitted with real data, return dummy recommendations for demo
        if not _model.is_fitted:
            recs = [101, 102, 103, 104, 105][: req.top_k]
        else:
            res_dict = _model.recommend(
                np.array([req.user_id]), top_k=req.top_k, filter_seen=req.filter_seen
            )
            recs = res_dict.get(req.user_id, [])

        items = [RecommendedItem(item_id=item_id, rank=idx + 1) for idx, item_id in enumerate(recs)]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return RecommendationResponse(
            user_id=req.user_id,
            recommendations=items,
            model_name="popularity",
            latency_ms=round(latency_ms, 3),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
