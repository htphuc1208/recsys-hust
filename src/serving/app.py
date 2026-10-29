import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

from src.serving.config import load_serving_config
from src.serving.pipeline import RecommendationPipeline
from src.serving.resources import ServingResources
from src.serving.schemas import (
    HealthResponse,
    ReadinessResponse,
    RecommendationRequest,
    RecommendationResponse,
)

logger = logging.getLogger(__name__)


def create_app(pipeline: RecommendationPipeline | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resources: ServingResources | None = None
        if pipeline is not None:
            app.state.pipeline = pipeline
            app.state.load_error = None
        else:
            try:
                resources = ServingResources.load(load_serving_config())
                app.state.pipeline = RecommendationPipeline(resources)
                app.state.load_error = None
            except Exception as exc:  # noqa: BLE001 - liveness remains available
                logger.exception("Serving bundle failed to load")
                app.state.pipeline = None
                app.state.load_error = type(exc).__name__
        yield
        if resources is not None:
            resources.close()

    api = FastAPI(
        title="RecSys HUST Recommendation API",
        description="Two-stage recommendation inference service",
        version="0.2.0",
        lifespan=lifespan,
    )

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        api.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @api.get("/", response_class=FileResponse)
    def index() -> FileResponse:
        index_file = static_dir / "index.html"
        if not index_file.exists():
            raise HTTPException(status_code=404, detail="Web UI not found")
        return FileResponse(str(index_file))

    @api.get("/health", response_model=HealthResponse)
    def health_check() -> HealthResponse:
        return HealthResponse(status="healthy", version="0.2.0")

    @api.get("/ready", response_model=ReadinessResponse)
    def readiness(request: Request) -> ReadinessResponse:
        active_pipeline = request.app.state.pipeline
        if active_pipeline is None:
            raise HTTPException(status_code=503, detail="Recommendation resources are not ready")
        return ReadinessResponse(
            status="ready",
            model_version=active_pipeline.resources.manifest.model_version,
        )

    @api.get("/api/personas")
    def sample_personas() -> list[dict]:
        from src.serving.metadata import get_sample_personas

        return get_sample_personas()

    @api.post("/recommend", response_model=RecommendationResponse)
    async def recommend(payload: RecommendationRequest, request: Request) -> RecommendationResponse:
        active_pipeline = request.app.state.pipeline
        if active_pipeline is None:
            raise HTTPException(status_code=503, detail="Recommendation resources are not ready")
        try:
            return await run_in_threadpool(active_pipeline.recommend, payload)
        except Exception as exc:
            logger.exception("Recommendation request failed")
            raise HTTPException(status_code=500, detail="Recommendation request failed") from exc

    @api.post("/api/recommend-rich")
    async def recommend_rich(payload: RecommendationRequest, request: Request) -> dict:
        active_pipeline = request.app.state.pipeline
        if active_pipeline is None:
            raise HTTPException(status_code=503, detail="Recommendation resources are not ready")
        try:
            resp: RecommendationResponse = await run_in_threadpool(
                active_pipeline.recommend, payload
            )
            data = resp.model_dump()
            from src.serving.metadata import get_news_metadata_batch

            rec_ids = [item["news_id"] for item in data.get("recommendations", [])]
            meta = get_news_metadata_batch(rec_ids)
            for item in data.get("recommendations", []):
                nid = item["news_id"]
                item["article"] = meta.get(
                    nid,
                    {
                        "title": f"Bản tin {nid}",
                        "category": "news",
                        "subcategory": "",
                        "abstract": "Thông tin bài viết từ hệ thống gợi ý RecSys HUST.",
                        "url": "#",
                    },
                )
            return data
        except Exception as exc:
            logger.exception("Rich recommendation request failed")
            raise HTTPException(status_code=500, detail="Rich recommendation failed") from exc

    return api


app = create_app()
