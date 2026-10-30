"""Load and own all immutable resources used by the online pipeline."""

import gc
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from src.indexing.branches import ModelBranches, model_branches
from src.indexing.faiss_index import FaissVectorIndex
from src.indexing.multi_source import MultiSourceRetriever
from src.indexing.retrievers import PopularRetriever, Retriever, VectorRetriever
from src.indexing.types import RetrievalQuery, RetrievalSource
from src.models.base import BaseRecommender
from src.serving.artifacts import ArtifactSpec, ServingManifest
from src.serving.config import ServingConfig

logger = logging.getLogger(__name__)

QueryEncoder = Callable[[RetrievalQuery], np.ndarray | None]


def _limit_intra_op_threads(num_threads: int) -> None:
    """Cap per-call threading in FAISS, BLAS and torch to avoid oversubscription."""
    from threadpoolctl import threadpool_limits

    from src.indexing.faiss_index import _import_faiss

    threadpool_limits(limits=num_threads)
    _import_faiss().omp_set_num_threads(num_threads)
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(num_threads)


@dataclass(frozen=True, slots=True)
class IdMappings:
    user_to_idx: dict[str, int]
    news_to_idx: dict[str, int]
    idx_to_news: dict[int, str]

    @classmethod
    def load(cls, users_path: Path, items_path: Path) -> "IdMappings":
        users = pq.read_table(users_path, columns=["user_id", "user_idx"]).to_pydict()
        items = pq.read_table(items_path, columns=["news_id", "item_idx"]).to_pydict()
        user_to_idx = dict(zip(users["user_id"], users["user_idx"], strict=True))
        news_to_idx = dict(zip(items["news_id"], items["item_idx"], strict=True))
        return cls(
            user_to_idx={str(key): int(value) for key, value in user_to_idx.items()},
            news_to_idx={str(key): int(value) for key, value in news_to_idx.items()},
            idx_to_news={int(value): str(key) for key, value in news_to_idx.items()},
        )


def _nonzero(vector: np.ndarray) -> np.ndarray | None:
    vector = np.asarray(vector, dtype=np.float32)
    return vector if np.isfinite(vector).all() and np.any(vector) else None


def _collaborative_encoder(user_vectors: np.ndarray) -> QueryEncoder:
    def encode(query: RetrievalQuery) -> np.ndarray | None:
        if query.user_idx is None or query.user_idx >= len(user_vectors):
            return None
        return _nonzero(user_vectors[query.user_idx])

    return encode


def _cache_history_encoding(model: Any, maxsize: int = 4096) -> None:
    """Memoize ``encode_history`` on the instance.

    Retrieval, the gating network and the sequential branch each encode the same history
    within one request; the SASRec forward pass is the dominant per-request cost.
    Callers must treat the returned vector as read-only since it is shared.
    """
    encode = model.encode_history

    @lru_cache(maxsize=maxsize)
    def cached(history: tuple[int, ...]) -> np.ndarray:
        return encode(history)

    model.encode_history = lambda history_item_idxs: cached(tuple(history_item_idxs))


def _sequential_encoder(model: Any) -> QueryEncoder:
    def encode(query: RetrievalQuery) -> np.ndarray | None:
        if not query.history_item_idxs:
            return None
        return _nonzero(model.encode_history(query.history_item_idxs))

    return encode


def _content_encoder(item_vectors: np.ndarray) -> QueryEncoder:
    def encode(query: RetrievalQuery) -> np.ndarray | None:
        history = [idx for idx in query.history_item_idxs if 0 <= idx < len(item_vectors)]
        if not history:
            return None
        return _nonzero(np.asarray(item_vectors[history], dtype=np.float32).mean(axis=0))

    return encode


@dataclass(slots=True)
class ServingResources:
    manifest: ServingManifest
    mappings: IdMappings
    retriever: MultiSourceRetriever
    ranker: BaseRecommender | None = None
    config: ServingConfig | None = None
    ranker_executor: ThreadPoolExecutor | None = None

    def __post_init__(self) -> None:
        if self.ranker_executor is None:
            workers = self.config.rank_workers if self.config is not None else 4
            self.ranker_executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="ranker"
            )

    @staticmethod
    def _artifact_path(
        root: Path,
        manifest: ServingManifest,
        name: str,
        *,
        required: bool = True,
    ) -> Path | None:
        spec: ArtifactSpec | None = manifest.artifacts.get(name)
        if spec is None:
            if required:
                raise ValueError(f"manifest does not define required artifact '{name}'")
            return None
        path = root / spec.path
        if not path.is_file() and not required and not spec.required:
            return None
        return path

    @classmethod
    def _load_index(
        cls, root: Path, manifest: ServingManifest, source: str
    ) -> FaissVectorIndex | None:
        index_path = cls._artifact_path(root, manifest, f"{source}_index", required=False)
        metadata_path = cls._artifact_path(root, manifest, f"{source}_metadata", required=False)
        if index_path is None or metadata_path is None:
            return None
        return FaissVectorIndex.load(index_path, metadata_path)

    @classmethod
    def _vector_retrievers(
        cls,
        root: Path,
        manifest: ServingManifest,
        ranker: BaseRecommender | None,
        branches: ModelBranches,
    ) -> list[Retriever]:
        retrievers: list[Retriever] = []

        collaborative_index = cls._load_index(root, manifest, "collaborative")
        if collaborative_index is not None:
            user_vectors_path = cls._artifact_path(
                root, manifest, "cf_user_vectors", required=False
            )
            if user_vectors_path is not None:
                user_vectors = np.load(user_vectors_path, mmap_mode="r", allow_pickle=False)
            else:
                # Bundles built before cf_user_vectors existed read factors from the ranker.
                user_vectors = getattr(ranker, "user_factors", None)
                if user_vectors is None:
                    raise ValueError("collaborative index requires user query vectors")
            retrievers.append(
                VectorRetriever(
                    RetrievalSource.COLLABORATIVE,
                    collaborative_index,
                    _collaborative_encoder(user_vectors),
                )
            )

        sequential_index = cls._load_index(root, manifest, "sequential")
        if sequential_index is not None:
            if branches.sequential is None:
                raise ValueError("sequential index requires a ranker with a sequential branch")
            retrievers.append(
                VectorRetriever(
                    RetrievalSource.SEQUENTIAL,
                    sequential_index,
                    _sequential_encoder(branches.sequential),
                )
            )

        content_index = cls._load_index(root, manifest, "content")
        if content_index is not None:
            vectors_path = cls._artifact_path(root, manifest, "content_item_vectors")
            assert vectors_path is not None
            item_vectors = np.load(vectors_path, mmap_mode="r", allow_pickle=False)
            retrievers.append(
                VectorRetriever(
                    RetrievalSource.CONTENT, content_index, _content_encoder(item_vectors)
                )
            )
        return retrievers

    @classmethod
    def load(cls, config: ServingConfig) -> "ServingResources":
        _limit_intra_op_threads(config.intra_op_threads)
        root = config.bundle_dir.resolve()
        manifest = ServingManifest.load(root)
        manifest.validate_files(root)

        users_path = cls._artifact_path(root, manifest, "users_mapping")
        items_path = cls._artifact_path(root, manifest, "items_mapping")
        popularity_path = cls._artifact_path(root, manifest, "popularity")
        assert users_path is not None and items_path is not None and popularity_path is not None
        mappings = IdMappings.load(users_path, items_path)

        with np.load(popularity_path, allow_pickle=False) as payload:
            popular_items = payload["item_ids"].astype(np.int64).tolist()
            popular_scores = payload["scores"].astype(np.float32).tolist()

        ranker_path = cls._artifact_path(root, manifest, "ranker", required=False)
        ranker = BaseRecommender.load(ranker_path) if ranker_path is not None else None
        branches = model_branches(ranker)
        if branches.sequential is not None:
            _cache_history_encoding(branches.sequential)
        features_manifest = cls._artifact_path(
            root, manifest, "content_feature:manifest.json", required=False
        )
        if branches.content is not None and features_manifest is not None:
            # Content features ship inside the bundle; never read the training data directory.
            branches.content.feature_dir = features_manifest.parent

        retrievers: list[Retriever] = [PopularRetriever(popular_items, popular_scores)]
        retrievers.extend(cls._vector_retrievers(root, manifest, ranker, branches))

        resources = cls(
            manifest=manifest,
            mappings=mappings,
            retriever=MultiSourceRetriever(retrievers, config.retrieval),
            ranker=ranker,
            config=config,
        )
        if config.warmup:
            resources.warm_up()
        # Model objects live for the whole process; exclude them from cyclic GC scans,
        # which otherwise cause multi-hundred-millisecond tail-latency pauses.
        gc.collect()
        gc.freeze()
        return resources

    def warm_up(self) -> None:
        """Run one request through retrieval and ranking so lazy loads happen at startup."""
        popular = self.retriever.retrieve(RetrievalQuery(user_idx=None)).item_ids
        history = tuple(int(item_idx) for item_idx in popular[:5])
        query = RetrievalQuery(user_idx=0, history_item_idxs=history)
        try:
            retrieval = self.retriever.retrieve(query)
            if self.ranker is not None and len(retrieval):
                limit = self.config.rerank_limit if self.config is not None else 200
                item_ids = retrieval.item_ids[:limit]
                self.ranker.predict(
                    np.array([0], dtype=np.int64), item_ids, history_item_idxs=list(history)
                )
        except Exception:  # noqa: BLE001 - warm-up must never block startup
            logger.warning("Serving warm-up failed", exc_info=True)

    def close(self) -> None:
        self.retriever.close()
        if self.ranker_executor is not None:
            self.ranker_executor.shutdown(wait=False, cancel_futures=True)
