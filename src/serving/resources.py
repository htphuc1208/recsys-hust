"""Load and own all immutable resources used by the online pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.indexing.faiss_index import FaissVectorIndex
from src.indexing.multi_source import MultiSourceRetriever
from src.indexing.retrievers import PopularRetriever, VectorRetriever
from src.indexing.types import RetrievalQuery, RetrievalSource
from src.models.base import BaseRecommender
from src.serving.artifacts import ArtifactSpec, ServingManifest
from src.serving.config import ServingConfig


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


@dataclass(slots=True)
class ServingResources:
    manifest: ServingManifest
    mappings: IdMappings
    retriever: MultiSourceRetriever
    ranker: BaseRecommender | None = None
    config: ServingConfig | None = None

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
    def load(cls, config: ServingConfig) -> "ServingResources":
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
        retrievers = [PopularRetriever(popular_items, popular_scores)]

        ranker_path = cls._artifact_path(root, manifest, "ranker", required=False)
        ranker = BaseRecommender.load(ranker_path) if ranker_path is not None else None

        collaborative_path = cls._artifact_path(
            root, manifest, "collaborative_index", required=False
        )
        collaborative_metadata_path = cls._artifact_path(
            root, manifest, "collaborative_metadata", required=False
        )
        if collaborative_path is not None and collaborative_metadata_path is not None:
            index = FaissVectorIndex.load(collaborative_path, collaborative_metadata_path)
            user_factors = getattr(ranker, "user_factors", None)
            if user_factors is None:
                raise ValueError("collaborative index requires ranker.user_factors")

            def encode_user(query: RetrievalQuery) -> np.ndarray | None:
                if query.user_idx is None or query.user_idx >= len(user_factors):
                    return None
                return np.asarray(user_factors[query.user_idx], dtype=np.float32)

            retrievers.append(VectorRetriever(RetrievalSource.COLLABORATIVE, index, encode_user))

        return cls(
            manifest=manifest,
            mappings=mappings,
            retriever=MultiSourceRetriever(retrievers, config.retrieval),
            ranker=ranker,
            config=config,
        )

    def close(self) -> None:
        self.retriever.close()
