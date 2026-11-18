"""Build a self-contained, checksummed serving bundle from a trained recommender.

Supports single-branch checkpoints (e.g. BPR) and ``AdaptiveGatingHybrid``. For each
retrieval branch found in the model the bundle stores the embedding matrices and an exact
FAISS index:

- collaborative: BPR/MF item factors (+ item bias column) and per-user query vectors
- sequential:    SASRec item embeddings; queries are encoded online from the history
- content:       weighted TF-IDF/one-hot features reduced to ``content_dim`` with SVD
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.indexing.branches import model_branches, observed_item_ids
from src.indexing.faiss_index import FaissVectorIndex
from src.models.base import BaseRecommender
from src.serving.artifacts import ArtifactSpec, ServingManifest, restore_path, sha256_file

CONTENT_FEATURE_DIR = "features/content"
CONTENT_METADATA_FILES = ("manifest.json", "item_index.json")


class _PortablePickler(pickle.Pickler):
    """Pickle OS-specific Path objects (e.g. WindowsPath) as plain POSIX strings."""

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, PurePath):
            return restore_path, (obj.as_posix(),)
        return NotImplemented


def _write_mapping(source: Path, destination: Path, columns: list[str]) -> int:
    table = ds.dataset(str(source), format="parquet").to_table(columns=columns)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
    return len(table)


def _popular_items(model: BaseRecommender) -> tuple[np.ndarray, np.ndarray]:
    items = getattr(model, "popular_items", None) or getattr(model, "popular_items_fallback", None)
    if not items:
        raise ValueError("model does not contain a popularity fallback")
    item_ids = np.asarray(items, dtype=np.int64)
    item_scores = getattr(model, "item_scores", {})
    if item_scores:
        scores = np.asarray(
            [item_scores.get(int(item), 0.0) for item in item_ids], dtype=np.float32
        )
    else:
        scores = np.arange(len(item_ids), 0, -1, dtype=np.float32)
    return item_ids, scores


def _move_to_cpu(*models: Any) -> None:
    """Serving runs on CPU; move torch modules so the checkpoint unpickles without CUDA."""
    import torch

    for model in models:
        if model is None:
            continue
        for name, value in vars(model).items():
            if isinstance(value, torch.nn.Module):
                value.to("cpu")
            elif isinstance(value, torch.device) and value.type != "cpu":
                setattr(model, name, torch.device("cpu"))


def _save_ranker(model: BaseRecommender, destination: Path) -> None:
    with destination.open("wb") as file:
        _PortablePickler(file, protocol=pickle.HIGHEST_PROTOCOL).dump(model)


def _collaborative_vectors(model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (item_ids, item_vectors, user_vectors) so that user·item equals the CF score."""
    item_factors = np.asarray(model.item_factors, dtype=np.float32)
    user_factors = np.asarray(model.user_factors, dtype=np.float32)
    item_biases = getattr(model, "item_biases", None)
    if item_biases is not None:
        item_factors = np.hstack(
            [item_factors, np.asarray(item_biases, dtype=np.float32).reshape(-1, 1)]
        )
        user_factors = np.hstack([user_factors, np.ones((len(user_factors), 1), np.float32)])

    user_mask = getattr(model, "observed_user_mask", None)
    if user_mask is not None and len(user_mask) == len(user_factors):
        # Untrained users get a zero vector so serving treats them as cold.
        user_factors[~np.asarray(user_mask, dtype=bool)] = 0.0

    item_ids = observed_item_ids(model, len(item_factors))
    return item_ids, item_factors[item_ids], user_factors


def _content_item_vectors(model: Any, dimension: int, seed: int) -> np.ndarray:
    """Dense item vectors whose dot products approximate the weighted per-feature cosine."""
    model._ensure_features_loaded()
    blocks = []
    for name, weight in model.feature_weights.items():
        if weight <= 0.0:
            continue
        matrix = sparse.csr_matrix(model._features[name], dtype=np.float32)
        norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        inverse = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        blocks.append(sparse.diags(inverse * np.sqrt(weight)) @ matrix)
    combined = sparse.hstack(blocks, format="csr", dtype=np.float32)

    if combined.shape[1] <= dimension:
        vectors = combined.toarray()
    else:
        svd = TruncatedSVD(n_components=dimension, random_state=seed)
        vectors = svd.fit_transform(combined)
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def _content_feature_files(model: Any) -> list[str]:
    from src.models.content_based.recommender import FEATURE_FILES

    weighted = [FEATURE_FILES[name] for name, w in model.feature_weights.items() if w > 0.0]
    return [*CONTENT_METADATA_FILES, *weighted]


def build_serving_bundle(
    *,
    checkpoint: str | Path,
    users_mapping: str | Path,
    items_mapping: str | Path,
    output_dir: str | Path,
    model_version: str,
    git_commit: str = "unknown",
    dataset: str = "MINDlarge",
    content_dim: int = 64,
    seed: int = 42,
    overwrite: bool = False,
) -> Path:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"serving bundle already exists: {root}")
    for directory in ("mappings", "models", "indexes", "embeddings"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    model = BaseRecommender.load(checkpoint)
    branches = model_branches(model)
    users_output = root / "mappings" / "users.parquet"
    items_output = root / "mappings" / "items.parquet"
    num_users = _write_mapping(Path(users_mapping), users_output, ["user_id", "user_idx"])
    num_items = _write_mapping(Path(items_mapping), items_output, ["news_id", "item_idx"])

    popular_ids, popular_scores = _popular_items(model)
    popularity_output = root / "popularity.npz"
    np.savez_compressed(popularity_output, item_ids=popular_ids, scores=popular_scores)

    artifacts: dict[str, ArtifactSpec] = {}
    retrieval_sources = ["popular"]

    def add_artifact(name: str, path: Path, *, required: bool = True) -> None:
        artifacts[name] = ArtifactSpec(
            path=path.relative_to(root).as_posix(),
            sha256=sha256_file(path),
            required=required,
        )

    def add_index(source: str, vectors: np.ndarray, item_ids: np.ndarray, metric: str) -> int:
        index = FaissVectorIndex.build(
            vectors,
            item_ids,
            metric=metric,
            source_model=model.__class__.__name__,
            model_version=model_version,
        )
        index_output = root / "indexes" / f"{source}.faiss"
        metadata_output = root / "indexes" / f"{source}.meta.json"
        index.save(index_output, metadata_output)
        add_artifact(f"{source}_index", index_output)
        add_artifact(f"{source}_metadata", metadata_output)
        retrieval_sources.append(source)
        return int(vectors.shape[1])

    add_artifact("users_mapping", users_output)
    add_artifact("items_mapping", items_output)
    add_artifact("popularity", popularity_output)

    embedding_dimension = None
    if branches.collaborative is not None:
        item_ids, item_vectors, user_vectors = _collaborative_vectors(branches.collaborative)
        user_output = root / "embeddings" / "cf_user_vectors.npy"
        np.save(user_output, user_vectors)
        add_artifact("cf_user_vectors", user_output)
        embedding_dimension = add_index("collaborative", item_vectors, item_ids, "inner_product")

    if branches.sequential is not None:
        item_factors = np.asarray(branches.sequential.item_factors, dtype=np.float32)
        item_ids = observed_item_ids(branches.sequential, len(item_factors))
        add_index("sequential", item_factors[item_ids], item_ids, "inner_product")

    if branches.content is not None:
        vectors = _content_item_vectors(branches.content, content_dim, seed)
        vectors_output = root / "embeddings" / "content_item_vectors.npy"
        np.save(vectors_output, vectors)
        add_artifact("content_item_vectors", vectors_output)
        item_ids = np.flatnonzero(np.linalg.norm(vectors, axis=1) > 0).astype(np.int64)
        add_index("content", vectors[item_ids], item_ids, "cosine")

        feature_root = root / CONTENT_FEATURE_DIR
        feature_root.mkdir(parents=True, exist_ok=True)
        for filename in _content_feature_files(branches.content):
            destination = feature_root / filename
            shutil.copy2(Path(branches.content.feature_dir) / filename, destination)
            add_artifact(f"content_feature:{filename}", destination)
        branches.content.feature_dir = Path(CONTENT_FEATURE_DIR)
        branches.content._features = {}

    _move_to_cpu(model, branches.collaborative, branches.sequential, branches.content)
    ranker_output = root / "models" / "ranker.pkl"
    _save_ranker(model, ranker_output)
    add_artifact("ranker", ranker_output)

    manifest = ServingManifest(
        model_version=model_version,
        dataset=dataset,
        git_commit=git_commit,
        item_mapping_version=sha256_file(items_output),
        num_users=num_users,
        num_items=num_items,
        embedding_dimension=embedding_dimension,
        model_type=model.__class__.__name__,
        retrieval_sources=retrieval_sources,
        artifacts=artifacts,
    )
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest.model_dump(mode="json"), file, indent=2, sort_keys=True)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a production serving bundle")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--users-mapping", required=True)
    parser.add_argument("--items-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument("--dataset", default="MINDlarge")
    parser.add_argument("--content-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_serving_bundle(
        checkpoint=args.checkpoint,
        users_mapping=args.users_mapping,
        items_mapping=args.items_mapping,
        output_dir=args.output_dir,
        model_version=args.model_version,
        git_commit=args.git_commit,
        dataset=args.dataset,
        content_dim=args.content_dim,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Wrote serving manifest: {manifest}")


if __name__ == "__main__":
    main()
