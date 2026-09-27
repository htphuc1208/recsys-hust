"""Build a self-contained, checksummed serving bundle from a trained recommender."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.indexing.faiss_index import FaissVectorIndex
from src.models.base import BaseRecommender
from src.serving.artifacts import ArtifactSpec, ServingManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def build_serving_bundle(
    *,
    checkpoint: str | Path,
    users_mapping: str | Path,
    items_mapping: str | Path,
    output_dir: str | Path,
    model_version: str,
    git_commit: str = "unknown",
    dataset: str = "MINDlarge",
    overwrite: bool = False,
) -> Path:
    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"serving bundle already exists: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "mappings").mkdir(exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    (root / "indexes").mkdir(exist_ok=True)

    model = BaseRecommender.load(checkpoint)
    users_output = root / "mappings" / "users.parquet"
    items_output = root / "mappings" / "items.parquet"
    num_users = _write_mapping(Path(users_mapping), users_output, ["user_id", "user_idx"])
    num_items = _write_mapping(Path(items_mapping), items_output, ["news_id", "item_idx"])

    ranker_output = root / "models" / "ranker.pkl"
    shutil.copy2(checkpoint, ranker_output)
    popular_ids, popular_scores = _popular_items(model)
    popularity_output = root / "popularity.npz"
    np.savez_compressed(popularity_output, item_ids=popular_ids, scores=popular_scores)

    artifacts: dict[str, ArtifactSpec] = {}

    def add_artifact(name: str, path: Path, *, required: bool = True) -> None:
        artifacts[name] = ArtifactSpec(
            path=path.relative_to(root).as_posix(),
            sha256=_sha256(path),
            required=required,
        )

    add_artifact("users_mapping", users_output)
    add_artifact("items_mapping", items_output)
    add_artifact("ranker", ranker_output)
    add_artifact("popularity", popularity_output)

    item_factors = getattr(model, "item_factors", None)
    embedding_dimension = None
    if item_factors is not None:
        item_matrix = np.asarray(item_factors, dtype=np.float32)
        embedding_dimension = int(item_matrix.shape[1])
        vector_index = FaissVectorIndex.build(
            item_matrix,
            np.arange(len(item_matrix), dtype=np.int64),
            metric="inner_product",
            source_model=model.__class__.__name__,
            model_version=model_version,
        )
        index_output = root / "indexes" / "collaborative.faiss"
        metadata_output = root / "indexes" / "collaborative.meta.json"
        vector_index.save(index_output, metadata_output)
        add_artifact("collaborative_index", index_output, required=False)
        add_artifact("collaborative_metadata", metadata_output, required=False)

    manifest = ServingManifest(
        model_version=model_version,
        dataset=dataset,
        git_commit=git_commit,
        item_mapping_version=_sha256(items_output),
        num_users=num_users,
        num_items=num_items,
        embedding_dimension=embedding_dimension,
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
        overwrite=args.overwrite,
    )
    print(f"Wrote serving manifest: {manifest}")


if __name__ == "__main__":
    main()
