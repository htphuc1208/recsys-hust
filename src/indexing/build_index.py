"""CLI for building a versioned FAISS index from model or standalone embeddings."""

import argparse
from pathlib import Path

import numpy as np

from src.indexing.faiss_index import FaissVectorIndex
from src.models.base import BaseRecommender


def _load_vectors(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str]:
    if args.checkpoint:
        model = BaseRecommender.load(args.checkpoint)
        vectors = getattr(model, "item_factors", None)
        if vectors is None:
            raise ValueError("checkpoint does not expose item_factors")
        item_ids = np.arange(len(vectors), dtype=np.int64)
        return np.asarray(vectors), item_ids, model.__class__.__name__

    vectors = np.load(args.vectors, allow_pickle=False)
    item_ids = (
        np.load(args.item_ids, allow_pickle=False)
        if args.item_ids
        else np.arange(len(vectors), dtype=np.int64)
    )
    return vectors, item_ids, args.source_model


def build_index(args: argparse.Namespace) -> tuple[Path, Path]:
    vectors, item_ids, source_model = _load_vectors(args)
    index = FaissVectorIndex.build(
        vectors,
        item_ids,
        metric=args.metric,
        source_model=source_model,
        model_version=args.model_version,
    )
    output_path = Path(args.output)
    metadata_path = output_path.with_suffix(".meta.json")
    index.save(output_path, metadata_path)
    return output_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an exact FAISS candidate index")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--checkpoint", help="Pickled recommender exposing item_factors")
    inputs.add_argument("--vectors", help="Two-dimensional .npy embedding matrix")
    parser.add_argument("--item-ids", help="Optional aligned int64 .npy item ID array")
    parser.add_argument("--output", required=True, help="Output .faiss path")
    parser.add_argument("--metric", choices=("inner_product", "cosine"), default="inner_product")
    parser.add_argument("--source-model", default="standalone-embeddings")
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def main() -> None:
    index_path, metadata_path = build_index(parse_args())
    print(f"Wrote FAISS index: {index_path}")
    print(f"Wrote index metadata: {metadata_path}")


if __name__ == "__main__":
    main()
