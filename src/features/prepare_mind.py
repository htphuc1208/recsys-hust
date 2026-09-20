"""Build leakage-safe, train-ready features from interim MINDlarge Parquet files."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.common.logger import setup_logger
from src.ingest.mind import ParquetPartWriter

SPLITS = ("train", "dev", "test")
UNKNOWN_INDEX = -1


def _load_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required. Install the project first with: pip install -e ."
        ) from exc
    return pa, pq


def _parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {directory}")
    return files


def _iter_batches(
    directory: Path, columns: list[str], batch_size: int
) -> Iterable[Any]:
    _, pq = _load_pyarrow()
    for path in _parquet_files(directory):
        parquet_file = pq.ParquetFile(path)
        yield from parquet_file.iter_batches(batch_size=batch_size, columns=columns)


def build_mappings(
    input_root: Path, *, batch_size: int
) -> tuple[dict[str, int], dict[str, int], set[str], dict[str, str]]:
    """Fit users on train and assign distinct indices to every available news item.

    News IDs and metadata are observable at serving time, so indexing catalogs
    from dev/test does not leak click labels. Train items are indexed first to
    keep their indices stable and are marked separately for collaborative models.
    """
    train_root = input_root / "train"
    item_to_index: dict[str, int] = {}
    item_first_seen_split: dict[str, str] = {}
    user_to_index: dict[str, int] = {}

    for split in SPLITS:
        news_dir = input_root / split / "news"
        if not news_dir.is_dir():
            continue
        for batch in _iter_batches(news_dir, ["news_id"], batch_size):
            for news_id in batch.column("news_id").to_pylist():
                if news_id and news_id not in item_to_index:
                    item_to_index[news_id] = len(item_to_index)
                    item_first_seen_split[news_id] = split

    train_item_ids = {
        news_id
        for news_id, first_seen_split in item_first_seen_split.items()
        if first_seen_split == "train"
    }

    for batch in _iter_batches(train_root / "behaviors", ["user_id"], batch_size):
        for user_id in batch.column("user_id").to_pylist():
            if user_id and user_id not in user_to_index:
                user_to_index[user_id] = len(user_to_index)

    return user_to_index, item_to_index, train_item_ids, item_first_seen_split


def _schemas() -> dict[str, Any]:
    pa, _ = _load_pyarrow()
    return {
        "user_mapping": pa.schema(
            [("user_id", pa.string()), ("user_idx", pa.int64())]
        ),
        "item_mapping": pa.schema(
            [
                ("news_id", pa.string()),
                ("item_idx", pa.int64()),
                ("seen_in_train", pa.bool_()),
                ("first_seen_split", pa.string()),
            ]
        ),
        "news": pa.schema(
            [
                ("news_id", pa.string()),
                ("item_idx", pa.int64()),
                ("is_known_item", pa.bool_()),
                ("seen_in_train", pa.bool_()),
                ("category", pa.string()),
                ("subcategory", pa.string()),
                ("title", pa.string()),
                ("abstract", pa.string()),
                ("url", pa.string()),
                ("title_entity_ids", pa.list_(pa.string())),
                ("abstract_entity_ids", pa.list_(pa.string())),
            ]
        ),
        "impressions": pa.schema(
            [
                ("impression_id", pa.int64()),
                ("user_id", pa.string()),
                ("user_idx", pa.int64()),
                ("is_known_user", pa.bool_()),
                ("timestamp", pa.timestamp("ms")),
                ("history_news_ids", pa.list_(pa.string())),
                ("history_item_idxs", pa.list_(pa.int64())),
                ("history_known_mask", pa.list_(pa.bool_())),
                ("history_seen_in_train_mask", pa.list_(pa.bool_())),
                ("candidate_news_ids", pa.list_(pa.string())),
                ("candidate_item_idxs", pa.list_(pa.int64())),
                ("candidate_known_mask", pa.list_(pa.bool_())),
                ("candidate_seen_in_train_mask", pa.list_(pa.bool_())),
                ("labels", pa.list_(pa.int8())),
            ]
        ),
        "interaction": pa.schema(
            [
                ("impression_id", pa.int64()),
                ("user_id", pa.string()),
                ("user_idx", pa.int64()),
                ("timestamp", pa.timestamp("ms")),
                ("news_id", pa.string()),
                ("item_idx", pa.int64()),
                ("is_known_item", pa.bool_()),
                ("seen_in_train", pa.bool_()),
                ("position", pa.int32()),
                ("label", pa.int8()),
            ]
        ),
    }


def _write_mapping(
    mapping: dict[str, int],
    output_dir: Path,
    *,
    id_column: str,
    index_column: str,
    schema: Any,
    chunk_size: int,
) -> None:
    writer = ParquetPartWriter(output_dir, schema, chunk_size)
    for raw_id, index in mapping.items():
        writer.append({id_column: raw_id, index_column: index})
    writer.close()


def _write_item_mapping(
    mapping: dict[str, int],
    train_item_ids: set[str],
    first_seen_split: dict[str, str],
    output_dir: Path,
    *,
    schema: Any,
    chunk_size: int,
) -> None:
    writer = ParquetPartWriter(output_dir, schema, chunk_size)
    for news_id, item_idx in mapping.items():
        writer.append(
            {
                "news_id": news_id,
                "item_idx": item_idx,
                "seen_in_train": news_id in train_item_ids,
                "first_seen_split": first_seen_split[news_id],
            }
        )
    writer.close()


def _prepare_news(
    input_dir: Path,
    output_dir: Path,
    item_to_index: dict[str, int],
    train_item_ids: set[str],
    *,
    batch_size: int,
    chunk_size: int,
) -> dict[str, int]:
    columns = [
        "news_id",
        "category",
        "subcategory",
        "title",
        "abstract",
        "url",
        "title_entity_ids",
        "abstract_entity_ids",
    ]
    writer = ParquetPartWriter(output_dir, _schemas()["news"], chunk_size)
    rows = unknown_items = 0
    for batch in _iter_batches(input_dir, columns, batch_size):
        data = batch.to_pydict()
        for values in zip(*(data[column] for column in columns), strict=True):
            row = dict(zip(columns, values, strict=True))
            item_idx = item_to_index.get(row["news_id"], UNKNOWN_INDEX)
            writer.append(
                {
                    **row,
                    "item_idx": item_idx,
                    "is_known_item": item_idx != UNKNOWN_INDEX,
                    "seen_in_train": row["news_id"] in train_item_ids,
                }
            )
            rows += 1
            unknown_items += item_idx == UNKNOWN_INDEX
    writer.close()
    return {"news_rows": rows, "unknown_news_rows": unknown_items}


def _sample_positions(
    labels: list[int], negative_ratio: int, rng: random.Random
) -> list[int]:
    positive_positions = [index for index, label in enumerate(labels) if label == 1]
    negative_positions = [index for index, label in enumerate(labels) if label == 0]
    negative_count = min(len(negative_positions), negative_ratio * len(positive_positions))
    sampled_negatives = rng.sample(negative_positions, negative_count)
    return sorted([*positive_positions, *sampled_negatives])


def _interaction_row(
    *,
    impression_id: int,
    user_id: str,
    user_idx: int,
    timestamp: Any,
    news_id: str,
    item_idx: int,
    seen_in_train: bool,
    position: int,
    label: int,
) -> dict[str, Any]:
    return {
        "impression_id": impression_id,
        "user_id": user_id,
        "user_idx": user_idx,
        "timestamp": timestamp,
        "news_id": news_id,
        "item_idx": item_idx,
        "is_known_item": item_idx != UNKNOWN_INDEX,
        "seen_in_train": seen_in_train,
        "position": position,
        "label": label,
    }


def _prepare_behaviors(
    input_dir: Path,
    output_dir: Path,
    user_to_index: dict[str, int],
    item_to_index: dict[str, int],
    train_item_ids: set[str],
    *,
    split: str,
    batch_size: int,
    chunk_size: int,
    negative_ratio: int,
    seed: int,
) -> dict[str, int | float]:
    schemas = _schemas()
    impressions_writer = ParquetPartWriter(
        output_dir / "impressions", schemas["impressions"], chunk_size
    )
    positives_writer = (
        ParquetPartWriter(
            output_dir / "positive_interactions", schemas["interaction"], chunk_size
        )
        if split in {"train", "dev"}
        else None
    )
    pointwise_writer = (
        ParquetPartWriter(output_dir / "pointwise", schemas["interaction"], chunk_size)
        if split == "train"
        else None
    )
    rng = random.Random(seed)
    columns = [
        "impression_id",
        "user_id",
        "timestamp",
        "history",
        "candidate_news_ids",
        "labels",
    ]
    rows = unknown_users = history_items = unknown_history_items = 0
    candidate_items = unknown_candidate_items = positive_rows = pointwise_rows = 0
    unseen_train_history_items = unseen_train_candidate_items = 0

    for batch in _iter_batches(input_dir, columns, batch_size):
        data = batch.to_pydict()
        for values in zip(*(data[column] for column in columns), strict=True):
            row = dict(zip(columns, values, strict=True))
            user_idx = user_to_index.get(row["user_id"], UNKNOWN_INDEX)
            history_ids = row["history"] or []
            candidate_ids = row["candidate_news_ids"] or []
            history_indices = [
                item_to_index.get(news_id, UNKNOWN_INDEX) for news_id in history_ids
            ]
            candidate_indices = [
                item_to_index.get(news_id, UNKNOWN_INDEX) for news_id in candidate_ids
            ]
            history_mask = [index != UNKNOWN_INDEX for index in history_indices]
            candidate_mask = [index != UNKNOWN_INDEX for index in candidate_indices]
            history_train_mask = [news_id in train_item_ids for news_id in history_ids]
            candidate_train_mask = [
                news_id in train_item_ids for news_id in candidate_ids
            ]
            impressions_writer.append(
                {
                    "impression_id": row["impression_id"],
                    "user_id": row["user_id"],
                    "user_idx": user_idx,
                    "is_known_user": user_idx != UNKNOWN_INDEX,
                    "timestamp": row["timestamp"],
                    "history_news_ids": history_ids,
                    "history_item_idxs": history_indices,
                    "history_known_mask": history_mask,
                    "history_seen_in_train_mask": history_train_mask,
                    "candidate_news_ids": candidate_ids,
                    "candidate_item_idxs": candidate_indices,
                    "candidate_known_mask": candidate_mask,
                    "candidate_seen_in_train_mask": candidate_train_mask,
                    "labels": row["labels"],
                }
            )

            labels = row["labels"]
            if labels is not None:
                positive_positions = [
                    position for position, label in enumerate(labels) if label == 1
                ]
                for position in positive_positions:
                    positives_writer.append(
                        _interaction_row(
                            impression_id=row["impression_id"],
                            user_id=row["user_id"],
                            user_idx=user_idx,
                            timestamp=row["timestamp"],
                            news_id=candidate_ids[position],
                            item_idx=candidate_indices[position],
                            seen_in_train=candidate_train_mask[position],
                            position=position,
                            label=1,
                        )
                    )
                    positive_rows += 1

                if pointwise_writer is not None:
                    for position in _sample_positions(labels, negative_ratio, rng):
                        pointwise_writer.append(
                            _interaction_row(
                                impression_id=row["impression_id"],
                                user_id=row["user_id"],
                                user_idx=user_idx,
                                timestamp=row["timestamp"],
                                news_id=candidate_ids[position],
                                item_idx=candidate_indices[position],
                                seen_in_train=candidate_train_mask[position],
                                position=position,
                                label=labels[position],
                            )
                        )
                        pointwise_rows += 1

            rows += 1
            unknown_users += user_idx == UNKNOWN_INDEX
            history_items += len(history_indices)
            unknown_history_items += history_mask.count(False)
            unseen_train_history_items += history_train_mask.count(False)
            candidate_items += len(candidate_indices)
            unknown_candidate_items += candidate_mask.count(False)
            unseen_train_candidate_items += candidate_train_mask.count(False)

    impressions_writer.close()
    if positives_writer is not None:
        positives_writer.close()
    if pointwise_writer is not None:
        pointwise_writer.close()

    return {
        "impression_rows": rows,
        "positive_interaction_rows": positive_rows,
        "pointwise_rows": pointwise_rows,
        "unknown_user_rows": unknown_users,
        "history_items": history_items,
        "unknown_history_items": unknown_history_items,
        "unseen_train_history_items": unseen_train_history_items,
        "candidate_items": candidate_items,
        "unknown_candidate_items": unknown_candidate_items,
        "unseen_train_candidate_items": unseen_train_candidate_items,
        "unknown_user_pct": round(100 * unknown_users / (rows or 1), 4),
        "unknown_history_item_pct": round(
            100 * unknown_history_items / (history_items or 1), 4
        ),
        "unknown_candidate_item_pct": round(
            100 * unknown_candidate_items / (candidate_items or 1), 4
        ),
        "unseen_train_history_item_pct": round(
            100 * unseen_train_history_items / (history_items or 1), 4
        ),
        "unseen_train_candidate_item_pct": round(
            100 * unseen_train_candidate_items / (candidate_items or 1), 4
        ),
    }


def prepare_dataset(
    input_root: Path,
    output_root: Path,
    *,
    splits: tuple[str, ...] = SPLITS,
    batch_size: int = 10_000,
    chunk_size: int = 10_000,
    negative_ratio: int = 4,
    seed: int = 42,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Prepare selected splits using mappings fitted exclusively on train."""
    if input_root.resolve() == output_root.resolve():
        raise ValueError("Input and output roots must be different")
    if output_root.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_root}. Pass --overwrite to rebuild it."
        )

    if output_root.exists() and overwrite:
        if set(splits) == set(SPLITS):
            shutil.rmtree(output_root)
            output_root.mkdir(parents=True)
        else:
            for s in splits:
                split_dir = output_root / s
                if split_dir.exists():
                    shutil.rmtree(split_dir)
    else:
        output_root.mkdir(parents=True)

    (
        user_to_index,
        item_to_index,
        train_item_ids,
        item_first_seen_split,
    ) = build_mappings(input_root, batch_size=batch_size)
    schemas = _schemas()
    _write_mapping(
        user_to_index,
        output_root / "mappings" / "users",
        id_column="user_id",
        index_column="user_idx",
        schema=schemas["user_mapping"],
        chunk_size=chunk_size,
    )
    _write_item_mapping(
        item_to_index,
        train_item_ids,
        item_first_seen_split,
        output_root / "mappings" / "items",
        schema=schemas["item_mapping"],
        chunk_size=chunk_size,
    )

    manifest: dict[str, Any] = {
        "dataset": "MINDlarge",
        "mapping_sources": {
            "users": "train behaviors",
            "items": "all available news catalogs (IDs and metadata only)",
        },
        "unknown_index": UNKNOWN_INDEX,
        "num_train_users": len(user_to_index),
        "num_items": len(item_to_index),
        "num_train_items": len(train_item_ids),
        "negative_ratio": negative_ratio,
        "seed": seed,
        "splits": {},
    }
    for split in splits:
        split_input = input_root / split
        if not split_input.is_dir():
            raise FileNotFoundError(f"Interim split does not exist: {split_input}")
        split_output = output_root / split
        news_stats = _prepare_news(
            split_input / "news",
            split_output / "news",
            item_to_index,
            train_item_ids,
            batch_size=batch_size,
            chunk_size=chunk_size,
        )
        behavior_stats = _prepare_behaviors(
            split_input / "behaviors",
            split_output,
            user_to_index,
            item_to_index,
            train_item_ids,
            split=split,
            batch_size=batch_size,
            chunk_size=chunk_size,
            negative_ratio=negative_ratio,
            seed=seed,
        )
        manifest["splits"][split] = {**news_stats, **behavior_stats}

    with (output_root / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare leakage-safe model features from interim MINDlarge data"
    )
    parser.add_argument("--input-root", type=Path, default=Path("data/interim/mind_large"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/mind_large"))
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--negative-ratio", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("--batch-size and --chunk-size must be positive")
    if args.negative_ratio < 0:
        raise ValueError("--negative-ratio cannot be negative")

    logger = setup_logger("prepare_mind")
    splits = SPLITS if args.split == "all" else (args.split,)
    logger.info("Fitting users from train and indexing all available news catalogs")
    manifest = prepare_dataset(
        args.input_root,
        args.output_root,
        splits=splits,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    logger.info(
        "Prepared %s train users and %s catalog items (%s seen in train) in %s",
        manifest["num_train_users"],
        manifest["num_items"],
        manifest["num_train_items"],
        args.output_root,
    )


if __name__ == "__main__":
    main()
