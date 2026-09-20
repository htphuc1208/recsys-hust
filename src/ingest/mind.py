"""Stream the MINDlarge TSV files into training-friendly Parquet datasets.

The raw archives already provide temporal train/dev/test splits. This module keeps
those splits separate and never modifies the source files under ``data/raw``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.common.logger import setup_logger

NEWS_COLUMNS = (
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
)
BEHAVIOR_COLUMNS = (
    "impression_id",
    "user_id",
    "timestamp",
    "history",
    "impressions",
)
SPLIT_DIRECTORIES = {
    "train": "MINDlarge_train",
    "dev": "MINDlarge_dev",
    "test": "MINDlarge_test",
}
TIMESTAMP_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def parse_history(raw_history: str) -> list[str]:
    """Return previously clicked news IDs in chronological order."""
    return raw_history.split() if raw_history else []


def parse_entity_ids(raw_entities: str) -> list[str]:
    """Extract Wikidata IDs from a MIND entity JSON column."""
    if not raw_entities:
        return []

    try:
        entities = json.loads(raw_entities)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid entity JSON") from exc

    return [entity["WikidataId"] for entity in entities if entity.get("WikidataId")]


def parse_impressions(
    raw_impressions: str, *, require_labels: bool
) -> tuple[list[str], list[int] | None]:
    """Parse candidate IDs and optional click labels from one impression.

    Train/dev candidates have the form ``N123-0`` or ``N123-1``. Test
    candidates contain only ``N123``. Mixed labeled and unlabeled candidates are
    rejected because they usually indicate a corrupt row or the wrong split.
    """
    news_ids: list[str] = []
    labels: list[int] = []
    saw_labeled = False
    saw_unlabeled = False

    for token in raw_impressions.split():
        news_id, separator, raw_label = token.rpartition("-")
        if separator and raw_label in {"0", "1"}:
            news_ids.append(news_id)
            labels.append(int(raw_label))
            saw_labeled = True
        else:
            news_ids.append(token)
            saw_unlabeled = True

    if saw_labeled and saw_unlabeled:
        raise ValueError("An impression mixes labeled and unlabeled candidates")
    if require_labels and saw_unlabeled:
        raise ValueError("Expected click labels in train/dev impressions")
    if not require_labels and saw_labeled:
        raise ValueError("Expected unlabeled candidates in test impressions")

    return news_ids, labels if saw_labeled else None


def _load_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required. Install the project first with: pip install -e ."
        ) from exc
    return pa, pq


@dataclass
class ParquetPartWriter:
    """Buffer dictionaries and write bounded-size Parquet part files."""

    output_dir: Path
    schema: Any
    chunk_size: int

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._rows: list[dict[str, Any]] = []
        self._part_number = 0
        self.row_count = 0

    def append(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self._rows:
            return

        pa, pq = _load_pyarrow()
        table = pa.Table.from_pylist(self._rows, schema=self.schema)
        part_path = self.output_dir / f"part-{self._part_number:05d}.parquet"
        pq.write_table(table, part_path, compression="zstd")
        self.row_count += len(self._rows)
        self._rows.clear()
        self._part_number += 1

    def close(self) -> None:
        self.flush()


def _iter_tsv(path: Path, expected_columns: int) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        # MIND is an unquoted TSV. News titles/abstracts may contain unmatched
        # double quotes (for example, an abstract starting with ``"What's``).
        # The csv module's default Excel dialect would treat that quote as the
        # start of a multi-line field and corrupt this row and the following row.
        reader = csv.reader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_number, row in enumerate(reader, start=1):
            if len(row) != expected_columns:
                raise ValueError(
                    f"{path}:{line_number} has {len(row)} columns; expected {expected_columns}"
                )
            yield row


def _schemas() -> tuple[Any, Any, Any]:
    pa, _ = _load_pyarrow()
    news_schema = pa.schema(
        [
            ("news_id", pa.string()),
            ("category", pa.string()),
            ("subcategory", pa.string()),
            ("title", pa.string()),
            ("abstract", pa.string()),
            ("url", pa.string()),
            ("title_entity_ids", pa.list_(pa.string())),
            ("abstract_entity_ids", pa.list_(pa.string())),
        ]
    )
    behaviors_schema = pa.schema(
        [
            ("impression_id", pa.int64()),
            ("user_id", pa.string()),
            ("timestamp", pa.timestamp("ms")),
            ("history", pa.list_(pa.string())),
            ("candidate_news_ids", pa.list_(pa.string())),
            ("labels", pa.list_(pa.int8())),
            ("history_length", pa.int32()),
            ("candidate_count", pa.int32()),
        ]
    )
    candidates_schema = pa.schema(
        [
            ("impression_id", pa.int64()),
            ("user_id", pa.string()),
            ("timestamp", pa.timestamp("ms")),
            ("news_id", pa.string()),
            ("position", pa.int32()),
            ("label", pa.int8()),
        ]
    )
    return news_schema, behaviors_schema, candidates_schema


def _process_news(raw_path: Path, output_dir: Path, chunk_size: int) -> int:
    news_schema, _, _ = _schemas()
    writer = ParquetPartWriter(output_dir, news_schema, chunk_size)

    for values in _iter_tsv(raw_path, len(NEWS_COLUMNS)):
        row = dict(zip(NEWS_COLUMNS, values, strict=True))
        writer.append(
            {
                "news_id": row["news_id"],
                "category": row["category"],
                "subcategory": row["subcategory"],
                "title": row["title"],
                "abstract": row["abstract"],
                "url": row["url"],
                "title_entity_ids": parse_entity_ids(row["title_entities"]),
                "abstract_entity_ids": parse_entity_ids(row["abstract_entities"]),
            }
        )

    writer.close()
    return writer.row_count


def _process_behaviors(
    raw_path: Path,
    output_dir: Path,
    *,
    split: str,
    chunk_size: int,
    explode_candidates: bool,
) -> tuple[int, int]:
    _, behaviors_schema, candidates_schema = _schemas()
    behaviors_writer = ParquetPartWriter(output_dir / "behaviors", behaviors_schema, chunk_size)
    candidates_writer = (
        ParquetPartWriter(output_dir / "candidates", candidates_schema, chunk_size)
        if explode_candidates
        else None
    )
    require_labels = split in {"train", "dev"}

    for values in _iter_tsv(raw_path, len(BEHAVIOR_COLUMNS)):
        row = dict(zip(BEHAVIOR_COLUMNS, values, strict=True))
        try:
            impression_id = int(row["impression_id"])
            timestamp = datetime.strptime(row["timestamp"], TIMESTAMP_FORMAT)
            history = parse_history(row["history"])
            news_ids, labels = parse_impressions(row["impressions"], require_labels=require_labels)
        except ValueError as exc:
            raise ValueError(
                f"Could not parse impression {row['impression_id']} in {raw_path}: {exc}"
            ) from exc

        behaviors_writer.append(
            {
                "impression_id": impression_id,
                "user_id": row["user_id"],
                "timestamp": timestamp,
                "history": history,
                "candidate_news_ids": news_ids,
                "labels": labels,
                "history_length": len(history),
                "candidate_count": len(news_ids),
            }
        )

        if candidates_writer is not None:
            row_labels: list[int | None]
            row_labels = labels if labels is not None else [None] * len(news_ids)
            for position, (news_id, label) in enumerate(zip(news_ids, row_labels, strict=True)):
                candidates_writer.append(
                    {
                        "impression_id": impression_id,
                        "user_id": row["user_id"],
                        "timestamp": timestamp,
                        "news_id": news_id,
                        "position": position,
                        "label": label,
                    }
                )

    behaviors_writer.close()
    if candidates_writer is not None:
        candidates_writer.close()

    candidate_count = candidates_writer.row_count if candidates_writer is not None else 0
    return behaviors_writer.row_count, candidate_count


def process_split(
    raw_root: Path,
    output_root: Path,
    split: str,
    *,
    chunk_size: int = 50_000,
    explode_candidates: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Process one official MIND split and return a small manifest."""
    if split not in SPLIT_DIRECTORIES:
        raise ValueError(f"Unknown split: {split}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    raw_split_dir = raw_root / SPLIT_DIRECTORIES[split]
    news_path = raw_split_dir / "news.tsv"
    behaviors_path = raw_split_dir / "behaviors.tsv"
    for required_path in (news_path, behaviors_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Missing MIND file: {required_path}")

    output_dir = output_root / split
    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_dir}. Pass --overwrite to rebuild it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    news_count = _process_news(news_path, output_dir / "news", chunk_size)
    behavior_count, candidate_count = _process_behaviors(
        behaviors_path,
        output_dir,
        split=split,
        chunk_size=chunk_size,
        explode_candidates=explode_candidates,
    )
    manifest = {
        "dataset": "MINDlarge",
        "split": split,
        "news_rows": news_count,
        "behavior_rows": behavior_count,
        "candidate_rows": candidate_count if explode_candidates else None,
        "candidates_exploded": explode_candidates,
        "raw_directory": str(raw_split_dir),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess the Microsoft MINDlarge dataset")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("data/raw/mind_large"),
        help="Directory containing MINDlarge_train/dev/test",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/interim/mind_large"),
        help="Destination for Parquet datasets",
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test", "all"),
        default="all",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument(
        "--explode-candidates",
        action="store_true",
        help="Also write one row per ranking candidate (large output)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing processed split",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logger("mind_ingest")
    splits = SPLIT_DIRECTORIES if args.split == "all" else (args.split,)

    for split in splits:
        logger.info("Processing MINDlarge %s", split)
        manifest = process_split(
            args.raw_root,
            args.output_root,
            split,
            chunk_size=args.chunk_size,
            explode_candidates=args.explode_candidates,
            overwrite=args.overwrite,
        )
        logger.info("Completed %s: %s", split, manifest)


if __name__ == "__main__":
    main()
