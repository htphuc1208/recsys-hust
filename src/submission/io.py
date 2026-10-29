"""Đọc impressions theo đúng thứ tự Parquet và kiểm tra thứ tự TSV gốc."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterator

REQUIRED_COLUMNS = (
    "impression_id",
    "user_idx",
    "history_item_idxs",
    "candidate_item_idxs",
    "candidate_news_ids",
    "labels",
)


def iter_impressions(path: str | Path, batch_size: int = 2048) -> Iterator[dict[str, Any]]:
    """Không dùng dataset scanning song song: đọc part và row theo thứ tự cố định."""
    import pyarrow.parquet as pq

    directory = Path(path)
    files = sorted(directory.glob("*.parquet")) if directory.is_dir() else [directory]
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError(f"Không tìm thấy impressions Parquet: {directory}")
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")
    for file in files:
        parquet = pq.ParquetFile(file)
        missing = set(REQUIRED_COLUMNS) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{file}: thiếu cột {sorted(missing)}")
        for batch in parquet.iter_batches(columns=list(REQUIRED_COLUMNS), batch_size=batch_size):
            pydict = batch.to_pydict()
            for index in range(batch.num_rows):
                yield {name: pydict[name][index] for name in REQUIRED_COLUMNS}


def iter_original_behaviors(path: str | Path, *, split: str) -> Iterator[tuple[int, list[str]]]:
    """Đối chiếu raw behaviors: Dev có nhãn, Test không có nhãn.

    Không tự đoán split: Test phải từ chối nhãn click, kể cả khi dữ liệu sai.
    """
    if split not in {"dev", "test"}:
        raise ValueError(f"Split không hỗ trợ để xác minh behaviors: {split}")
    from src.ingest.mind import parse_impressions

    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for line_no, parts in enumerate(reader, start=1):
            if len(parts) != 5:
                raise ValueError(f"behaviors.tsv dòng {line_no}: yêu cầu 5 cột.")
            try:
                impression_id = int(parts[0])
            except ValueError as exc:
                raise ValueError(f"behaviors.tsv dòng {line_no}: impression ID không hợp lệ.") from exc
            news_ids, _ = parse_impressions(parts[4], require_labels=(split == "dev"))
            yield impression_id, news_ids


def original_behavior_path(split: str) -> Path:
    names = {"test": "MINDlarge_test", "dev": "MINDlarge_dev"}
    return Path("data/raw/mind_large") / names[split] / "behaviors.tsv"


def verify_original_row(
    impression_id: int, candidate_news_ids: list[str], reference: tuple[int, list[str]], row_number: int
) -> None:
    expected_id, expected_news_ids = reference
    if impression_id != expected_id:
        raise ValueError(
            f"Dòng {row_number}: impression ID {impression_id} khác behaviors.tsv ({expected_id})."
        )
    if candidate_news_ids != expected_news_ids:
        raise ValueError(f"Dòng {row_number}: candidate order khác behaviors.tsv gốc.")
