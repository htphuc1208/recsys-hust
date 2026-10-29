"""Kiểm định TXT/ZIP và thứ tự impression trước khi nộp MIND."""

from __future__ import annotations

import hashlib
from pathlib import Path
from itertools import zip_longest
from zipfile import ZipFile

from src.submission.io import iter_impressions, iter_original_behaviors, verify_original_row
from src.submission.ranking import parse_prediction_line


def validate_submission(
    prediction_txt: str | Path,
    impressions_path: str | Path,
    *,
    prediction_zip: str | Path | None = None,
    original_behaviors: str | Path | None = None,
    batch_size: int = 2048,
    expected_count: int | None = None,
    split: str = "test",
) -> dict:
    """Duyệt lại *toàn bộ* file, kiểm tra rank/ID/order/độ dài/ZIP."""
    txt = Path(prediction_txt)
    if not txt.is_file():
        raise FileNotFoundError(f"Không có prediction.txt: {txt}")
    original = iter_original_behaviors(original_behaviors, split=split) if original_behaviors else None
    raw_iter = iter(original) if original is not None else None
    seen_ids: set[int] = set()
    count = 0
    candidates = 0

    with txt.open("r", encoding="utf-8", newline="") as text_file:
        impressions = iter_impressions(impressions_path, batch_size=batch_size)

        for row, line in zip_longest(impressions, text_file):
            if row is None or line is None:
                raise ValueError("Số dòng prediction không bằng số impressions trong Test.")
            count += 1
            impression_id, ranks = parse_prediction_line(line)
            if impression_id in seen_ids:
                raise ValueError(f"Impression ID trùng: {impression_id}")
            seen_ids.add(impression_id)
            candidates_ids = row["candidate_news_ids"] or []
            if len(ranks) != len(candidates_ids):
                raise ValueError(f"Impression {impression_id}: số rank != số candidate.")
            if impression_id != int(row["impression_id"]):
                raise ValueError(f"Dòng {count}: thứ tự impression bị thay đổi.")
            candidates += len(candidates_ids)
            if raw_iter is not None:
                reference = next(raw_iter, None)
                if reference is None:
                    raise ValueError("TSV gốc có ít dòng hơn predictions.")
                verify_original_row(impression_id, candidates_ids, reference, count)

    if raw_iter is not None and next(raw_iter, None) is not None:
        raise ValueError("TSV gốc có nhiều impression hơn predictions.")
    if count == 0 or (expected_count is not None and count != expected_count):
        raise ValueError("Số impression rỗng hoặc không khớp số dự kiến.")

    sha = hashlib.sha256()
    with txt.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(block)
    digest = sha.hexdigest()
    if prediction_zip is not None:
        path = Path(prediction_zip)
        with ZipFile(path) as archive:
            if archive.namelist() != ["prediction.txt"]:
                raise ValueError("ZIP chỉ được chứa prediction.txt tại thư mục gốc.")
            if archive.testzip() is not None:
                raise ValueError("ZIP bị lỗi CRC.")
            with archive.open("prediction.txt") as member, txt.open("rb") as source:
                while True:
                    a, b = member.read(1024 * 1024), source.read(1024 * 1024)
                    if a != b:
                        raise ValueError("Nội dung ZIP khác prediction.txt.")
                    if not a:
                        break

    return {
        "impressions": count,
        "candidates": candidates,
        "txt_sha256": digest,
        "zip_valid": prediction_zip is not None,
        "original_order_verified": original_behaviors is not None,
    }


def validate_partial_submission(
    prediction_txt: str | Path,
    impressions_path: str | Path,
    *,
    prediction_zip: str | Path | None = None,
    original_behaviors: str | Path | None = None,
    batch_size: int = 2048,
    expected_count: int,
    split: str = "dev",
) -> dict:
    """Kiểm tra prefix impressions cho Dev smoke-test; không coi là submission hoàn chỉnh."""
    txt = Path(prediction_txt)
    seen: set[int] = set()
    count = 0
    candidates = 0
    original = iter_original_behaviors(original_behaviors, split=split) if original_behaviors else None
    with txt.open("r", encoding="utf-8") as source:
        impressions = iter_impressions(impressions_path, batch_size=batch_size)
        for line in source:
            row = next(impressions, None)
            if row is None:
                raise ValueError("Có nhiều prediction hơn impression trong Dev.")
            count += 1
            identity, ranks = parse_prediction_line(line)
            if identity in seen or identity != int(row["impression_id"]):
                raise ValueError("Impression ID trùng hoặc sai thứ tự trong Dev.")
            seen.add(identity)
            candidates_ids = row["candidate_news_ids"] or []
            if len(candidates_ids) != len(ranks):
                raise ValueError("Số rank sai trong Dev.")
            candidates += len(ranks)
            if original is not None:
                reference = next(original, None)
                if reference is None:
                    raise ValueError("TSV gốc có ít dòng hơn Dev sample.")
                verify_original_row(identity, candidates_ids, reference, count)
    if count == 0 or count != expected_count:
        raise ValueError("Số dòng Dev sample không khớp.")
    if prediction_zip is not None:
        with ZipFile(prediction_zip) as archive:
            if archive.namelist() != ["prediction.txt"] or archive.testzip() is not None:
                raise ValueError("ZIP Dev sample không hợp lệ.")
            if archive.read("prediction.txt") != txt.read_bytes():
                raise ValueError("ZIP và TXT Dev sample khác nhau.")
    return {
        "impressions": count, "candidates": candidates,
        "txt_sha256": hashlib.sha256(txt.read_bytes()).hexdigest(),
        "zip_valid": prediction_zip is not None,
        "original_order_verified": original_behaviors is not None,
        "partial": True,
    }
