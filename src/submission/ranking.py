"""Xếp hạng theo *vị trí candidate gốc*, không xuất danh sách news ID."""

from __future__ import annotations

import re

import numpy as np

LINE_PATTERN = re.compile(r"^([1-9]\d*) \[([1-9]\d*(?:,[1-9]\d*)*)\]$")


def scores_to_ranks(scores: np.ndarray | list[float]) -> list[int]:
    """Rank bắt đầu từ 1, score cao đứng trước, đồng điểm giữ vị trí gốc."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Danh sách score phải không rỗng và chỉ chứa số hữu hạn.")
    order = np.argsort(-values, kind="stable")
    return positions_to_ranks(order.tolist(), values.size)


def positions_to_ranks(order: list[int], candidate_count: int) -> list[int]:
    """Đổi thứ tự các vị trí [1, 2, 0] thành rank tại vị trí gốc [3, 1, 2]."""
    if candidate_count <= 0 or len(order) != candidate_count:
        raise ValueError("Thứ tự dự đoán không khớp số candidate hoặc rỗng.")
    if sorted(order) != list(range(candidate_count)):
        raise ValueError("Thứ tự dự đoán phải là hoán vị đầy đủ các vị trí candidate.")
    result = [0] * candidate_count
    for rank, original_position in enumerate(order, start=1):
        result[original_position] = rank
    return result


def format_prediction_line(impression_id: int, ranks: list[int]) -> str:
    if not isinstance(impression_id, int) or impression_id <= 0:
        raise ValueError("impression_id phải là số nguyên dương.")
    if not ranks or sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("Ranks phải là hoán vị của 1..N.")
    return f"{impression_id} [{','.join(map(str, ranks))}]\n"


def parse_prediction_line(line: str) -> tuple[int, list[int]]:
    match = LINE_PATTERN.fullmatch(line.rstrip("\r\n"))
    if match is None:
        raise ValueError(f"Dòng không đúng định dạng Microsoft MIND: {line[:120]!r}")
    impression_id = int(match.group(1))
    ranks = [int(value) for value in match.group(2).split(",")]
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError(f"Impression {impression_id}: rank thiếu hoặc trùng.")
    return impression_id, ranks
