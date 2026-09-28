"""Các hàm chia nhóm item theo độ phổ biến trong tập train."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEAD = 0
LONG_TAIL = 1
ZERO_CLICK = 2


@dataclass(frozen=True)
class ItemPopularityGroups:
    """Lưu nhãn nhóm popularity cho từng item_idx."""

    group_codes: np.ndarray
    head_items: int
    long_tail_items: int
    zero_click_items: int
    head_fraction: float


def build_item_popularity_groups(
    train_click_counts: np.ndarray,
    head_fraction: float = 0.20,
) -> ItemPopularityGroups:
    """Chia item có click thành Head/Long-tail; zero-click được giữ thành nhóm riêng."""
    counts = np.asarray(train_click_counts, dtype=np.int64).reshape(-1)
    if counts.size == 0:
        raise ValueError("train_click_counts không được rỗng.")
    if np.any(counts < 0):
        raise ValueError("Số click train không được âm.")
    if not 0.0 < head_fraction < 1.0:
        raise ValueError("head_fraction phải nằm trong khoảng (0, 1).")

    codes = np.full(counts.size, ZERO_CLICK, dtype=np.int8)
    clicked_items = np.flatnonzero(counts > 0)

    if clicked_items.size == 0:
        return ItemPopularityGroups(
            group_codes=codes,
            head_items=0,
            long_tail_items=0,
            zero_click_items=int(counts.size),
            head_fraction=head_fraction,
        )

    # Stable sort để item_idx quyết định thứ tự nếu cùng tần suất.
    order = clicked_items[np.argsort(-counts[clicked_items], kind="stable")]
    head_count = max(1, int(np.ceil(clicked_items.size * head_fraction)))
    head_ids = order[:head_count]
    long_tail_ids = order[head_count:]

    codes[head_ids] = HEAD
    codes[long_tail_ids] = LONG_TAIL

    return ItemPopularityGroups(
        group_codes=codes,
        head_items=int(head_ids.size),
        long_tail_items=int(long_tail_ids.size),
        zero_click_items=int(np.sum(counts == 0)),
        head_fraction=head_fraction,
    )


def split_relevant_items_by_popularity(
    true_items: set[int],
    group_codes: np.ndarray,
) -> dict[str, set[int]]:
    """Tách ground-truth item thành Head, Long-tail và Zero-click."""
    groups = {
        "head_item": set(),
        "long_tail_item": set(),
        "zero_click_item": set(),
    }

    for item_id in true_items:
        if item_id < 0 or item_id >= group_codes.size:
            raise ValueError(f"item_idx ngoài phạm vi popularity groups: {item_id}")
        code = int(group_codes[item_id])
        if code == HEAD:
            groups["head_item"].add(item_id)
        elif code == LONG_TAIL:
            groups["long_tail_item"].add(item_id)
        else:
            groups["zero_click_item"].add(item_id)

    return groups


def split_relevant_items_by_catalog_status(
    candidates: list[int],
    labels: list[int],
    candidate_seen_in_train_mask: list[bool],
) -> dict[str, set[int]]:
    """Tách relevant item thành catalog-cold và catalog-warm."""
    if not (
        len(candidates) == len(labels) == len(candidate_seen_in_train_mask)
    ):
        raise ValueError("Candidate, label và seen mask phải cùng độ dài.")

    cold_items: set[int] = set()
    warm_items: set[int] = set()

    for item_id, label, seen_in_train in zip(
        candidates,
        labels,
        candidate_seen_in_train_mask,
    ):
        if int(label) != 1:
            continue
        if bool(seen_in_train):
            warm_items.add(int(item_id))
        else:
            cold_items.add(int(item_id))

    return {
        "catalog_cold_item": cold_items,
        "catalog_warm_item": warm_items,
    }
