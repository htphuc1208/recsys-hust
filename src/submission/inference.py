"""Dự đoán ứng viên MIND và giữ đầy đủ vị trí để nộp leaderboard."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.models.base import BaseRecommender
from src.reranking.pipeline import RerankingPipeline
from src.submission.ranking import positions_to_ranks


@dataclass
class RankResult:
    ranks: list[int]
    unknown_candidates: int = 0
    p3_fallback_reason: str | None = None


def infer_ranks(
    model: BaseRecommender,
    *,
    user_idx: int,
    candidate_item_idxs: list[int],
    history_item_idxs: list[int] | None,
    reranker: RerankingPipeline | None = None,
) -> RankResult:
    """Không bỏ candidate âm: đưa item chưa mapping xuống cuối, vẫn đủ rank.

    P3 không nhận candidate ID trùng. Trường hợp đó dùng stable baseline và
    ghi rõ fallback; không tự loại trùng vì Microsoft kiểm tra theo vị trí.
    """
    item_ids = np.asarray(candidate_item_idxs, dtype=np.int64).reshape(-1)
    if item_ids.size == 0:
        raise ValueError("Một impression không có candidate.")
    known_positions = np.flatnonzero(item_ids >= 0)
    unknown_positions = np.flatnonzero(item_ids < 0)

    if known_positions.size:
        scores = np.asarray(
            model.predict(
                np.array([int(user_idx)], dtype=np.int64),
                item_ids[known_positions],
                history_item_idxs=history_item_idxs or [],
            ),
            dtype=np.float64,
        ).reshape(-1)
        if scores.size != known_positions.size or not np.all(np.isfinite(scores)):
            raise ValueError("Model trả số score sai hoặc chứa NaN/Inf.")
        baseline_order = np.argsort(-scores, kind="stable")
        ordered_known = known_positions[baseline_order].tolist()
    else:
        scores = np.empty(0, dtype=np.float64)
        ordered_known = []

    reason = None
    if reranker is not None and known_positions.size:
        known_ids = item_ids[known_positions].tolist()
        if len(set(known_ids)) != len(known_ids):
            reason = "DUPLICATE_CANDIDATE_ITEM_IDS"
        else:
            reordered_ids = reranker.rerank(known_ids, scores)
            if Counter(reordered_ids) != Counter(known_ids):
                raise ValueError("P3 thay đổi tập ứng viên; không thể xuất submission hợp lệ.")
            index_by_id = {
                item_id: int(known_positions[pos]) for pos, item_id in enumerate(known_ids)
            }
            ordered_known = [index_by_id[item_id] for item_id in reordered_ids]

    order = ordered_known + unknown_positions.tolist()
    return RankResult(
        ranks=positions_to_ranks(order, int(item_ids.size)),
        unknown_candidates=int(unknown_positions.size),
        p3_fallback_reason=reason,
    )
