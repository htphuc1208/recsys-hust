"""Locate the retrieval branches (CF, sequential, content) inside a trained recommender.

Hybrid models expose their sub-models as ``cf_model``, ``seq_model`` and ``content_model``;
single-branch checkpoints (BPR, SASRec, content-based) are treated as one branch.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ModelBranches:
    collaborative: Any | None = None
    sequential: Any | None = None
    content: Any | None = None


def _is_collaborative(model: Any) -> bool:
    return (
        getattr(model, "user_factors", None) is not None
        and getattr(model, "item_factors", None) is not None
    )


def _is_sequential(model: Any) -> bool:
    return (
        callable(getattr(model, "encode_history", None))
        and getattr(model, "item_factors", None) is not None
    )


def _is_content(model: Any) -> bool:
    return hasattr(model, "feature_dir") and hasattr(model, "feature_weights")


def model_branches(model: Any | None) -> ModelBranches:
    if model is None:
        return ModelBranches()
    if any(hasattr(model, name) for name in ("cf_model", "seq_model", "content_model")):
        cf = getattr(model, "cf_model", None)
        seq = getattr(model, "seq_model", None)
        content = getattr(model, "content_model", None)
    else:
        cf = seq = content = model
    return ModelBranches(
        collaborative=cf if cf is not None and _is_collaborative(cf) else None,
        sequential=seq if seq is not None and _is_sequential(seq) else None,
        content=content if content is not None and _is_content(content) else None,
    )


def observed_item_ids(model: Any, num_rows: int) -> np.ndarray:
    """Item rows that were actually trained; untouched rows hold random initial weights."""
    mask = getattr(model, "observed_item_mask", None)
    if mask is not None and len(mask) == num_rows:
        return np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)
    seen = getattr(model, "popular_items_fallback", None) or getattr(model, "popular_items", None)
    if seen:
        ids = np.unique(np.asarray(seen, dtype=np.int64))
        return ids[(ids >= 0) & (ids < num_rows)]
    return np.arange(num_rows, dtype=np.int64)
