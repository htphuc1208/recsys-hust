"""Deterministic final ordering utilities."""

from src.indexing.types import Candidate


def stable_top_k(
    candidates: list[Candidate], scores: dict[int, float], top_k: int
) -> list[Candidate]:
    """Rank by model score with stable item-ID tie breaking."""
    return sorted(candidates, key=lambda item: (-scores[item.item_idx], item.item_idx))[:top_k]
