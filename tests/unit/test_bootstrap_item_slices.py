"""Các test không phụ thuộc PyArrow cho Bootstrap và item slicing."""

import numpy as np
import pytest

from src.evaluation.bootstrap import BootstrapSettings, paired_bootstrap_comparison
from src.evaluation.item_slices import HEAD, LONG_TAIL, ZERO_CLICK, build_item_popularity_groups


def test_item_popularity_groups_keep_zero_click_separate() -> None:
    """Head/Long-tail chỉ xét item có click; zero-click được giữ riêng."""
    groups = build_item_popularity_groups(np.array([10, 5, 1, 0, 0]), head_fraction=0.4)
    assert groups.group_codes.tolist() == [HEAD, HEAD, LONG_TAIL, ZERO_CLICK, ZERO_CLICK]
    assert groups.head_items == 2
    assert groups.long_tail_items == 1
    assert groups.zero_click_items == 2


def test_paired_bootstrap_zero_delta_has_zero_interval() -> None:
    """Baseline giống P3 thì paired delta và CI phải bằng 0."""
    users = np.array([1, 1, 2, 3], dtype=np.int64)
    values = np.array([[1.0], [0.5], [0.0], [1.0]], dtype=np.float64)
    settings = BootstrapSettings(num_resamples=100, unit="user", seed=7, batch_size=10)
    result = paired_bootstrap_comparison(users, values, values, ["NDCG@10"], settings)
    assert result["NDCG@10"]["delta"] == 0.0
    assert result["NDCG@10"]["delta_ci"] == [0.0, 0.0]


def test_head_fraction_validation() -> None:
    """Head fraction phải nằm trong (0, 1)."""
    with pytest.raises(ValueError):
        build_item_popularity_groups(np.array([1, 2]), head_fraction=1.0)
