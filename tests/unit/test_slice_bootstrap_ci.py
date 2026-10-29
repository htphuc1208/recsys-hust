"""Unit tests cho Bootstrap theo đúng mẫu của từng slice, kể cả P3 paired."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

from src.evaluation.bootstrap import BootstrapSettings


class _Schema:
    names = [
        "impression_id", "user_idx", "history_item_idxs", "candidate_item_idxs",
        "labels", "candidate_seen_in_train_mask",
    ]


class _Batch:
    def __init__(self, rows):
        self.rows = rows

    def to_pydict(self):
        return {key: [row[key] for row in self.rows] for key in _Schema.names}


class _Dataset:
    schema = _Schema()

    def __init__(self, rows):
        self.rows = rows

    def count_rows(self):
        return len(self.rows)

    def to_batches(self, columns):
        return [_Batch(self.rows)]


class _Model:
    def predict(self, user_ids, item_ids, history_item_idxs=None):
        return np.array([10 - int(item) for item in item_ids], dtype=float)


class _Reranker:
    categories = ["tech", "sports", "tech", "finance", "finance"]
    train_click_counts = np.array([10, 5, 1, 0, 0], dtype=np.int64)


class _Pipeline:
    reranker = _Reranker()

    def score_and_rerank(self, model, user_idx, candidate_item_idxs, history_item_idxs, top_k):
        scores = model.predict(np.array([user_idx]), np.array(candidate_item_idxs))
        order = np.argsort(-scores, kind="stable")
        baseline = [candidate_item_idxs[i] for i in order]
        # Dời item cuối lên đầu để có thay đổi rõ ở một số slice.
        after = baseline[-1:] + baseline[:-1]
        return baseline, after, scores


def _load_modules(monkeypatch):
    if "pyarrow" not in sys.modules and importlib.util.find_spec("pyarrow") is None:
        # Chỉ stub cho unit test in-memory trong môi trường thiếu PyArrow.
        stub = types.ModuleType("pyarrow")
        stub.__path__ = []
        stub_dataset = types.ModuleType("pyarrow.dataset")
        stub_dataset.dataset = lambda *args, **kwargs: None
        monkeypatch.setitem(sys.modules, "pyarrow", stub)
        monkeypatch.setitem(sys.modules, "pyarrow.dataset", stub_dataset)
    import src.evaluate as evaluator
    import src.evaluation.p3_evaluation as p3
    return evaluator, p3


def _rows():
    return [
        {"impression_id": 1, "user_idx": 10, "history_item_idxs": [0, 1, 2, 0, 1, 2],
         "candidate_item_idxs": [0, 1, 2, 3, 4], "labels": [1, 0, 0, 0, 0],
         "candidate_seen_in_train_mask": [True, True, True, False, False]},
        {"impression_id": 2, "user_idx": 10, "history_item_idxs": [0, 1, 2, 0, 1, 2],
         "candidate_item_idxs": [0, 1, 2, 3, 4], "labels": [0, 0, 1, 0, 0],
         "candidate_seen_in_train_mask": [True, True, True, False, False]},
        {"impression_id": 3, "user_idx": -1, "history_item_idxs": [0],
         "candidate_item_idxs": [0, 1, 2, 3, 4], "labels": [0, 0, 0, 1, 0],
         "candidate_seen_in_train_mask": [True, True, True, False, False]},
        {"impression_id": 4, "user_idx": -1, "history_item_idxs": [0],
         "candidate_item_idxs": [0, 1, 2, 3, 4], "labels": [0, 0, 0, 0, 1],
         "candidate_seen_in_train_mask": [True, True, True, False, False]},
    ]


def test_single_model_ci_by_slice_and_missing_slice(monkeypatch):
    evaluator, _ = _load_modules(monkeypatch)
    monkeypatch.setattr(evaluator.ds, "dataset", lambda *args, **kwargs: _Dataset(_rows()))
    setting = BootstrapSettings(num_resamples=70, seed=42, unit="user", batch_size=10)
    result = evaluator.evaluate_impressions(
        _Model(), Path("unused"), k_list=[5, 10],
        train_click_counts=np.array([10, 5, 1, 0, 0]),
        bootstrap_settings=setting,
    )
    per_slice = result["bootstrap"]["slices"]
    assert per_slice["overall"]["metrics"] == result["bootstrap"]["metrics"]
    assert per_slice["overall"]["count"] == 4
    assert per_slice["cold_user"]["count"] == 2
    assert per_slice["warm_user"]["count"] == 2
    assert per_slice["head_item"]["count"] == 1
    assert per_slice["long_tail_item"]["count"] == 1
    assert per_slice["zero_click_item"]["count"] == 2
    assert per_slice["catalog_cold_item"]["count"] == 2
    assert per_slice["catalog_warm_item"]["count"] == 2
    assert per_slice["warm_item"]["count"] == 0
    assert per_slice["warm_item"]["metrics"] == {}
    for name, info in per_slice.items():
        assert info["count"] == result["slices"][name]["count"]
        for metric, summary in info["metrics"].items():
            assert abs(summary["mean"] - result["slices"][name]["metrics"][metric]) < 2e-5
            assert summary["ci"][0] <= summary["ci"][1]


def test_p3_paired_ci_by_slice(monkeypatch):
    _, p3 = _load_modules(monkeypatch)
    monkeypatch.setattr(p3.ds, "dataset", lambda *args, **kwargs: _Dataset(_rows()))
    settings = BootstrapSettings(num_resamples=70, seed=42, unit="user", batch_size=10)
    results = p3.evaluate_p3_impressions(
        _Model(), Path("unused"), _Pipeline(), k_list=[5, 10],
        all_catalog_items=set(range(5)), bootstrap_settings=settings,
    )
    per_slice = results["comparison"]["bootstrap"]["slices"]
    assert per_slice["overall"]["metrics"]["NDCG@10"] == results["comparison"]["bootstrap"]["metrics"]["NDCG@10"]
    assert per_slice["catalog_cold_item"]["count"] == 2
    assert per_slice["warm_item"]["metrics"] == {}
    for name, info in per_slice.items():
        assert info["count"] == results["baseline"]["slices"][name]["count"]
        for metric, summary in info["metrics"].items():
            before = results["baseline"]["slices"][name]["metrics"][metric]
            after = results["reranked"]["slices"][name]["metrics"][metric]
            assert abs(summary["baseline_mean"] - before) < 2e-5
            assert abs(summary["reranked_mean"] - after) < 2e-5
            assert summary["delta_ci"][0] <= summary["delta_ci"][1]
