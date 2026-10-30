"""
import sys
import types
"""

import numpy as np

"""
# Container test environment may not ship PyArrow. Stub only enough to import src.evaluate.
pyarrow_module = types.ModuleType("pyarrow")
pyarrow_dataset_module = types.ModuleType("pyarrow.dataset")
pyarrow_dataset_module.dataset = lambda *args, **kwargs: None
sys.modules.setdefault("pyarrow", pyarrow_module)
sys.modules.setdefault("pyarrow.dataset", pyarrow_dataset_module)
"""

from src.evaluate import evaluate_impressions
from src.evaluation.bootstrap import BootstrapSettings, bootstrap_confidence_intervals


class _FakeSchema:
    def __init__(self, names: list[str]):
        self.names = names


class _FakeBatch:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def to_pydict(self) -> dict[str, list]:
        keys = self._rows[0].keys()
        return {key: [row[key] for row in self._rows] for key in keys}


class _FakeDataset:
    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.schema = _FakeSchema(list(rows[0].keys()))

    def count_rows(self) -> int:
        return len(self._rows)

    def to_batches(self, columns: list[str]):
        selected = [{key: row[key] for key in columns} for row in self._rows]
        return [_FakeBatch(selected)]


class _FakeModel:
    def predict(self, user_ids, item_ids, history_item_idxs=None):
        # Fixed scores: item 0 > 1 > 2 > unseen items 3/4 (= neutral zero).
        score_map = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0, 4: 0.0}
        return np.array([score_map[int(item)] for item in item_ids], dtype=np.float64)


def test_single_model_bootstrap_is_deterministic() -> None:
    user_ids = np.array([1, 1, 2, 3], dtype=np.int64)
    values = np.array(
        [[1.0, 1.0], [0.5, 0.0], [0.0, 1.0], [1.0, 0.5]],
        dtype=np.float64,
    )
    settings = BootstrapSettings(
        enabled=True,
        num_resamples=100,
        confidence_level=0.95,
        unit="user",
        seed=42,
        batch_size=16,
    )

    first = bootstrap_confidence_intervals(
        user_ids=user_ids,
        values=values,
        metric_names=["NDCG@10", "Recall@10"],
        settings=settings,
    )
    second = bootstrap_confidence_intervals(
        user_ids=user_ids,
        values=values,
        metric_names=["NDCG@10", "Recall@10"],
        settings=settings,
    )

    assert first == second
    assert first["NDCG@10"]["mean"] == 0.625
    assert len(first["NDCG@10"]["ci"]) == 2


def test_regular_evaluator_adds_popularity_slices_and_bootstrap(monkeypatch) -> None:
    rows = [
        {
            "impression_id": 1,
            "user_idx": 10,
            "history_item_idxs": [0, 1, 2, 0, 1, 2],
            "candidate_item_idxs": [0, 1, 2, 3, 4],
            "labels": [1, 0, 0, 0, 0],
            "candidate_seen_in_train_mask": [True, True, True, False, False],
        },
        {
            "impression_id": 2,
            "user_idx": 10,
            "history_item_idxs": [0, 1, 2, 0, 1, 2],
            "candidate_item_idxs": [0, 1, 2, 3, 4],
            "labels": [0, 0, 1, 0, 0],
            "candidate_seen_in_train_mask": [True, True, True, False, False],
        },
        {
            "impression_id": 3,
            "user_idx": 20,
            "history_item_idxs": [0, 1],
            "candidate_item_idxs": [0, 1, 2, 3, 4],
            "labels": [0, 0, 0, 1, 0],
            "candidate_seen_in_train_mask": [True, True, True, False, False],
        },
        {
            "impression_id": 4,
            "user_idx": 20,
            "history_item_idxs": [0, 1],
            "candidate_item_idxs": [0, 1, 2, 3, 4],
            "labels": [0, 0, 0, 0, 1],
            "candidate_seen_in_train_mask": [True, True, True, False, False],
        },
    ]

    import src.evaluate as evaluate_module

    fake_dataset = _FakeDataset(rows)
    monkeypatch.setattr(evaluate_module.ds, "dataset", lambda *args, **kwargs: fake_dataset)

    settings = BootstrapSettings(
        enabled=True,
        num_resamples=50,
        confidence_level=0.95,
        unit="user",
        seed=42,
        batch_size=8,
    )
    results = evaluate_impressions(
        model=_FakeModel(),
        impressions_path="ignored",
        k_list=[5, 10],
        all_catalog_items={0, 1, 2, 3, 4},
        train_click_counts=np.array([10, 5, 1, 0, 0], dtype=np.int64),
        head_fraction=0.5,
        bootstrap_settings=settings,
    )

    slices = results["slices"]
    assert slices["head_item"]["count"] == 1
    assert slices["long_tail_item"]["count"] == 1
    assert slices["zero_click_item"]["count"] == 2
    assert slices["catalog_cold_item"]["count"] == 2
    assert slices["catalog_warm_item"]["count"] == 2

    # Convention chung của project vẫn giữ nguyên: chỉ cần có cold candidate.
    assert slices["cold_item"]["count"] == 4
    assert slices["warm_item"]["count"] == 0

    popularity = results["item_popularity_definition"]
    assert popularity["head_items"] == 2
    assert popularity["long_tail_items"] == 1
    assert popularity["zero_click_items"] == 2

    bootstrap = results["bootstrap"]
    assert bootstrap["unit"] == "user"
    assert bootstrap["num_resamples"] == 50
    assert set(bootstrap["metrics"]) == {"NDCG@10", "Recall@10", "MRR@10"}
    assert len(bootstrap["metrics"]["NDCG@10"]["ci"]) == 2
