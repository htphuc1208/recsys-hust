"""Đánh giá paired baseline vs P3 trên cùng impression MIND."""

from __future__ import annotations

import time
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds

from src.evaluation.beyond_accuracy import gini_from_exposure_counts
from src.evaluation.bootstrap import BootstrapSettings, paired_bootstrap_comparison
from src.evaluation.item_slices import (
    LONG_TAIL,
    ZERO_CLICK,
    build_item_popularity_groups,
    split_relevant_items_by_catalog_status,
    split_relevant_items_by_popularity,
)
from src.evaluation.ranking_metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.models.base import BaseRecommender
from src.reranking.pipeline import RerankingPipeline

RANKING_METRICS = ("NDCG", "Recall", "HitRate", "MRR")
BASE_SLICES = ("overall", "cold_item", "warm_item", "cold_user", "warm_user")
ITEM_SLICES = (
    "head_item",
    "long_tail_item",
    "zero_click_item",
    "catalog_cold_item",
    "catalog_warm_item",
)
ALL_SLICES = BASE_SLICES + ITEM_SLICES


@dataclass
class ExposureAccumulator:
    """Tích lũy beyond-accuracy mà không giữ toàn bộ recommendation trong RAM."""

    item_exposure_counts: np.ndarray
    diversity_sum: float = 0.0
    category_coverage_sum: float = 0.0
    long_tail_hits: int = 0
    clicked_long_tail_hits: int = 0
    zero_click_hits: int = 0
    long_tail_exposure_sum: float = 0.0
    clicked_long_tail_exposure_sum: float = 0.0
    zero_click_exposure_sum: float = 0.0
    total_positions: int = 0
    list_count: int = 0

    @classmethod
    def create(cls, num_items: int) -> ExposureAccumulator:
        return cls(item_exposure_counts=np.zeros(num_items, dtype=np.int64))

    def update(
        self,
        ranking: list[int],
        categories: list[str],
        popularity_codes: np.ndarray,
        k: int,
    ) -> tuple[float, float, float]:
        """Cập nhật exposure và trả về per-impression diversity/LTE/zero-click exposure."""
        top_items = [int(item) for item in ranking[:k]]
        if not top_items:
            return 0.0, 0.0, 0.0

        ids = np.asarray(top_items, dtype=np.int64)
        np.add.at(self.item_exposure_counts, ids, 1)

        category_values = [categories[item] for item in top_items]
        non_empty_categories = [category for category in category_values if category]
        unique_categories = len(set(non_empty_categories))
        category_coverage = unique_categories / len(top_items)

        if len(top_items) < 2:
            diversity = 0.0
        else:
            category_counts: dict[str, int] = {}
            for category in category_values:
                category_counts[category] = category_counts.get(category, 0) + 1
            total_pairs = len(top_items) * (len(top_items) - 1) / 2.0
            same_category_pairs = sum(
                count * (count - 1) / 2.0 for count in category_counts.values()
            )
            diversity = 1.0 - same_category_pairs / total_pairs

        codes = popularity_codes[ids]
        clicked_long_tail_hits = int(np.sum(codes == LONG_TAIL))
        zero_click_hits = int(np.sum(codes == ZERO_CLICK))
        # Long-tail exposure phục vụ P3 gồm mọi item không thuộc Head;
        # zero-click được báo cáo riêng để nhìn rõ cold/novel exposure.
        long_tail_hits = clicked_long_tail_hits + zero_click_hits
        long_tail_exposure = long_tail_hits / len(top_items)
        clicked_long_tail_exposure = clicked_long_tail_hits / len(top_items)
        zero_click_exposure = zero_click_hits / len(top_items)

        self.diversity_sum += diversity
        self.category_coverage_sum += category_coverage
        self.long_tail_hits += long_tail_hits
        self.clicked_long_tail_hits += clicked_long_tail_hits
        self.zero_click_hits += zero_click_hits
        self.long_tail_exposure_sum += long_tail_exposure
        self.clicked_long_tail_exposure_sum += clicked_long_tail_exposure
        self.zero_click_exposure_sum += zero_click_exposure
        self.total_positions += len(top_items)
        self.list_count += 1
        return diversity, long_tail_exposure, zero_click_exposure

    def finalize(self, total_catalog_items: int) -> dict[str, float | int]:
        """Trả về aggregate beyond-accuracy metrics."""
        if self.list_count == 0:
            return {
                "catalog_coverage@10": 0.0,
                "gini_index@10": 0.0,
                "category_diversity@10": 0.0,
                "category_coverage@10": 0.0,
                "long_tail_exposure@10": 0.0,
                "clicked_long_tail_exposure@10": 0.0,
                "zero_click_exposure@10": 0.0,
                "total_catalog_items": total_catalog_items,
            }

        exposed_items = int(np.count_nonzero(self.item_exposure_counts))
        catalog_coverage = exposed_items / total_catalog_items if total_catalog_items else 0.0
        return {
            "catalog_coverage@10": round(catalog_coverage, 5),
            "gini_index@10": round(gini_from_exposure_counts(self.item_exposure_counts), 5),
            "category_diversity@10": round(self.diversity_sum / self.list_count, 5),
            "category_coverage@10": round(self.category_coverage_sum / self.list_count, 5),
            # Dùng mean theo impression để khớp đúng với Bootstrap CI.
            "long_tail_exposure@10": round(
                self.long_tail_exposure_sum / self.list_count,
                5,
            ),
            "clicked_long_tail_exposure@10": round(
                self.clicked_long_tail_exposure_sum / self.list_count,
                5,
            ),
            "zero_click_exposure@10": round(
                self.zero_click_exposure_sum / self.list_count,
                5,
            ),
            "total_catalog_items": total_catalog_items,
        }


def _empty_slice_store(k_list: list[int]) -> dict[str, dict[str, list[float]]]:
    """Tạo vùng tích lũy metric cho mọi slice."""
    return {
        slice_name: {f"{metric_name}@{k}": [] for metric_name in RANKING_METRICS for k in k_list}
        for slice_name in ALL_SLICES
    }


def _ranking_metrics(
    ranked_candidates: list[int],
    true_items: set[int],
    k_list: list[int],
) -> dict[str, float]:
    """Tính ranking metrics của một impression."""
    values: dict[str, float] = {}
    for k in k_list:
        values[f"NDCG@{k}"] = ndcg_at_k(ranked_candidates, true_items, k)
        values[f"Recall@{k}"] = recall_at_k(ranked_candidates, true_items, k)
        values[f"HitRate@{k}"] = hit_rate_at_k(ranked_candidates, true_items, k)
        values[f"MRR@{k}"] = mrr_at_k(ranked_candidates, true_items, k)
    return values


def _append_metrics(
    store: dict[str, dict[str, list[float]]],
    slice_name: str,
    values: dict[str, float],
) -> None:
    """Thêm metric vào một slice."""
    for metric_name, metric_value in values.items():
        store[slice_name][metric_name].append(float(metric_value))


def _aggregate_slices(
    store: dict[str, dict[str, list[float]]],
    total_evaluated: int,
    count_metric: str,
) -> dict[str, Any]:
    """Tính mean và sample count cho mọi slice."""
    output: dict[str, Any] = {}
    for slice_name, metrics in store.items():
        count = len(metrics[count_metric])
        output[slice_name] = {
            "count": count,
            "pct": round(100.0 * count / (total_evaluated or 1), 2),
            "metrics": {
                metric_name: round(float(np.mean(values)), 5) if values else 0.0
                for metric_name, values in metrics.items()
            },
        }
    return output


def evaluate_p3_impressions(
    model: BaseRecommender,
    impressions_path: Path,
    reranking_pipeline: RerankingPipeline,
    k_list: list[int] | None = None,
    sample_size: int | None = None,
    all_catalog_items: set[int] | None = None,
    head_fraction: float = 0.20,
    bootstrap_settings: BootstrapSettings | None = None,
) -> dict[str, Any]:
    """Đánh giá baseline và P3 trong cùng một lượt để so sánh paired công bằng."""
    if k_list is None:
        k_list = [5, 10, 20]
    if not k_list:
        raise ValueError("k_list không được rỗng.")

    dataset = ds.dataset(str(impressions_path), format="parquet")
    required_cols = [
        "impression_id",
        "user_idx",
        "history_item_idxs",
        "candidate_item_idxs",
        "labels",
        "candidate_seen_in_train_mask",
    ]
    missing_columns = set(required_cols) - set(dataset.schema.names)
    if missing_columns:
        raise ValueError(f"Impressions thiếu cột: {sorted(missing_columns)}")

    reranker = reranking_pipeline.reranker
    categories = reranker.categories
    click_counts = reranker.train_click_counts
    popularity_groups = build_item_popularity_groups(click_counts, head_fraction=head_fraction)

    baseline_store = _empty_slice_store(k_list)
    reranked_store = _empty_slice_store(k_list)

    catalog_size = len(categories)
    if all_catalog_items:
        catalog_size = len(all_catalog_items)
    baseline_exposure = ExposureAccumulator.create(len(categories))
    reranked_exposure = ExposureAccumulator.create(len(categories))

    max_records = dataset.count_rows()
    if sample_size is not None:
        max_records = min(max_records, sample_size)

    key_k = 10 if 10 in k_list else max(k_list)
    bootstrap_metric_names = [f"NDCG@{key_k}", f"Recall@{key_k}", f"MRR@{key_k}"]
    bootstrap_metric_names += [f"Diversity@{key_k}", f"LongTailExposure@{key_k}"]
    user_ids = np.empty(max_records, dtype=np.int64)
    baseline_boot_values = np.empty((max_records, len(bootstrap_metric_names)), dtype=np.float32)
    reranked_boot_values = np.empty_like(baseline_boot_values)

    settings = bootstrap_settings or BootstrapSettings(enabled=False)
    # Mảng nhị phân giúp giữ bootstrap theo slice gọn bộ nhớ trên MIND Large.
    slice_boot_users = {name: array("q") for name in ALL_SLICES} if settings.enabled else {}
    slice_boot_before = {name: array("f") for name in ALL_SLICES} if settings.enabled else {}
    slice_boot_after = {name: array("f") for name in ALL_SLICES} if settings.enabled else {}
    ranking_boot_names = bootstrap_metric_names[:3]

    total_evaluated = 0
    start_time = time.perf_counter()

    for batch in dataset.to_batches(columns=required_cols):
        rows = batch.to_pydict()
        row_count = len(rows["impression_id"])

        for row_index in range(row_count):
            labels = rows["labels"][row_index]
            if labels is None or sum(labels) == 0:
                continue

            candidates = [int(item) for item in rows["candidate_item_idxs"][row_index]]
            user_idx = int(rows["user_idx"][row_index])
            history = rows["history_item_idxs"][row_index] or []
            seen_mask = rows["candidate_seen_in_train_mask"][row_index]

            baseline, reranked, _scores = reranking_pipeline.score_and_rerank(
                model=model,
                user_idx=user_idx,
                candidate_item_idxs=candidates,
                history_item_idxs=history,
                top_k=max(k_list),
            )

            true_items = {
                candidates[index] for index, label in enumerate(labels) if int(label) == 1
            }
            baseline_metrics = _ranking_metrics(baseline, true_items, k_list)
            reranked_metrics = _ranking_metrics(reranked, true_items, k_list)

            active_base_slices = ["overall"]
            has_cold_candidate = False in seen_mask
            active_base_slices.append("cold_item" if has_cold_candidate else "warm_item")
            is_cold_user = len(history) <= 5 or user_idx == -1
            active_base_slices.append("cold_user" if is_cold_user else "warm_user")

            bootstrap_user = user_idx if user_idx >= 0 else -(total_evaluated + 1)
            for slice_name in active_base_slices:
                _append_metrics(baseline_store, slice_name, baseline_metrics)
                _append_metrics(reranked_store, slice_name, reranked_metrics)
                if settings.enabled and slice_name != "overall":
                    slice_boot_users[slice_name].append(bootstrap_user)
                    slice_boot_before[slice_name].extend(
                        [baseline_metrics[name] for name in ranking_boot_names]
                    )
                    slice_boot_after[slice_name].extend(
                        [reranked_metrics[name] for name in ranking_boot_names]
                    )

            popularity_truth = split_relevant_items_by_popularity(
                true_items,
                popularity_groups.group_codes,
            )
            catalog_truth = split_relevant_items_by_catalog_status(
                candidates,
                labels,
                seen_mask,
            )
            item_slice_truth = {**popularity_truth, **catalog_truth}

            # Giữ nguyên ranking đầy đủ, chỉ thay tập relevant cho từng item slice.
            for slice_name, slice_true_items in item_slice_truth.items():
                if not slice_true_items:
                    continue
                before_metrics = _ranking_metrics(baseline, slice_true_items, k_list)
                after_metrics = _ranking_metrics(reranked, slice_true_items, k_list)
                _append_metrics(baseline_store, slice_name, before_metrics)
                _append_metrics(reranked_store, slice_name, after_metrics)
                if settings.enabled and slice_name != "overall":
                    slice_boot_users[slice_name].append(bootstrap_user)
                    slice_boot_before[slice_name].extend(
                        [before_metrics[name] for name in ranking_boot_names]
                    )
                    slice_boot_after[slice_name].extend(
                        [after_metrics[name] for name in ranking_boot_names]
                    )

            baseline_div, baseline_lte, _baseline_zero = baseline_exposure.update(
                baseline,
                categories,
                popularity_groups.group_codes,
                k=key_k,
            )
            reranked_div, reranked_lte, _reranked_zero = reranked_exposure.update(
                reranked,
                categories,
                popularity_groups.group_codes,
                k=key_k,
            )

            # Unknown user (-1) được xem như cluster riêng cho từng impression.
            user_ids[total_evaluated] = bootstrap_user
            baseline_boot_values[total_evaluated] = [
                baseline_metrics[f"NDCG@{key_k}"],
                baseline_metrics[f"Recall@{key_k}"],
                baseline_metrics[f"MRR@{key_k}"],
                baseline_div,
                baseline_lte,
            ]
            reranked_boot_values[total_evaluated] = [
                reranked_metrics[f"NDCG@{key_k}"],
                reranked_metrics[f"Recall@{key_k}"],
                reranked_metrics[f"MRR@{key_k}"],
                reranked_div,
                reranked_lte,
            ]

            total_evaluated += 1
            if sample_size and total_evaluated >= sample_size:
                break

        if sample_size and total_evaluated >= sample_size:
            break

    elapsed = time.perf_counter() - start_time
    throughput = total_evaluated / elapsed if elapsed > 0 else 0.0
    count_metric = f"NDCG@{key_k}"

    baseline_result = {
        "slices": _aggregate_slices(baseline_store, total_evaluated, count_metric),
        "beyond_accuracy": baseline_exposure.finalize(catalog_size),
    }
    reranked_result = {
        "slices": _aggregate_slices(reranked_store, total_evaluated, count_metric),
        "beyond_accuracy": reranked_exposure.finalize(catalog_size),
    }

    comparison: dict[str, Any] = {
        "overall_delta": {},
    }
    for metric_name in baseline_result["slices"]["overall"]["metrics"]:
        before = baseline_result["slices"]["overall"]["metrics"][metric_name]
        after = reranked_result["slices"]["overall"]["metrics"][metric_name]
        comparison["overall_delta"][metric_name] = round(after - before, 5)
    for metric_name in baseline_result["beyond_accuracy"]:
        if metric_name == "total_catalog_items":
            continue
        before = float(baseline_result["beyond_accuracy"][metric_name])
        after = float(reranked_result["beyond_accuracy"][metric_name])
        comparison["overall_delta"][metric_name] = round(after - before, 5)

    if settings.enabled and total_evaluated > 0:
        comparison["bootstrap"] = {
            "unit": settings.unit,
            "num_resamples": settings.num_resamples,
            "confidence_level": settings.confidence_level,
            "metrics": paired_bootstrap_comparison(
                user_ids=user_ids[:total_evaluated],
                baseline_values=baseline_boot_values[:total_evaluated],
                reranked_values=reranked_boot_values[:total_evaluated],
                metric_names=bootstrap_metric_names,
                settings=settings,
            ),
        }
        # Mẫu paired được căn thẳng hàng và resample theo cùng user IDs
        # trong từng slice để delta CI có ý nghĩa thống kê.
        comparison["bootstrap"]["slices"] = {
            name: {
                "count": total_evaluated if name == "overall" else len(slice_boot_users[name]),
                "metrics": {
                    key: comparison["bootstrap"]["metrics"][key]
                    for key in ranking_boot_names
                } if name == "overall" else paired_bootstrap_comparison(
                    user_ids=np.frombuffer(slice_boot_users[name], dtype=np.int64),
                    baseline_values=np.frombuffer(slice_boot_before[name], dtype=np.float32).reshape(
                        -1, len(ranking_boot_names)
                    ),
                    reranked_values=np.frombuffer(slice_boot_after[name], dtype=np.float32).reshape(
                        -1, len(ranking_boot_names)
                    ),
                    metric_names=ranking_boot_names,
                    settings=settings,
                ) if slice_boot_users[name] else {},
            }
            for name in ALL_SLICES
        }

    return {
        "evaluation_mode": "paired_baseline_vs_p3",
        "total_evaluated_impressions": total_evaluated,
        "eval_time_seconds": round(elapsed, 2),
        "throughput_impressions_per_sec": round(throughput, 1),
        "item_popularity_definition": {
            "head_fraction_of_clicked_items": head_fraction,
            "head_items": popularity_groups.head_items,
            "long_tail_items": popularity_groups.long_tail_items,
            "zero_click_items": popularity_groups.zero_click_items,
        },
        "baseline": baseline_result,
        "reranked": reranked_result,
        "comparison": comparison,
    }
