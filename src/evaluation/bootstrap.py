"""Bootstrap confidence interval cho so sánh baseline và P3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapSettings:
    """Cấu hình bootstrap cho evaluation."""

    enabled: bool = True
    num_resamples: int = 1000
    confidence_level: float = 0.95
    unit: str = "user"
    seed: int = 42
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.num_resamples <= 0:
            raise ValueError("num_resamples phải lớn hơn 0.")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level phải nằm trong khoảng (0, 1).")
        if self.unit not in {"user", "impression"}:
            raise ValueError("bootstrap unit chỉ hỗ trợ 'user' hoặc 'impression'.")
        if self.batch_size <= 0:
            raise ValueError("bootstrap batch_size phải lớn hơn 0.")


def _aggregate_clusters(
    user_ids: np.ndarray,
    values: np.ndarray,
    unit: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Gom tổng metric và số impression theo đơn vị bootstrap."""
    if values.ndim != 2:
        raise ValueError("values phải có dạng [num_impressions, num_metrics].")
    if user_ids.shape[0] != values.shape[0]:
        raise ValueError("user_ids và values phải cùng số dòng.")

    if unit == "impression":
        return values.astype(np.float64, copy=False), np.ones(values.shape[0], dtype=np.float64)

    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    cluster_sums = np.zeros((unique_users.size, values.shape[1]), dtype=np.float64)
    cluster_counts = np.zeros(unique_users.size, dtype=np.float64)
    np.add.at(cluster_sums, inverse, values)
    np.add.at(cluster_counts, inverse, 1.0)
    return cluster_sums, cluster_counts


def _percentile_interval(
    samples: np.ndarray,
    confidence_level: float,
) -> tuple[float, float]:
    """Lấy percentile confidence interval hai phía."""
    alpha = 1.0 - confidence_level
    low = float(np.quantile(samples, alpha / 2.0))
    high = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return low, high



def bootstrap_confidence_intervals(
    user_ids: np.ndarray,
    values: np.ndarray,
    metric_names: list[str],
    settings: BootstrapSettings,
) -> dict[str, dict[str, float | list[float]]]:
    """Tính bootstrap CI cho một model trên các impression đã đánh giá."""
    users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    metric_values = np.asarray(values, dtype=np.float64)

    if metric_values.ndim != 2:
        raise ValueError("values phải có dạng [num_impressions, num_metrics].")
    if metric_values.shape[0] != users.size:
        raise ValueError("Số user_ids không khớp số impression.")
    if metric_values.shape[1] != len(metric_names):
        raise ValueError("metric_names không khớp số cột metric.")
    if metric_values.shape[0] == 0:
        return {}

    cluster_sums, cluster_counts = _aggregate_clusters(
        users,
        metric_values,
        settings.unit,
    )
    cluster_count = cluster_counts.size
    rng = np.random.default_rng(settings.seed)
    bootstrap_values = np.empty(
        (settings.num_resamples, metric_values.shape[1]),
        dtype=np.float64,
    )

    # Giới hạn bộ nhớ của ma trận chỉ số cluster được lấy mẫu.
    bytes_per_cluster_per_resample = 4 + 8 * metric_values.shape[1]
    memory_safe_batch = max(
        1,
        int(64_000_000 // max(1, cluster_count * bytes_per_cluster_per_resample)),
    )
    effective_batch_size = min(settings.batch_size, memory_safe_batch)

    written = 0
    while written < settings.num_resamples:
        current_batch = min(effective_batch_size, settings.num_resamples - written)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(current_batch, cluster_count),
            dtype=np.int32,
        )
        sampled_counts = cluster_counts[sampled].sum(axis=1)
        sampled_totals = cluster_sums[sampled].sum(axis=1)
        bootstrap_values[written : written + current_batch] = (
            sampled_totals / sampled_counts[:, None]
        )
        written += current_batch

    results: dict[str, dict[str, float | list[float]]] = {}
    for metric_index, metric_name in enumerate(metric_names):
        low, high = _percentile_interval(
            bootstrap_values[:, metric_index],
            settings.confidence_level,
        )
        mean = float(np.mean(metric_values[:, metric_index]))
        results[metric_name] = {
            "mean": round(mean, 6),
            "ci": [round(low, 6), round(high, 6)],
        }

    return results

def paired_bootstrap_comparison(
    user_ids: np.ndarray,
    baseline_values: np.ndarray,
    reranked_values: np.ndarray,
    metric_names: list[str],
    settings: BootstrapSettings,
) -> dict[str, dict[str, float | list[float]]]:
    """Tính paired bootstrap CI cho baseline, P3 và delta trên cùng mẫu."""
    users = np.asarray(user_ids, dtype=np.int64).reshape(-1)
    baseline = np.asarray(baseline_values, dtype=np.float64)
    reranked = np.asarray(reranked_values, dtype=np.float64)

    if baseline.shape != reranked.shape:
        raise ValueError("baseline_values và reranked_values phải cùng shape.")
    if baseline.ndim != 2:
        raise ValueError("Metric matrix phải có hai chiều.")
    if baseline.shape[0] != users.size:
        raise ValueError("Số user_ids không khớp số impression.")
    if baseline.shape[1] != len(metric_names):
        raise ValueError("metric_names không khớp số cột metric.")
    if baseline.shape[0] == 0:
        return {}

    baseline_sums, cluster_counts = _aggregate_clusters(users, baseline, settings.unit)
    reranked_sums, reranked_counts = _aggregate_clusters(users, reranked, settings.unit)
    if not np.array_equal(cluster_counts, reranked_counts):
        raise RuntimeError("Paired bootstrap mất đồng bộ cluster giữa baseline và P3.")

    cluster_count = cluster_counts.size
    rng = np.random.default_rng(settings.seed)
    baseline_boot = np.empty((settings.num_resamples, baseline.shape[1]), dtype=np.float64)
    reranked_boot = np.empty_like(baseline_boot)

    # Giới hạn bộ nhớ tạm khi số cluster lớn.
    bytes_per_cluster_per_resample = 4 + 16 * baseline.shape[1]
    memory_safe_batch = max(
        1,
        int(64_000_000 // max(1, cluster_count * bytes_per_cluster_per_resample)),
    )
    effective_batch_size = min(settings.batch_size, memory_safe_batch)

    written = 0
    while written < settings.num_resamples:
        current_batch = min(effective_batch_size, settings.num_resamples - written)
        sampled = rng.integers(
            0,
            cluster_count,
            size=(current_batch, cluster_count),
            dtype=np.int32,
        )
        sampled_counts = cluster_counts[sampled].sum(axis=1)
        baseline_totals = baseline_sums[sampled].sum(axis=1)
        reranked_totals = reranked_sums[sampled].sum(axis=1)
        baseline_boot[written : written + current_batch] = baseline_totals / sampled_counts[:, None]
        reranked_boot[written : written + current_batch] = reranked_totals / sampled_counts[:, None]
        written += current_batch

    delta_boot = reranked_boot - baseline_boot
    results: dict[str, dict[str, float | list[float]]] = {}

    for metric_index, metric_name in enumerate(metric_names):
        baseline_ci = _percentile_interval(
            baseline_boot[:, metric_index], settings.confidence_level
        )
        reranked_ci = _percentile_interval(
            reranked_boot[:, metric_index], settings.confidence_level
        )
        delta_ci = _percentile_interval(
            delta_boot[:, metric_index], settings.confidence_level
        )
        baseline_mean = float(np.mean(baseline[:, metric_index]))
        reranked_mean = float(np.mean(reranked[:, metric_index]))

        results[metric_name] = {
            "baseline_mean": round(baseline_mean, 6),
            "baseline_ci": [round(baseline_ci[0], 6), round(baseline_ci[1], 6)],
            "reranked_mean": round(reranked_mean, 6),
            "reranked_ci": [round(reranked_ci[0], 6), round(reranked_ci[1], 6)],
            "delta": round(reranked_mean - baseline_mean, 6),
            "delta_ci": [round(delta_ci[0], 6), round(delta_ci[1], 6)],
        }

    return results
