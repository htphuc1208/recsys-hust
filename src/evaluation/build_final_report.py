"""Tổng hợp toàn bộ kết quả cuối của project thành bảng, hình và báo cáo Markdown."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

MAIN_RESULT_FILES = {
    "bpr": "exp04_mind_bpr_dev.json",
    "content": "exp05_mind_content_dev.json",
    "content_p3": "exp06_mind_content_p3_dev.json",
    "bpr_p3": "exp07_mind_bpr_p3_dev.json",
}

BASELINE_LABELS = {
    "bpr": "BPR",
    "content": "Content-based",
}

P3_LABELS = {
    "bpr_p3": "BPR + P3",
    "content_p3": "Content-based + P3",
}

ITEM_SLICE_ORDER = [
    "head_item",
    "long_tail_item",
    "zero_click_item",
    "catalog_cold_item",
    "catalog_warm_item",
]

ITEM_SLICE_LABELS = {
    "head_item": "Head item",
    "long_tail_item": "Long-tail item",
    "zero_click_item": "Zero-click item",
    "catalog_cold_item": "Catalog cold item",
    "catalog_warm_item": "Catalog warm item",
}

P3_METRICS = [
    "NDCG@10",
    "Recall@10",
    "MRR@10",
    "category_diversity@10",
    "long_tail_exposure@10",
    "zero_click_exposure@10",
    "catalog_coverage@10",
    "gini_index@10",
]

# Khoảng timestamp train đã audit trực tiếp trên MIND Large hiện tại của project.
MIND_TRAIN_TIMESTAMP_START = "2019-11-09 00:00:00"
MIND_TRAIN_TIMESTAMP_END = "2019-11-14 23:59:59"


@dataclass(frozen=True)
class ReportPaths:
    """Các đường dẫn input/output của bộ báo cáo."""

    project_root: Path
    metrics_dir: Path
    data_minimization_dir: Path
    config_dir: Path
    tables_dir: Path
    figures_dir: Path
    reports_root: Path


@dataclass(frozen=True)
class ReportInputs:
    """Toàn bộ dữ liệu đầu vào đã được kiểm tra."""

    main_results: dict[str, dict[str, Any]]
    data_minimization: pd.DataFrame
    dataset_stats: pd.DataFrame
    configs: dict[str, dict[str, Any]]
    full_dev_count: int


def _read_json(path: Path) -> dict[str, Any]:
    """Đọc một file JSON và báo lỗi rõ đường dẫn."""
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file kết quả bắt buộc: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _read_yaml(path: Path) -> dict[str, Any]:
    """Đọc YAML nếu file tồn tại, nếu không trả về dict rỗng."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    return loaded if isinstance(loaded, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    """Đọc CSV bắt buộc."""
    if not path.exists():
        raise FileNotFoundError(f"Không thấy file CSV bắt buộc: {path}")
    return pd.read_csv(path)


def _single_slices(result: dict[str, Any]) -> dict[str, Any]:
    """Lấy slices của model thường."""
    return result.get("slices", {})


def _p3_slices(result: dict[str, Any], side: str) -> dict[str, Any]:
    """Lấy slices baseline hoặc reranked trong paired P3 result."""
    return result.get(side, {}).get("slices", {})


def _metric_from_slice(
    slices: dict[str, Any],
    slice_name: str,
    metric_name: str,
) -> float:
    """Lấy metric từ một slice; thiếu dữ liệu thì trả NaN."""
    value = slices.get(slice_name, {}).get("metrics", {}).get(metric_name)
    return float(value) if value is not None else float("nan")


def _beyond_metric(result: dict[str, Any], metric_name: str) -> float:
    """Lấy beyond-accuracy metric của model thường."""
    value = result.get("beyond_accuracy", {}).get(metric_name)
    return float(value) if value is not None else float("nan")


def _p3_beyond_metric(result: dict[str, Any], side: str, metric_name: str) -> float:
    """Lấy beyond-accuracy metric từ baseline/reranked của P3."""
    value = result.get(side, {}).get("beyond_accuracy", {}).get(metric_name)
    return float(value) if value is not None else float("nan")


def _bootstrap_ci(result: dict[str, Any], metric_name: str) -> tuple[float, float]:
    """Lấy confidence interval của model thường."""
    ci = result.get("bootstrap", {}).get("metrics", {}).get(metric_name, {}).get("ci")
    if not isinstance(ci, list) or len(ci) != 2:
        return float("nan"), float("nan")
    return float(ci[0]), float(ci[1])


def _paired_delta_ci(result: dict[str, Any], metric_name: str) -> tuple[float, float]:
    """Lấy paired bootstrap CI của delta P3."""
    ci = (
        result.get("comparison", {})
        .get("bootstrap", {})
        .get("metrics", {})
        .get(metric_name, {})
        .get("delta_ci")
    )
    if not isinstance(ci, list) or len(ci) != 2:
        return float("nan"), float("nan")
    return float(ci[0]), float(ci[1])


def _validate_full_dev_results(results: dict[str, dict[str, Any]]) -> int:
    """Đảm bảo bốn result chính cùng đánh giá trên full dev, không phải preview sample."""
    counts = {
        name: int(result.get("total_evaluated_impressions", 0))
        for name, result in results.items()
    }
    if any(count <= 0 for count in counts.values()):
        raise ValueError(f"Có result không chứa total_evaluated_impressions hợp lệ: {counts}")

    unique_counts = set(counts.values())
    if len(unique_counts) != 1:
        raise ValueError(
            "Bốn result chính không cùng số impression. "
            f"Có thể đang trộn preview/sample với full dev: {counts}"
        )

    full_dev_count = next(iter(unique_counts))
    if full_dev_count < 10_000:
        raise ValueError(
            "Result hiện tại có quá ít impression để coi là full dev "
            f"({full_dev_count:,}). Hãy chạy lại exp04/05/06/07 trên full dev trước khi build report."
        )
    return full_dev_count


def _validate_data_minimization(summary: pd.DataFrame) -> None:
    """Kiểm tra Data Minimization có đủ 1d/3d/5d/all cho cả hai model."""
    required_columns = {
        "window",
        "model",
        "data_retained_pct",
        "training_storage_mb",
        "storage_reduction_pct",
        "train_wall_seconds",
        "NDCG@10",
        "Recall@10",
        "MRR@10",
    }
    missing = required_columns - set(summary.columns)
    if missing:
        raise ValueError(f"summary.csv thiếu cột: {sorted(missing)}")

    expected_windows = {"1d", "3d", "5d", "all"}
    expected_models = {"bpr", "content_based"}
    present_pairs = set(zip(summary["window"].astype(str), summary["model"].astype(str)))
    missing_pairs = [
        (window, model)
        for model in sorted(expected_models)
        for window in ("1d", "3d", "5d", "all")
        if (window, model) not in present_pairs
    ]
    if missing_pairs:
        raise ValueError(f"summary.csv thiếu các experiment: {missing_pairs}")



def load_inputs(paths: ReportPaths) -> ReportInputs:
    """Đọc và kiểm tra toàn bộ nguồn số liệu chính thức."""
    main_results = {
        key: _read_json(paths.metrics_dir / filename)
        for key, filename in MAIN_RESULT_FILES.items()
    }
    full_dev_count = _validate_full_dev_results(main_results)

    # Baseline bên trong paired P3 phải khớp model nền đã đánh giá độc lập.
    consistency_pairs = [
        ("bpr", "bpr_p3"),
        ("content", "content_p3"),
    ]
    for baseline_key, p3_key in consistency_pairs:
        independent = _metric_from_slice(
            _single_slices(main_results[baseline_key]),
            "overall",
            "NDCG@10",
        )
        paired = _metric_from_slice(
            _p3_slices(main_results[p3_key], "baseline"),
            "overall",
            "NDCG@10",
        )
        if not np.isfinite(independent) or not np.isfinite(paired):
            raise ValueError(
                f"Thiếu baseline NDCG@10 để kiểm tra {baseline_key} vs {p3_key}."
            )
        if abs(independent - paired) > 1e-4:
            raise ValueError(
                "Baseline độc lập và baseline trong paired P3 không khớp: "
                f"{baseline_key}={independent:.6f}, {p3_key} baseline={paired:.6f}. "
                "Hãy kiểm tra checkpoint/config/result trước khi build report."
            )

    dm_summary = _read_csv(paths.data_minimization_dir / "summary.csv")
    dataset_stats = _read_csv(paths.data_minimization_dir / "dataset_stats.csv")
    _validate_data_minimization(dm_summary)

    configs = {
        "bpr": _read_yaml(paths.config_dir / "exp04_mind_bpr.yaml"),
        "content": _read_yaml(paths.config_dir / "exp05_mind_content.yaml"),
        "content_p3": _read_yaml(paths.config_dir / "exp06_mind_content_p3.yaml"),
        "bpr_p3": _read_yaml(paths.config_dir / "exp07_mind_bpr_p3.yaml"),
        "data_minimization": _read_yaml(paths.config_dir / "data_minimization.yaml"),
    }

    return ReportInputs(
        main_results=main_results,
        data_minimization=dm_summary,
        dataset_stats=dataset_stats,
        configs=configs,
        full_dev_count=full_dev_count,
    )


def _relative_delta(before: float, after: float) -> float:
    """Tính thay đổi tương đối theo phần trăm."""
    if not np.isfinite(before) or before == 0.0 or not np.isfinite(after):
        return float("nan")
    return 100.0 * (after - before) / before


def build_baseline_table(inputs: ReportInputs) -> pd.DataFrame:
    """Tạo bảng so sánh BPR và Content-based."""
    rows: list[dict[str, Any]] = []
    for key in ("bpr", "content"):
        result = inputs.main_results[key]
        slices = _single_slices(result)
        ndcg_ci = _bootstrap_ci(result, "NDCG@10")
        recall_ci = _bootstrap_ci(result, "Recall@10")
        mrr_ci = _bootstrap_ci(result, "MRR@10")
        rows.append(
            {
                "model": BASELINE_LABELS[key],
                "impressions": inputs.full_dev_count,
                "NDCG@10": _metric_from_slice(slices, "overall", "NDCG@10"),
                "NDCG@10_CI_low": ndcg_ci[0],
                "NDCG@10_CI_high": ndcg_ci[1],
                "Recall@10": _metric_from_slice(slices, "overall", "Recall@10"),
                "Recall@10_CI_low": recall_ci[0],
                "Recall@10_CI_high": recall_ci[1],
                "MRR@10": _metric_from_slice(slices, "overall", "MRR@10"),
                "MRR@10_CI_low": mrr_ci[0],
                "MRR@10_CI_high": mrr_ci[1],
                "ColdItem_NDCG@10": _metric_from_slice(slices, "cold_item", "NDCG@10"),
                "WarmItem_NDCG@10": _metric_from_slice(slices, "warm_item", "NDCG@10"),
                "ColdUser_NDCG@10": _metric_from_slice(slices, "cold_user", "NDCG@10"),
                "WarmUser_NDCG@10": _metric_from_slice(slices, "warm_user", "NDCG@10"),
                "CatalogColdItem_NDCG@10": _metric_from_slice(
                    slices, "catalog_cold_item", "NDCG@10"
                ),
                "CatalogWarmItem_NDCG@10": _metric_from_slice(
                    slices, "catalog_warm_item", "NDCG@10"
                ),
                "CatalogCoverage@10": _beyond_metric(result, "catalog_coverage@10"),
                "GiniIndex@10": _beyond_metric(result, "gini_index@10"),
            }
        )
    return pd.DataFrame(rows)


def build_p3_table(result: dict[str, Any], label: str) -> pd.DataFrame:
    """Tạo bảng baseline vs P3 cho một recommender."""
    baseline_slices = _p3_slices(result, "baseline")
    reranked_slices = _p3_slices(result, "reranked")
    rows: list[dict[str, Any]] = []

    for metric_name in P3_METRICS:
        if metric_name in {"NDCG@10", "Recall@10", "MRR@10"}:
            before = _metric_from_slice(baseline_slices, "overall", metric_name)
            after = _metric_from_slice(reranked_slices, "overall", metric_name)
        else:
            before = _p3_beyond_metric(result, "baseline", metric_name)
            after = _p3_beyond_metric(result, "reranked", metric_name)

        ci_low, ci_high = _paired_delta_ci(result, metric_name)
        rows.append(
            {
                "system": label,
                "metric": metric_name,
                "baseline": before,
                "p3": after,
                "delta": after - before if np.isfinite(before) and np.isfinite(after) else np.nan,
                "relative_delta_pct": _relative_delta(before, after),
                "paired_delta_CI_low": ci_low,
                "paired_delta_CI_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def build_slice_table(inputs: ReportInputs) -> pd.DataFrame:
    """Tạo bảng NDCG@10 theo item slice cho BPR/Content và hai nhánh P3."""
    systems = {
        "BPR": _single_slices(inputs.main_results["bpr"]),
        "Content-based": _single_slices(inputs.main_results["content"]),
        "BPR + P3": _p3_slices(inputs.main_results["bpr_p3"], "reranked"),
        "Content-based + P3": _p3_slices(inputs.main_results["content_p3"], "reranked"),
    }
    rows: list[dict[str, Any]] = []
    for slice_name in ITEM_SLICE_ORDER:
        row: dict[str, Any] = {
            "slice": slice_name,
            "slice_label": ITEM_SLICE_LABELS[slice_name],
        }
        for label, slices in systems.items():
            row[label] = _metric_from_slice(slices, slice_name, "NDCG@10")
        rows.append(row)
    return pd.DataFrame(rows)


def build_data_minimization_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung utility loss/retained so với cửa sổ all cho Data Minimization."""
    result = summary.copy()
    result["window"] = result["window"].astype(str)

    all_metrics = (
        result[result["window"] == "all"]
        .set_index("model")["NDCG@10"]
        .to_dict()
    )
    all_train_times = (
        result[result["window"] == "all"]
        .set_index("model")["train_wall_seconds"]
        .to_dict()
    )

    ndcg_loss_pct: list[float] = []
    utility_retained_pct: list[float] = []
    train_time_reduction_pct: list[float] = []
    for _, row in result.iterrows():
        model_name = str(row["model"])
        baseline_ndcg = float(all_metrics[model_name])
        baseline_time = float(all_train_times[model_name])
        current_ndcg = float(row["NDCG@10"])
        current_time = float(row["train_wall_seconds"])
        loss = 100.0 * (baseline_ndcg - current_ndcg) / baseline_ndcg
        ndcg_loss_pct.append(loss)
        utility_retained_pct.append(100.0 * current_ndcg / baseline_ndcg)
        if baseline_time > 0:
            train_time_reduction_pct.append(100.0 * (baseline_time - current_time) / baseline_time)
        else:
            train_time_reduction_pct.append(float("nan"))

    result["NDCG_loss_vs_all_pct"] = ndcg_loss_pct
    result["utility_retained_pct"] = utility_retained_pct
    result["train_time_reduction_pct"] = train_time_reduction_pct
    return result


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Ghi DataFrame ra CSV với định dạng số ổn định."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")


def _save_figure(path: Path) -> None:
    """Lưu figure hiện tại và đóng để không tích lũy RAM."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_baseline_accuracy(table: pd.DataFrame, output_path: Path) -> None:
    """Vẽ NDCG/Recall/MRR của hai baseline."""
    metrics = ["NDCG@10", "Recall@10", "MRR@10"]
    x = np.arange(len(metrics))
    width = 0.36

    plt.figure(figsize=(8.0, 5.0))
    for index, row in table.reset_index(drop=True).iterrows():
        values = [float(row[metric]) for metric in metrics]
        offset = (index - 0.5) * width
        plt.bar(x + offset, values, width=width, label=str(row["model"]))
    plt.xticks(x, metrics)
    plt.ylabel("Metric value")
    plt.title("Baseline recommendation accuracy on full dev")
    plt.ylim(0.0, 0.75)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save_figure(output_path)


def plot_catalog_cold_start(table: pd.DataFrame, output_path: Path) -> None:
    """Vẽ khả năng xử lý catalog cold/warm item của hai baseline."""
    labels = ["Catalog cold item", "Catalog warm item"]
    x = np.arange(len(labels))
    width = 0.36

    plt.figure(figsize=(8.0, 5.0))
    for index, row in table.reset_index(drop=True).iterrows():
        values = [
            float(row["CatalogColdItem_NDCG@10"]),
            float(row["CatalogWarmItem_NDCG@10"]),
        ]
        offset = (index - 0.5) * width
        plt.bar(x + offset, values, width=width, label=str(row["model"]))
    plt.xticks(x, labels)
    plt.ylabel("NDCG@10")
    plt.title("Cold-start item performance")
    plt.ylim(0.0, 0.6)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save_figure(output_path)


def plot_p3_tradeoff(table: pd.DataFrame, title: str, output_path: Path) -> None:
    """Vẽ trade-off relevance/diversity/long-tail của P3."""
    selected = table[
        table["metric"].isin(
            [
                "NDCG@10",
                "category_diversity@10",
                "long_tail_exposure@10",
                "zero_click_exposure@10",
            ]
        )
    ].copy()
    labels = [
        "NDCG@10",
        "Category\nDiversity@10",
        "Long-tail\nExposure@10",
        "Zero-click\nExposure@10",
    ]
    selected = selected.set_index("metric").loc[
        [
            "NDCG@10",
            "category_diversity@10",
            "long_tail_exposure@10",
            "zero_click_exposure@10",
        ]
    ]

    x = np.arange(len(labels))
    width = 0.36
    plt.figure(figsize=(9.0, 5.2))
    plt.bar(x - width / 2, selected["baseline"], width=width, label="Baseline")
    plt.bar(x + width / 2, selected["p3"], width=width, label="P3")
    plt.xticks(x, labels)
    plt.ylabel("Metric value")
    plt.title(title)
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    _save_figure(output_path)


def plot_item_slices(table: pd.DataFrame, output_path: Path) -> None:
    """Vẽ NDCG@10 theo các item slice chính."""
    systems = ["BPR", "Content-based", "BPR + P3", "Content-based + P3"]
    x = np.arange(len(table))
    width = 0.19

    plt.figure(figsize=(11.0, 5.5))
    for index, system in enumerate(systems):
        offset = (index - 1.5) * width
        plt.bar(x + offset, table[system].astype(float), width=width, label=system)
    plt.xticks(x, table["slice_label"], rotation=15, ha="right")
    plt.ylabel("NDCG@10")
    plt.title("NDCG@10 across item slices")
    plt.ylim(0.0, 0.65)
    plt.legend(ncol=2)
    plt.grid(axis="y", alpha=0.25)
    _save_figure(output_path)


def _ordered_dm_rows(table: pd.DataFrame, model: str) -> pd.DataFrame:
    """Sắp Data Minimization theo 1d/3d/5d/all."""
    order = {"1d": 1, "3d": 3, "5d": 5, "all": 6}
    subset = table[table["model"] == model].copy()
    subset["_order"] = subset["window"].map(order)
    return subset.sort_values("_order")


def plot_dm_ndcg_vs_retained(table: pd.DataFrame, output_path: Path) -> None:
    """Vẽ NDCG theo phần trăm dữ liệu được giữ lại."""
    plt.figure(figsize=(8.0, 5.0))
    for model, label in (("bpr", "BPR"), ("content_based", "Content-based")):
        subset = _ordered_dm_rows(table, model)
        plt.plot(
            subset["data_retained_pct"],
            subset["NDCG@10"],
            marker="o",
            label=label,
        )
        for _, row in subset.iterrows():
            plt.annotate(
                str(row["window"]),
                (float(row["data_retained_pct"]), float(row["NDCG@10"])),
            )
    plt.xlabel("Training data retained (%)")
    plt.ylabel("NDCG@10")
    plt.title("Data minimization: utility vs retained data")
    plt.legend()
    plt.grid(alpha=0.25)
    _save_figure(output_path)


def plot_dm_ndcg_vs_storage(table: pd.DataFrame, output_path: Path) -> None:
    """Vẽ NDCG theo storage dùng cho training."""
    plt.figure(figsize=(8.0, 5.0))
    for model, label in (("bpr", "BPR"), ("content_based", "Content-based")):
        subset = _ordered_dm_rows(table, model)
        plt.plot(
            subset["training_storage_mb"],
            subset["NDCG@10"],
            marker="o",
            label=label,
        )
    plt.xlabel("Training storage (MB)")
    plt.ylabel("NDCG@10")
    plt.title("Data minimization: utility vs storage")
    plt.legend()
    plt.grid(alpha=0.25)
    _save_figure(output_path)


def plot_dm_ndcg_vs_train_time(table: pd.DataFrame, output_path: Path) -> None:
    """Vẽ NDCG theo thời gian train."""
    plt.figure(figsize=(8.0, 5.0))
    for model, label in (("bpr", "BPR"), ("content_based", "Content-based")):
        subset = _ordered_dm_rows(table, model)
        plt.plot(
            subset["train_wall_seconds"],
            subset["NDCG@10"],
            marker="o",
            label=label,
        )
    plt.xlabel("Training wall time (seconds)")
    plt.ylabel("NDCG@10")
    plt.title("Data minimization: utility vs training time")
    plt.legend()
    plt.grid(alpha=0.25)
    _save_figure(output_path)


def _format_float(value: float, digits: int = 5) -> str:
    """Format số, tránh in 'nan' khó đọc."""
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}"


def _format_pct(value: float, digits: int = 2) -> str:
    """Format phần trăm."""
    return "N/A" if not np.isfinite(value) else f"{value:.{digits}f}%"


def _weights_from_config(config: dict[str, Any], section: str) -> dict[str, float]:
    """Đọc weights từ config và chuẩn hóa sang float."""
    if section == "content":
        raw = config.get("model", {}).get("feature_weights", {})
    else:
        raw = config.get("reranking", {}).get("weights", {})
    return {str(key): float(value) for key, value in raw.items()}


def _markdown_table_from_dataframe(
    dataframe: pd.DataFrame,
    columns: list[str],
    headers: list[str],
    digits: int = 5,
) -> str:
    """Tạo bảng Markdown nhỏ từ DataFrame."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in dataframe.iterrows():
        values: list[str] = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(_format_float(float(value), digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_model_card(
    inputs: ReportInputs,
    baseline_table: pd.DataFrame,
    bpr_p3_table: pd.DataFrame,
    content_p3_table: pd.DataFrame,
) -> str:
    """Tạo Model Card bằng tiếng Việt từ kết quả thực nghiệm chính thức."""
    bpr_row = baseline_table[baseline_table["model"] == "BPR"].iloc[0]
    content_row = baseline_table[baseline_table["model"] == "Content-based"].iloc[0]
    content_weights = _weights_from_config(inputs.configs["content"], "content")
    bpr_p3_weights = _weights_from_config(inputs.configs["bpr_p3"], "p3")
    content_p3_weights = _weights_from_config(inputs.configs["content_p3"], "p3")

    bpr_p3_ndcg = bpr_p3_table[bpr_p3_table["metric"] == "NDCG@10"].iloc[0]
    bpr_p3_diversity = bpr_p3_table[
        bpr_p3_table["metric"] == "category_diversity@10"
    ].iloc[0]
    bpr_p3_long_tail = bpr_p3_table[
        bpr_p3_table["metric"] == "long_tail_exposure@10"
    ].iloc[0]

    content_p3_ndcg = content_p3_table[
        content_p3_table["metric"] == "NDCG@10"
    ].iloc[0]
    content_p3_diversity = content_p3_table[
        content_p3_table["metric"] == "category_diversity@10"
    ].iloc[0]
    content_p3_long_tail = content_p3_table[
        content_p3_table["metric"] == "long_tail_exposure@10"
    ].iloc[0]

    return f"""# Model Card — Hệ gợi ý MIND Large

## 1. Phạm vi

Bộ Model Card này mô tả bốn hệ thống được đánh giá trên cùng **{inputs.full_dev_count:,}** impression của tập dev:

- BPR.
- Content-based.
- BPR + P3 re-ranker.
- Content-based + P3 re-ranker.

Mục tiêu là đánh giá đồng thời độ liên quan (relevance), cold-start, diversity và exposure của long-tail item.

## 2. Mô hình nền

### BPR

BPR học latent representation từ interaction train. Hyperparameter cuối được lấy từ `configs/experiment/exp04_mind_bpr.yaml`.

### Content-based

Content-based dùng metadata item và user profile từ lịch sử tương tác. Feature weights cuối:

```text
{json.dumps(content_weights, ensure_ascii=False, indent=2)}
```

`cold_user_strategy = {inputs.configs['content'].get('model', {}).get('cold_user_strategy', 'N/A')}`.

## 3. P3 re-ranking

P3 tái xếp hạng candidate bằng ba thành phần: relevance, category diversity và long-tail exposure. Hai nhánh được tuning độc lập.

BPR + P3 weights:

```text
{json.dumps(bpr_p3_weights, ensure_ascii=False, indent=2)}
```

Content-based + P3 weights:

```text
{json.dumps(content_p3_weights, ensure_ascii=False, indent=2)}
```

## 4. Kết quả chính

| Model | NDCG@10 | Recall@10 | MRR@10 | Catalog cold NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| BPR | {_format_float(float(bpr_row['NDCG@10']))} | {_format_float(float(bpr_row['Recall@10']))} | {_format_float(float(bpr_row['MRR@10']))} | {_format_float(float(bpr_row['CatalogColdItem_NDCG@10']))} |
| Content-based | {_format_float(float(content_row['NDCG@10']))} | {_format_float(float(content_row['Recall@10']))} | {_format_float(float(content_row['MRR@10']))} | {_format_float(float(content_row['CatalogColdItem_NDCG@10']))} |

Bootstrap CI được tính theo user với cấu hình trong experiment YAML. BPR NDCG@10 CI = [{_format_float(float(bpr_row['NDCG@10_CI_low']), 6)}, {_format_float(float(bpr_row['NDCG@10_CI_high']), 6)}]; Content-based NDCG@10 CI = [{_format_float(float(content_row['NDCG@10_CI_low']), 6)}, {_format_float(float(content_row['NDCG@10_CI_high']), 6)}].

## 5. Trade-off khi dùng P3

- BPR + P3: NDCG@10 delta = {_format_float(float(bpr_p3_ndcg['delta']), 6)} ({_format_pct(float(bpr_p3_ndcg['relative_delta_pct']))}); Category Diversity@10 = {_format_pct(float(bpr_p3_diversity['relative_delta_pct']))}; Long-tail Exposure@10 = {_format_pct(float(bpr_p3_long_tail['relative_delta_pct']))}.
- Content-based + P3: NDCG@10 delta = {_format_float(float(content_p3_ndcg['delta']), 6)} ({_format_pct(float(content_p3_ndcg['relative_delta_pct']))}); Category Diversity@10 = {_format_pct(float(content_p3_diversity['relative_delta_pct']))}; Long-tail Exposure@10 = {_format_pct(float(content_p3_long_tail['relative_delta_pct']))}.

P3 được thiết kế để chấp nhận một mức giảm relevance có kiểm soát nhằm tăng diversity và long-tail exposure. Chi tiết nằm trong `reports/tables/p3_bpr_comparison.csv` và `reports/tables/p3_content_comparison.csv`.

## 6. Cold-start và slicing

`cold_item` của project là **impression-level slice**: impression được xếp vào cold item nếu có ít nhất một candidate chưa xuất hiện trong train. `catalog_cold_item` là phân tích bổ sung, chỉ xét relevant/clicked item có thực sự chưa thấy trong train hay không.

`cold_user` theo evaluator là user có history length <= 5 hoặc `user_idx == -1`.

Head item được định nghĩa là top 20% các item có click trong train theo click count; long-tail là 80% còn lại trong nhóm đã có click; zero-click được báo cáo riêng.

## 7. Data Minimization / GDPR-motivated experiment

MIND Large train interactions có timestamp thực tế chỉ phủ khoảng sáu ngày, vì vậy experiment dùng 1d/3d/5d/all thay vì 7d/30d/90d/all. Individual click trong `history` không có timestamp riêng nên không được gán timestamp giả.

Riêng Content-based trong experiment này đặt `use_impression_history = false` để user profile chỉ dựa trên retained train history; điều này tránh sử dụng full dev history làm đường tắt.

Experiment này đánh giá trade-off theo nguyên tắc data minimization và **không** được diễn giải như bằng chứng rằng toàn bộ hệ thống tuân thủ GDPR.

## 8. Intended use

Hệ thống phục vụ nghiên cứu/đánh giá recommender trên MIND Large, đặc biệt cho cold-start item, re-ranking responsible exposure và data minimization. Không nên diễn giải metric offline như bằng chứng trực tiếp cho chất lượng sản phẩm online hoặc trải nghiệm người dùng thực tế.

## 9. Hạn chế

- Kết quả là offline evaluation trên MIND Large dev.
- Khoảng thời gian train có timestamp ngắn nên retention window bị giới hạn ở 1/3/5/all ngày.
- P3 tối ưu trade-off theo metric proxy; diversity/exposure cao hơn không tự động đồng nghĩa với mức hài lòng người dùng cao hơn.
- Gini rất gần 1 do catalog lớn và recommendation exposure tập trung; cần diễn giải cùng catalog coverage và các exposure metric khác.
- Kết quả Content-based trong Data Minimization không trực tiếp so sánh tuyệt đối với exp05 thông thường vì experiment data minimization cố ý tắt full dev impression history.
- Content feature weights và P3 re-ranking weights đều được lựa chọn dựa trên dev data; project hiện không dùng validation/holdout riêng cho hyperparameter selection. Vì vậy các metric cuối trên dev có thể có optimistic bias và nên được xem là development estimates, không phải unbiased test estimates.
"""


def build_datasheet(inputs: ReportInputs) -> str:
    """Tạo Data Sheet cho dữ liệu MIND đã dùng trong project."""
    stats = inputs.dataset_stats.copy()
    stats["window"] = stats["window"].astype(str)
    stats_md = _markdown_table_from_dataframe(
        stats,
        columns=[
            "window",
            "pointwise_rows",
            "positive_rows",
            "pointwise_retention_pct",
            "positive_retention_pct",
            "pointwise_storage_mb",
            "positive_storage_mb",
        ],
        headers=[
            "Window",
            "Pointwise rows",
            "Positive rows",
            "Pointwise retained (%)",
            "Positive retained (%)",
            "Pointwise MB",
            "Positive MB",
        ],
        digits=2,
    )

    cutoff_values = stats.get("cutoff_timestamp", pd.Series(dtype=str)).astype(str).tolist()
    cutoff = cutoff_values[0] if cutoff_values else "N/A"
    content_weights = _weights_from_config(inputs.configs["content"], "content")

    return f"""# Data Sheet — MIND Large trong project

## 1. Dataset được sử dụng

Project sử dụng MIND Large đã được preprocess vào `data/processed/mind_large/`. Các experiment chính dùng train interactions, dev impressions, item mapping và metadata item.

## 2. Các trường evaluator sử dụng

Dev impression evaluator đọc các trường:

- `impression_id`
- `user_idx`
- `history_item_idxs`
- `candidate_item_idxs`
- `labels`
- `candidate_seen_in_train_mask`

Train interaction cho Data Minimization sử dụng `timestamp`; BPR dùng pointwise interactions còn Content-based dùng positive interactions.

## 3. Phạm vi thời gian có timestamp

Dữ liệu timestamp dùng cho experiment retention bắt đầu từ **{MIND_TRAIN_TIMESTAMP_START}** và kết thúc ở **{MIND_TRAIN_TIMESTAMP_END}**; cutoff là **{cutoff}**. Do release hiện tại chỉ cho interaction timestamp trong khoảng gần sáu ngày, experiment không thể tái tạo trung thực retention 30 hoặc 90 ngày.

## 4. Retention datasets

{stats_md}

Các variant 1d/3d/5d/all được materialize bằng cùng Parquet writer/compression để phép đo storage có thể so sánh công bằng.

## 5. Slicing convention

- `cold_item`: impression có ít nhất một candidate chưa thấy trong train.
- `warm_item`: tất cả candidate trong impression đã thấy trong train.
- `cold_user`: history length <= 5 hoặc `user_idx == -1`.
- `head_item`: top 20% item đã có click theo train click count.
- `long_tail_item`: 80% item đã có click còn lại.
- `zero_click_item`: item không có click trong train.
- `catalog_cold_item`: relevant item chưa thấy trong train.
- `catalog_warm_item`: relevant item đã thấy trong train.

## 6. Feature dùng cho Content-based

Content-based sử dụng feature đã chuẩn bị trong `data/processed/mind_large/features/b4`, gồm title TF-IDF, abstract TF-IDF, category one-hot, subcategory one-hot và average pretrained title entity embedding. Final feature weights:

```text
{json.dumps(content_weights, ensure_ascii=False, indent=2)}
```

Các trọng số này được lưu trong `configs/experiment/exp05_mind_content.yaml`.

## 7. Data minimization và privacy framing

Experiment chỉ giảm lượng interaction được giữ lại để train/profile user. Không tạo timestamp giả cho item trong historical click list. Với Content-based, full dev impression history được tắt riêng trong experiment retention để tránh làm mất ý nghĩa data minimization.

Experiment không thu thập thêm dữ liệu cá nhân và không chứng minh GDPR compliance; nó chỉ đánh giá trade-off của nguyên tắc data minimization trong thiết lập offline hiện tại.

## 8. Hạn chế dữ liệu

- Timestamp riêng của từng historical click không có trong representation đang dùng.
- Offline impression logs không phản ánh đầy đủ preference, satisfaction hoặc causal effect của recommendation.
- Popularity group phụ thuộc vào train click frequency nên có thể thay đổi nếu preprocessing hoặc retention window thay đổi.
"""


def _p3_summary_values(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Đổi bảng P3 thành lookup theo metric."""
    values: dict[str, dict[str, float]] = {}
    for row in table.itertuples(index=False):
        values[str(row.metric)] = {
            "baseline": float(row.baseline),
            "p3": float(row.p3),
            "delta": float(row.delta),
            "relative_delta_pct": float(row.relative_delta_pct),
        }
    return values


def build_final_summary(
    inputs: ReportInputs,
    baseline_table: pd.DataFrame,
    bpr_p3_table: pd.DataFrame,
    content_p3_table: pd.DataFrame,
    dm_table: pd.DataFrame,
) -> str:
    """Tạo báo cáo Markdown tổng hợp cuối cùng."""
    bpr = baseline_table[baseline_table["model"] == "BPR"].iloc[0]
    content = baseline_table[baseline_table["model"] == "Content-based"].iloc[0]
    bpr_p3 = _p3_summary_values(bpr_p3_table)
    content_p3 = _p3_summary_values(content_p3_table)
    content_weights = _weights_from_config(inputs.configs["content"], "content")
    bpr_p3_weights = _weights_from_config(inputs.configs["bpr_p3"], "p3")
    content_p3_weights = _weights_from_config(inputs.configs["content_p3"], "p3")

    dm_bpr = _ordered_dm_rows(dm_table, "bpr")
    dm_content = _ordered_dm_rows(dm_table, "content_based")

    def dm_line(table: pd.DataFrame, window: str) -> pd.Series:
        return table[table["window"] == window].iloc[0]

    bpr_1d = dm_line(dm_bpr, "1d")
    bpr_3d = dm_line(dm_bpr, "3d")
    content_1d = dm_line(dm_content, "1d")
    content_3d = dm_line(dm_content, "3d")

    content_advantage = 100.0 * (
        float(content["NDCG@10"]) - float(bpr["NDCG@10"])
    ) / float(bpr["NDCG@10"])

    return f"""# Final Experimental Summary

## 1. Experimental setup

Bốn hệ thống chính được đánh giá trên cùng **{inputs.full_dev_count:,}** MIND Large dev impressions: BPR, Content-based, BPR + P3 và Content-based + P3. Accuracy dùng NDCG/Recall/MRR; beyond-accuracy dùng catalog coverage, Gini, category diversity, long-tail exposure và zero-click exposure. Bootstrap CI dùng resampling theo user theo cấu hình experiment.

## 2. B4 — Content-based Recommender

Content-based đạt NDCG@10 = **{_format_float(float(content['NDCG@10']))}**, so với BPR = **{_format_float(float(bpr['NDCG@10']))}**, tương ứng chênh lệch tương đối **{_format_pct(content_advantage)}** trên full dev.

Khác biệt nổi bật ở `catalog_cold_item`: BPR = **{_format_float(float(bpr['CatalogColdItem_NDCG@10']))}**, Content-based = **{_format_float(float(content['CatalogColdItem_NDCG@10']))}**. Slice này xét relevant item thực sự chưa thấy trong train và bổ sung cho convention `cold_item` ở mức impression.

Final feature weights:

```text
{json.dumps(content_weights, ensure_ascii=False, indent=2)}
```

Các trọng số này được lưu trong `configs/experiment/exp05_mind_content.yaml`.

## 3. P3 — Popularity-Aware & Diversity Re-ranking

### BPR + P3

Final P3 weights:

```text
{json.dumps(bpr_p3_weights, ensure_ascii=False, indent=2)}
```

- NDCG@10: {_format_float(bpr_p3['NDCG@10']['baseline'])} → {_format_float(bpr_p3['NDCG@10']['p3'])} ({_format_pct(bpr_p3['NDCG@10']['relative_delta_pct'])}).
- Category Diversity@10: {_format_float(bpr_p3['category_diversity@10']['baseline'])} → {_format_float(bpr_p3['category_diversity@10']['p3'])} ({_format_pct(bpr_p3['category_diversity@10']['relative_delta_pct'])}).
- Long-tail Exposure@10: {_format_float(bpr_p3['long_tail_exposure@10']['baseline'])} → {_format_float(bpr_p3['long_tail_exposure@10']['p3'])} ({_format_pct(bpr_p3['long_tail_exposure@10']['relative_delta_pct'])}).

### Content-based + P3

Final P3 weights:

```text
{json.dumps(content_p3_weights, ensure_ascii=False, indent=2)}
```

- NDCG@10: {_format_float(content_p3['NDCG@10']['baseline'])} → {_format_float(content_p3['NDCG@10']['p3'])} ({_format_pct(content_p3['NDCG@10']['relative_delta_pct'])}).
- Category Diversity@10: {_format_float(content_p3['category_diversity@10']['baseline'])} → {_format_float(content_p3['category_diversity@10']['p3'])} ({_format_pct(content_p3['category_diversity@10']['relative_delta_pct'])}).
- Long-tail Exposure@10: {_format_float(content_p3['long_tail_exposure@10']['baseline'])} → {_format_float(content_p3['long_tail_exposure@10']['p3'])} ({_format_pct(content_p3['long_tail_exposure@10']['relative_delta_pct'])}).

Kết quả cho thấy P3 tạo trade-off có chủ đích: relevance giảm ở mức nhất định trong khi diversity và exposure của non-head item tăng. Paired bootstrap CI được lưu trong hai JSON exp06/exp07 và bảng CSV tương ứng.

## 4. Evaluation rigor

Evaluator báo cáo:

- project `cold_item` / `warm_item` convention ở mức impression;
- `cold_user` / `warm_user`;
- head / long-tail / zero-click item;
- catalog cold / warm relevant item;
- Bootstrap 95% CI cho model thường;
- paired Bootstrap CI cho P3 delta.

Việc giữ cả `cold_item` và `catalog_cold_item` giúp vừa tương thích convention chung của project vừa phân tích đúng cold-start của relevant item.

## 5. Data Minimization / GDPR-motivated experiment

MIND train interaction có timestamp chỉ phủ khoảng sáu ngày nên experiment dùng **1d / 3d / 5d / all**, thay vì 7/30/90/all. Không gán timestamp giả cho historical clicks.

### BPR

- 1d giữ {_format_pct(float(bpr_1d['data_retained_pct']))} dữ liệu, giảm storage {_format_pct(float(bpr_1d['storage_reduction_pct']))}, NDCG@10 = {_format_float(float(bpr_1d['NDCG@10']))}, utility retained = {_format_pct(float(bpr_1d['utility_retained_pct']))}.
- 3d giữ {_format_pct(float(bpr_3d['data_retained_pct']))} dữ liệu, giảm storage {_format_pct(float(bpr_3d['storage_reduction_pct']))}, NDCG@10 = {_format_float(float(bpr_3d['NDCG@10']))}, utility retained = {_format_pct(float(bpr_3d['utility_retained_pct']))}.

### Content-based

Lưu ý: Content-based `all` trong Data Minimization không trực tiếp so sánh với exp05 thông thường. Experiment này cố ý đặt `use_impression_history = false` để user profile chỉ sử dụng retained training interactions, tránh full dev history làm che mất tác động của retention window.

- 1d giữ {_format_pct(float(content_1d['data_retained_pct']))} dữ liệu, giảm storage {_format_pct(float(content_1d['storage_reduction_pct']))}, NDCG@10 = {_format_float(float(content_1d['NDCG@10']))}, utility retained = {_format_pct(float(content_1d['utility_retained_pct']))}.
- 3d giữ {_format_pct(float(content_3d['data_retained_pct']))} dữ liệu, giảm storage {_format_pct(float(content_3d['storage_reduction_pct']))}, NDCG@10 = {_format_float(float(content_3d['NDCG@10']))}, utility retained = {_format_pct(float(content_3d['utility_retained_pct']))}.

BPR giữ utility tốt hơn khi retention window ngắn. Content-based nhạy hơn vì user content profile phụ thuộc trực tiếp vào lượng retained history. Training time của Content-based chỉ vài giây nên không nên diễn giải quá mạnh các khác biệt nhỏ về thời gian giữa windows.

## 6. Responsible AI discussion

P3 cải thiện category diversity và long-tail exposure, qua đó giảm xu hướng recommendation exposure tập trung vào các item phổ biến. Tuy nhiên, thay đổi của Gini index tương đối nhỏ; vì vậy kết quả không nên được diễn giải rằng concentration bias đã được loại bỏ.

Các beyond-accuracy metrics là offline proxies và không trực tiếp chứng minh user satisfaction, fairness hay chất lượng trải nghiệm trong triển khai thực tế.

Data Minimization experiment cho thấy trong một số setting có thể giảm đáng kể lượng interaction được lưu trữ và sử dụng để train trong khi vẫn giữ phần lớn offline recommendation utility. Đây là một thí nghiệm được thúc đẩy bởi nguyên tắc data minimization; nó không tự thân chứng minh hệ thống tuân thủ đầy đủ GDPR.

Mọi kết luận về giảm dữ liệu cần được trình bày cùng utility cost thay vì chỉ báo cáo lợi ích về privacy/storage.

## 7. Limitations

- Offline MIND Large evaluation không thay thế online A/B testing.
- Train timestamp span ngắn giới hạn retention experiment.
- Individual historical clicks không có timestamp riêng trong representation sử dụng.
- Data Minimization Content baseline không bằng exp05 thông thường vì experiment cố ý không dùng full dev impression history.
- Head/long-tail phụ thuộc train click frequency và cutoff cụ thể.
- Content feature weights và P3 re-ranking weights đều được lựa chọn dựa trên dev data; project hiện không dùng validation/holdout riêng cho hyperparameter selection. Vì vậy các final dev estimates có thể có optimistic bias và nên được xem là offline development results, không phải unbiased test estimates.

## 8. Generated artifacts

- `reports/tables/model_accuracy_comparison.csv`
- `reports/tables/p3_bpr_comparison.csv`
- `reports/tables/p3_content_comparison.csv`
- `reports/tables/item_slice_comparison.csv`
- `reports/tables/data_minimization_summary.csv`
- `reports/figures/`
- `reports/model_card.md`
- `reports/datasheet.md`

Các bảng/biểu đồ được sinh trực tiếp từ JSON/CSV experiment để giảm lỗi copy số liệu thủ công.
"""


def write_reports(paths: ReportPaths, inputs: ReportInputs) -> dict[str, Path]:
    """Sinh toàn bộ tables, figures và Markdown report."""
    paths.tables_dir.mkdir(parents=True, exist_ok=True)
    paths.figures_dir.mkdir(parents=True, exist_ok=True)

    baseline_table = build_baseline_table(inputs)
    bpr_p3_table = build_p3_table(inputs.main_results["bpr_p3"], "BPR")
    content_p3_table = build_p3_table(inputs.main_results["content_p3"], "Content-based")
    slice_table = build_slice_table(inputs)
    dm_table = build_data_minimization_table(inputs.data_minimization)

    table_paths = {
        "baseline": paths.tables_dir / "model_accuracy_comparison.csv",
        "bpr_p3": paths.tables_dir / "p3_bpr_comparison.csv",
        "content_p3": paths.tables_dir / "p3_content_comparison.csv",
        "slices": paths.tables_dir / "item_slice_comparison.csv",
        "data_minimization": paths.tables_dir / "data_minimization_summary.csv",
    }
    _save_dataframe(baseline_table, table_paths["baseline"])
    _save_dataframe(bpr_p3_table, table_paths["bpr_p3"])
    _save_dataframe(content_p3_table, table_paths["content_p3"])
    _save_dataframe(slice_table, table_paths["slices"])
    _save_dataframe(dm_table, table_paths["data_minimization"])

    plot_baseline_accuracy(
        baseline_table,
        paths.figures_dir / "model_accuracy_comparison.png",
    )
    plot_catalog_cold_start(
        baseline_table,
        paths.figures_dir / "cold_item_comparison.png",
    )
    plot_p3_tradeoff(
        bpr_p3_table,
        "BPR: baseline vs P3 trade-off",
        paths.figures_dir / "p3_bpr_tradeoff.png",
    )
    plot_p3_tradeoff(
        content_p3_table,
        "Content-based: baseline vs P3 trade-off",
        paths.figures_dir / "p3_content_tradeoff.png",
    )
    plot_item_slices(
        slice_table,
        paths.figures_dir / "item_slice_comparison.png",
    )
    plot_dm_ndcg_vs_retained(
        dm_table,
        paths.figures_dir / "data_minimization_ndcg_retained.png",
    )
    plot_dm_ndcg_vs_storage(
        dm_table,
        paths.figures_dir / "data_minimization_ndcg_storage.png",
    )
    plot_dm_ndcg_vs_train_time(
        dm_table,
        paths.figures_dir / "data_minimization_train_time.png",
    )

    model_card_path = paths.reports_root / "model_card.md"
    datasheet_path = paths.reports_root / "datasheet.md"
    final_summary_path = paths.reports_root / "final_summary.md"

    model_card_path.write_text(
        build_model_card(inputs, baseline_table, bpr_p3_table, content_p3_table),
        encoding="utf-8",
    )
    datasheet_path.write_text(build_datasheet(inputs), encoding="utf-8")
    final_summary_path.write_text(
        build_final_summary(
            inputs,
            baseline_table,
            bpr_p3_table,
            content_p3_table,
            dm_table,
        ),
        encoding="utf-8",
    )

    return {
        **table_paths,
        "model_card": model_card_path,
        "datasheet": datasheet_path,
        "final_summary": final_summary_path,
    }


def parse_args() -> argparse.Namespace:
    """Đọc CLI args."""
    parser = argparse.ArgumentParser(
        description="Tổng hợp kết quả cuối thành tables, figures, Model Card và Data Sheet."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root của project recsys-hust. Mặc định là thư mục hiện tại.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point của final report builder."""
    args = parse_args()
    project_root = args.project_root.resolve()
    reports_root = project_root / "reports"
    paths = ReportPaths(
        project_root=project_root,
        metrics_dir=reports_root / "metrics",
        data_minimization_dir=reports_root / "data_minimization",
        config_dir=project_root / "configs" / "experiment",
        tables_dir=reports_root / "tables",
        figures_dir=reports_root / "figures",
        reports_root=reports_root,
    )

    print("=" * 72)
    print("BUILD FINAL RECOMMENDER REPORT")
    print("=" * 72)
    print(f"Project root : {project_root}")

    inputs = load_inputs(paths)
    print(f"Full dev     : {inputs.full_dev_count:,} impressions")
    print("Inputs       : OK")

    outputs = write_reports(paths, inputs)

    print("\nGenerated:")
    for path in outputs.values():
        print(f"  - {path.relative_to(project_root)}")
    print(f"  - {paths.figures_dir.relative_to(project_root)}/*.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
