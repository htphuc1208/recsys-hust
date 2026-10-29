"""Chạy có chọn lọc các cấu hình Dev, tổng hợp metric và CI của mọi model.

Không tái tạo CI từ JSON cũ: CI theo slice chỉ có sau khi đánh giá
lại model với bản evaluator mới (có giá trị per-impression).
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

SLICES = (
    "overall", "cold_user", "warm_user", "cold_item", "warm_item",
    "catalog_cold_item", "catalog_warm_item", "head_item", "long_tail_item",
    "zero_click_item",
)
METRICS = ("NDCG@10", "Recall@10", "MRR@10")
BEYOND_METRICS = (
    "catalog_coverage@10", "gini_index@10", "category_diversity@10",
    "long_tail_exposure@10", "zero_click_exposure@10",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict) or not isinstance(result.get("slices", result.get("reranked", {}).get("slices")), dict):
        raise ValueError(f"Không phải result Dev hợp lệ: {path}")
    return result


def _read_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML không hợp lệ: {path}")
    return data


def _checkpoint_from_config(cfg: dict[str, Any]) -> Path:
    reranking = cfg.get("reranking", {})
    if reranking.get("enabled", False):
        return Path(reranking["base_checkpoint"])
    directory = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")) / "checkpoints"
    return directory / f"{cfg['experiment_name']}_{cfg['model']['name']}.pkl"


def evaluate_configs(
    configs: list[Path],
    metrics_dir: Path,
    *,
    rerun: bool = False,
    missing_ci_only: bool = False,
    allow_training: bool = False,
    sample_size: int | None = None,
) -> list[str]:
    """Chạy có chọn lọc, không tự train model thiếu checkpoint trừ khi được cho phép."""
    messages: list[str] = []
    for config in dict.fromkeys(configs):
        cfg = _read_config(config)
        name = str(cfg.get("experiment_name", config.stem))
        target = Path(cfg.get("paths", {}).get("reports_dir", "reports")) / "metrics" / f"{name}_dev.json"
        # Nếu --metrics-dir dùng để tập hợp file từ vị trí khác, không thay đổi
        # đường dẫn output của evaluate(); chỉ cảnh báo vì nguồn dữ liệu có thể khác.
        if sample_size is not None and target.exists() and (rerun or missing_ci_only):
            raise ValueError(
                f"Không ghi đè Full Dev bằng sample-size lên {target}. "
                "Hãy chạy bản sao cấu hình với experiment_name riêng."
            )
        if target.exists() and not rerun:
            if not missing_ci_only:
                messages.append(f"SKIP đã có: {target}")
                continue
            old = _read_json(target)
            ci_root = old.get("comparison", {}).get("bootstrap", {}) if old.get("evaluation_mode") else old.get("bootstrap", {})
            if ci_root.get("slices"):
                messages.append(f"SKIP đã có slice CI: {target}")
                continue
        checkpoint = _checkpoint_from_config(cfg)
        if not checkpoint.is_file() and not allow_training:
            messages.append(f"SKIP thiếu checkpoint (không tự train): {name} ({checkpoint})")
            continue
        if metrics_dir.resolve() != target.parent.resolve():
            messages.append(f"LƯU Ý evaluate ghi vào {target.parent}, không phải {metrics_dir}")
        from src.evaluate import evaluate

        evaluate(config, sample_size=sample_size)
        messages.append(f"DONE: {name} -> {target}")
    return messages


def _metric_ci(
    result: dict[str, Any],
    *,
    slice_name: str,
    metric_name: str,
    side: str,
) -> tuple[Any, Any]:
    """Lấy CI đúng nguồn; JSON cũ không có CI theo slice thì giữ trống."""
    if result.get("evaluation_mode") == "paired_baseline_vs_p3":
        boot = result.get("comparison", {}).get("bootstrap", {})
        if slice_name == "overall":
            # Paired overall đã có từ bản đánh giá cũ.
            info = boot.get("metrics", {}).get(metric_name, {})
        else:
            info = boot.get("slices", {}).get(slice_name, {}).get("metrics", {}).get(metric_name, {})
        interval = info.get("baseline_ci" if side == "baseline" else "reranked_ci")
    else:
        boot = result.get("bootstrap", {})
        if slice_name == "overall":
            info = boot.get("metrics", {}).get(metric_name, {})
        else:
            info = boot.get("slices", {}).get(slice_name, {}).get("metrics", {}).get(metric_name, {})
        interval = info.get("ci")
    if isinstance(interval, list) and len(interval) == 2:
        return interval[0], interval[1]
    return "", ""


def collect_reports(
    metrics_dir: Path,
    *,
    include_p3_baselines: bool = False,
    allow_mixed_counts: bool = False,
    selected_models: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Đọc những kết quả *_dev.json. Không cần model, checkpoint hay PyArrow."""
    sources: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(metrics_dir.glob("*_dev.json")):
        if selected_models is not None and path.stem.removesuffix("_dev") not in selected_models:
            continue
        result = _read_json(path)
        if int(result.get("total_evaluated_impressions", 0)) <= 0:
            raise ValueError(f"Thiếu total_evaluated_impressions: {path}")
        sources.append((path, result))
    if not sources:
        raise FileNotFoundError(f"Không có *_dev.json trong {metrics_dir}")

    sizes = Counter(int(obj["total_evaluated_impressions"]) for _, obj in sources)
    reference_count = max(sizes)  # Ưu tiên Full Dev, không để nhiều sample lấn át Full Dev.
    excluded = []
    models: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []

    for source, result in sources:
        count = int(result["total_evaluated_impressions"])
        if not allow_mixed_counts and count != reference_count:
            excluded.append({"source": source.name, "count": count, "reason": "mismatched impressions"})
            continue

        is_p3 = result.get("evaluation_mode") == "paired_baseline_vs_p3"
        sides = ("baseline", "reranked") if is_p3 and include_p3_baselines else ("reranked",) if is_p3 else ("single",)
        for side in sides:
            model_id = source.stem.removesuffix("_dev")
            if side == "baseline":
                model_id += "__baseline_in_p3"
            slice_info = result.get(side, {}).get("slices", {}) if is_p3 else result.get("slices", {})
            beyond = result.get(side, {}).get("beyond_accuracy", {}) if is_p3 else result.get("beyond_accuracy", {})
            boot = result.get("comparison", {}).get("bootstrap", {}) if is_p3 else result.get("bootstrap", {})
            overall_metrics = slice_info.get("overall", {}).get("metrics", {})
            comparison: dict[str, Any] = {
                "model": model_id,
                "variant": side,
                "source_file": source.name,
                "impressions": count,
                "bootstrap_unit": boot.get("unit", ""),
                "num_resamples": boot.get("num_resamples", ""),
            }
            for metric in METRICS:
                comparison[metric] = overall_metrics.get(metric, "")
                low, high = _metric_ci(result, slice_name="overall", metric_name=metric, side=side)
                comparison[f"{metric}_ci_low"] = low
                comparison[f"{metric}_ci_high"] = high
            for metric in BEYOND_METRICS:
                comparison[metric] = beyond.get(metric, "")
            models.append(comparison)

            for slice_name in SLICES:
                entry = slice_info.get(slice_name, {})
                metrics = entry.get("metrics", {})
                for metric in METRICS:
                    low, high = _metric_ci(
                        result, slice_name=slice_name, metric_name=metric, side=side,
                    )
                    value = metrics.get(metric, "")
                    status = "available" if low != "" and high != "" else "not_recorded"
                    if value == "" or entry.get("count", 0) == 0:
                        status = "no_data"
                        low, high = "", ""
                    slices.append({
                        "model": model_id,
                        "variant": side,
                        "source_file": source.name,
                        "slice": slice_name,
                        "count": entry.get("count", 0),
                        "metric": metric,
                        "mean": value,
                        "ci_low": low,
                        "ci_high": high,
                        "ci_status": status,
                        "bootstrap_unit": boot.get("unit", ""),
                        "num_resamples": boot.get("num_resamples", ""),
                    })

    manifest = {
        "metrics_dir": str(metrics_dir),
        "reference_impressions": reference_count,
        "input_files": len(sources),
        "model_rows": len(models),
        "slice_metric_rows": len(slices),
        "ci_available_rows": sum(r["ci_status"] == "available" for r in slices),
        "ci_not_recorded_rows": sum(r["ci_status"] == "not_recorded" for r in slices),
        "excluded_mismatched": excluded,
        "methodology": (
            "CI chỉ lấy từ bootstrap per-impression khi chạy evaluate; "
            "không tính lại CI từ mean JSON. Item slices sử dụng relevant/clicked "
            "items riêng, có thể chồng nhau; không cộng số lượng slice."
        ),
    }
    return models, slices, manifest


def _save_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_reports(
    output_dir: Path, models: list[dict[str, Any]], slices: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_file = output_dir / "model_comparison.csv"
    slice_file = output_dir / "model_slice_ci.csv"
    manifest_file = output_dir / "model_evaluation_manifest.json"
    model_fields = ["model", "variant", "source_file", "impressions", "bootstrap_unit", "num_resamples"]
    for metric in METRICS:
        model_fields.extend((metric, f"{metric}_ci_low", f"{metric}_ci_high"))
    model_fields.extend(BEYOND_METRICS)
    slice_fields = ["model", "variant", "source_file", "slice", "count", "metric", "mean", "ci_low", "ci_high", "ci_status", "bootstrap_unit", "num_resamples"]
    _save_csv(model_file, models, model_fields)
    _save_csv(slice_file, slices, slice_fields)
    manifest_file.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"models": model_file, "slices": slice_file, "manifest": manifest_file}



def collect_p3_delta_rows(
    metrics_dir: Path,
    *,
    reference_impressions: int,
    selected_models: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Tổng hợp paired CI đã lưu; không ước lượng CI từ các số trung bình.

    Chỉ đưa các thí nghiệm Full Dev cùng kích thước với bảng so sánh chính.
    """
    rows: list[dict[str, Any]] = []
    for source in sorted(metrics_dir.glob("*_dev.json")):
        if selected_models is not None and source.stem.removesuffix("_dev") not in selected_models:
            continue
        result = _read_json(source)
        if result.get("evaluation_mode") != "paired_baseline_vs_p3":
            continue
        if int(result.get("total_evaluated_impressions", 0)) != reference_impressions:
            continue

        bootstrap = result.get("comparison", {}).get("bootstrap", {})
        before_slices = result.get("baseline", {}).get("slices", {})
        after_slices = result.get("reranked", {}).get("slices", {})
        model_id = source.stem.removesuffix("_dev")

        for slice_name in SLICES:
            before_slice = before_slices.get(slice_name, {})
            after_slice = after_slices.get(slice_name, {})
            before_metrics = before_slice.get("metrics", {})
            after_metrics = after_slice.get("metrics", {})
            count = after_slice.get("count", 0)
            if slice_name == "overall":
                bootstrap_metrics = bootstrap.get("metrics", {})
            else:
                bootstrap_metrics = (
                    bootstrap.get("slices", {})
                    .get(slice_name, {})
                    .get("metrics", {})
                )

            for metric in METRICS:
                before = before_metrics.get(metric, "")
                after = after_metrics.get(metric, "")
                info = bootstrap_metrics.get(metric, {})
                interval = info.get("delta_ci")
                valid_interval = isinstance(interval, list) and len(interval) == 2
                if count == 0 or before == "" or after == "":
                    status = "no_data"
                elif valid_interval:
                    status = "available"
                else:
                    status = "not_recorded"

                rows.append({
                    "model": model_id,
                    "source_file": source.name,
                    "slice": slice_name,
                    "count": count,
                    "metric": metric,
                    "baseline": before,
                    "reranked": after,
                    "delta": info.get("delta", float(after) - float(before))
                    if status != "no_data" else "",
                    "delta_ci_low": interval[0] if status == "available" else "",
                    "delta_ci_high": interval[1] if status == "available" else "",
                    "ci_status": status,
                    "bootstrap_unit": bootstrap.get("unit", ""),
                    "num_resamples": bootstrap.get("num_resamples", ""),
                })
    return rows


def write_p3_delta_report(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    """Xuất bảng trước/sau P3 và CI hiệu số theo từng lát cắt."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "p3_paired_delta_ci.csv"
    columns = [
        "model", "source_file", "slice", "count", "metric", "baseline",
        "reranked", "delta", "delta_ci_low", "delta_ci_high", "ci_status",
        "bootstrap_unit", "num_resamples",
    ]
    _save_csv(path, rows, columns)
    return path

def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá đa model và tổng hợp CI theo slice")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--run-config", type=Path, action="append", default=[], help="YAML cần đánh giá")
    parser.add_argument("--run-all-configs", action="store_true", help="Chạy mọi YAML exp*_mind_*.yaml")
    parser.add_argument("--configs-dir", type=Path, default=Path("configs/experiment"))
    parser.add_argument("--rerun-existing", action="store_true", help="Chạy lại dù đã có result")
    parser.add_argument("--missing-ci-only", action="store_true", help="Chạy lại model chưa có CI theo slice")
    parser.add_argument("--allow-training", action="store_true", help="Cho phép huấn luyện checkpoint còn thiếu")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--include-p3-baselines", action="store_true")
    parser.add_argument("--allow-mixed-counts", action="store_true", help="Cho phép trộn Full Dev với sample")
    args = parser.parse_args()

    configs = list(args.run_config)
    if args.run_all_configs:
        configs.extend(sorted(args.configs_dir.glob("exp*_mind_*.yaml")))
    if configs:
        for message in evaluate_configs(
            configs, args.metrics_dir,
            rerun=args.rerun_existing,
            missing_ci_only=args.missing_ci_only,
            allow_training=args.allow_training,
            sample_size=args.sample_size,
        ):
            print(message)
    models, slices, manifest = collect_reports(
        args.metrics_dir,
        include_p3_baselines=args.include_p3_baselines,
        allow_mixed_counts=args.allow_mixed_counts,
    )
    files = write_reports(args.output_dir, models, slices, manifest)
    p3_delta_rows = collect_p3_delta_rows(
        args.metrics_dir, reference_impressions=manifest["reference_impressions"],
    )
    files["p3_paired_deltas"] = write_p3_delta_report(args.output_dir, p3_delta_rows)
    for name, file in files.items():
        print(f"{name}: {file}")
    print(
        f"Tổng hợp {manifest['model_rows']} mô hình / {manifest['reference_impressions']:,} impressions; "
        f"CI theo slice có {manifest['ci_available_rows']}/{manifest['slice_metric_rows']} dòng."
    )
    if manifest["excluded_mismatched"]:
        print(f"CẢNH BÁO: bỏ qua {len(manifest['excluded_mismatched'])} file lệch số impressions")


if __name__ == "__main__":
    main()
