"""Xuất báo cáo đa model chuẩn hóa từ JSON Dev, không cần checkpoint/dataset.

- core: BPR, Content-Based và hai phiên bản P3 đã hoàn thiện.
- all: tự nhận bất kỳ mô hình *_dev.json mới nào (không sửa code).
- --model: chọn danh sách mô hình tùy ý, kể cả model được thêm về sau.

Không suy diễn bootstrap CI từ giá trị mean, không trộn sample và Full Dev.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.run_all import (
    BEYOND_METRICS,
    METRICS,
    SLICES,
    collect_p3_delta_rows,
    collect_reports,
    write_p3_delta_report,
    write_reports,
)

CORE_MODELS = (
    "exp04_mind_bpr",
    "exp05_mind_content",
    "exp06_mind_content_p3",
    "exp07_mind_bpr_p3",
)


# Không buộc các mô hình trong tương lai phải đổi code để được xuất báo cáo.
def choose_model_ids(
    metrics_dir: Path, *, profile: str = "core", selected_models: list[str] | None = None,
) -> tuple[set[str] | None, list[str]]:
    if selected_models:
        requested = {name.removesuffix("_dev.json").removesuffix("_dev") for name in selected_models}
    elif profile == "core":
        requested = set(CORE_MODELS)
    elif profile == "all":
        return None, []
    else:
        raise ValueError(f"Profile không hợp lệ: {profile}")
    available = {p.stem.removesuffix("_dev") for p in metrics_dir.glob("*_dev.json")}
    return requested & available, sorted(requested - available)


def _num(value: Any) -> float | None:
    if value == "" or value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _fmt(value: Any, places: int = 5) -> str:
    x = _num(value)
    return f"{x:.{places}f}" if x is not None else "—"


def _csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_p3_tradeoff_rows(metrics_dir: Path, included: set[str]) -> list[dict[str, Any]]:
    """Một hàng / chỉ số / mô hình P3. Khi thêm P3 mới, tự nhận schema cũ."""
    rows: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob("*_dev.json")):
        model_id = path.stem.removesuffix("_dev")
        if model_id not in included:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("evaluation_mode") != "paired_baseline_vs_p3":
            continue
        baseline = result.get("baseline", {}).get("beyond_accuracy", {})
        reranked = result.get("reranked", {}).get("beyond_accuracy", {})
        for metric in BEYOND_METRICS:
            before = _num(baseline.get(metric))
            after = _num(reranked.get(metric))
            rows.append({
                "model": model_id,
                "metric": metric,
                "baseline": before if before is not None else "",
                "reranked": after if after is not None else "",
                "delta": after - before if before is not None and after is not None else "",
                "status": "available" if before is not None and after is not None else "no_data",
                "source_file": path.name,
            })
    return rows


def _summary_text(
    models: list[dict[str, Any]],
    slice_rows: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    tradeoffs: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    profile: str,
    missing_models: list[str],
) -> str:
    lines = [
        "# Báo cáo đánh giá hệ gợi ý – RecSys HUST",
        "",
        f"**Chế độ:** `{profile}` | **Dev impressions:** {manifest['reference_impressions']:,} | "
        f"**Kết quả mô hình:** {manifest['model_rows']}",
        "",
        "Báo cáo tái tạo trực tiếp từ JSON đánh giá có sẵn, không chạy huấn luyện hay suy luận. "
        "Các khoảng tin cậy chỉ được hiển thị khi đã được lưu bởi evaluator.",
        "",
        "## 1. Bảng tổng hợp độ chính xác",
        "",
        "| Mô hình | NDCG@10 | 95% CI | Recall@10 | MRR@10 |",
        "|---|---:|---|---:|---:|",
    ]
    for m in models:
        low, high = _num(m.get("NDCG@10_ci_low")), _num(m.get("NDCG@10_ci_high"))
        ci = f"[{low:.5f}, {high:.5f}]" if low is not None and high is not None else "chưa ghi"
        lines.append(
            f"| `{m['model']}` | {_fmt(m.get('NDCG@10'))} | {ci} | "
            f"{_fmt(m.get('Recall@10'))} | {_fmt(m.get('MRR@10'))} |"
        )
    lines.extend(["", "## 2. Bootstrap CI theo lát cắt", ""])
    counts = Counter(row["ci_status"] for row in slice_rows)
    lines.append(
        f"Có **{counts['available']} / {len(slice_rows)}** dòng metric–lát cắt có CI 95%; "
        f"**{counts['not_recorded']}** dòng chưa ghi CI và **{counts['no_data']}** dòng không có dữ liệu."
    )
    lines.extend([
        "", "- `available`: khoảng tin cậy lấy từ JSON đã đánh giá.",
        "- `not_recorded`: có metric nhưng chưa lưu Bootstrap CI; **không được nội suy CI từ mean**.",
        "- `no_data`: không có metric/lát cắt hợp lệ trong báo cáo nguồn.",
        "", "## 3. P3: Paired Bootstrap trên cùng impression", "",
        "| Mô hình | Lát cắt | ΔNDCG@10 | Paired CI 95% | Nhận xét |",
        "|---|---|---:|---|---|",
    ])
    important = {"overall", "catalog_cold_item", "zero_click_item", "head_item", "long_tail_item"}
    for row in paired:
        if row["metric"] != "NDCG@10" or row["slice"] not in important:
            continue
        low, high = _num(row.get("delta_ci_low")), _num(row.get("delta_ci_high"))
        if low is None or high is None:
            ci = "chưa ghi"
            statement = "chưa kết luận"
        elif low > 0:
            ci, statement = f"[{low:.5f}, {high:.5f}]", "CI > 0"
        elif high < 0:
            ci, statement = f"[{low:.5f}, {high:.5f}]", "CI < 0"
        else:
            ci, statement = f"[{low:.5f}, {high:.5f}]", "CI chứa 0"
        lines.append(f"| `{row['model']}` | `{row['slice']}` | {_fmt(row['delta'])} | {ci} | {statement} |")
    if not paired:
        lines.append("| Chưa có thí nghiệm P3 | — | — | — | — |")
    lines.extend([
        "", "## 4. Beyond-accuracy và các đánh đổi", "",
        "| Mô hình P3 | Chỉ số | Trước P3 | Sau P3 | Chênh lệch |",
        "|---|---|---:|---:|---:|",
    ])
    for row in tradeoffs:
        if row["status"] != "available":
            continue
        lines.append(
            f"| `{row['model']}` | `{row['metric']}` | {_fmt(row['baseline'])} | "
            f"{_fmt(row['reranked'])} | {_fmt(row['delta'])} |"
        )
    if not tradeoffs:
        lines.append("| Chưa có thí nghiệm P3 | — | — | — | — |")
    lines.extend([
        "", "Giá trị ở bảng này là chênh lệch quan sát được; không suy diễn độ tin cậy "
        "thống kê cho các chỉ số chưa có Bootstrap CI.",
        "", "## 5. Phạm vi và giới hạn", "",
        "- Chỉ đối chiếu các báo cáo cùng số impression Full Dev; mẫu khác được loại khỏi so sánh mặc định.",
        "- Các lát cắt theo item có thể chồng nhau; không cộng số lượng slice để suy ra tổng số impression.",
        "- Không suy diễn ý nghĩa thống kê cho chênh lệch **hai model khác nhau** từ hai CI riêng. "
        "Muốn kết luận cần paired Bootstrap của hiệu số theo cùng impression.",
        "- Kết quả Dev đã được dùng trong quá trình lựa chọn cấu hình; không coi là Test độc lập.",
    ])
    if missing_models:
        lines.append("- Chưa tìm thấy báo cáo yêu cầu: " + ", ".join(f"`{m}`" for m in missing_models))
    if manifest["excluded_mismatched"]:
        lines.append("- Đã loại các file lệch số impression: " + ", ".join(r["source"] for r in manifest["excluded_mismatched"]))
    lines.extend([
        "", "## 6. Khả năng mở rộng", "",
        "- `--profile core`: chỉ chọn BPR, Content-Based và hai kết quả P3 hiện có.",
        "- `--profile all`: tự đọc tất cả JSON `*_dev.json` hợp lệ được thêm vào trong tương lai.",
        "- `--model <experiment_name>` (lặp lại): chọn một tập mô hình tùy ý, không sửa mã nguồn.",
        "- Khi thêm một mô hình, chỉ cần chạy evaluator để có `reports/metrics/<experiment_name>_dev.json` "
        "cùng schema hiện hành rồi chạy lại script; không cần checkpoint khi chỉ xuất báo cáo.",
        "",
    ])
    return "\n".join(lines)


def plot_figures(
    models: list[dict[str, Any]], slices: list[dict[str, Any]],
    deltas: list[dict[str, Any]], tradeoffs: list[dict[str, Any]], output_dir: Path,
) -> dict[str, Path]:
    """Vẽ biểu đồ riêng; không giả định sẵn số lượng hoặc tên mô hình."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    if models:
        metrics = ["NDCG@10", "Recall@10", "MRR@10"]
        labels = [m["model"].replace("exp", "E", 1) for m in models]
        x = np.arange(len(models))
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        width = 0.23
        for i, metric in enumerate(metrics):
            vals = [_num(m.get(metric)) or 0 for m in models]
            ax.bar(x + (i - 1) * width, vals, width, label=metric)
        ax.set_xticks(x, labels, rotation=17, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.set_title("Dev accuracy - model comparison")
        ax.legend()
        ax.grid(axis="y", alpha=0.22)
        fig.tight_layout()
        saved["accuracy"] = output_dir / "task2_model_accuracy.png"
        fig.savefig(saved["accuracy"], dpi=170)
        plt.close(fig)

    n_rows = [r for r in slices if r["metric"] == "NDCG@10" and r["ci_status"] != "no_data"]
    model_names = [m["model"] for m in models]
    non_p3 = [m for m in model_names if "p3" not in m.lower()]
    selected = non_p3 if non_p3 else model_names[: min(4, len(model_names))]
    data = {(r["model"], r["slice"]): _num(r["mean"]) for r in n_rows}
    active_slices = [s for s in SLICES if s != "overall" and any(data.get((m, s)) is not None for m in selected)]
    if selected and active_slices:
        x = np.arange(len(active_slices))
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        width = 0.74 / len(selected)
        for i, m in enumerate(selected):
            vals = [data.get((m, s)) if data.get((m, s)) is not None else np.nan for s in active_slices]
            ax.bar(x - 0.37 + (i + 0.5) * width, vals, width, label=m)
        ax.set_xticks(x, [s.replace("_", "\n") for s in active_slices], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("NDCG@10")
        ax.set_title("NDCG@10 across evaluation slices")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        saved["slices"] = output_dir / "task2_slice_ndcg.png"
        fig.savefig(saved["slices"], dpi=170)
        plt.close(fig)

    filtered = [r for r in deltas if r["metric"] == "NDCG@10" and r["slice"] != "overall" and _num(r["delta"]) is not None]
    if filtered:
        p3s = list(dict.fromkeys(r["model"] for r in filtered))
        unique_slices = list(dict.fromkeys(r["slice"] for r in filtered))
        lookup = {(r["model"], r["slice"]): r for r in filtered}
        x = np.arange(len(unique_slices))
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        width = 0.78 / len(p3s)
        for i, model in enumerate(p3s):
            vals = []
            lower, upper = [], []
            for sl in unique_slices:
                r = lookup.get((model, sl))
                val = _num(r["delta"]) if r else None
                low = _num(r["delta_ci_low"]) if r else None
                high = _num(r["delta_ci_high"]) if r else None
                vals.append(val if val is not None else np.nan)
                lower.append((val - low) if val is not None and low is not None else 0.0)
                upper.append((high - val) if val is not None and high is not None else 0.0)
            positions = x - 0.39 + (i + 0.5) * width
            ax.bar(positions, vals, width, label=model)
            for j, sl in enumerate(unique_slices):
                r = lookup.get((model, sl))
                low = _num(r["delta_ci_low"]) if r else None
                high = _num(r["delta_ci_high"]) if r else None
                if low is not None and high is not None:
                    # Vẽ đoạn CI thực thay vì giả định estimate nằm giữa hai cận.
                    ax.vlines(positions[j], low, high, color="black", linewidth=0.9)
                    ax.hlines([low, high], positions[j] - width * 0.07,
                              positions[j] + width * 0.07, color="black", linewidth=0.9)
        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x, [s.replace("_", "\n") for s in unique_slices], fontsize=9)
        ax.set_ylabel("Delta NDCG@10 (P3 - baseline)")
        ax.set_title("P3 paired changes by slice (95% CI when available)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        saved["p3_delta"] = output_dir / "task2_p3_paired_delta.png"
        fig.savefig(saved["p3_delta"], dpi=170)
        plt.close(fig)

    active_tradeoffs = [r for r in tradeoffs if r["metric"] in (
        "category_diversity@10", "long_tail_exposure@10", "zero_click_exposure@10",
    ) and r["status"] == "available"]
    if active_tradeoffs:
        groups = list(dict.fromkeys(r["model"] for r in active_tradeoffs))
        names = ["Category diversity", "Long-tail exposure", "Zero-click exposure"]
        metrics = ["category_diversity@10", "long_tail_exposure@10", "zero_click_exposure@10"]
        x = np.arange(len(groups) * len(metrics))
        base, rerank = [], []
        for group in groups:
            for metric in metrics:
                r = next((r for r in active_tradeoffs if r["model"] == group and r["metric"] == metric), None)
                base.append(_num(r["baseline"]) if r else np.nan)
                rerank.append(_num(r["reranked"]) if r else np.nan)
        fig, ax = plt.subplots(figsize=(12.8, 7.2))
        ax.bar(x - 0.18, base, 0.36, label="Before P3")
        ax.bar(x + 0.18, rerank, 0.36, label="After P3")
        ax.set_xticks(x, [f"{group}\n{names[i % len(names)]}" for group in groups for i in range(len(metrics))],
                      fontsize=8, rotation=12, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Exposure / diversity ratio")
        ax.set_title("Beyond-accuracy tradeoff before and after P3")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        fig.tight_layout()
        saved["p3_tradeoff"] = output_dir / "task2_p3_tradeoff.png"
        fig.savefig(saved["p3_tradeoff"], dpi=170)
        plt.close(fig)

    return saved


def build_final_report(
    metrics_dir: Path,
    output_dir: Path,
    figures_dir: Path,
    *,
    profile: str = "core",
    selected_models: list[str] | None = None,
    allow_mixed_counts: bool = False,
    generate_figures: bool = True,
) -> dict[str, Path]:
    model_filter, missing = choose_model_ids(metrics_dir, profile=profile, selected_models=selected_models)
    if model_filter is not None and not model_filter:
        raise FileNotFoundError("Không tìm thấy JSON đánh giá cho bất kỳ model nào được chọn.")
    models, slices, manifest = collect_reports(
        metrics_dir, selected_models=model_filter, allow_mixed_counts=allow_mixed_counts,
    )
    included = {m["model"] for m in models}
    manifest.update({"profile": profile, "missing_requested_models": missing, "selected_models": sorted(included)})
    files = write_reports(output_dir, models, slices, manifest)
    p3_rows = collect_p3_delta_rows(
        metrics_dir, reference_impressions=manifest["reference_impressions"],
        selected_models=included,
    )
    files["p3_paired_deltas"] = write_p3_delta_report(output_dir, p3_rows)
    tradeoff_rows = build_p3_tradeoff_rows(metrics_dir, included)
    # Đặt tên chuẩn cho bộ báo cáo cuối; run_all toàn hệ thống không bị ghi đè.
    copies = {
        "models": "task2_model_comparison.csv",
        "slices": "task2_slice_ci.csv",
        "p3_paired_deltas": "task2_p3_paired_delta_ci.csv",
    }
    for key, filename in copies.items():
        src = files.pop(key)
        dst = output_dir / filename
        src.replace(dst)
        files[key] = dst
    manifest_path = files["manifest"]
    named_manifest = output_dir / "task2_report_manifest.json"
    manifest_path.replace(named_manifest)
    files["manifest"] = named_manifest
    tradeoff_path = output_dir / "task2_p3_tradeoff.csv"
    _csv(tradeoff_path, tradeoff_rows, ["model", "metric", "baseline", "reranked", "delta", "status", "source_file"])
    files["p3_tradeoff"] = tradeoff_path
    summary_path = output_dir / "task2_summary.md"
    summary_path.write_text(
        _summary_text(models, slices, p3_rows, tradeoff_rows, manifest, profile=profile, missing_models=missing),
        encoding="utf-8",
    )
    files["summary"] = summary_path
    if generate_figures:
        files.update({f"figure_{k}": v for k, v in plot_figures(models, slices, p3_rows, tradeoff_rows, figures_dir).items()})
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Báo cáo cuối Task 2, hỗ trợ model mới tự động")
    parser.add_argument("--profile", choices=("core", "all"), default="core")
    parser.add_argument("--model", action="append", default=[], help="Tên thí nghiệm, lặp lại để chọn nhiều model")
    parser.add_argument("--metrics-dir", type=Path, default=Path("reports/metrics"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/tables/task2_final"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures/task2_final"))
    parser.add_argument("--allow-mixed-counts", action="store_true", help="Không khuyến nghị: chấp nhận so sánh kết quả khác sample size")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    outputs = build_final_report(
        args.metrics_dir, args.output_dir, args.figures_dir,
        profile=args.profile, selected_models=args.model or None,
        allow_mixed_counts=args.allow_mixed_counts, generate_figures=not args.no_figures,
    )
    for kind, path in outputs.items():
        print(f"{kind}: {path}")
    manifest = json.loads(outputs["manifest"].read_text(encoding="utf-8"))
    print(f"Đã tổng hợp {manifest['model_rows']} model, "
          f"{manifest['ci_available_rows']}/{manifest['slice_metric_rows']} CI theo lát cắt, "
          f"Full Dev {manifest['reference_impressions']:,} impressions")
    if manifest["missing_requested_models"]:
        print("LƯU Ý - chưa có kết quả:", ", ".join(manifest["missing_requested_models"]))


if __name__ == "__main__":
    main()
