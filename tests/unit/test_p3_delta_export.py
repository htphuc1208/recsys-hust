"""Kiểm tra báo cáo P3 theo slice không tạo CI giả."""

import csv
import json
from pathlib import Path

from src.evaluation.run_all import collect_p3_delta_rows, write_p3_delta_report


def _create_result(path: Path, *, impressions: int, with_slice_ci: bool) -> None:
    overall = {"count": impressions, "metrics": {"NDCG@10": 0.4, "Recall@10": 0.7}}
    after = {"count": impressions, "metrics": {"NDCG@10": 0.39, "Recall@10": 0.69}}
    slice_before = {"count": 10, "metrics": {"NDCG@10": 0.5}}
    slice_after = {"count": 10, "metrics": {"NDCG@10": 0.52}}
    boot = {
        "unit": "user",
        "num_resamples": 1000,
        "metrics": {"NDCG@10": {"delta": -0.01, "delta_ci": [-0.012, -0.008]}},
    }
    if with_slice_ci:
        boot["slices"] = {
            "catalog_cold_item": {
                "count": 10,
                "metrics": {"NDCG@10": {"delta": 0.02, "delta_ci": [0.01, 0.03]}},
            }
        }
    data = {
        "evaluation_mode": "paired_baseline_vs_p3",
        "total_evaluated_impressions": impressions,
        "baseline": {"slices": {"overall": overall, "catalog_cold_item": slice_before}},
        "reranked": {"slices": {"overall": after, "catalog_cold_item": slice_after}},
        "comparison": {"bootstrap": boot},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_p3_delta_export_retains_real_ci(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    _create_result(metrics_dir / "exp06_dev.json", impressions=100, with_slice_ci=True)
    _create_result(metrics_dir / "exp07_dev.json", impressions=50, with_slice_ci=True)

    rows = collect_p3_delta_rows(metrics_dir, reference_impressions=100)
    assert len(rows) == 30  # 1 valid experiment x 10 slices x 3 metrics
    assert {r["model"] for r in rows} == {"exp06"}

    overall = next(r for r in rows if r["slice"] == "overall" and r["metric"] == "NDCG@10")
    assert overall["ci_status"] == "available"
    assert overall["delta_ci_low"] == -0.012
    assert overall["delta_ci_high"] == -0.008

    cold = next(r for r in rows if r["slice"] == "catalog_cold_item" and r["metric"] == "NDCG@10")
    assert cold["ci_status"] == "available"
    assert cold["delta_ci_low"] == 0.01
    assert cold["delta"] == 0.02

    recall = next(
        r for r in rows if r["slice"] == "catalog_cold_item" and r["metric"] == "Recall@10"
    )
    assert recall["ci_status"] == "no_data"
    assert recall["delta_ci_low"] == ""

    csv_path = write_p3_delta_report(tmp_path / "tables", rows)
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 30
    assert saved[0]["model"] == "exp06"


def test_old_p3_report_does_not_fabricate_ci(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    _create_result(metrics_dir / "exp07_dev.json", impressions=100, with_slice_ci=False)

    rows = collect_p3_delta_rows(metrics_dir, reference_impressions=100)
    cold = next(r for r in rows if r["slice"] == "catalog_cold_item" and r["metric"] == "NDCG@10")
    assert cold["ci_status"] == "not_recorded"
    assert cold["delta_ci_low"] == ""
    assert abs(cold["delta"] - 0.02) < 1e-8
