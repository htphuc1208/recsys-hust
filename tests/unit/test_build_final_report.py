"""Test final report builder bằng bộ result nhỏ tổng hợp."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.evaluation.build_final_report import (
    ReportPaths,
    load_inputs,
    write_reports,
)


def _slice(ndcg: float, recall: float, mrr: float, count: int = 20_000) -> dict:
    """Tạo một slice metric tối thiểu."""
    return {
        "count": count,
        "pct": 100.0,
        "metrics": {
            "NDCG@10": ndcg,
            "Recall@10": recall,
            "MRR@10": mrr,
        },
    }


def _single_result(ndcg: float, recall: float, mrr: float) -> dict:
    """Tạo result của model thường."""
    slices = {
        "overall": _slice(ndcg, recall, mrr),
        "cold_item": _slice(ndcg - 0.03, recall - 0.03, mrr - 0.03),
        "warm_item": _slice(ndcg + 0.03, recall + 0.03, mrr + 0.03),
        "cold_user": _slice(ndcg - 0.01, recall - 0.01, mrr - 0.01),
        "warm_user": _slice(ndcg + 0.01, recall + 0.01, mrr + 0.01),
        "head_item": _slice(ndcg + 0.08, recall, mrr),
        "long_tail_item": _slice(ndcg - 0.08, recall, mrr),
        "zero_click_item": _slice(ndcg - 0.04, recall, mrr),
        "catalog_cold_item": _slice(ndcg + 0.10, recall, mrr),
        "catalog_warm_item": _slice(ndcg, recall, mrr),
    }
    return {
        "total_evaluated_impressions": 20_000,
        "slices": slices,
        "beyond_accuracy": {
            "catalog_coverage@10": 0.03,
            "gini_index@10": 0.998,
        },
        "bootstrap": {
            "unit": "user",
            "num_resamples": 1000,
            "confidence_level": 0.95,
            "metrics": {
                "NDCG@10": {"mean": ndcg, "ci": [ndcg - 0.001, ndcg + 0.001]},
                "Recall@10": {"mean": recall, "ci": [recall - 0.001, recall + 0.001]},
                "MRR@10": {"mean": mrr, "ci": [mrr - 0.001, mrr + 0.001]},
            },
        },
    }


def _p3_result(ndcg: float, recall: float, mrr: float) -> dict:
    """Tạo paired baseline vs P3 result."""
    baseline = _single_result(ndcg, recall, mrr)
    reranked = _single_result(ndcg - 0.005, recall - 0.006, mrr - 0.002)
    baseline["beyond_accuracy"].update(
        {
            "category_diversity@10": 0.70,
            "long_tail_exposure@10": 0.45,
            "zero_click_exposure@10": 0.22,
        }
    )
    reranked["beyond_accuracy"].update(
        {
            "category_diversity@10": 0.76,
            "long_tail_exposure@10": 0.50,
            "zero_click_exposure@10": 0.27,
        }
    )
    return {
        "evaluation_mode": "paired_baseline_vs_p3",
        "total_evaluated_impressions": 20_000,
        "baseline": {
            "slices": baseline["slices"],
            "beyond_accuracy": baseline["beyond_accuracy"],
        },
        "reranked": {
            "slices": reranked["slices"],
            "beyond_accuracy": reranked["beyond_accuracy"],
        },
        "comparison": {
            "bootstrap": {
                "unit": "user",
                "num_resamples": 1000,
                "confidence_level": 0.95,
                "metrics": {
                    "NDCG@10": {"delta_ci": [-0.006, -0.004]},
                    "Recall@10": {"delta_ci": [-0.007, -0.005]},
                    "MRR@10": {"delta_ci": [-0.003, -0.001]},
                    "category_diversity@10": {"delta_ci": [0.05, 0.07]},
                    "long_tail_exposure@10": {"delta_ci": [0.04, 0.06]},
                },
            }
        },
    }


def _write_json(path: Path, data: dict) -> None:
    """Ghi JSON fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_config(path: Path, data: dict) -> None:
    """Ghi YAML fixture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def _prepare_project(tmp_path: Path) -> ReportPaths:
    """Tạo project fixture đủ input cho final report."""
    metrics_dir = tmp_path / "reports" / "metrics"
    dm_dir = tmp_path / "reports" / "data_minimization"
    config_dir = tmp_path / "configs" / "experiment"

    _write_json(metrics_dir / "exp04_mind_bpr_dev.json", _single_result(0.357, 0.615, 0.295))
    _write_json(metrics_dir / "exp05_mind_content_dev.json", _single_result(0.391, 0.634, 0.337))
    _write_json(metrics_dir / "exp06_mind_content_p3_dev.json", _p3_result(0.391, 0.634, 0.337))
    _write_json(metrics_dir / "exp07_mind_bpr_p3_dev.json", _p3_result(0.357, 0.615, 0.295))

    dm_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, all_ndcg, storage in (
        ("bpr", 0.357, 194.0),
        ("content_based", 0.359, 86.0),
    ):
        for window, retained, reduction, ndcg, train_time in (
            ("1d", 20.0, 80.0, all_ndcg * 0.98, 100.0),
            ("3d", 62.0, 40.0, all_ndcg * 0.99, 200.0),
            ("5d", 91.0, 9.0, all_ndcg * 0.997, 300.0),
            ("all", 100.0, 0.0, all_ndcg, 400.0),
        ):
            rows.append(
                {
                    "window": window,
                    "days": window,
                    "model": model,
                    "train_rows": 100,
                    "data_retained_pct": retained,
                    "training_storage_mb": storage * retained / 100.0,
                    "storage_reduction_pct": reduction,
                    "train_wall_seconds": train_time,
                    "eval_wall_seconds": 10.0,
                    "NDCG@10": ndcg,
                    "Recall@10": 0.60,
                    "MRR@10": 0.30,
                    "checkpoint_path": "x.pkl",
                    "metrics_path": "x.json",
                }
            )
    pd.DataFrame(rows).to_csv(dm_dir / "summary.csv", index=False)

    stats_rows = []
    for window, days, start, p_ret, pos_ret in (
        ("1d", 1, "2019-11-14 00:00:00", 20.0, 20.0),
        ("3d", 3, "2019-11-12 00:00:00", 62.0, 62.0),
        ("5d", 5, "2019-11-10 00:00:00", 91.0, 91.0),
        ("all", "all", "all", 100.0, 100.0),
    ):
        stats_rows.append(
            {
                "window": window,
                "days": days,
                "start_timestamp": start,
                "cutoff_timestamp": "2019-11-15 00:00:00",
                "pointwise_rows": 1000,
                "positive_rows": 200,
                "pointwise_retention_pct": p_ret,
                "positive_retention_pct": pos_ret,
                "pointwise_storage_mb": p_ret,
                "positive_storage_mb": pos_ret / 2,
                "pointwise_storage_reduction_pct": 100 - p_ret,
                "positive_storage_reduction_pct": 100 - pos_ret,
            }
        )
    pd.DataFrame(stats_rows).to_csv(dm_dir / "dataset_stats.csv", index=False)

    _write_config(
        config_dir / "exp04_mind_bpr.yaml",
        {"model": {"name": "bpr"}},
    )
    _write_config(
        config_dir / "exp05_mind_content.yaml",
        {
            "model": {
                "name": "content_based",
                "cold_user_strategy": "zero",
                "feature_weights": {
                    "title": 0.70,
                    "abstract": 0.0,
                    "category": 0.15,
                    "subcategory": 0.15,
                    "title_entity": 0.0,
                },
            }
        },
    )
    _write_config(
        config_dir / "exp06_mind_content_p3.yaml",
        {"reranking": {"weights": {"relevance": 0.80, "diversity": 0.15, "long_tail": 0.05}}},
    )
    _write_config(
        config_dir / "exp07_mind_bpr_p3.yaml",
        {"reranking": {"weights": {"relevance": 0.75, "diversity": 0.05, "long_tail": 0.20}}},
    )
    _write_config(config_dir / "data_minimization.yaml", {"windows": ["1d", "3d", "5d", "all"]})

    return ReportPaths(
        project_root=tmp_path,
        metrics_dir=metrics_dir,
        data_minimization_dir=dm_dir,
        config_dir=config_dir,
        tables_dir=tmp_path / "reports" / "tables",
        figures_dir=tmp_path / "reports" / "figures",
        reports_root=tmp_path / "reports",
    )


def test_build_final_report_outputs(tmp_path: Path) -> None:
    """Sinh đủ tables, figures và Markdown report."""
    paths = _prepare_project(tmp_path)
    inputs = load_inputs(paths)
    outputs = write_reports(paths, inputs)

    assert inputs.full_dev_count == 20_000
    assert all(path.exists() for path in outputs.values())
    assert (paths.figures_dir / "model_accuracy_comparison.png").exists()
    assert (paths.figures_dir / "p3_bpr_tradeoff.png").exists()
    assert (paths.figures_dir / "data_minimization_ndcg_retained.png").exists()

    final_summary = (paths.reports_root / "final_summary.md").read_text(encoding="utf-8")
    assert "Data Minimization / GDPR-motivated experiment" in final_summary
    assert "catalog_cold_item" in final_summary
    assert "optimistic bias" in final_summary
    assert '"relevance": 0.75' in final_summary

    model_card = (paths.reports_root / "model_card.md").read_text(encoding="utf-8")
    assert "GDPR-motivated experiment" in model_card
    assert "Long-tail Exposure@10" in model_card
    assert "unbiased test estimates" in model_card

    datasheet = (paths.reports_root / "datasheet.md").read_text(encoding="utf-8")
    assert "2019-11-09 00:00:00" in datasheet
    assert "category one-hot" in datasheet
    assert "không chứng minh GDPR compliance" in datasheet

    dm_table = pd.read_csv(paths.tables_dir / "data_minimization_summary.csv")
    all_bpr = dm_table[(dm_table["model"] == "bpr") & (dm_table["window"] == "all")].iloc[0]
    assert all_bpr["NDCG_loss_vs_all_pct"] == pytest.approx(0.0)
    assert all_bpr["utility_retained_pct"] == pytest.approx(100.0)


def test_rejects_preview_sample(tmp_path: Path) -> None:
    """Không cho build final report từ result preview 1.000 impression."""
    paths = _prepare_project(tmp_path)
    for path in paths.metrics_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["total_evaluated_impressions"] = 1000
        path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="full dev"):
        load_inputs(paths)


def test_rejects_mismatched_p3_baseline(tmp_path: Path) -> None:
    """Không cho trộn paired P3 result với baseline checkpoint/config khác."""
    paths = _prepare_project(tmp_path)
    p3_path = paths.metrics_dir / "exp07_mind_bpr_p3_dev.json"
    result = json.loads(p3_path.read_text(encoding="utf-8"))
    result["baseline"]["slices"]["overall"]["metrics"]["NDCG@10"] = 0.300
    p3_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(ValueError, match="không khớp"):
        load_inputs(paths)
