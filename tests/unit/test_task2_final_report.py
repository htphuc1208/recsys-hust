"""Kiểm thử báo cáo Task 2: đúng dữ liệu, thêm model không sửa code, không chế CI."""
import csv
import json
from pathlib import Path

import pytest

from src.evaluation.final_report import (
    build_final_report,
    build_p3_tradeoff_rows,
    choose_model_ids,
)


def simple_result(n=50, *, with_ci=True):
    d = {
        "total_evaluated_impressions": n,
        "slices": {
            "overall": {"count": n, "metrics": {"NDCG@10": 0.4, "Recall@10": 0.6, "MRR@10": 0.3}},
            "cold_user": {"count": 10, "metrics": {"NDCG@10": 0.3, "Recall@10": 0.5, "MRR@10": 0.2}},
        },
        "beyond_accuracy": {"catalog_coverage@10": 0.02},
        "bootstrap": {"unit": "user", "num_resamples": 1000, "metrics": {
            "NDCG@10": {"mean": 0.4, "ci": [0.39, 0.41]}
        }},
    }
    if with_ci:
        d["bootstrap"]["slices"] = {
            "cold_user": {"count": 10, "metrics": {
                "NDCG@10": {"mean": 0.3, "ci": [0.28, 0.32]}
            }}
        }
    return d


def p3_result(n=50):
    b = simple_result(n)
    before = b["slices"]
    after = json.loads(json.dumps(before))
    after["overall"]["metrics"]["NDCG@10"] = 0.41
    after["cold_user"]["metrics"]["NDCG@10"] = 0.32
    return {
        "evaluation_mode": "paired_baseline_vs_p3",
        "total_evaluated_impressions": n,
        "baseline": {"slices": before, "beyond_accuracy": {"category_diversity@10": 0.5}},
        "reranked": {"slices": after, "beyond_accuracy": {"category_diversity@10": 0.6}},
        "comparison": {"bootstrap": {"unit": "user", "num_resamples": 1000,
            "metrics": {"NDCG@10": {"delta": 0.01, "delta_ci": [0.004, 0.016],
                "baseline_ci": [0.39, 0.41], "reranked_ci": [0.40, 0.42]}},
            "slices": {"cold_user": {"count": 10, "metrics": {"NDCG@10": {
                "delta": 0.02, "delta_ci": [0.005, 0.035],
                "baseline_ci": [0.28, 0.32], "reranked_ci": [0.30, 0.34]}}}},
        }},
    }


def save(d: Path, name: str, data: dict):
    (d / (name + "_dev.json")).write_text(json.dumps(data), encoding="utf-8")


def read_csv(p: Path):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_default_core_and_p3_tradeoff(tmp_path):
    md = tmp_path / "metrics"
    md.mkdir()
    save(md, "exp04_mind_bpr", simple_result())
    save(md, "exp05_mind_content", simple_result(with_ci=False))
    save(md, "exp06_mind_content_p3", p3_result())
    save(md, "exp07_mind_bpr_p3", p3_result())
    save(md, "exp03_mind_mf", simple_result())
    outputs = build_final_report(md, tmp_path / "tables", tmp_path / "figures", generate_figures=False)
    models = read_csv(outputs["models"])
    assert len(models) == 4
    assert "exp03_mind_mf" not in {m["model"] for m in models}
    delta = read_csv(outputs["p3_paired_deltas"])
    r = next(x for x in delta if x["model"] == "exp06_mind_content_p3" and x["slice"] == "cold_user" and x["metric"] == "NDCG@10")
    assert r["delta_ci_low"] == "0.005"
    tradeoffs = read_csv(outputs["p3_tradeoff"])
    x = next(r for r in tradeoffs if r["model"] == "exp06_mind_content_p3" and r["metric"] == "category_diversity@10")
    assert abs(float(x["delta"]) - 0.1) < 1e-10
    rows = read_csv(outputs["slices"])
    assert next(r for r in rows if r["model"] == "exp05_mind_content" and r["slice"] == "cold_user" and r["metric"] == "NDCG@10")["ci_status"] == "not_recorded"
    assert "chưa ghi" in outputs["summary"].read_text(encoding="utf-8")


def test_all_discovers_future_model_without_edit(tmp_path):
    md = tmp_path / "metrics"
    md.mkdir()
    save(md, "exp04_mind_bpr", simple_result())
    save(md, "exp93_future_transformer", simple_result())
    outs = build_final_report(md, tmp_path / "tables", tmp_path / "figures", profile="all", generate_figures=False)
    ids = {m["model"] for m in read_csv(outs["models"])}
    assert ids == {"exp04_mind_bpr", "exp93_future_transformer"}
    assert len(read_csv(outs["p3_paired_deltas"])) == 0  # only header
    assert len(read_csv(outs["p3_tradeoff"])) == 0


def test_custom_model_name_and_missing_warning(tmp_path):
    md = tmp_path / "metrics"
    md.mkdir()
    save(md, "exp91_new_model", simple_result())
    outs = build_final_report(md, tmp_path / "out", tmp_path / "figs", selected_models=["exp91_new_model", "missing"], generate_figures=False)
    assert [r["model"] for r in read_csv(outs["models"])] == ["exp91_new_model"]
    manifest = json.loads(outs["manifest"].read_text(encoding="utf-8"))
    assert manifest["missing_requested_models"] == ["missing"]


def test_mixed_sample_excluded(tmp_path):
    md = tmp_path / "metrics"
    md.mkdir()
    save(md, "exp04_mind_bpr", simple_result(n=50))
    save(md, "exp05_mind_content", simple_result(n=10))
    outs = build_final_report(md, tmp_path / "out", tmp_path / "figs", generate_figures=False)
    assert len(read_csv(outs["models"])) == 1
    manifest = json.loads(outs["manifest"].read_text(encoding="utf-8"))
    assert manifest["excluded_mismatched"][0]["source"] == "exp05_mind_content_dev.json"


def test_missing_all_selected_is_clear_error(tmp_path):
    md = tmp_path / "metrics"
    md.mkdir()
    save(md, "exp99_only", simple_result())
    with pytest.raises(FileNotFoundError, match="Không tìm thấy"):
        build_final_report(md, tmp_path / "o", tmp_path / "f", generate_figures=False)


def test_figures_created(tmp_path):
    pytest.importorskip("matplotlib")
    md = tmp_path / "metrics"
    md.mkdir()
    save(md, "exp04_mind_bpr", simple_result())
    save(md, "exp05_mind_content", simple_result())
    save(md, "exp06_mind_content_p3", p3_result())
    outs = build_final_report(md, tmp_path / "tables", tmp_path / "figures")
    assert outs["figure_accuracy"].stat().st_size > 1000
    assert outs["figure_slices"].stat().st_size > 1000
    assert outs["figure_p3_delta"].stat().st_size > 1000
