"""Đọc JSON cũ/mới, không bịa CI theo lát cắt."""

import json
from pathlib import Path

from src.evaluation.run_all import collect_reports, evaluate_configs, write_reports


def _basic(count: int = 20, ci: bool = False):
    result = {
        "total_evaluated_impressions": count,
        "slices": {
            "overall": {
                "count": count,
                "metrics": {"NDCG@10": 0.5, "Recall@10": 0.7, "MRR@10": 0.3},
            },
            "cold_user": {
                "count": 10,
                "metrics": {"NDCG@10": 0.4, "Recall@10": 0.6, "MRR@10": 0.2},
            },
        },
        "bootstrap": {
            "unit": "user",
            "num_resamples": 100,
            "metrics": {"NDCG@10": {"mean": 0.5, "ci": [0.45, 0.55]}},
        },
    }
    if ci:
        result["bootstrap"]["slices"] = {
            "cold_user": {"count": 10, "metrics": {"NDCG@10": {"mean": 0.4, "ci": [0.32, 0.48]}}}
        }
    return result


def _p3():
    return {
        "evaluation_mode": "paired_baseline_vs_p3",
        "total_evaluated_impressions": 20,
        "baseline": {"slices": _basic()["slices"], "beyond_accuracy": {}},
        "reranked": {
            "slices": _basic()["slices"],
            "beyond_accuracy": {"category_diversity@10": 0.7},
        },
        "comparison": {
            "bootstrap": {
                "unit": "user",
                "num_resamples": 100,
                "metrics": {"NDCG@10": {"baseline_ci": [0.45, 0.55], "reranked_ci": [0.46, 0.56]}},
                "slices": {
                    "cold_user": {
                        "count": 10,
                        "metrics": {
                            "NDCG@10": {
                                "baseline_ci": [0.31, 0.47],
                                "reranked_ci": [0.32, 0.48],
                                "delta_ci": [-0.02, 0.05],
                            }
                        },
                    }
                },
            }
        },
    }


def _write(path: Path, data: dict):
    path.write_text(json.dumps(data), encoding="utf-8")


def test_collect_old_new_and_p3_without_fabricating_ci(tmp_path):
    _write(tmp_path / "old_dev.json", _basic())
    _write(tmp_path / "new_dev.json", _basic(ci=True))
    _write(tmp_path / "p3_dev.json", _p3())
    models, slices, metadata = collect_reports(tmp_path)
    assert len(models) == 3
    assert metadata["reference_impressions"] == 20
    lookup = {(row["model"], row["slice"], row["metric"]): row for row in slices}
    assert lookup["old", "overall", "NDCG@10"]["ci_low"] == 0.45
    assert lookup["old", "cold_user", "NDCG@10"]["ci_status"] == "not_recorded"
    assert lookup["new", "cold_user", "NDCG@10"]["ci_low"] == 0.32
    assert lookup["p3", "cold_user", "NDCG@10"]["ci_low"] == 0.32
    assert lookup["p3", "cold_user", "NDCG@10"]["variant"] == "reranked"
    paths = write_reports(tmp_path / "tables", models, slices, metadata)
    for path in paths.values():
        assert path.exists()


def test_mismatched_sample_filtered(tmp_path):
    _write(tmp_path / "full_dev.json", _basic(count=20))
    _write(tmp_path / "sample_dev.json", _basic(count=3))
    models, _, manifest = collect_reports(tmp_path)
    assert len(models) == 1
    assert manifest["excluded_mismatched"][0]["source"] == "sample_dev.json"
    models, _, manifest = collect_reports(tmp_path, allow_mixed_counts=True)
    assert len(models) == 2


def test_missing_checkpoint_skips_without_training(tmp_path):
    import yaml

    cfg = tmp_path / "experiment.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "missing",
                "model": {"name": "popularity"},
                "paths": {
                    "artifacts_dir": str(tmp_path / "artifacts"),
                    "reports_dir": str(tmp_path / "reports"),
                },
            }
        ),
        encoding="utf-8",
    )
    messages = evaluate_configs([cfg], tmp_path / "metrics")
    assert "thiếu checkpoint" in messages[0]


def test_no_fabricated_ci_on_empty_slice(tmp_path):
    data = _basic(ci=True)
    data["slices"]["cold_user"]["count"] = 0
    _write(tmp_path / "only_dev.json", data)
    _, rows, _ = collect_reports(tmp_path)
    row = next(r for r in rows if r["slice"] == "cold_user" and r["metric"] == "NDCG@10")
    assert row["ci_status"] == "no_data"
    assert row["ci_low"] == ""
