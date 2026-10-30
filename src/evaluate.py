import argparse
import json
import time
from array import array
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.dataset as ds
import yaml

from src.common.logger import setup_logger
from src.evaluation.beyond_accuracy import catalog_coverage, gini_index
from src.evaluation.bootstrap import BootstrapSettings, bootstrap_confidence_intervals
from src.evaluation.item_slices import (
    build_item_popularity_groups,
    split_relevant_items_by_catalog_status,
    split_relevant_items_by_popularity,
)
from src.evaluation.p3_evaluation import evaluate_p3_impressions
from src.evaluation.ranking_metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.models.base import BaseRecommender
from src.reranking.pipeline import RerankingPipeline, load_train_click_counts


def evaluate_impressions(
    model: BaseRecommender,
    impressions_path: Path,
    k_list: list[int] | None = None,
    sample_size: int | None = None,
    all_catalog_items: set[int] | None = None,
    reranking_pipeline: RerankingPipeline | None = None,
    train_click_counts: np.ndarray | None = None,
    head_fraction: float = 0.20,
    bootstrap_settings: BootstrapSettings | None = None,
    use_impression_history: bool = True,
) -> dict[str, Any]:
    """Đánh giá một model với slicing đầy đủ và bootstrap confidence interval."""
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

    base_slices = ("overall", "cold_item", "warm_item", "cold_user", "warm_user")
    item_slices = (
        "head_item",
        "long_tail_item",
        "zero_click_item",
        "catalog_cold_item",
        "catalog_warm_item",
    )
    slices = base_slices + item_slices
    slice_metrics: dict[str, dict[str, list[float]]] = {
        slice_name: {
            f"{metric_name}@{k}": []
            for metric_name in ("NDCG", "Recall", "HitRate", "MRR")
            for k in k_list
        }
        for slice_name in slices
    }

    popularity_groups = None
    if train_click_counts is not None:
        popularity_groups = build_item_popularity_groups(
            train_click_counts,
            head_fraction=head_fraction,
        )

    recommendations_at_10: dict[int, list[int]] = {}
    key_k = 10 if 10 in k_list else max(k_list)
    key_metric = f"NDCG@{key_k}"

    settings = bootstrap_settings or BootstrapSettings(enabled=False)
    bootstrap_metric_names = [f"NDCG@{key_k}", f"Recall@{key_k}", f"MRR@{key_k}"]
    max_records = dataset.count_rows()
    if sample_size is not None:
        max_records = min(max_records, sample_size)
    if settings.enabled:
        bootstrap_user_ids = np.empty(max_records, dtype=np.int64)
        bootstrap_values = np.empty(
            (max_records, len(bootstrap_metric_names)),
            dtype=np.float32,
        )
    else:
        bootstrap_user_ids = None
        bootstrap_values = None

    # Mỗi slice có tập impression và user IDs riêng để Bootstrap theo đúng
    # định nghĩa slice; không thể tái tạo CI từ các mean JSON đã làm tròn.
    # Mảng nhị phân tránh giữ hàng triệu Python int/float trong RAM.
    slice_bootstrap_users = {name: array("q") for name in slices} if settings.enabled else {}
    slice_bootstrap_values = {name: array("f") for name in slices} if settings.enabled else {}

    total_evaluated = 0
    start_time = time.perf_counter()

    for batch in dataset.to_batches(columns=required_cols):
        pydict = batch.to_pydict()
        num_rows = len(pydict["impression_id"])

        for i in range(num_rows):
            labels = pydict["labels"][i]
            if labels is None or sum(labels) == 0:
                continue

            candidates = [int(item) for item in pydict["candidate_item_idxs"][i]]
            user_idx = int(pydict["user_idx"][i])
            history = pydict["history_item_idxs"][i] or []
            candidate_train_mask = pydict["candidate_seen_in_train_mask"][i]

            model_history = history if use_impression_history else None
            scores = np.asarray(
                model.predict(
                    np.array([user_idx]),
                    np.array(candidates),
                    history_item_idxs=model_history,
                ),
                dtype=np.float64,
            ).reshape(-1)
            if scores.size != len(candidates):
                raise ValueError("Mô hình trả về số score không khớp số candidate.")

            if reranking_pipeline is None:
                ranked_indices = np.argsort(-scores, kind="stable")
                ranked_candidates = [candidates[idx] for idx in ranked_indices]
            else:
                ranked_candidates = reranking_pipeline.rerank(
                    candidates,
                    scores,
                    top_k=max(k_list),
                )

            true_items = {candidates[idx] for idx, label in enumerate(labels) if int(label) == 1}

            impression_id = int(pydict["impression_id"][i])
            recommendations_at_10[impression_id] = ranked_candidates[:10]

            impression_metrics: dict[str, float] = {}
            for k in k_list:
                impression_metrics[f"NDCG@{k}"] = ndcg_at_k(ranked_candidates, true_items, k)
                impression_metrics[f"Recall@{k}"] = recall_at_k(ranked_candidates, true_items, k)
                impression_metrics[f"HitRate@{k}"] = hit_rate_at_k(ranked_candidates, true_items, k)
                impression_metrics[f"MRR@{k}"] = mrr_at_k(ranked_candidates, true_items, k)

            # Giữ nguyên convention chung của project cho cold/warm impression.
            has_cold_candidate = False in candidate_train_mask
            is_cold_user = len(history) <= 5 or user_idx == -1
            active_base_slices = ["overall"]
            active_base_slices.append("cold_item" if has_cold_candidate else "warm_item")
            active_base_slices.append("cold_user" if is_cold_user else "warm_user")

            # User không có mapping (-1) không được xem là một cluster chung.
            bootstrap_user = user_idx if user_idx >= 0 else -(total_evaluated + 1)
            for slice_name in active_base_slices:
                for metric_name, metric_value in impression_metrics.items():
                    slice_metrics[slice_name][metric_name].append(float(metric_value))
                if settings.enabled and slice_name != "overall":
                    slice_bootstrap_users[slice_name].append(bootstrap_user)
                    slice_bootstrap_values[slice_name].extend(
                        [impression_metrics[name] for name in bootstrap_metric_names]
                    )

            # Item popularity slices dùng chính relevant/clicked items.
            item_slice_truth: dict[str, set[int]] = {}
            if popularity_groups is not None:
                item_slice_truth.update(
                    split_relevant_items_by_popularity(
                        true_items,
                        popularity_groups.group_codes,
                    )
                )
            item_slice_truth.update(
                split_relevant_items_by_catalog_status(
                    candidates,
                    labels,
                    candidate_train_mask,
                )
            )

            for slice_name, slice_true_items in item_slice_truth.items():
                if not slice_true_items:
                    continue
                item_metrics: dict[str, float] = {}
                for k in k_list:
                    item_metrics[f"NDCG@{k}"] = ndcg_at_k(ranked_candidates, slice_true_items, k)
                    item_metrics[f"Recall@{k}"] = recall_at_k(
                        ranked_candidates, slice_true_items, k
                    )
                    item_metrics[f"HitRate@{k}"] = hit_rate_at_k(
                        ranked_candidates, slice_true_items, k
                    )
                    item_metrics[f"MRR@{k}"] = mrr_at_k(ranked_candidates, slice_true_items, k)
                for metric_name, value in item_metrics.items():
                    slice_metrics[slice_name][metric_name].append(value)
                if settings.enabled and slice_name != "overall":
                    slice_bootstrap_users[slice_name].append(bootstrap_user)
                    slice_bootstrap_values[slice_name].extend(
                        [item_metrics[name] for name in bootstrap_metric_names]
                    )

            if settings.enabled and bootstrap_user_ids is not None and bootstrap_values is not None:
                # Unknown user (-1) được tách thành cluster riêng cho từng impression.
                bootstrap_user_ids[total_evaluated] = bootstrap_user
                bootstrap_values[total_evaluated] = [
                    impression_metrics[f"NDCG@{key_k}"],
                    impression_metrics[f"Recall@{key_k}"],
                    impression_metrics[f"MRR@{key_k}"],
                ]

            total_evaluated += 1
            if sample_size and total_evaluated >= sample_size:
                break

        if sample_size and total_evaluated >= sample_size:
            break

    elapsed = time.perf_counter() - start_time
    throughput = total_evaluated / elapsed if elapsed > 0 else 0.0

    results: dict[str, Any] = {
        "total_evaluated_impressions": total_evaluated,
        "eval_time_seconds": round(elapsed, 2),
        "throughput_impressions_per_sec": round(throughput, 1),
        "slices": {},
    }

    for slice_name in slices:
        metrics = slice_metrics[slice_name]
        count = len(metrics[key_metric])
        results["slices"][slice_name] = {
            "count": count,
            "pct": round(100.0 * count / (total_evaluated or 1), 2),
            "metrics": {
                metric_name: round(float(np.mean(values)), 5) if values else 0.0
                for metric_name, values in metrics.items()
            },
        }

    if popularity_groups is not None:
        results["item_popularity_definition"] = {
            "head_fraction_of_clicked_items": head_fraction,
            "head_items": popularity_groups.head_items,
            "long_tail_items": popularity_groups.long_tail_items,
            "zero_click_items": popularity_groups.zero_click_items,
        }

    if all_catalog_items:
        cov_10 = catalog_coverage(recommendations_at_10, all_catalog_items, k=10)
        gini_10 = gini_index(recommendations_at_10, all_catalog_items, k=10)
        results["beyond_accuracy"] = {
            "catalog_coverage@10": round(cov_10, 5),
            "gini_index@10": round(gini_10, 5),
            "total_catalog_items": len(all_catalog_items),
        }

    if (
        settings.enabled
        and total_evaluated > 0
        and bootstrap_user_ids is not None
        and bootstrap_values is not None
    ):
        results["bootstrap"] = {
            "unit": settings.unit,
            "num_resamples": settings.num_resamples,
            "confidence_level": settings.confidence_level,
            "metrics": bootstrap_confidence_intervals(
                user_ids=bootstrap_user_ids[:total_evaluated],
                values=bootstrap_values[:total_evaluated],
                metric_names=bootstrap_metric_names,
                settings=settings,
            ),
        }
        results["bootstrap"]["slices"] = {
            name: {
                "count": total_evaluated if name == "overall" else len(slice_bootstrap_users[name]),
                "metrics": results["bootstrap"]["metrics"]
                if name == "overall"
                else bootstrap_confidence_intervals(
                    user_ids=np.frombuffer(slice_bootstrap_users[name], dtype=np.int64),
                    values=np.frombuffer(slice_bootstrap_values[name], dtype=np.float32).reshape(
                        -1, len(bootstrap_metric_names)
                    ),
                    metric_names=bootstrap_metric_names,
                    settings=settings,
                )
                if slice_bootstrap_users[name]
                else {},
            }
            for name in slices
        }

    return results


def format_markdown_table(results: dict[str, Any]) -> str:
    """Tạo report cho model thường, gồm slicing và bootstrap CI."""
    slices = results.get("slices", {})
    metrics_to_show = ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@10", "Recall@20", "MRR@10"]

    headers = ["Metric", "Overall", "Cold Item", "Warm Item", "Cold User", "Warm User"]
    rows = [
        [
            "**Impression Count**",
            f"**{slices.get('overall', {}).get('count', 0):,}**",
            f"{slices.get('cold_item', {}).get('count', 0):,} ({slices.get('cold_item', {}).get('pct', 0)}%)",
            f"{slices.get('warm_item', {}).get('count', 0):,} ({slices.get('warm_item', {}).get('pct', 0)}%)",
            f"{slices.get('cold_user', {}).get('count', 0):,} ({slices.get('cold_user', {}).get('pct', 0)}%)",
            f"{slices.get('warm_user', {}).get('count', 0):,} ({slices.get('warm_user', {}).get('pct', 0)}%)",
        ]
    ]

    for metric_name in metrics_to_show:
        row = [f"**{metric_name}**"]
        for slice_name in ("overall", "cold_item", "warm_item", "cold_user", "warm_user"):
            value = slices.get(slice_name, {}).get("metrics", {}).get(metric_name, 0.0)
            row.append(f"{value:.4f}")
        rows.append(row)

    beyond = results.get("beyond_accuracy", {})
    if beyond:
        rows.append(["---"] * len(headers))
        rows.append(
            [
                "**Catalog Coverage@10**",
                f"{beyond.get('catalog_coverage@10', 0.0):.4f}",
                "-",
                "-",
                "-",
                "-",
            ]
        )
        rows.append(
            ["**Gini Index@10**", f"{beyond.get('gini_index@10', 0.0):.4f}", "-", "-", "-", "-"]
        )

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        if row[0] == "---":
            lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        else:
            lines.append("| " + " | ".join(row) + " |")

    item_slice_order = [
        "head_item",
        "long_tail_item",
        "zero_click_item",
        "catalog_cold_item",
        "catalog_warm_item",
    ]
    if any(slices.get(name, {}).get("count", 0) > 0 for name in item_slice_order):
        lines.extend(
            [
                "",
                "### NDCG@10 theo item slice",
                "| Slice | Count | NDCG@10 | Recall@10 | MRR@10 |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
        )
        for slice_name in item_slice_order:
            info = slices.get(slice_name, {})
            metrics = info.get("metrics", {})
            lines.append(
                f"| {slice_name} | {int(info.get('count', 0)):,} | "
                f"{float(metrics.get('NDCG@10', 0.0)):.5f} | "
                f"{float(metrics.get('Recall@10', 0.0)):.5f} | "
                f"{float(metrics.get('MRR@10', 0.0)):.5f} |"
            )

    bootstrap = results.get("bootstrap", {})
    bootstrap_metrics = bootstrap.get("metrics", {})
    if bootstrap_metrics:
        confidence_pct = 100.0 * float(bootstrap.get("confidence_level", 0.95))
        lines.extend(
            [
                "",
                f"Bootstrap: unit={bootstrap.get('unit')}, resamples={bootstrap.get('num_resamples')}, confidence={bootstrap.get('confidence_level')}",
                f"| Metric | Mean | {confidence_pct:.0f}% CI |",
                "| :--- | ---: | :---: |",
            ]
        )
        for metric_name, metric_info in bootstrap_metrics.items():
            ci = metric_info.get("ci", [0.0, 0.0])
            lines.append(
                f"| {metric_name} | {float(metric_info.get('mean', 0.0)):.6f} | "
                f"[{float(ci[0]):.6f}, {float(ci[1]):.6f}] |"
            )

    return "\n".join(lines)


def format_p3_markdown_table(results: dict[str, Any]) -> str:
    """Tạo bảng trước/sau P3 và bảng slicing chính."""
    baseline = results.get("baseline", {})
    reranked = results.get("reranked", {})
    comparison = results.get("comparison", {})

    baseline_overall = baseline.get("slices", {}).get("overall", {}).get("metrics", {})
    reranked_overall = reranked.get("slices", {}).get("overall", {}).get("metrics", {})
    delta = comparison.get("overall_delta", {})

    metrics_to_show = [
        "NDCG@10",
        "Recall@10",
        "MRR@10",
    ]
    beyond_metrics = [
        "category_diversity@10",
        "long_tail_exposure@10",
        "zero_click_exposure@10",
        "catalog_coverage@10",
        "gini_index@10",
    ]

    lines = [
        "| Metric | Baseline | P3 | Delta |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for metric_name in metrics_to_show:
        before = float(baseline_overall.get(metric_name, 0.0))
        after = float(reranked_overall.get(metric_name, 0.0))
        change = float(delta.get(metric_name, after - before))
        lines.append(f"| **{metric_name}** | {before:.5f} | {after:.5f} | {change:+.5f} |")

    baseline_beyond = baseline.get("beyond_accuracy", {})
    reranked_beyond = reranked.get("beyond_accuracy", {})
    for metric_name in beyond_metrics:
        before = float(baseline_beyond.get(metric_name, 0.0))
        after = float(reranked_beyond.get(metric_name, 0.0))
        change = float(delta.get(metric_name, after - before))
        lines.append(f"| **{metric_name}** | {before:.5f} | {after:.5f} | {change:+.5f} |")

    lines.append("")
    lines.append("### NDCG@10 theo slice")
    lines.append("| Slice | Count | Baseline | P3 | Delta |")
    lines.append("| :--- | ---: | ---: | ---: | ---: |")

    slice_order = [
        "overall",
        "cold_item",
        "warm_item",
        "head_item",
        "long_tail_item",
        "zero_click_item",
        "catalog_cold_item",
        "catalog_warm_item",
        "cold_user",
        "warm_user",
    ]
    baseline_slices = baseline.get("slices", {})
    reranked_slices = reranked.get("slices", {})
    for slice_name in slice_order:
        before_info = baseline_slices.get(slice_name, {})
        after_info = reranked_slices.get(slice_name, {})
        count = int(before_info.get("count", 0))
        before = float(before_info.get("metrics", {}).get("NDCG@10", 0.0))
        after = float(after_info.get("metrics", {}).get("NDCG@10", 0.0))
        lines.append(
            f"| {slice_name} | {count:,} | {before:.5f} | {after:.5f} | {after - before:+.5f} |"
        )

    bootstrap = comparison.get("bootstrap", {})
    bootstrap_metrics = bootstrap.get("metrics", {})
    if bootstrap_metrics:
        lines.append("")
        lines.append(
            f"Bootstrap: unit={bootstrap.get('unit')}, "
            f"resamples={bootstrap.get('num_resamples')}, "
            f"confidence={bootstrap.get('confidence_level')}"
        )
        ndcg_bootstrap = bootstrap_metrics.get("NDCG@10")
        if ndcg_bootstrap:
            ci = ndcg_bootstrap.get("delta_ci", [0.0, 0.0])
            lines.append(
                f"Paired delta NDCG@10 = {ndcg_bootstrap.get('delta', 0.0):+.6f} "
                f"CI [{ci[0]:+.6f}, {ci[1]:+.6f}]"
            )

    return "\n".join(lines)


def evaluate(config_path: str | Path, sample_size: int | None = None) -> dict[str, Any]:
    """Run full evaluation pipeline."""
    logger = setup_logger("evaluate")
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "unnamed_run")
    model_name = cfg.get("model", {}).get("name", "popularity")
    eval_cfg = cfg.get("evaluation", {})
    k_list = eval_cfg.get("top_k_list", [5, 10, 20])

    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")) / "checkpoints"
    checkpoint_path = artifacts_dir / f"{exp_name}_{model_name}.pkl"
    rerank_cfg = cfg.get("reranking", {})
    reranking_pipeline = None
    if rerank_cfg.get("enabled", False):
        # Dùng checkpoint đã huấn luyện của mô hình nền, không train lại theo tên P3.
        base_checkpoint = rerank_cfg.get("base_checkpoint")
        if not base_checkpoint:
            raise ValueError("P3 yêu cầu reranking.base_checkpoint trong YAML.")
        checkpoint_path = Path(base_checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint P3: {checkpoint_path}")
        reranking_pipeline = RerankingPipeline.from_config(cfg)

    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Training model on the fly...")
        from src.train import train

        checkpoint_path = train(config_path)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    model = BaseRecommender.load(checkpoint_path)

    dataset_cfg = cfg.get("dataset", {})
    dev_path = Path(dataset_cfg.get("dev_impressions", "data/processed/mind_large/dev/impressions"))
    items_path = Path(dataset_cfg.get("items_mapping", "data/processed/mind_large/mappings/items"))

    catalog_items: set[int] = set()
    if items_path.exists():
        logger.info(f"Loading item catalog for beyond-accuracy metrics from: {items_path}")
        items_ds = ds.dataset(str(items_path), format="parquet")
        catalog_items = set(items_ds.to_table(columns=["item_idx"])["item_idx"].to_pylist())
        logger.info(f"Loaded {len(catalog_items):,} total catalog items.")

    slicing_cfg = eval_cfg.get("item_slicing", {})
    head_fraction = float(slicing_cfg.get("head_fraction", 0.20))
    use_impression_history = bool(eval_cfg.get("use_impression_history", True))
    bootstrap_cfg = eval_cfg.get("bootstrap", {})
    bootstrap_settings = BootstrapSettings(
        enabled=bool(bootstrap_cfg.get("enabled", True)),
        num_resamples=int(bootstrap_cfg.get("num_resamples", 1000)),
        confidence_level=float(bootstrap_cfg.get("confidence_level", 0.95)),
        unit=str(bootstrap_cfg.get("unit", "user")),
        seed=int(bootstrap_cfg.get("seed", cfg.get("seed", 42))),
        batch_size=int(bootstrap_cfg.get("batch_size", 32)),
    )

    logger.info(f"Starting evaluation on {dev_path} (top_k={k_list}, sample_size={sample_size})...")
    if reranking_pipeline is None:
        train_click_counts = None
        train_path_value = dataset_cfg.get("train_interactions")
        if train_path_value and catalog_items:
            logger.info(
                f"Loading train click counts for Head/Long-tail slices from: {train_path_value}"
            )
            train_click_counts = load_train_click_counts(
                train_path_value,
                len(catalog_items),
            )

        results = evaluate_impressions(
            model=model,
            impressions_path=dev_path,
            k_list=k_list,
            sample_size=sample_size,
            all_catalog_items=catalog_items,
            train_click_counts=train_click_counts,
            head_fraction=head_fraction,
            bootstrap_settings=bootstrap_settings,
            use_impression_history=use_impression_history,
        )
        report_md = format_markdown_table(results)
    else:
        results = evaluate_p3_impressions(
            model=model,
            impressions_path=dev_path,
            reranking_pipeline=reranking_pipeline,
            k_list=k_list,
            sample_size=sample_size,
            all_catalog_items=catalog_items,
            head_fraction=head_fraction,
            bootstrap_settings=bootstrap_settings,
        )
        report_md = format_p3_markdown_table(results)
    logger.info(
        "\n"
        + "=" * 60
        + "\nEVALUATION RESULTS REPORT:\n"
        + "=" * 60
        + "\n"
        + report_md
        + "\n"
        + "=" * 60
    )

    reports_dir = Path(cfg.get("paths", {}).get("reports_dir", "reports")) / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_json_path = reports_dir / f"{exp_name}_dev.json"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation metrics JSON saved to: {report_json_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="RecSys Offline Evaluation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/exp01_mind_popularity.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional max number of impressions to evaluate (useful for smoke test)",
    )
    args = parser.parse_args()
    evaluate(args.config, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
