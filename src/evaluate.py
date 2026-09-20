import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Set
import numpy as np
import pyarrow.dataset as ds
import yaml

from src.common.logger import setup_logger
from src.evaluation.beyond_accuracy import catalog_coverage, gini_index
from src.evaluation.ranking_metrics import hit_rate_at_k, mrr_at_k, ndcg_at_k, recall_at_k
from src.models.base import BaseRecommender


def evaluate_impressions(
    model: BaseRecommender,
    impressions_path: Path,
    k_list: List[int] = [5, 10, 20],
    sample_size: int | None = None,
    all_catalog_items: Set[int] | None = None,
) -> Dict[str, Any]:
    """Evaluate recommender model on impression dataset across multiple subgroups/slices."""
    dataset = ds.dataset(str(impressions_path), format="parquet")
    required_cols = [
        "impression_id",
        "user_idx",
        "history_item_idxs",
        "candidate_item_idxs",
        "labels",
        "candidate_seen_in_train_mask",
    ]

    slices = ("overall", "cold_item", "warm_item", "cold_user", "warm_user")
    slice_metrics: Dict[str, Dict[str, List[float]]] = {
        s: {f"{m}@{k}": [] for m in ("NDCG", "Recall", "HitRate", "MRR") for k in k_list}
        for s in slices
    }

    recommendations_at_10: Dict[int, List[int]] = {}
    total_evaluated = 0
    start_time = time.perf_counter()

    for batch in dataset.to_batches(columns=required_cols):
        pydict = batch.to_pydict()
        num_rows = len(pydict["impression_id"])

        for i in range(num_rows):
            labels = pydict["labels"][i]
            if labels is None or sum(labels) == 0:
                continue

            candidates = pydict["candidate_item_idxs"][i]
            user_idx = pydict["user_idx"][i]
            history = pydict["history_item_idxs"][i] or []
            candidate_train_mask = pydict["candidate_seen_in_train_mask"][i]

            # Pointwise scoring and ranking (passes user history for personalized models)
            scores = model.predict(
                np.array([user_idx]), np.array(candidates), history_item_idxs=history
            )
            # Sort descending by score
            ranked_indices = np.argsort(-scores)
            ranked_candidates = [candidates[idx] for idx in ranked_indices]
            true_items = {candidates[idx] for idx, l in enumerate(labels) if l == 1}

            imp_id = pydict["impression_id"][i]
            recommendations_at_10[imp_id] = ranked_candidates[:10]

            # Compute metrics for this impression
            imp_metrics = {}
            for k in k_list:
                imp_metrics[f"NDCG@{k}"] = ndcg_at_k(ranked_candidates, true_items, k)
                imp_metrics[f"Recall@{k}"] = recall_at_k(ranked_candidates, true_items, k)
                imp_metrics[f"HitRate@{k}"] = hit_rate_at_k(ranked_candidates, true_items, k)
                imp_metrics[f"MRR@{k}"] = mrr_at_k(ranked_candidates, true_items, k)

            # Determine slice membership
            has_cold_candidate = False in candidate_train_mask
            is_cold_user = len(history) <= 5 or user_idx == -1

            active_slices = ["overall"]
            if has_cold_candidate:
                active_slices.append("cold_item")
            else:
                active_slices.append("warm_item")

            if is_cold_user:
                active_slices.append("cold_user")
            else:
                active_slices.append("warm_user")

            for s in active_slices:
                for metric_name, val in imp_metrics.items():
                    slice_metrics[s][metric_name].append(val)

            total_evaluated += 1
            if sample_size and total_evaluated >= sample_size:
                break

        if sample_size and total_evaluated >= sample_size:
            break

    elapsed = time.perf_counter() - start_time
    throughput = total_evaluated / elapsed if elapsed > 0 else 0.0

    # Aggregate means
    results: Dict[str, Any] = {
        "total_evaluated_impressions": total_evaluated,
        "eval_time_seconds": round(elapsed, 2),
        "throughput_impressions_per_sec": round(throughput, 1),
        "slices": {},
    }

    for s in slices:
        count = len(slice_metrics[s]["NDCG@10"])
        results["slices"][s] = {
            "count": count,
            "pct": round(100.0 * count / (total_evaluated or 1), 2),
            "metrics": {
                m: round(float(np.mean(vals)), 5) if vals else 0.0
                for m, vals in slice_metrics[s].items()
            },
        }

    # Beyond-accuracy metrics (if catalog available)
    if all_catalog_items:
        cov_10 = catalog_coverage(recommendations_at_10, all_catalog_items, k=10)
        gini_10 = gini_index(recommendations_at_10, all_catalog_items, k=10)
        results["beyond_accuracy"] = {
            "catalog_coverage@10": round(cov_10, 5),
            "gini_index@10": round(gini_10, 5),
            "total_catalog_items": len(all_catalog_items),
        }

    return results


def format_markdown_table(results: Dict[str, Any]) -> str:
    """Format evaluation results into a Markdown report table matching capstone guidelines."""
    slices = results.get("slices", {})
    metrics_to_show = ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@10", "Recall@20", "MRR@10"]

    headers = ["Metric", "Overall", "Cold Item", "Warm Item", "Cold User", "Warm User"]
    rows = []

    # Impression counts row
    count_row = [
        "**Impression Count**",
        f"**{slices.get('overall', {}).get('count', 0):,}**",
        f"{slices.get('cold_item', {}).get('count', 0):,} ({slices.get('cold_item', {}).get('pct', 0)}%)",
        f"{slices.get('warm_item', {}).get('count', 0):,} ({slices.get('warm_item', {}).get('pct', 0)}%)",
        f"{slices.get('cold_user', {}).get('count', 0):,} ({slices.get('cold_user', {}).get('pct', 0)}%)",
        f"{slices.get('warm_user', {}).get('count', 0):,} ({slices.get('warm_user', {}).get('pct', 0)}%)",
    ]
    rows.append(count_row)

    for m in metrics_to_show:
        row = [f"**{m}**"]
        for s in ("overall", "cold_item", "warm_item", "cold_user", "warm_user"):
            val = slices.get(s, {}).get("metrics", {}).get(m, 0.0)
            row.append(f"{val:.4f}")
        rows.append(row)

    # Beyond accuracy if present
    beyond = results.get("beyond_accuracy", {})
    if beyond:
        rows.append(["---"] * len(headers))
        rows.append([
            "**Catalog Coverage@10**",
            f"{beyond.get('catalog_coverage@10', 0.0):.4f}",
            "-", "-", "-", "-",
        ])
        rows.append([
            "**Gini Index@10**",
            f"{beyond.get('gini_index@10', 0.0):.4f}",
            "-", "-", "-", "-",
        ])

    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join([":---"] + [":---:" for _ in headers[1:]]) + " |",
    ]
    for r in rows:
        if r[0] == "---":
            table_lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        else:
            table_lines.append("| " + " | ".join(r) + " |")

    return "\n".join(table_lines)


def evaluate(config_path: str | Path, sample_size: int | None = None) -> Dict[str, Any]:
    """Run full evaluation pipeline."""
    logger = setup_logger("evaluate")
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment_name", "unnamed_run")
    model_name = cfg.get("model", {}).get("name", "popularity")
    eval_cfg = cfg.get("evaluation", {})
    k_list = eval_cfg.get("top_k_list", [5, 10, 20])

    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")) / "checkpoints"
    checkpoint_path = artifacts_dir / f"{exp_name}_{model_name}.pkl"

    if not checkpoint_path.exists():
        logger.warning(
            f"Checkpoint not found at {checkpoint_path}. Training model on the fly..."
        )
        from src.train import train
        checkpoint_path = train(config_path)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    model = BaseRecommender.load(checkpoint_path)

    dataset_cfg = cfg.get("dataset", {})
    dev_path = Path(dataset_cfg.get("dev_impressions", "data/processed/mind_large/dev/impressions"))
    items_path = Path(dataset_cfg.get("items_mapping", "data/processed/mind_large/mappings/items"))

    catalog_items: Set[int] = set()
    if items_path.exists():
        logger.info(f"Loading item catalog for beyond-accuracy metrics from: {items_path}")
        items_ds = ds.dataset(str(items_path), format="parquet")
        catalog_items = set(items_ds.to_table(columns=["item_idx"])["item_idx"].to_pylist())
        logger.info(f"Loaded {len(catalog_items):,} total catalog items.")

    logger.info(f"Starting evaluation on {dev_path} (top_k={k_list}, sample_size={sample_size})...")
    results = evaluate_impressions(
        model=model,
        impressions_path=dev_path,
        k_list=k_list,
        sample_size=sample_size,
        all_catalog_items=catalog_items,
    )

    report_md = format_markdown_table(results)
    logger.info("\n" + "=" * 60 + "\nEVALUATION RESULTS REPORT:\n" + "=" * 60 + "\n" + report_md + "\n" + "=" * 60)

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
