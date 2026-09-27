"""Offline catalog-retrieval evaluation on labeled MIND impressions."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from src.indexing.types import RetrievalQuery, RetrievalSource
from src.serving.config import load_serving_config
from src.serving.resources import ServingResources


def evaluate_retrieval(
    resources: ServingResources,
    impressions_path: str | Path,
    *,
    k_values: list[int],
    sample_size: int | None = None,
) -> dict[str, object]:
    dataset = ds.dataset(str(impressions_path), format="parquet")
    columns = ["user_idx", "history_item_idxs", "candidate_item_idxs", "labels"]
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    hit_rates: dict[int, list[float]] = {k: [] for k in k_values}
    source_hits: dict[RetrievalSource, int] = defaultdict(int)
    latencies_ms: list[float] = []
    fallback_queries = 0
    evaluated = 0

    for batch in dataset.to_batches(columns=columns):
        rows = batch.to_pydict()
        for row_idx in range(batch.num_rows):
            labels = rows["labels"][row_idx]
            if labels is None:
                continue
            positives = {
                int(item_idx)
                for item_idx, label in zip(
                    rows["candidate_item_idxs"][row_idx], labels, strict=True
                )
                if label == 1 and item_idx >= 0
            }
            if not positives:
                continue
            user_idx = int(rows["user_idx"][row_idx])
            history = tuple(
                int(item_idx)
                for item_idx in (rows["history_item_idxs"][row_idx] or [])
                if item_idx >= 0
            )
            query = RetrievalQuery(
                user_idx=user_idx if user_idx >= 0 else None,
                history_item_idxs=history,
                filter_item_idxs=frozenset(history),
            )
            start = time.perf_counter()
            result = resources.retriever.retrieve(query)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)
            fallback_queries += int(result.used_popularity_fill)

            candidates = list(result.candidates)
            for k in k_values:
                retrieved = {candidate.item_idx for candidate in candidates[:k]}
                hits = positives & retrieved
                recalls[k].append(len(hits) / len(positives))
                hit_rates[k].append(float(bool(hits)))
            for candidate in candidates:
                if candidate.item_idx in positives:
                    for source in candidate.sources:
                        source_hits[source] += 1

            evaluated += 1
            if sample_size is not None and evaluated >= sample_size:
                break
        if sample_size is not None and evaluated >= sample_size:
            break

    latency_array = np.asarray(latencies_ms, dtype=float)
    return {
        "evaluated_impressions": evaluated,
        "metrics": {
            f"recall@{k}": round(float(np.mean(recalls[k])), 6) if recalls[k] else 0.0
            for k in k_values
        }
        | {
            f"hit_rate@{k}": round(float(np.mean(hit_rates[k])), 6) if hit_rates[k] else 0.0
            for k in k_values
        },
        "retrieval_latency_ms": {
            "p50": round(float(np.percentile(latency_array, 50)), 3) if evaluated else 0.0,
            "p95": round(float(np.percentile(latency_array, 95)), 3) if evaluated else 0.0,
            "p99": round(float(np.percentile(latency_array, 99)), 3) if evaluated else 0.0,
        },
        "source_positive_hits": {source.value: count for source, count in source_hits.items()},
        "popularity_fill_rate": round(fallback_queries / evaluated, 6) if evaluated else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate full-catalog candidate retrieval")
    parser.add_argument("--config", default="configs/serving/mind_production.yaml")
    parser.add_argument("--impressions", default="data/processed/mind_large/dev/impressions")
    parser.add_argument("--k", nargs="+", type=int, default=[500, 800])
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--output", default="reports/metrics/retrieval_dev.json")
    args = parser.parse_args()

    config = load_serving_config(args.config)
    resources = ServingResources.load(config)
    try:
        results = evaluate_retrieval(
            resources,
            args.impressions,
            k_values=sorted(set(args.k)),
            sample_size=args.sample_size,
        )
    finally:
        resources.close()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
