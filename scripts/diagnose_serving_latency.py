"""Read-only serving diagnostics. Writes new reports, never changes the serving bundle.

Run with .venv/Scripts/python.exe scripts/diagnose_serving_latency.py.
Timings include instrumentation overhead; compare variants in the same process.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import json
import pstats
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import psutil
import pyarrow.dataset as ds

from src.indexing.types import RetrievalQuery
from src.serving.config import load_serving_config
from src.serving.pipeline import RecommendationPipeline
from src.serving.resources import ServingResources
from src.serving.schemas import RecommendationRequest


def stats(values):
    if not values:
        return None
    return dict(
        zip(
            ("p50", "p95", "p99", "max"),
            [round(float(x), 3) for x in np.percentile(values, [50, 95, 99, 100])],
            strict=True,
        )
    )


class Recorder:
    def __init__(self):
        self.values = defaultdict(list)
        self.enabled = True

    def wrap(self, obj, name, label):
        original = getattr(obj, name)

        def measured(*args, **kwargs):
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                if self.enabled:
                    self.values[label].append((time.perf_counter() - start) * 1000)

        setattr(obj, name, measured)

    def executor(self, executor, label):
        submit = executor.submit

        def measured_submit(fn, *args, **kwargs):
            queued = time.perf_counter()

            def run():
                if self.enabled:
                    self.values[label + "_queue"].append((time.perf_counter() - queued) * 1000)
                return fn(*args, **kwargs)

            return submit(run)

        executor.submit = measured_submit


def payloads(resources, limit=512):
    rows = []
    dataset = ds.dataset("data/processed/mind_large/dev/impressions", format="parquet")
    for batch in dataset.to_batches(columns=["user_idx", "history_item_idxs"], batch_size=1024):
        for row in batch.to_pylist():
            if row["user_idx"] >= 0 and row["history_item_idxs"]:
                history = tuple(
                    i for i in row["history_item_idxs"] if i in resources.mappings.idx_to_news
                )
                if history:
                    rows.append((int(row["user_idx"]), history))
            if len(rows) >= limit:
                break
        if len(rows) >= limit:
            break
    reverse_users = {v: k for k, v in resources.mappings.user_to_idx.items()}
    requests = [
        RecommendationRequest(
            user_id=reverse_users[u],
            history_news_ids=[resources.mappings.idx_to_news[i] for i in h],
            top_k=10,
        )
        for u, h in rows
    ]
    queries = [
        RetrievalQuery(user_idx=u, history_item_idxs=h, filter_item_idxs=frozenset(h))
        for u, h in rows
    ]
    return requests, queries


def run_level(fn, concurrency, duration, recorder, label):
    # Let native work that outlived a timeout finish before the next measurement.
    time.sleep(0.5)
    recorder.values.clear()
    process = psutil.Process()
    cpu_start = process.cpu_times()
    start = time.perf_counter()
    deadline = start + duration

    def worker(worker_id):
        latencies = []
        reasons = Counter()
        count = worker_id
        while time.perf_counter() < deadline:
            t = time.perf_counter()
            result = fn(count)
            latencies.append((time.perf_counter() - t) * 1000)
            if getattr(result, "degraded", False):
                reasons[str(result.fallback_reason)] += 1
            count += concurrency
        return latencies, reasons

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        results = list(executor.map(worker, range(concurrency)))
    elapsed = time.perf_counter() - start
    cpu_end = process.cpu_times()
    latencies = [t for times, _ in results for t in times]
    reasons = sum((r for _, r in results), Counter())
    summary = {
        "variant": label,
        "concurrency": concurrency,
        "requests": len(latencies),
        "elapsed_s": round(elapsed, 3),
        "rps": round(len(latencies) / elapsed, 2),
        "latency_ms": stats(latencies),
        "fallback_reasons": dict(reasons),
        "process_cpu_percent": round(
            100 * ((cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system)) / elapsed,
            1,
        ),
        "stages_ms": {
            k: {"count": len(v), **stats(v)} for k, v in sorted(recorder.values.items()) if v
        },
    }
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=4)
    parser.add_argument("--output-dir", default="reports/benchmarks/diagnosis")
    parser.add_argument(
        "--suite", choices=["baseline", "ablation", "overhead", "fusion"], default="baseline"
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resources = ServingResources.load(load_serving_config())
    requests, queries = payloads(resources)
    print(
        "DATA",
        json.dumps(
            {
                "payloads": len(requests),
                "unique_histories": len({q.history_item_idxs for q in queries}),
                "history_lengths": stats([len(q.history_item_idxs) for q in queries]),
            }
        ),
        flush=True,
    )
    recorder = Recorder()
    for source, retriever in resources.retriever._retrievers.items():
        recorder.wrap(retriever, "retrieve", "source_" + source.value)
        if hasattr(retriever, "_index"):
            recorder.wrap(retriever, "_query_encoder", "encode_" + source.value)
            recorder.wrap(retriever._index, "search", "search_" + source.value)
    recorder.wrap(resources.retriever, "retrieve", "retrieval_total")
    recorder.executor(resources.retriever._executor, "retrieval")
    recorder.executor(resources.ranker_executor, "ranker")
    ranker = resources.ranker
    recorder.wrap(ranker, "predict", "ranker_total")
    for name in ("cf", "seq", "content"):
        recorder.wrap(getattr(ranker, name + "_model"), "predict", "rank_" + name)
    recorder.wrap(ranker.content_model, "_build_profiles", "content_profile")
    recorder.wrap(ranker.content_model, "_score_candidates", "content_scores")
    recorder.wrap(ranker.gating_network, "forward", "gating_forward")
    pipeline = RecommendationPipeline(resources)
    levels = []
    try:
        count = len(requests)
        # Warm every request, to separate steady-state capacity from first-use cost.
        for request in requests:
            pipeline.recommend(request)
        candidates = np.asarray(
            [c.item_idx for c in resources.retriever.retrieve(queries[0]).candidates[:100]]
        )
        if args.suite == "baseline":
            for c in (1, 5, 10, 20):
                levels.append(
                    run_level(
                        lambda i: pipeline.recommend(requests[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "full_pipeline",
                    )
                )
            for c in (1, 20):
                levels.append(
                    run_level(
                        lambda i: resources.retriever.retrieve(queries[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "retrieval_only",
                    )
                )
            for c in (1, 20):
                levels.append(
                    run_level(
                        lambda i: ranker.predict(
                            np.asarray([queries[i % count].user_idx]),
                            candidates,
                            history_item_idxs=list(queries[i % count].history_item_idxs),
                        ),
                        c,
                        args.duration,
                        recorder,
                        "ranker_only",
                    )
                )
        elif args.suite == "fusion":
            queries = queries[:32]
            count = len(queries)
            for retriever in resources.retriever._retrievers.values():
                source_config = resources.config.retrieval.sources[retriever.source]
                cache = {
                    (q.user_idx, q.history_item_idxs): retriever.retrieve(q, source_config.limit)
                    for q in queries
                }
                retriever.retrieve = lambda query, limit, cache=cache: cache[
                    (query.user_idx, query.history_item_idxs)
                ]
            gc.collect()
            gc.freeze()
            recorder.wrap(resources.retriever._executor, "submit", "retrieval_submit")
            for c in (1, 20):
                levels.append(
                    run_level(
                        lambda i: resources.retriever.retrieve(queries[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "cached_sources_rrf_only_frozen",
                    )
                )
            original_executor = resources.retriever._executor
            from concurrent.futures import Future

            class InlineExecutor:
                def submit(self, fn, *args, **kwargs):
                    future = Future()
                    try:
                        future.set_result(fn(*args, **kwargs))
                    except BaseException as exc:
                        future.set_exception(exc)
                    return future

                def shutdown(self, **kwargs):
                    pass

            resources.retriever._executor = InlineExecutor()
            for c in (1, 20):
                levels.append(
                    run_level(
                        lambda i: resources.retriever.retrieve(queries[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "cached_sources_inline_rrf",
                    )
                )
            resources.retriever._executor = original_executor
        elif args.suite == "overhead":
            print("GIL_SWITCH_INTERVAL_SECONDS", sys.getswitchinterval(), flush=True)
            for interval in (0.001, 0.0001):
                sys.setswitchinterval(interval)
                for c in (5, 20):
                    levels.append(
                        run_level(
                            lambda i: pipeline.recommend(requests[i % count]),
                            c,
                            args.duration,
                            recorder,
                            "gil_switch_" + str(interval),
                        )
                    )
            sys.setswitchinterval(0.005)
            content_retriever = next(
                r for s, r in resources.retriever._retrievers.items() if s.value == "content"
            )
            vectors = [content_retriever._query_encoder(q) for q in queries]
            for c in (1, 20):
                levels.append(
                    run_level(
                        lambda i: content_retriever._index.search(vectors[i % count], 300),
                        c,
                        args.duration,
                        recorder,
                        "content_faiss_only",
                    )
                )
            # Cache full source results, leaving executor dispatch + Python RRF fusion intact.
            for retriever in resources.retriever._retrievers.values():
                source_config = resources.config.retrieval.sources[retriever.source]
                cache = {
                    (q.user_idx, q.history_item_idxs): retriever.retrieve(q, source_config.limit)
                    for q in queries
                }
                retriever.retrieve = lambda query, limit, cache=cache: cache[
                    (query.user_idx, query.history_item_idxs)
                ]
            recorder.wrap(resources.retriever._executor, "submit", "retrieval_submit")
            for c in (1, 20):
                levels.append(
                    run_level(
                        lambda i: resources.retriever.retrieve(queries[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "cached_sources_rrf_only",
                    )
                )
        else:
            # Temporary, diagnostic variants only; none is written into the bundle/source.
            resources.ranker_executor.shutdown(wait=True)
            resources.ranker_executor = ThreadPoolExecutor(max_workers=1)
            recorder.executor(resources.ranker_executor, "ranker")
            for c in (5, 20):
                levels.append(
                    run_level(
                        lambda i: pipeline.recommend(requests[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "rank_workers_1",
                    )
                )
            resources.ranker_executor.shutdown(wait=True)
            resources.ranker_executor = ThreadPoolExecutor(max_workers=8)
            recorder.executor(resources.ranker_executor, "ranker")
            install_numpy_gate(ranker)
            for c in (1, 5, 20):
                levels.append(
                    run_level(
                        lambda i: pipeline.recommend(requests[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "numpy_gate",
                    )
                )
            build_profile = ranker.content_model._build_profiles
            cached_profile = lru_cache(maxsize=4096)(lambda history: build_profile(history))
            ranker.content_model._build_profiles = lambda history: cached_profile(tuple(history))
            for query in queries:
                ranker.content_model._build_profiles(query.history_item_idxs)
            for c in (1, 5, 20):
                levels.append(
                    run_level(
                        lambda i: pipeline.recommend(requests[i % count]),
                        c,
                        args.duration,
                        recorder,
                        "numpy_gate_and_warm_profile_cache",
                    )
                )
        # Python call counts and cumulative time are diagnostic, not benchmark latencies.
        recorder.enabled = False
        profiler = cProfile.Profile()
        profiler.enable()
        for request in requests[: min(count, 100)]:
            pipeline.recommend(request)
        profiler.disable()
        profiler.dump_stats(str(output / (args.suite + "_serial.pstats")))
        with (output / (args.suite + "_serial_profile.txt")).open("w", encoding="utf-8") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(
                65
            )
        (output / (args.suite + ".json")).write_text(
            json.dumps(
                {
                    "environment": {
                        "python": sys.version,
                        "cpu_count": psutil.cpu_count(),
                        "bundle": resources.manifest.model_version,
                        "note": "Instrumented direct calls, all 512 warm-user payloads warmed first; not HTTP.",
                    },
                    "levels": levels,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        resources.close()


def install_numpy_gate(ranker):
    """Exact eval-mode MLP equivalent for a CPU-only diagnostic counterfactual."""
    import torch
    from scipy.special import erf

    gate = ranker.gating_network
    original = gate.forward
    params = []
    for net in (gate.gate_cf, gate.gate_seq, gate.gate_content):
        params.append(
            tuple(
                x.detach().cpu().numpy().copy()
                for x in (net[0].weight, net[0].bias, net[3].weight, net[3].bias)
            )
        )

    def mlp(x, weights):
        w1, b1, w2, b2 = weights
        hidden = x @ w1.T + b1
        hidden = 0.5 * hidden * (1 + erf(hidden / np.float32(np.sqrt(2))))
        return hidden @ w2.T + b2

    def forward(user_repr, item_repr, branch_mask=None):
        u = user_repr.detach().cpu().numpy().reshape(1, -1)
        items = item_repr.detach().cpu().numpy()
        users = np.broadcast_to(u, items.shape)
        logits = np.concatenate(
            (
                mlp(np.concatenate((users, items), axis=1), params[0]),
                np.broadcast_to(mlp(u, params[1]), (len(items), 1)),
                mlp(items, params[2]),
            ),
            axis=1,
        )
        if branch_mask is not None:
            logits[:, ~branch_mask.cpu().numpy()] = -1e9
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        return torch.from_numpy(exps / exps.sum(axis=1, keepdims=True))

    rng = np.random.default_rng(42)
    user = torch.from_numpy(rng.normal(size=ranker.embedding_dim).astype(np.float32))
    items = torch.from_numpy(rng.normal(size=(100, ranker.embedding_dim)).astype(np.float32))
    with torch.no_grad():
        expected = original(user, items).cpu().numpy()
        actual = forward(user, items).cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    print("NUMPY_GATE_MAX_ABS_ERROR", float(np.max(np.abs(expected - actual))), flush=True)
    gate.forward = forward


if __name__ == "__main__":
    main()
