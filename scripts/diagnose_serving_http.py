"""Isolated localhost HTTP check; does not overwrite the official benchmark report."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from diagnose_serving_latency import install_numpy_gate, stats


def load_payloads():
    root = Path("artifacts/serving/current")
    table = pq.read_table(root / "mappings/users.parquet").to_pydict()
    users = dict(zip(table["user_idx"], table["user_id"], strict=True))
    table = pq.read_table(root / "mappings/items.parquet").to_pydict()
    items = dict(zip(table["item_idx"], table["news_id"], strict=True))
    dataset = ds.dataset("data/processed/mind_large/dev/impressions", format="parquet")
    payloads = []
    for batch in dataset.to_batches(columns=["user_idx", "history_item_idxs"], batch_size=1024):
        for row in batch.to_pylist():
            if row["user_idx"] >= 0 and row["history_item_idxs"]:
                payloads.append(
                    {
                        "user_id": users[row["user_idx"]],
                        "history_news_ids": [
                            items[i] for i in row["history_item_idxs"] if i in items
                        ],
                        "top_k": 10,
                    }
                )
            if len(payloads) >= 512:
                return payloads
    return payloads


async def benchmark(payloads, duration):
    levels = []
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:18081",
        timeout=10,
        trust_env=False,
        limits=httpx.Limits(max_connections=20, max_keepalive_connections=20),
    ) as client:
        # Warm all histories before measuring steady-state HTTP capacity.
        for payload in payloads:
            response = await client.post("/recommend", json=payload)
            response.raise_for_status()
        await client.get("/__diagnostic_timings")
        for concurrency in (1, 5, 10, 20):
            start = time.perf_counter()
            deadline = start + duration

            async def worker(worker_id, deadline=deadline, concurrency=concurrency):
                http_ms, pipeline_ms, outside_ms = [], [], []
                statuses, reasons = Counter(), Counter()
                count = worker_id
                while time.perf_counter() < deadline:
                    t = time.perf_counter()
                    response = await client.post("/recommend", json=payloads[count % len(payloads)])
                    http_ms.append((time.perf_counter() - t) * 1000)
                    statuses[response.status_code] += 1
                    if response.status_code == 200:
                        body = response.json()
                        pipeline_ms.append(body["latency_ms"])
                        outside_ms.append(http_ms[-1] - body["latency_ms"])
                        if body["degraded"]:
                            reasons[body["fallback_reason"]] += 1
                    count += concurrency
                return http_ms, pipeline_ms, statuses, reasons, outside_ms

            results = await asyncio.gather(*(worker(i) for i in range(concurrency)))
            elapsed = time.perf_counter() - start
            http_ms = [v for r in results for v in r[0]]
            pipeline_ms = [v for r in results for v in r[1]]
            level = {
                "concurrency": concurrency,
                "requests": len(http_ms),
                "rps": round(len(http_ms) / elapsed, 2),
                "http_ms": stats(http_ms),
                "pipeline_ms": stats(pipeline_ms),
                "outside_pipeline_ms": stats([v for r in results for v in r[4]]),
                "statuses": dict(sum((r[2] for r in results), Counter())),
                "fallback_reasons": dict(sum((r[3] for r in results), Counter())),
            }
            level["server_timings"] = (await client.get("/__diagnostic_timings")).json()
            levels.append(level)
            print(json.dumps(level), flush=True)
    return levels


def serve(variant):
    import uvicorn

    import src.serving.app as app_module
    from src.serving.app import create_app
    from src.serving.config import load_serving_config
    from src.serving.pipeline import RecommendationPipeline
    from src.serving.resources import ServingResources

    resources = ServingResources.load(load_serving_config())
    if variant == "numpy_gate":
        install_numpy_gate(resources.ranker)
    values = {"offload_queue_ms": [], "execution_ms": [], "event_loop_resume_ms": []}
    original_offload = app_module.run_in_threadpool

    async def measured_offload(fn, *args, **kwargs):
        queued = time.perf_counter()

        def execute():
            entered = time.perf_counter()
            result = fn(*args, **kwargs)
            finished = time.perf_counter()
            values["offload_queue_ms"].append((entered - queued) * 1000)
            values["execution_ms"].append((finished - entered) * 1000)
            return result, finished

        result, finished = await original_offload(execute)
        values["event_loop_resume_ms"].append((time.perf_counter() - finished) * 1000)
        return result

    app_module.run_in_threadpool = measured_offload
    api = create_app(RecommendationPipeline(resources))

    @api.get("/__diagnostic_timings")
    async def timings():
        result = {k: stats(v) for k, v in values.items()}
        for v in values.values():
            v.clear()
        return result

    try:
        uvicorn.run(api, host="127.0.0.1", port=18081, access_log=False, log_level="warning")
    finally:
        resources.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--variant", choices=["baseline", "numpy_gate"], default="baseline")
    parser.add_argument("--duration", type=float, default=8)
    args = parser.parse_args()
    if args.serve:
        serve(args.variant)
        return
    output = Path("reports/benchmarks/diagnosis")
    output.mkdir(parents=True, exist_ok=True)
    payloads = load_payloads()
    with (output / ("http_instrumented_" + args.variant + "_server.log")).open(
        "w", encoding="utf-8"
    ) as log:
        child = subprocess.Popen(
            [sys.executable, __file__, "--serve", "--variant", args.variant],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            started = time.perf_counter()
            while time.perf_counter() - started < 45:
                if child.poll() is not None:
                    raise RuntimeError("Diagnostic server exited; inspect server log")
                try:
                    response = httpx.get("http://127.0.0.1:18081/ready", timeout=1, trust_env=False)
                    if response.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.25)
            else:
                raise RuntimeError("Diagnostic server readiness timed out")
            levels = asyncio.run(benchmark(payloads, args.duration))
            (output / ("http_instrumented_" + args.variant + ".json")).write_text(
                json.dumps(
                    {
                        "variant": args.variant,
                        "server_pid": child.pid,
                        "payload_count": len(payloads),
                        "duration_per_level_s": args.duration,
                        "note": "Separate localhost server/client; warmed histories; no access logging.",
                        "levels": levels,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        finally:
            child.terminate()
            child.wait(timeout=15)


if __name__ == "__main__":
    main()
