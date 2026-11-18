"""HTTP load benchmark with latency, throughput, and resource measurements."""

import argparse
import asyncio
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import psutil
import yaml


def _gpu_memory_mb() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
        values = [float(value.strip()) for value in result.stdout.splitlines() if value.strip()]
        return sum(values) if values else 0.0
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


async def _sample_resources(
    process: psutil.Process,
    stop: asyncio.Event,
    samples: list[dict[str, float | None]],
) -> None:
    process.cpu_percent(None)
    while not stop.is_set():
        try:
            samples.append(
                {
                    "rss_mb": process.memory_info().rss / (1024 * 1024),
                    "cpu_percent": process.cpu_percent(None),
                    "gpu_memory_mb": await asyncio.to_thread(_gpu_memory_mb),
                }
            )
        except psutil.Error:
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
        except TimeoutError:
            pass


async def _run_level(
    *,
    client: httpx.AsyncClient,
    endpoint: str,
    payloads: list[dict[str, Any]],
    concurrency: int,
    duration_seconds: float,
    process: psutil.Process | None,
) -> dict[str, object]:
    latencies_ms: list[float] = []
    statuses: dict[int, int] = {}
    fallback_reasons: dict[str, int] = {}
    stop_resources = asyncio.Event()
    resource_samples: list[dict[str, float | None]] = []
    sampler = (
        asyncio.create_task(_sample_resources(process, stop_resources, resource_samples))
        if process is not None
        else None
    )
    deadline = time.perf_counter() + duration_seconds

    async def worker(worker_id: int) -> None:
        request_number = worker_id
        while time.perf_counter() < deadline:
            payload = payloads[request_number % len(payloads)]
            request_number += concurrency
            started = time.perf_counter()
            try:
                response = await client.post(endpoint, json=payload)
                status = response.status_code
                if 200 <= status < 300:
                    body = response.json()
                    if body.get("degraded"):
                        reason = str(body.get("fallback_reason"))
                        fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
            except httpx.HTTPError:
                status = 0
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            statuses[status] = statuses.get(status, 0) + 1

    started = time.perf_counter()
    await asyncio.gather(*(worker(worker_id) for worker_id in range(concurrency)))
    elapsed = time.perf_counter() - started
    stop_resources.set()
    if sampler is not None:
        await sampler

    values = np.asarray(latencies_ms, dtype=float)
    successes = sum(count for status, count in statuses.items() if 200 <= status < 300)
    rss = [float(sample["rss_mb"]) for sample in resource_samples if sample["rss_mb"] is not None]
    cpu = [
        float(sample["cpu_percent"])
        for sample in resource_samples
        if sample["cpu_percent"] is not None
    ]
    gpu = [
        float(sample["gpu_memory_mb"])
        for sample in resource_samples
        if sample["gpu_memory_mb"] is not None
    ]
    return {
        "concurrency": concurrency,
        "requests": len(latencies_ms),
        "successful_requests": successes,
        "error_rate": round(1.0 - successes / max(1, len(latencies_ms)), 6),
        # COLD_USER is expected for unknown users; the other reasons mean a broken budget.
        "fallback_rate": round(sum(fallback_reasons.values()) / max(1, successes), 6),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "rps": round(successes / elapsed, 3),
        "latency_ms": {
            "p50": round(float(np.percentile(values, 50)), 3),
            "p95": round(float(np.percentile(values, 95)), 3),
            "p99": round(float(np.percentile(values, 99)), 3),
        },
        "resources": {
            "rss_mb_peak": round(max(rss), 3) if rss else None,
            "cpu_percent_mean": round(float(np.mean(cpu)), 3) if cpu else None,
            "gpu_memory_mb_peak": round(max(gpu), 3) if gpu else None,
        },
        "status_counts": {str(status): count for status, count in sorted(statuses.items())},
    }


def _load_payloads(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [{"user_id": None, "top_k": 10}]
    payloads = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not payloads:
        raise ValueError("payload file must contain at least one JSON request")
    return payloads


async def run_benchmark(config: dict[str, Any]) -> dict[str, object]:
    base_url = config["base_url"].rstrip("/")
    endpoint = f"{base_url}/recommend"
    payloads = _load_payloads(config.get("payloads_file"))
    server_pid = config.get("server_pid")
    process = psutil.Process(int(server_pid)) if server_pid is not None else None
    timeout = httpx.Timeout(float(config.get("request_timeout_seconds", 5)))
    max_concurrency = max(config["concurrency"])
    limits = httpx.Limits(
        max_connections=max_concurrency,
        max_keepalive_connections=max_concurrency,
    )
    results: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for _ in range(int(config.get("warmup_requests", 20))):
            await client.post(endpoint, json=payloads[0])
        for concurrency in config["concurrency"]:
            results.append(
                await _run_level(
                    client=client,
                    endpoint=endpoint,
                    payloads=payloads,
                    concurrency=int(concurrency),
                    duration_seconds=float(config["duration_seconds"]),
                    process=process,
                )
            )
    return {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "server_pid": process.pid if process is not None else None,
            "resource_measurements_enabled": process is not None,
        },
        "levels": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark recommendation HTTP serving")
    parser.add_argument("--config", default="configs/benchmark/serving.yaml")
    parser.add_argument("--server-pid", type=int)
    args = parser.parse_args()
    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if args.server_pid is not None:
        config["server_pid"] = args.server_pid
    results = asyncio.run(run_benchmark(config))
    output = Path(config.get("output", "reports/benchmarks/serving.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
