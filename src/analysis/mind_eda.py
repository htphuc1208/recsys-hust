"""Memory-conscious exploratory data analysis for processed MINDlarge data."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

from src.common.logger import setup_logger

SPLITS = ("train", "dev", "test")
TOP_N = 15


def _load_parquet() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required. Install the project first with: pip install -e ."
        ) from exc
    return pq


@dataclass
class Reservoir:
    """Keep a uniform bounded sample from a stream of numeric values."""

    size: int
    seed: int
    values: list[int] = field(default_factory=list)
    seen: int = 0

    def __post_init__(self) -> None:
        self._random = random.Random(self.seed)

    def add_many(self, values: Iterable[int]) -> None:
        for value in values:
            self.seen += 1
            if len(self.values) < self.size:
                self.values.append(value)
                continue
            index = self._random.randrange(self.seen)
            if index < self.size:
                self.values[index] = value


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"mean": None, "median": None, "p95": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    return {
        "mean": round(fmean(values), 3),
        "median": median(values),
        "p95": ordered[p95_index],
        "max": max(values),
    }


def _parquet_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet files found in {directory}")
    return files


def _iter_batches(
    directory: Path,
    columns: list[str],
    *,
    batch_size: int,
    max_rows: int | None,
) -> Iterable[Any]:
    pq = _load_parquet()
    rows_read = 0
    for path in _parquet_files(directory):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=columns):
            if max_rows is not None:
                remaining = max_rows - rows_read
                if remaining <= 0:
                    return
                if batch.num_rows > remaining:
                    batch = batch.slice(0, remaining)
            rows_read += batch.num_rows
            yield batch


def _analyze_news(
    directory: Path, *, batch_size: int, max_rows: int | None
) -> tuple[dict[str, Any], Counter[str]]:
    categories: Counter[str] = Counter()
    news_ids: set[str] = set()
    rows = missing_title = missing_abstract = 0
    title_chars = abstract_chars = title_entities = abstract_entities = 0

    columns = [
        "news_id",
        "category",
        "title",
        "abstract",
        "title_entity_ids",
        "abstract_entity_ids",
    ]
    for batch in _iter_batches(directory, columns, batch_size=batch_size, max_rows=max_rows):
        data = batch.to_pydict()
        rows += batch.num_rows
        news_ids.update(value for value in data["news_id"] if value)
        categories.update(value or "<missing>" for value in data["category"])
        for title, abstract, title_ids, abstract_ids in zip(
            data["title"],
            data["abstract"],
            data["title_entity_ids"],
            data["abstract_entity_ids"],
            strict=True,
        ):
            missing_title += not bool(title)
            missing_abstract += not bool(abstract)
            title_chars += len(title or "")
            abstract_chars += len(abstract or "")
            title_entities += len(title_ids or [])
            abstract_entities += len(abstract_ids or [])

    denominator = rows or 1
    return (
        {
            "rows": rows,
            "unique_news": len(news_ids),
            "duplicate_news_rows": rows - len(news_ids),
            "categories": len(categories),
            "missing_title_pct": round(100 * missing_title / denominator, 3),
            "missing_abstract_pct": round(100 * missing_abstract / denominator, 3),
            "avg_title_chars": round(title_chars / denominator, 3),
            "avg_abstract_chars": round(abstract_chars / denominator, 3),
            "avg_title_entities": round(title_entities / denominator, 3),
            "avg_abstract_entities": round(abstract_entities / denominator, 3),
        },
        categories,
    )


def _analyze_behaviors(
    directory: Path,
    *,
    batch_size: int,
    max_rows: int | None,
    sample_size: int,
    seed: int,
) -> tuple[dict[str, Any], Counter[str], dict[str, list[int]]]:
    users: set[str] = set()
    daily: Counter[str] = Counter()
    history_sample = Reservoir(sample_size, seed)
    candidate_sample = Reservoir(sample_size, seed + 1)
    rows = empty_histories = history_total = total_candidates = total_clicks = 0
    labeled_rows = 0
    history_max = candidate_max = 0
    no_click_rows = 0
    timestamp_min: datetime | None = None
    timestamp_max: datetime | None = None

    columns = [
        "user_id",
        "timestamp",
        "labels",
        "history_length",
        "candidate_count",
    ]
    for batch in _iter_batches(directory, columns, batch_size=batch_size, max_rows=max_rows):
        data = batch.to_pydict()
        rows += batch.num_rows
        users.update(value for value in data["user_id"] if value)
        history_lengths = [int(value or 0) for value in data["history_length"]]
        candidate_counts = [int(value or 0) for value in data["candidate_count"]]
        history_sample.add_many(history_lengths)
        candidate_sample.add_many(candidate_counts)
        empty_histories += sum(value == 0 for value in history_lengths)
        history_total += sum(history_lengths)
        total_candidates += sum(candidate_counts)
        history_max = max(history_max, max(history_lengths, default=0))
        candidate_max = max(candidate_max, max(candidate_counts, default=0))

        for timestamp in data["timestamp"]:
            if timestamp is None:
                continue
            timestamp_min = timestamp if timestamp_min is None else min(timestamp_min, timestamp)
            timestamp_max = timestamp if timestamp_max is None else max(timestamp_max, timestamp)
            daily[timestamp.date().isoformat()] += 1

        for labels in data["labels"]:
            if labels is None:
                continue
            clicks = sum(labels)
            labeled_rows += 1
            total_clicks += clicks
            no_click_rows += clicks == 0

    denominator = rows or 1
    click_denominator = total_candidates or 1
    labeled_denominator = labeled_rows or 1
    history_distribution = _distribution(history_sample.values)
    candidate_distribution = _distribution(candidate_sample.values)
    history_distribution.update(mean=round(history_total / denominator, 3), max=history_max)
    candidate_distribution.update(mean=round(total_candidates / denominator, 3), max=candidate_max)
    stats = {
        "rows": rows,
        "unique_users": len(users),
        "timestamp_min": timestamp_min.isoformat() if timestamp_min else None,
        "timestamp_max": timestamp_max.isoformat() if timestamp_max else None,
        "empty_history_pct": round(100 * empty_histories / denominator, 3),
        "history_length": history_distribution,
        "candidate_count": candidate_distribution,
        "total_candidates": total_candidates,
        "labeled": labeled_rows > 0,
        "total_clicks": total_clicks if labeled_rows else None,
        "click_through_rate": (
            round(total_clicks / click_denominator, 6) if labeled_rows else None
        ),
        "avg_clicks_per_impression": (
            round(total_clicks / labeled_denominator, 3) if labeled_rows else None
        ),
        "no_click_impressions_pct": (
            round(100 * no_click_rows / labeled_denominator, 3) if labeled_rows else None
        ),
        "quantile_sample_size": len(history_sample.values),
        "quantiles_approximate": rows > sample_size,
    }
    samples = {
        "history_length": history_sample.values,
        "candidate_count": candidate_sample.values,
    }
    return stats, daily, samples


def analyze_split(
    input_root: Path,
    split: str,
    *,
    batch_size: int = 65_536,
    max_rows: int | None = None,
    sample_size: int = 100_000,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze one processed split and return serializable statistics plus plot data."""
    split_root = input_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Processed split does not exist: {split_root}")

    news, categories = _analyze_news(split_root / "news", batch_size=batch_size, max_rows=max_rows)
    behaviors, daily, samples = _analyze_behaviors(
        split_root / "behaviors",
        batch_size=batch_size,
        max_rows=max_rows,
        sample_size=sample_size,
        seed=seed,
    )
    report = {
        "split": split,
        "sampled": max_rows is not None,
        "max_rows_per_table": max_rows,
        "news": news,
        "behaviors": behaviors,
        "top_categories": [
            {"category": category, "news_count": count}
            for category, count in categories.most_common(TOP_N)
        ],
        "daily_impressions": dict(sorted(daily.items())),
    }
    plot_data = {"categories": categories, "daily": daily, **samples}
    return report, plot_data


def _write_outputs(reports: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"splits": reports}, file, indent=2, ensure_ascii=False)

    fields = [
        "split",
        "news_rows",
        "unique_news",
        "behavior_rows",
        "unique_users",
        "timestamp_min",
        "timestamp_max",
        "empty_history_pct",
        "history_median",
        "history_p95",
        "candidate_median",
        "candidate_p95",
        "click_through_rate",
    ]
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for report in reports:
            news = report["news"]
            behaviors = report["behaviors"]
            writer.writerow(
                {
                    "split": report["split"],
                    "news_rows": news["rows"],
                    "unique_news": news["unique_news"],
                    "behavior_rows": behaviors["rows"],
                    "unique_users": behaviors["unique_users"],
                    "timestamp_min": behaviors["timestamp_min"],
                    "timestamp_max": behaviors["timestamp_max"],
                    "empty_history_pct": behaviors["empty_history_pct"],
                    "history_median": behaviors["history_length"]["median"],
                    "history_p95": behaviors["history_length"]["p95"],
                    "candidate_median": behaviors["candidate_count"]["median"],
                    "candidate_p95": behaviors["candidate_count"]["p95"],
                    "click_through_rate": behaviors["click_through_rate"],
                }
            )


def _save_plots(
    reports: list[dict[str, Any]], plot_data: dict[str, dict[str, Any]], output_dir: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for EDA plots. Install with: pip install -e ."
        ) from exc

    splits = [report["split"] for report in reports]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(splits, [report["news"]["rows"] for report in reports])
    axes[0].set_title("News rows")
    axes[1].bar(splits, [report["behaviors"]["rows"] for report in reports])
    axes[1].set_title("Impressions")
    for axis in axes:
        axis.set_xlabel("Split")
        axis.ticklabel_format(axis="y", style="plain")
    figure.tight_layout()
    figure.savefig(output_dir / "dataset_overview.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(len(splits), 2, figsize=(12, 3.5 * len(splits)), squeeze=False)
    for row, split in enumerate(splits):
        axes[row, 0].hist(plot_data[split]["history_length"], bins=40)
        axes[row, 0].set_title(f"{split}: history length")
        axes[row, 1].hist(plot_data[split]["candidate_count"], bins=40)
        axes[row, 1].set_title(f"{split}: candidates per impression")
    figure.tight_layout()
    figure.savefig(output_dir / "behavior_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(len(splits), 1, figsize=(11, 4 * len(splits)), squeeze=False)
    for row, split in enumerate(splits):
        common = plot_data[split]["categories"].most_common(TOP_N)
        labels = [item[0] for item in reversed(common)]
        counts = [item[1] for item in reversed(common)]
        axes[row, 0].barh(labels, counts)
        axes[row, 0].set_title(f"{split}: top {TOP_N} news categories")
    figure.tight_layout()
    figure.savefig(output_dir / "top_categories.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for split in splits:
        dates = sorted(plot_data[split]["daily"])
        axis.plot(dates, [plot_data[split]["daily"][date] for date in dates], label=split)
    axis.set_title("Impressions over time")
    axis.set_xlabel("Date")
    axis.set_ylabel("Impressions")
    axis.tick_params(axis="x", rotation=45)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "impressions_over_time.png", dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EDA on processed MINDlarge Parquet data")
    parser.add_argument("--input-root", type=Path, default=Path("data/interim/mind_large"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/mind_eda"))
    parser.add_argument("--split", choices=(*SPLITS, "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Analyze at most this many rows per table (useful for a quick sample run)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.sample_size <= 0:
        raise ValueError("--batch-size and --sample-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive")

    logger = setup_logger("mind_eda")
    splits = SPLITS if args.split == "all" else (args.split,)
    reports: list[dict[str, Any]] = []
    all_plot_data: dict[str, dict[str, Any]] = {}
    for split in splits:
        logger.info("Analyzing MINDlarge %s", split)
        report, split_plot_data = analyze_split(
            args.input_root,
            split,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        reports.append(report)
        all_plot_data[split] = split_plot_data

    _write_outputs(reports, args.output_dir)
    _save_plots(reports, all_plot_data, args.output_dir)
    logger.info("EDA report written to %s", args.output_dir)


if __name__ == "__main__":
    main()
