"""Quét trọng số Content-based trên một mẫu cố định của toàn bộ tập dev."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.evaluation.ranking_metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.models.base import BaseRecommender
from src.models.content_based.recommender import (
    FEATURE_FILES,
    AVAILABLE_FEATURE_FILES,
    ContentBasedRecommender,
)

FEATURE_NAMES = tuple(FEATURE_FILES.keys())  # 5 feature mặc định, giữ nguyên hành vi cũ.
FEATURE_NAMES_WITH_ABSTRACT_ENTITY = tuple(AVAILABLE_FEATURE_FILES.keys())
MAX_SWEEP_CONFIGS = 10_000


@dataclass(frozen=True)
class CachedContentImpression:
    """Giữ component score của các feature cho một impression dev."""

    impression_id: int
    candidate_item_idxs: np.ndarray
    true_items: set[int]
    feature_scores: dict[str, np.ndarray]
    fallback_scores: np.ndarray | None
    has_cold_candidate: bool
    is_cold_user: bool


@dataclass(frozen=True)
class SweepSettings:
    """Thông số quét trọng số Content-based."""

    step: float = 0.20
    sample_size: int = 5000
    seed: int = 42
    top_k: int = 10
    weight_bounds: dict[str, tuple[float, float]] | None = None
    output_tag: str | None = None
    include_abstract_entity: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.step <= 1.0:
            raise ValueError("step phải nằm trong (0, 1].")
        if self.sample_size <= 0:
            raise ValueError("sample_size phải lớn hơn 0.")
        if self.top_k <= 0:
            raise ValueError("top_k phải lớn hơn 0.")
        if self.output_tag is not None and not self.output_tag.strip():
            raise ValueError("output_tag không được là chuỗi rỗng.")


def _validate_step(step: float) -> int:
    """Kiểm tra step và trả về số đơn vị tạo thành tổng 1."""
    divisions = int(round(1.0 / step))
    if divisions <= 0 or not np.isclose(divisions * step, 1.0, atol=1e-8):
        raise ValueError("step phải chia hết 1, ví dụ 0.5, 0.25, 0.2, 0.1 hoặc 0.05.")
    return divisions


def _normalize_weight_bounds(
    step: float,
    bounds: dict[str, tuple[float, float]] | None,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[str, tuple[int, int]]:
    """Đổi bounds sang đơn vị nguyên, phù hợp cả 5/6 nhóm feature."""
    divisions = _validate_step(step)
    normalized: dict[str, tuple[int, int]] = {}
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("Danh sách feature phải không rỗng và không trùng.")
    if set(feature_names) - set(AVAILABLE_FEATURE_FILES):
        raise ValueError("Danh sách feature có tên không hợp lệ.")
    if bounds and set(bounds) - set(feature_names):
        raise ValueError("Bounds chứa feature không được chọn.")

    for name in feature_names:
        lower, upper = (0.0, 1.0) if bounds is None else bounds.get(name, (0.0, 1.0))
        lower, upper = float(lower), float(upper)
        if not 0.0 <= lower <= upper <= 1.0:
            raise ValueError(f"Bounds của {name} phải thỏa 0 <= min <= max <= 1.")
        lower_units = int(round(lower * divisions))
        upper_units = int(round(upper * divisions))
        if not np.isclose(lower_units / divisions, lower, atol=1e-8):
            raise ValueError(f"Min bound của {name}={lower} không nằm trên lưới step={step}.")
        if not np.isclose(upper_units / divisions, upper, atol=1e-8):
            raise ValueError(f"Max bound của {name}={upper} không nằm trên lưới step={step}.")
        normalized[name] = (lower_units, upper_units)

    if sum(v[0] for v in normalized.values()) > divisions or sum(
        v[1] for v in normalized.values()
    ) < divisions:
        raise ValueError("Các bounds không thể tạo được bộ trọng số có tổng bằng 1.")
    return normalized


def generate_weight_grid(
    step: float,
    bounds: dict[str, tuple[float, float]] | None = None,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> list[dict[str, float]]:
    """Sinh lưới trọng số tổng bằng 1; giới hạn số cấu hình để tránh bùng nổ."""
    divisions = _validate_step(step)
    bound_units = _normalize_weight_bounds(step, bounds, feature_names)
    configurations: list[dict[str, float]] = []

    def add_units(index: int, remaining: int, assigned: list[int]) -> None:
        if index == len(feature_names):
            if remaining == 0:
                configurations.append({
                    name: value / divisions for name, value in zip(feature_names, assigned)
                })
                if len(configurations) > MAX_SWEEP_CONFIGS:
                    raise ValueError(
                        f"Lưới vượt {MAX_SWEEP_CONFIGS:,} cấu hình; hãy siết bounds hoặc tăng step."
                    )
            return

        name = feature_names[index]
        low, high = bound_units[name]
        others = feature_names[index + 1:]
        others_min = sum(bound_units[k][0] for k in others)
        others_max = sum(bound_units[k][1] for k in others)
        for units in range(max(low, remaining - others_max), min(high, remaining - others_min) + 1):
            add_units(index + 1, remaining - units, assigned + [units])

    add_units(0, divisions, [])
    if not configurations:
        raise ValueError("Không có cấu hình trọng số nào thỏa step và bounds đã chọn.")
    return configurations


def _reservoir_add(
    reservoir: list[dict[str, Any]],
    row: dict[str, Any],
    seen_count: int,
    sample_size: int,
    rng: np.random.Generator,
) -> None:
    """Cập nhật reservoir sample để không cần nạp toàn bộ dev vào RAM."""
    if len(reservoir) < sample_size:
        reservoir.append(row)
        return

    replacement_index = int(rng.integers(0, seen_count))
    if replacement_index < sample_size:
        reservoir[replacement_index] = row


def sample_dev_rows(
    impressions_path: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Lấy sample cố định từ toàn bộ dev có ít nhất một positive label."""
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(impressions_path), format="parquet")
    columns = [
        "impression_id",
        "user_idx",
        "history_item_idxs",
        "candidate_item_idxs",
        "labels",
        "candidate_seen_in_train_mask",
    ]
    missing = set(columns) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"Impressions thiếu cột: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    reservoir: list[dict[str, Any]] = []
    eligible_count = 0

    for batch in dataset.to_batches(columns=columns, batch_size=4096):
        rows = batch.to_pydict()
        row_count = len(rows["impression_id"])
        for index in range(row_count):
            labels = rows["labels"][index]
            if labels is None or sum(labels) == 0:
                continue

            eligible_count += 1
            row = {
                "impression_id": int(rows["impression_id"][index]),
                "user_idx": int(rows["user_idx"][index]),
                "history_item_idxs": rows["history_item_idxs"][index] or [],
                "candidate_item_idxs": rows["candidate_item_idxs"][index],
                "labels": labels,
                "candidate_seen_in_train_mask": rows["candidate_seen_in_train_mask"][index],
            }
            _reservoir_add(
                reservoir=reservoir,
                row=row,
                seen_count=eligible_count,
                sample_size=sample_size,
                rng=rng,
            )

    if not reservoir:
        raise RuntimeError("Không lấy được impression dev hợp lệ.")
    return reservoir, eligible_count


def cache_feature_scores(
    model: ContentBasedRecommender,
    rows: list[dict[str, Any]],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> list[CachedContentImpression]:
    """Chấm feature đúng một lần cho mỗi impression rồi lưu cache."""
    if not model.is_fitted:
        raise RuntimeError("Checkpoint Content-based chưa fitted.")

    # Quét 6 chiều phải tải abstract_entity dù trọng số baseline đang bằng 0.
    model._ensure_features_loaded(
        load_all=True, include_optional="abstract_entity" in feature_names
    )
    cached: list[CachedContentImpression] = []

    for row_number, row in enumerate(rows, start=1):
        candidates = np.asarray(row["candidate_item_idxs"], dtype=np.int64)
        history = row["history_item_idxs"]
        profiles = model._build_profiles(history)

        feature_scores: dict[str, np.ndarray] = {
            name: np.zeros(candidates.size, dtype=np.float32) for name in feature_names
        }
        fallback_scores: np.ndarray | None = None

        if profiles:
            scored = model._score_candidates_by_feature(candidates, profiles)
            for name, scores in scored.items():
                if name in feature_scores:
                    feature_scores[name] = np.asarray(scores, dtype=np.float32)
        elif model.cold_user_strategy == "popularity":
            fallback_scores = model.popularity_scores[candidates].astype(
                np.float32,
                copy=True,
            )
        else:
            fallback_scores = np.zeros(candidates.size, dtype=np.float32)

        labels = row["labels"]
        true_items = {
            int(candidates[index]) for index, label in enumerate(labels) if int(label) == 1
        }
        candidate_train_mask = row["candidate_seen_in_train_mask"]
        has_cold_candidate = any(not bool(value) for value in candidate_train_mask)
        user_idx = int(row["user_idx"])
        is_cold_user = len(history) <= 5 or user_idx == -1

        cached.append(
            CachedContentImpression(
                impression_id=int(row["impression_id"]),
                candidate_item_idxs=candidates,
                true_items=true_items,
                feature_scores=feature_scores,
                fallback_scores=fallback_scores,
                has_cold_candidate=has_cold_candidate,
                is_cold_user=is_cold_user,
            )
        )

        if row_number % 1000 == 0:
            print(f"  Đã cache feature scores: {row_number:,}/{len(rows):,}")

    return cached


def combine_feature_scores(
    impression: CachedContentImpression,
    weights: dict[str, float],
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> np.ndarray:
    """Ghép component score theo một bộ trọng số."""
    if impression.fallback_scores is not None:
        return impression.fallback_scores

    scores = np.zeros(impression.candidate_item_idxs.size, dtype=np.float32)
    for name in feature_names:
        scores += float(weights[name]) * impression.feature_scores[name]
    return scores


def _empty_metric_lists(top_k: int) -> dict[str, list[float]]:
    """Tạo bộ nhớ metric cho một slice."""
    return {
        f"NDCG@{top_k}": [],
        f"Recall@{top_k}": [],
        f"MRR@{top_k}": [],
    }


def evaluate_cached_weights(
    cached: list[CachedContentImpression],
    weights: dict[str, float],
    top_k: int = 10,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> dict[str, float]:
    """Đánh giá một bộ trọng số trên cache, dùng đúng tie-breaking stable."""
    slice_values = {
        "overall": _empty_metric_lists(top_k),
        "cold_item": _empty_metric_lists(top_k),
        "warm_item": _empty_metric_lists(top_k),
        "cold_user": _empty_metric_lists(top_k),
        "warm_user": _empty_metric_lists(top_k),
    }

    for impression in cached:
        scores = combine_feature_scores(impression, weights, feature_names=feature_names)
        order = np.argsort(-scores, kind="stable")
        ranking = [int(impression.candidate_item_idxs[index]) for index in order]

        metrics = {
            f"NDCG@{top_k}": ndcg_at_k(ranking, impression.true_items, top_k),
            f"Recall@{top_k}": recall_at_k(ranking, impression.true_items, top_k),
            f"MRR@{top_k}": mrr_at_k(ranking, impression.true_items, top_k),
        }

        active_slices = ["overall"]
        active_slices.append("cold_item" if impression.has_cold_candidate else "warm_item")
        active_slices.append("cold_user" if impression.is_cold_user else "warm_user")

        for slice_name in active_slices:
            for metric_name, value in metrics.items():
                slice_values[slice_name][metric_name].append(value)

    result: dict[str, float] = {}
    for slice_name, metrics in slice_values.items():
        for metric_name, values in metrics.items():
            key = metric_name if slice_name == "overall" else f"{slice_name}_{metric_name}"
            result[key] = float(np.mean(values)) if values else 0.0
    return result


def _weight_columns(
    weights: dict[str, float], feature_names: tuple[str, ...] = FEATURE_NAMES
) -> dict[str, float]:
    """Đổi tên trọng số thành cột CSV dễ đọc."""
    return {f"{name}_weight": float(weights[name]) for name in feature_names}


def run_weight_sweep(
    config_path: str | Path,
    settings: SweepSettings,
    checkpoint_path: str | Path | None = None,
) -> pd.DataFrame:
    """Chạy coarse/fine sweep cho Content-based trên một sample dev cố định."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    experiment_name = config.get("experiment_name", "exp05_mind_content")
    model_config = config.get("model", {})
    if model_config.get("name") != "content_based":
        raise ValueError("Config phải dùng model.name = 'content_based'.")

    dataset_config = config.get("dataset", {})
    impressions_path = Path(dataset_config["dev_impressions"])

    if checkpoint_path is None:
        artifacts_dir = Path(config.get("paths", {}).get("artifacts_dir", "artifacts"))
        checkpoint_path = artifacts_dir / "checkpoints" / f"{experiment_name}_content_based.pkl"
    checkpoint_path = Path(checkpoint_path)

    print("Nạp Content-based checkpoint...")
    loaded = BaseRecommender.load(checkpoint_path)
    if not isinstance(loaded, ContentBasedRecommender):
        raise TypeError("Checkpoint không phải ContentBasedRecommender.")
    model = loaded
    feature_names = (
        FEATURE_NAMES_WITH_ABSTRACT_ENTITY if settings.include_abstract_entity else FEATURE_NAMES
    )
    if not settings.include_abstract_entity and model.feature_weights.get("abstract_entity", 0.0) > 0:
        raise ValueError("Checkpoint đang dùng abstract_entity; hãy bật --include-abstract-entity.")

    print(
        "Lấy sample cố định trực tiếp từ toàn bộ dev: "
        f"sample_size={settings.sample_size:,}, seed={settings.seed}"
    )
    rows, eligible_count = sample_dev_rows(
        impressions_path=impressions_path,
        sample_size=settings.sample_size,
        seed=settings.seed,
    )
    print(f"Dev có {eligible_count:,} impression hợp lệ; dùng {len(rows):,} impression để sweep.")

    print(f"Chấm {len(feature_names)} feature một lần cho dev sample...")
    cached = cache_feature_scores(model, rows, feature_names=feature_names)

    current_weights = {name: float(model.feature_weights.get(name, 0.0)) for name in feature_names}
    baseline = evaluate_cached_weights(
        cached,
        current_weights,
        top_k=settings.top_k,
        feature_names=feature_names,
    )
    ndcg_key = f"NDCG@{settings.top_k}"
    recall_key = f"Recall@{settings.top_k}"
    mrr_key = f"MRR@{settings.top_k}"
    cold_ndcg_key = f"cold_item_NDCG@{settings.top_k}"

    weight_grid = generate_weight_grid(
        settings.step,
        bounds=settings.weight_bounds,
        feature_names=feature_names,
    )
    print(f"Số cấu hình trọng số: {len(weight_grid):,}")
    if settings.weight_bounds is not None:
        print("Bounds fine sweep:")
        for name in feature_names:
            lower, upper = settings.weight_bounds.get(name, (0.0, 1.0))
            print(f"  {name}: [{lower:.2f}, {upper:.2f}]")

    records: list[dict[str, float]] = []
    for index, weights in enumerate(weight_grid, start=1):
        metrics = evaluate_cached_weights(
            cached,
            weights,
            top_k=settings.top_k,
            feature_names=feature_names,
        )
        record = {
            **_weight_columns(weights, feature_names=feature_names),
            **metrics,
            f"Delta{ndcg_key}": metrics[ndcg_key] - baseline[ndcg_key],
            f"DeltaColdItemNDCG@{settings.top_k}": (
                metrics[cold_ndcg_key] - baseline[cold_ndcg_key]
            ),
        }
        records.append(record)

        if index == 1 or index % 10 == 0 or index == len(weight_grid):
            print(
                f"  [{index:>3}/{len(weight_grid)}] "
                f"w=({', '.join(f'{name}={weights[name]:.2f}' for name in feature_names)}) "
                f"NDCG={metrics[ndcg_key]:.5f} "
                f"Cold={metrics[cold_ndcg_key]:.5f}"
            )

    results = pd.DataFrame(records)
    results = results.sort_values(
        by=[ndcg_key, cold_ndcg_key, recall_key, mrr_key],
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    results.insert(0, "rank", np.arange(1, len(results) + 1))

    output_dir = Path("reports") / "content_tuning" / experiment_name
    if settings.output_tag is not None:
        output_dir = output_dir / settings.output_tag.strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "weight_sweep_results.csv"
    top_path = output_dir / "top_configs.csv"
    summary_path = output_dir / "weight_sweep_summary.json"

    results.to_csv(results_path, index=False)
    results.head(20).to_csv(top_path, index=False)

    # Xuất YAML ứng viên để chạy train/evaluate Full Dev một cách tái lập.
    candidate_path = output_dir / "best_config_candidate.yaml"
    candidate_config = dict(config)
    candidate_config["experiment_name"] = f"{experiment_name}_sweep_candidate"
    candidate_config["model"] = dict(model_config)
    candidate_config["model"]["feature_weights"] = {
        name: float(results.iloc[0][f"{name}_weight"]) for name in feature_names
    }
    # Việc sweep chỉ chọn cấu hình; chưa chứng minh tốt hơn trên Full Dev độc lập.
    with candidate_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(candidate_config, file, allow_unicode=True, sort_keys=False)

    summary = {
        "experiment_name": experiment_name,
        "checkpoint": str(checkpoint_path),
        "dev_impressions": str(impressions_path),
        "eligible_dev_impressions": eligible_count,
        "sample_size": len(rows),
        "seed": settings.seed,
        "step": settings.step,
        "weight_bounds": settings.weight_bounds,
        "output_tag": settings.output_tag,
        "num_weight_configs": len(weight_grid),
        "candidate_config": str(candidate_path),
        "feature_order": list(feature_names),
        "include_abstract_entity": settings.include_abstract_entity,
        "current_weights": current_weights,
        "current_metrics": baseline,
        "best_config": results.iloc[0].to_dict() if not results.empty else None,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\nHoàn thành Content-based weight sweep.")
    print("Current weights:", json.dumps(current_weights, ensure_ascii=False))
    print(
        "Current metrics:",
        json.dumps(
            {
                ndcg_key: round(baseline[ndcg_key], 6),
                f"ColdItemNDCG@{settings.top_k}": round(baseline[cold_ndcg_key], 6),
                recall_key: round(baseline[recall_key], 6),
                mrr_key: round(baseline[mrr_key], 6),
            },
            ensure_ascii=False,
        ),
    )
    print("Kết quả:", results_path)
    print("Top 20:", top_path)
    print("Cấu hình ứng viên để train/evaluate Full Dev:", candidate_path)
    return results


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Quét 5 hoặc 6 trọng số của Content-based recommender.")
    parser.add_argument(
        "--config",
        default="configs/experiment/exp05_mind_content.yaml",
        help="Config Content-based.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint tùy chọn; mặc định suy ra từ experiment_name.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.20,
        help="Bước lưới trọng số; coarse khuyến nghị 0.20.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Số impression dev dùng cho sweep.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-abstract-entity", action="store_true",
        help="Quét đủ 6 feature; yêu cầu Feature Store có abstract_entity_embeddings.npy.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output-tag",
        default=None,
        help="Thư mục con để không ghi đè sweep trước, ví dụ fine.",
    )

    for feature_name in FEATURE_NAMES_WITH_ABSTRACT_ENTITY:
        cli_name = feature_name.replace("_", "-")
        parser.add_argument(
            f"--{cli_name}-min",
            type=float,
            default=0.0,
            help=f"Giới hạn dưới cho {feature_name}.",
        )
        parser.add_argument(
            f"--{cli_name}-max",
            type=float,
            default=1.0,
            help=f"Giới hạn trên cho {feature_name}.",
        )
    return parser.parse_args()


def main() -> None:
    """Entry point của Content-based weight sweep."""
    args = parse_args()
    feature_names = (
        FEATURE_NAMES_WITH_ABSTRACT_ENTITY if args.include_abstract_entity else FEATURE_NAMES
    )
    if not args.include_abstract_entity and (
        args.abstract_entity_min != 0.0 or args.abstract_entity_max != 1.0
    ):
        raise ValueError("Dùng --include-abstract-entity để đặt bounds cho abstract_entity.")
    weight_bounds = {
        name: (
            float(getattr(args, f"{name}_min")),
            float(getattr(args, f"{name}_max")),
        )
        for name in feature_names
    }
    has_custom_bounds = any(lower != 0.0 or upper != 1.0 for lower, upper in weight_bounds.values())

    settings = SweepSettings(
        step=args.step,
        sample_size=args.sample_size,
        seed=args.seed,
        top_k=args.top_k,
        weight_bounds=weight_bounds if has_custom_bounds else None,
        output_tag=args.output_tag,
        include_abstract_entity=args.include_abstract_entity,
    )
    run_weight_sweep(
        config_path=args.config,
        settings=settings,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
