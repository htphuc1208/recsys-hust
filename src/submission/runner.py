"""Điều phối offline submission MIND Test: không train model, không đọc nhãn."""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from src.models.base import BaseRecommender
from src.reranking.pipeline import RerankingPipeline
from src.submission.inference import infer_ranks
from src.submission.io import (
    iter_impressions,
    iter_original_behaviors,
    original_behavior_path,
    verify_original_row,
)
from src.submission.ranking import format_prediction_line
from src.submission.validate import validate_submission


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    if not isinstance(configuration, dict):
        raise ValueError("Cấu hình YAML không hợp lệ.")
    return configuration


def resolve_checkpoint(config: dict[str, Any]) -> Path:
    reranking = config.get("reranking", {})
    if reranking.get("enabled"):
        checkpoint = reranking.get("base_checkpoint")
        if not checkpoint:
            raise ValueError("P3 yêu cầu reranking.base_checkpoint trong config.")
        return Path(checkpoint)
    name = config.get("experiment_name")
    model = config.get("model", {}).get("name")
    if not name or not model:
        raise ValueError("Thiếu experiment_name hoặc model.name.")
    directory = Path(config.get("paths", {}).get("artifacts_dir", "artifacts")) / "checkpoints"
    return directory / f"{name}_{model}.pkl"


def resolve_impressions(config: dict[str, Any], split: str) -> Path:
    dataset = config.get("dataset", {})
    explicit = dataset.get(f"{split}_impressions")
    if explicit:
        return Path(explicit)
    return Path(dataset.get("processed_root", "data/processed/mind_large")) / split / "impressions"


def resolve_output_dir(config: dict[str, Any], split: str, limit: int | None = None) -> Path:
    """Tách output Test mẫu khỏi submission Test đầy đủ để tránh ghi đè/nhầm nộp."""
    base = Path("reports/submissions/mind_large")
    experiment = str(config.get("experiment_name", "unnamed"))
    if split == "test" and limit is not None:
        return base / "test_samples" / experiment / f"first_{limit}"
    return base / split / experiment


def resolve_original_behaviors(
    *, split: str, raw_behaviors: str | Path | None, require_raw: bool
) -> Path | None:
    if raw_behaviors is not None:
        path = Path(raw_behaviors)
        if not path.is_file():
            raise FileNotFoundError(f"Không tìm thấy behaviors.tsv gốc: {path}")
        return path
    candidate = original_behavior_path(split)
    if candidate.is_file():
        return candidate
    if require_raw:
        raise FileNotFoundError(
            f"Không tìm thấy {candidate}. Cần behaviors.tsv gốc để chứng nhận thứ tự nộp. "
            "Có thể chỉ định --raw-behaviors PATH."
        )
    return None


def run_submission(
    *,
    config_path: str | Path,
    split: str = "test",
    output_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    impressions_path: str | Path | None = None,
    raw_behaviors: str | Path | None = None,
    require_raw: bool = False,
    batch_size: int = 2048,
    limit: int | None = None,
    overwrite: bool = False,
    model: BaseRecommender | None = None,
    reranker: RerankingPipeline | None = None,
) -> dict[str, Any]:
    """Sinh TXT/ZIP, tự kiểm tra lại từng hàng, và ghi manifest truy vết.

    `model` / `reranker` injection chỉ phục vụ kiểm thử; CLI luôn load checkpoint.
    """
    if split not in {"test", "dev"}:
        raise ValueError("Chỉ hỗ trợ split=test hoặc dev.")
    if batch_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("batch_size/limit phải lớn hơn 0.")

    config = load_config(config_path)
    impression_source = (
        Path(impressions_path) if impressions_path else resolve_impressions(config, split)
    )
    target = Path(output_dir) if output_dir else resolve_output_dir(config, split, limit)
    target.mkdir(parents=True, exist_ok=True)
    txt = target / "prediction.txt"
    zip_path = target / "prediction.zip"
    manifest_file = target / "submission_manifest.json"
    if not overwrite and any(p.exists() for p in (txt, zip_path, manifest_file)):
        raise FileExistsError(f"Đã có submission tại {target}; dùng --overwrite nếu muốn tạo lại.")

    original_path = resolve_original_behaviors(
        split=split, raw_behaviors=raw_behaviors, require_raw=require_raw
    )
    checkpoint = (
        Path(checkpoint_path) if checkpoint_path is not None else resolve_checkpoint(config)
    )
    if model is None:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint không tồn tại (không tự train): {checkpoint}")
        model = BaseRecommender.load(checkpoint)
    p3_enabled = bool(config.get("reranking", {}).get("enabled", False))
    if p3_enabled and reranker is None:
        reranker = RerankingPipeline.from_config(config)
    if not p3_enabled and reranker is not None:
        raise ValueError("Config không bật P3 nhưng lại truyền reranker.")

    # Viết file tạm và chỉ thay thế output sau khi đã kiểm tra dữ liệu đầu vào.
    temporary_txt = target / "prediction.txt.part"
    temporary_zip = target / "prediction.zip.part"
    original_iterator = (
        iter_original_behaviors(original_path, split=split) if original_path else None
    )
    total = 0
    unknown_candidates = 0
    p3_fallbacks: Counter[str] = Counter()
    seen_ids: set[int] = set()
    try:
        with temporary_txt.open("w", encoding="utf-8", newline="\n") as handle:
            for row in iter_impressions(impression_source, batch_size=batch_size):
                impression_id = int(row["impression_id"])
                if impression_id <= 0 or impression_id in seen_ids:
                    raise ValueError(f"Impression ID không hợp lệ hoặc trùng: {impression_id}")
                seen_ids.add(impression_id)
                candidate_ids = list(row["candidate_news_ids"] or [])
                candidate_items = [int(x) for x in (row["candidate_item_idxs"] or [])]
                if not candidate_ids or len(candidate_ids) != len(candidate_items):
                    raise ValueError(
                        f"Impression {impression_id}: candidate IDs rỗng hoặc lệch mapping."
                    )
                labels = row["labels"]
                if split == "test" and labels is not None:
                    raise ValueError(f"Test impression {impression_id} không được có nhãn.")
                if split == "dev" and labels is not None and len(labels) != len(candidate_items):
                    raise ValueError(
                        f"Dev impression {impression_id}: labels và candidates không khớp."
                    )
                if original_iterator is not None:
                    reference = next(original_iterator, None)
                    if reference is None:
                        raise ValueError("Parquet có nhiều impression hơn raw behaviors.tsv.")
                    verify_original_row(impression_id, candidate_ids, reference, total + 1)

                result = infer_ranks(
                    model,
                    user_idx=int(row["user_idx"]),
                    candidate_item_idxs=candidate_items,
                    history_item_idxs=row["history_item_idxs"] or [],
                    reranker=reranker,
                )
                handle.write(format_prediction_line(impression_id, result.ranks))
                total += 1
                unknown_candidates += result.unknown_candidates
                if result.p3_fallback_reason:
                    p3_fallbacks[result.p3_fallback_reason] += 1
                if total % 50000 == 0:
                    print(f"Đã dự đoán {total:,} impressions")
                if limit is not None and total >= limit:
                    break
            if (
                original_iterator is not None
                and limit is None
                and next(original_iterator, None) is not None
            ):
                raise ValueError("Raw behaviors.tsv có nhiều impression hơn Parquet.")
        if total == 0:
            raise ValueError("Không có impression nào để xuất.")
        if limit is not None and total != limit:
            raise ValueError(f"Yêu cầu {limit} impressions nhưng chỉ đọc được {total}.")
        # Validate trên file tạm trước khi công bố output chính thức.
        with ZipFile(temporary_zip, mode="w", compression=ZIP_DEFLATED) as archive:
            archive.write(temporary_txt, arcname="prediction.txt")
        if limit is None:
            validation = validate_submission(
                temporary_txt,
                impression_source,
                prediction_zip=temporary_zip,
                original_behaviors=original_path,
                batch_size=batch_size,
                expected_count=total,
                split=split,
            )
        else:
            from src.submission.validate import validate_partial_submission

            validation = validate_partial_submission(
                temporary_txt,
                impression_source,
                prediction_zip=temporary_zip,
                original_behaviors=original_path,
                batch_size=batch_size,
                expected_count=total,
                split=split,
            )
        os.replace(temporary_txt, txt)
        os.replace(temporary_zip, zip_path)
        result_manifest = {
            "format": "Microsoft MIND prediction.txt; impression_id [rank_per_original_candidate]",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": config.get("experiment_name"),
            "model_name": config.get("model", {}).get("name"),
            "split": split,
            "model_checkpoint": str(checkpoint),
            "config_path": str(config_path),
            "impressions_path": str(impression_source),
            "raw_behaviors_path": str(original_path) if original_path else None,
            "reranking_enabled": p3_enabled,
            "unknown_candidates": unknown_candidates,
            "p3_fallback_impressions": dict(p3_fallbacks),
            "complete_dataset": limit is None,
            "sample_limit": limit,
            "submission_ready": split == "test" and limit is None and bool(original_path),
            "validation": validation,
        }
        with manifest_file.open("w", encoding="utf-8") as file:
            json.dump(result_manifest, file, indent=2, ensure_ascii=False)
        return result_manifest
    except Exception:
        for path in (temporary_txt, temporary_zip):
            path.unlink(missing_ok=True)
        # File có trước vẫn được bảo vệ khi thiếu --overwrite.
        raise
