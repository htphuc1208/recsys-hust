"""Thí nghiệm Data Minimization cho MIND Large.

Module này tạo các cửa sổ retention theo timestamp của interaction train,
train lại BPR và Content-based trên từng cửa sổ, evaluate trên cùng dev split,
rồi tổng hợp trade-off giữa utility, storage và thời gian train.

Điểm quan trọng:
- Dữ liệu gốc không bị sửa hoặc ghi đè.
- Window ``all`` được ghi lại bằng cùng Parquet writer/compression với các window khác
  để phép đo storage công bằng; dữ liệu gốc vẫn không bị sửa.
- Content-based trong thí nghiệm này dùng history đã giữ lại trong train
  (evaluation.use_impression_history=false), tránh nhìn thấy full dev history.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml


@dataclass(frozen=True)
class WindowSpec:
    """Mô tả một cửa sổ retention."""

    name: str
    days: int | None


@dataclass(frozen=True)
class DatasetPaths:
    """Đường dẫn pointwise và positive interactions của một window."""

    pointwise: Path
    positive_interactions: Path


@dataclass(frozen=True)
class ExperimentLayout:
    """Các thư mục output dùng riêng cho thí nghiệm."""

    project_root: Path
    data_dir: Path
    reports_dir: Path
    artifacts_dir: Path
    generated_config_dir: Path
    metrics_dir: Path
    run_metadata_dir: Path
    figures_dir: Path


def load_yaml(path: Path) -> dict[str, Any]:
    """Đọc YAML và yêu cầu root là dictionary."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML phải có dạng mapping: {path}")
    return data


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Ghi YAML dễ đọc và giữ thứ tự key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Ghi JSON UTF-8 có indent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """Đổi đường dẫn tương đối trong config thành đường dẫn theo project root."""
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def to_project_relative(project_root: Path, path: Path) -> str:
    """Đưa đường dẫn về dạng tương đối để YAML portable giữa các máy."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_layout(project_root: Path, config: dict[str, Any]) -> ExperimentLayout:
    """Tạo layout output từ master config."""
    paths_cfg = config.get("paths", {})
    data_dir = resolve_project_path(
        project_root,
        paths_cfg.get("data_dir", "data/experiments/data_minimization"),
    )
    reports_dir = resolve_project_path(
        project_root,
        paths_cfg.get("reports_dir", "reports/data_minimization"),
    )
    artifacts_dir = resolve_project_path(
        project_root,
        paths_cfg.get("artifacts_dir", "artifacts/data_minimization"),
    )
    return ExperimentLayout(
        project_root=project_root,
        data_dir=data_dir,
        reports_dir=reports_dir,
        artifacts_dir=artifacts_dir,
        generated_config_dir=reports_dir / "generated_configs",
        metrics_dir=reports_dir / "metrics",
        run_metadata_dir=reports_dir / "run_metadata",
        figures_dir=reports_dir / "figures",
    )


def parse_windows(config: dict[str, Any]) -> list[WindowSpec]:
    """Đọc danh sách window, trong đó days=null nghĩa là all."""
    raw_windows = config.get("windows", [])
    if not raw_windows:
        raise ValueError("Config phải có ít nhất một window.")

    windows: list[WindowSpec] = []
    seen_names: set[str] = set()
    for entry in raw_windows:
        name = str(entry["name"])
        days_raw = entry.get("days")
        days = None if days_raw is None else int(days_raw)
        if days is not None and days <= 0:
            raise ValueError(f"Window {name}: days phải > 0 hoặc null.")
        if name in seen_names:
            raise ValueError(f"Window bị trùng tên: {name}")
        seen_names.add(name)
        windows.append(WindowSpec(name=name, days=days))
    return windows


def dataset_timestamp_bounds(path: Path, timestamp_col: str) -> tuple[datetime, datetime]:
    """Quét một cột timestamp để tìm min/max mà không nạp cả dataset vào Pandas."""
    dataset = ds.dataset(str(path), format="parquet")
    if timestamp_col not in dataset.schema.names:
        raise ValueError(f"{path} thiếu cột timestamp: {timestamp_col}")

    minimum: datetime | None = None
    maximum: datetime | None = None
    for batch in dataset.to_batches(columns=[timestamp_col], batch_size=262_144):
        column = batch.column(0)
        if column.null_count == len(column):
            continue
        min_max = pc.min_max(column).as_py()
        batch_min = min_max["min"]
        batch_max = min_max["max"]
        if batch_min is not None and (minimum is None or batch_min < minimum):
            minimum = batch_min
        if batch_max is not None and (maximum is None or batch_max > maximum):
            maximum = batch_max

    if minimum is None or maximum is None:
        raise ValueError(f"Không tìm thấy timestamp hợp lệ trong {path}")
    return minimum, maximum


def resolve_cutoff_timestamp(
    config: dict[str, Any],
    positive_path: Path,
    timestamp_col: str,
) -> datetime:
    """Lấy cutoff từ config hoặc tự động chọn max timestamp + 1 giây."""
    retention_cfg = config.get("retention", {})
    raw_cutoff = retention_cfg.get("cutoff_timestamp", "auto")
    if raw_cutoff not in (None, "auto"):
        return pd.Timestamp(raw_cutoff).to_pydatetime()

    _minimum, maximum = dataset_timestamp_bounds(positive_path, timestamp_col)
    return maximum + timedelta(seconds=1)


def build_time_filter(
    schema: pa.Schema,
    timestamp_col: str,
    start_time: datetime,
    cutoff_time: datetime,
) -> ds.Expression:
    """Tạo filter start <= timestamp < cutoff đúng type của Parquet."""
    timestamp_type = schema.field(timestamp_col).type
    start_scalar = pa.scalar(start_time, type=timestamp_type)
    cutoff_scalar = pa.scalar(cutoff_time, type=timestamp_type)
    return (ds.field(timestamp_col) >= start_scalar) & (ds.field(timestamp_col) < cutoff_scalar)


def parquet_storage_bytes(path: Path) -> int:
    """Tính dung lượng dữ liệu Parquet, không tính manifest/metadata phụ trợ."""
    if path.is_file():
        return int(path.stat().st_size) if path.suffix.lower() == ".parquet" else 0
    if not path.exists():
        return 0
    return int(sum(file.stat().st_size for file in path.rglob("*.parquet") if file.is_file()))


def clear_directory(path: Path) -> None:
    """Xóa directory cũ rồi tạo lại."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def list_parquet_files(path: Path) -> list[Path]:
    """Lấy file Parquet theo thứ tự tên để giữ nguyên thứ tự impression."""
    if path.is_file():
        files = [path]
    else:
        files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy file Parquet trong: {path}")
    return files


def write_normalized_parquet_dataset(
    source_path: Path,
    target_path: Path,
    rows_per_file: int,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Ghi lại toàn bộ dataset bằng cùng chuẩn Parquet của các retention window.

    ``all`` phải đi qua cùng writer/compression với 1d/3d/5d. Nếu dùng trực tiếp
    Parquet gốc thì chênh lệch row-group/compression có thể làm storage không còn
    so sánh công bằng. Hàm này giữ nguyên thứ tự file và thứ tự row của source.
    """
    manifest_path = target_path / "_retention_manifest.json"
    if target_path.exists() and manifest_path.is_file() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    source = ds.dataset(str(source_path), format="parquet")
    if rows_per_file <= 0:
        raise ValueError("rows_per_file phải > 0.")

    clear_directory(target_path)
    file_index = 0
    rows_in_file = 0
    total_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for source_file in list_parquet_files(source_path):
            parquet_file = pq.ParquetFile(source_file)
            for batch in parquet_file.iter_batches(batch_size=131_072):
                if batch.num_rows == 0:
                    continue

                if writer is None or rows_in_file >= rows_per_file:
                    if writer is not None:
                        writer.close()
                    output_file = target_path / f"part-{file_index:05d}.parquet"
                    writer = pq.ParquetWriter(
                        output_file,
                        source.schema,
                        compression=compression,
                    )
                    file_index += 1
                    rows_in_file = 0

                writer.write_batch(batch)
                rows_in_file += batch.num_rows
                total_rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise ValueError(f"Dataset source không có dòng nào: {source_path}")

    manifest = {
        "source_path": source_path.as_posix(),
        "rows": int(total_rows),
        "files": int(file_index),
        "compression": compression,
        "row_order_preserved": True,
        "normalized_full_copy": True,
    }
    save_json(manifest_path, manifest)
    return manifest


def write_filtered_parquet_dataset(
    source_path: Path,
    target_path: Path,
    timestamp_col: str,
    start_time: datetime,
    cutoff_time: datetime,
    rows_per_file: int,
    compression: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Lọc Parquet theo thời gian, giữ nguyên thứ tự row của source.

    BPR gom pair theo impression và có thể gặp impression ở ranh giới file,
    vì vậy source được đọc theo thứ tự tên file thay vì để Dataset scanner
    tự chọn thứ tự fragment.
    """
    manifest_path = target_path / "_retention_manifest.json"
    if target_path.exists() and manifest_path.is_file() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    source = ds.dataset(str(source_path), format="parquet")
    if timestamp_col not in source.schema.names:
        raise ValueError(f"{source_path} thiếu cột {timestamp_col}")
    if rows_per_file <= 0:
        raise ValueError("rows_per_file phải > 0.")

    clear_directory(target_path)
    timestamp_type = source.schema.field(timestamp_col).type
    start_scalar = pa.scalar(start_time, type=timestamp_type)
    cutoff_scalar = pa.scalar(cutoff_time, type=timestamp_type)

    file_index = 0
    rows_in_file = 0
    total_rows = 0
    writer: pq.ParquetWriter | None = None

    try:
        for source_file in list_parquet_files(source_path):
            parquet_file = pq.ParquetFile(source_file)
            for batch in parquet_file.iter_batches(batch_size=131_072):
                timestamp_index = batch.schema.get_field_index(timestamp_col)
                if timestamp_index < 0:
                    raise ValueError(f"{source_file} thiếu cột {timestamp_col}")
                timestamps = batch.column(timestamp_index)
                mask = pc.and_(
                    pc.greater_equal(timestamps, start_scalar),
                    pc.less(timestamps, cutoff_scalar),
                )
                filtered_batch = batch.filter(mask)
                if filtered_batch.num_rows == 0:
                    continue

                if writer is None or rows_in_file >= rows_per_file:
                    if writer is not None:
                        writer.close()
                    output_file = target_path / f"part-{file_index:05d}.parquet"
                    writer = pq.ParquetWriter(
                        output_file,
                        source.schema,
                        compression=compression,
                    )
                    file_index += 1
                    rows_in_file = 0

                writer.write_batch(filtered_batch)
                rows_in_file += filtered_batch.num_rows
                total_rows += filtered_batch.num_rows
    finally:
        if writer is not None:
            writer.close()

    if total_rows == 0:
        raise ValueError(
            f"Window {start_time} -> {cutoff_time} không giữ lại dòng nào từ {source_path}."
        )

    manifest = {
        "source_path": source_path.as_posix(),
        "start_timestamp": start_time.isoformat(sep=" "),
        "cutoff_timestamp": cutoff_time.isoformat(sep=" "),
        "rows": int(total_rows),
        "files": int(file_index),
        "compression": compression,
        "row_order_preserved": True,
    }
    save_json(manifest_path, manifest)
    return manifest


def prepare_retention_datasets(
    project_root: Path,
    config: dict[str, Any],
    layout: ExperimentLayout,
    selected_windows: set[str] | None,
    overwrite: bool,
) -> tuple[dict[str, DatasetPaths], pd.DataFrame, datetime]:
    """Tạo dataset retention và chuẩn hóa ``all`` để so sánh storage công bằng."""
    source_cfg = config.get("source", {})
    pointwise_source = resolve_project_path(
        project_root,
        source_cfg.get("pointwise", "data/processed/mind_large/train/pointwise"),
    )
    positive_source = resolve_project_path(
        project_root,
        source_cfg.get(
            "positive_interactions",
            "data/processed/mind_large/train/positive_interactions",
        ),
    )
    timestamp_col = str(config.get("retention", {}).get("timestamp_col", "timestamp"))
    prepare_cfg = config.get("prepare", {})
    rows_per_file = int(prepare_cfg.get("rows_per_file", 500_000))
    compression = str(prepare_cfg.get("compression", "snappy"))

    if not pointwise_source.exists():
        raise FileNotFoundError(f"Không tìm thấy pointwise source: {pointwise_source}")
    if not positive_source.exists():
        raise FileNotFoundError(f"Không tìm thấy positive source: {positive_source}")

    cutoff_time = resolve_cutoff_timestamp(
        config,
        positive_source,
        timestamp_col,
    )

    source_rows = {
        "pointwise": int(ds.dataset(str(pointwise_source), format="parquet").count_rows()),
        "positive_interactions": int(
            ds.dataset(str(positive_source), format="parquet").count_rows()
        ),
    }

    # ``all`` luôn được materialize bằng cùng writer/compression để làm baseline
    # storage. Việc này cũng được thực hiện khi CLI chỉ chọn một subset window.
    all_dir = layout.data_dir / "all"
    all_paths = DatasetPaths(
        pointwise=all_dir / "pointwise",
        positive_interactions=all_dir / "positive_interactions",
    )
    print("[prepare] all: chuẩn hóa toàn bộ train bằng cùng Parquet writer")
    write_normalized_parquet_dataset(
        source_path=pointwise_source,
        target_path=all_paths.pointwise,
        rows_per_file=rows_per_file,
        compression=compression,
        overwrite=overwrite,
    )
    write_normalized_parquet_dataset(
        source_path=positive_source,
        target_path=all_paths.positive_interactions,
        rows_per_file=rows_per_file,
        compression=compression,
        overwrite=overwrite,
    )
    normalized_all_storage = {
        "pointwise": parquet_storage_bytes(all_paths.pointwise),
        "positive_interactions": parquet_storage_bytes(all_paths.positive_interactions),
    }

    window_paths: dict[str, DatasetPaths] = {}
    stat_rows: list[dict[str, Any]] = []

    for window in parse_windows(config):
        if selected_windows is not None and window.name not in selected_windows:
            continue

        if window.days is None:
            paths = all_paths
            start_time = None
        else:
            start_time = cutoff_time - timedelta(days=window.days)
            window_dir = layout.data_dir / window.name
            pointwise_target = window_dir / "pointwise"
            positive_target = window_dir / "positive_interactions"

            print(
                f"[prepare] {window.name}: giữ interaction từ {start_time} đến trước {cutoff_time}"
            )
            write_filtered_parquet_dataset(
                source_path=pointwise_source,
                target_path=pointwise_target,
                timestamp_col=timestamp_col,
                start_time=start_time,
                cutoff_time=cutoff_time,
                rows_per_file=rows_per_file,
                compression=compression,
                overwrite=overwrite,
            )
            write_filtered_parquet_dataset(
                source_path=positive_source,
                target_path=positive_target,
                timestamp_col=timestamp_col,
                start_time=start_time,
                cutoff_time=cutoff_time,
                rows_per_file=rows_per_file,
                compression=compression,
                overwrite=overwrite,
            )
            paths = DatasetPaths(
                pointwise=pointwise_target,
                positive_interactions=positive_target,
            )

        window_paths[window.name] = paths

        pointwise_rows = int(ds.dataset(str(paths.pointwise), format="parquet").count_rows())
        positive_rows = int(
            ds.dataset(str(paths.positive_interactions), format="parquet").count_rows()
        )
        pointwise_bytes = parquet_storage_bytes(paths.pointwise)
        positive_bytes = parquet_storage_bytes(paths.positive_interactions)

        stat_rows.append(
            {
                "window": window.name,
                "days": window.days if window.days is not None else "all",
                "start_timestamp": (
                    start_time.isoformat(sep=" ") if start_time is not None else "all"
                ),
                "cutoff_timestamp": cutoff_time.isoformat(sep=" "),
                "pointwise_rows": pointwise_rows,
                "positive_rows": positive_rows,
                "pointwise_retention_pct": 100.0
                * pointwise_rows
                / max(source_rows["pointwise"], 1),
                "positive_retention_pct": 100.0
                * positive_rows
                / max(source_rows["positive_interactions"], 1),
                "pointwise_storage_mb": pointwise_bytes / (1024**2),
                "positive_storage_mb": positive_bytes / (1024**2),
                "pointwise_storage_reduction_pct": 100.0
                * (1.0 - pointwise_bytes / max(normalized_all_storage["pointwise"], 1)),
                "positive_storage_reduction_pct": 100.0
                * (1.0 - positive_bytes / max(normalized_all_storage["positive_interactions"], 1)),
                "is_materialized_copy": True,
                "storage_baseline": "normalized_all",
            }
        )

    stats_df = pd.DataFrame(stat_rows)
    layout.reports_dir.mkdir(parents=True, exist_ok=True)
    stats_path = layout.reports_dir / "dataset_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"[prepare] Dataset stats: {stats_path}")
    return window_paths, stats_df, cutoff_time


def deep_copy_config(data: dict[str, Any]) -> dict[str, Any]:
    """Copy config bằng round-trip YAML để tránh sửa object gốc."""
    return yaml.safe_load(yaml.safe_dump(data, sort_keys=False))


def build_model_config(
    project_root: Path,
    master_config: dict[str, Any],
    layout: ExperimentLayout,
    model_key: str,
    window_name: str,
    dataset_paths: DatasetPaths,
) -> tuple[dict[str, Any], Path]:
    """Sinh config train/evaluate từ config baseline final hiện tại."""
    models_cfg = master_config.get("models", {})
    if model_key not in models_cfg:
        raise KeyError(f"Không có model '{model_key}' trong master config.")

    model_entry = models_cfg[model_key]
    base_config_path = resolve_project_path(project_root, model_entry["base_config"])
    base_config = deep_copy_config(load_yaml(base_config_path))

    if model_key == "bpr":
        train_path = dataset_paths.pointwise
        model_slug = "bpr"
    elif model_key == "content_based":
        train_path = dataset_paths.positive_interactions
        model_slug = "content"
    else:
        raise ValueError(f"Model chưa hỗ trợ trong Data Minimization: {model_key}")

    experiment_name = f"dm_{model_slug}_{window_name}"
    base_config["experiment_name"] = experiment_name
    base_config.setdefault("dataset", {})["train_interactions"] = to_project_relative(
        project_root,
        train_path,
    )

    # Dùng chung full dev và catalog từ baseline config.
    base_config.setdefault("paths", {})["artifacts_dir"] = to_project_relative(
        project_root,
        layout.artifacts_dir,
    )
    base_config["paths"]["reports_dir"] = to_project_relative(
        project_root,
        layout.reports_dir,
    )

    if model_key == "content_based":
        # Trong thí nghiệm privacy, profile chỉ được phép dùng retained train history.
        use_impression_history = bool(model_entry.get("use_impression_history", False))
        base_config.setdefault("evaluation", {})["use_impression_history"] = use_impression_history

    config_path = layout.generated_config_dir / f"{experiment_name}.yaml"
    save_yaml(config_path, base_config)
    return base_config, config_path


def expected_checkpoint_path(
    project_root: Path,
    generated_config: dict[str, Any],
) -> Path:
    """Tính checkpoint path giống src.train."""
    artifacts_dir = resolve_project_path(
        project_root,
        generated_config.get("paths", {}).get("artifacts_dir", "artifacts"),
    )
    experiment_name = generated_config["experiment_name"]
    model_name = generated_config["model"]["name"]
    return artifacts_dir / "checkpoints" / f"{experiment_name}_{model_name}.pkl"


def expected_metrics_path(
    project_root: Path,
    generated_config: dict[str, Any],
) -> Path:
    """Tính JSON metrics path giống src.evaluate."""
    reports_dir = resolve_project_path(
        project_root,
        generated_config.get("paths", {}).get("reports_dir", "reports"),
    )
    return reports_dir / "metrics" / f"{generated_config['experiment_name']}_dev.json"


def run_command(command: list[str], cwd: Path) -> float:
    """Chạy subprocess trực tiếp ra terminal và trả wall-clock seconds."""
    print("[run] " + " ".join(command))
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, check=True)
    return time.perf_counter() - start


def load_overall_metrics(metrics_path: Path) -> dict[str, float]:
    """Đọc ba metric chính từ JSON evaluator."""
    with metrics_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    overall = payload.get("slices", {}).get("overall", {}).get("metrics", {})
    return {
        "NDCG@10": float(overall.get("NDCG@10", 0.0)),
        "Recall@10": float(overall.get("Recall@10", 0.0)),
        "MRR@10": float(overall.get("MRR@10", 0.0)),
    }


def run_one_model_window(
    project_root: Path,
    master_config: dict[str, Any],
    layout: ExperimentLayout,
    model_key: str,
    window_name: str,
    dataset_paths: DatasetPaths,
    overwrite_models: bool,
    evaluation_sample_size: int | None,
) -> dict[str, Any]:
    """Train + evaluate một model/window và lưu timing để có thể resume."""
    generated_config, config_path = build_model_config(
        project_root=project_root,
        master_config=master_config,
        layout=layout,
        model_key=model_key,
        window_name=window_name,
        dataset_paths=dataset_paths,
    )
    checkpoint_path = expected_checkpoint_path(project_root, generated_config)
    metrics_path = expected_metrics_path(project_root, generated_config)
    metadata_path = layout.run_metadata_dir / f"{model_key}_{window_name}.json"

    metadata: dict[str, Any] = {}
    if metadata_path.is_file() and not overwrite_models:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

    if overwrite_models and checkpoint_path.exists():
        checkpoint_path.unlink()
    if overwrite_models and metrics_path.exists():
        metrics_path.unlink()

    has_train_timing = metadata.get("train_wall_seconds") is not None
    if checkpoint_path.exists() and not overwrite_models and has_train_timing:
        print(f"[skip train] Đã có checkpoint + timing: {checkpoint_path}")
        train_seconds = metadata.get("train_wall_seconds")
    else:
        train_seconds = run_command(
            [sys.executable, "-m", "src.train", "--config", str(config_path)],
            cwd=project_root,
        )

    same_evaluation_scope = (
        bool(metadata)
        and metadata.get("eval_wall_seconds") is not None
        and metadata.get("evaluation_sample_size") == evaluation_sample_size
    )
    if metrics_path.exists() and not overwrite_models and same_evaluation_scope:
        print(f"[skip eval] Đã có metrics cùng evaluation scope: {metrics_path}")
        eval_seconds = metadata.get("eval_wall_seconds")
    else:
        command = [sys.executable, "-m", "src.evaluate", "--config", str(config_path)]
        if evaluation_sample_size is not None:
            command.extend(["--sample-size", str(evaluation_sample_size)])
        eval_seconds = run_command(command, cwd=project_root)

    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluate không tạo metrics JSON: {metrics_path}")

    metrics = load_overall_metrics(metrics_path)
    metadata = {
        "model": model_key,
        "window": window_name,
        "config_path": to_project_relative(project_root, config_path),
        "checkpoint_path": to_project_relative(project_root, checkpoint_path),
        "metrics_path": to_project_relative(project_root, metrics_path),
        "train_wall_seconds": train_seconds,
        "eval_wall_seconds": eval_seconds,
        "evaluation_sample_size": evaluation_sample_size,
        **metrics,
    }
    save_json(metadata_path, metadata)
    return metadata


def build_summary(
    stats_df: pd.DataFrame,
    run_rows: list[dict[str, Any]],
    layout: ExperimentLayout,
) -> pd.DataFrame:
    """Ghép dataset stats với kết quả model thành bảng long-format."""
    stats_lookup = stats_df.set_index("window").to_dict(orient="index")
    summary_rows: list[dict[str, Any]] = []

    for run in run_rows:
        window = str(run["window"])
        model = str(run["model"])
        stats = stats_lookup[window]

        if model == "bpr":
            train_rows = int(stats["pointwise_rows"])
            retention_pct = float(stats["pointwise_retention_pct"])
            storage_mb = float(stats["pointwise_storage_mb"])
            storage_reduction_pct = float(stats["pointwise_storage_reduction_pct"])
        else:
            train_rows = int(stats["positive_rows"])
            retention_pct = float(stats["positive_retention_pct"])
            storage_mb = float(stats["positive_storage_mb"])
            storage_reduction_pct = float(stats["positive_storage_reduction_pct"])

        summary_rows.append(
            {
                "window": window,
                "days": stats["days"],
                "model": model,
                "train_rows": train_rows,
                "data_retained_pct": retention_pct,
                "training_storage_mb": storage_mb,
                "storage_reduction_pct": storage_reduction_pct,
                "train_wall_seconds": run.get("train_wall_seconds"),
                "eval_wall_seconds": run.get("eval_wall_seconds"),
                "NDCG@10": run["NDCG@10"],
                "Recall@10": run["Recall@10"],
                "MRR@10": run["MRR@10"],
                "checkpoint_path": run["checkpoint_path"],
                "metrics_path": run["metrics_path"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    window_order = {name: index for index, name in enumerate(stats_df["window"].tolist())}
    summary_df["_window_order"] = summary_df["window"].map(window_order)
    summary_df = summary_df.sort_values(["model", "_window_order"]).drop(columns=["_window_order"])
    summary_path = layout.reports_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}")
    return summary_df


def plot_metric_tradeoff(
    summary_df: pd.DataFrame,
    x_column: str,
    x_label: str,
    output_path: Path,
) -> None:
    """Vẽ NDCG@10 theo một trục trade-off cho từng model."""
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    for model, group in summary_df.groupby("model", sort=False):
        valid = group.dropna(subset=[x_column, "NDCG@10"]).sort_values(x_column)
        if valid.empty:
            continue
        axis.plot(
            valid[x_column],
            valid["NDCG@10"],
            marker="o",
            label=model,
        )
        for _, row in valid.iterrows():
            axis.annotate(
                str(row["window"]),
                (row[x_column], row["NDCG@10"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xlabel(x_label)
    axis.set_ylabel("NDCG@10")
    axis.set_title("Data Minimization Trade-off")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def create_figures(summary_df: pd.DataFrame, layout: ExperimentLayout) -> None:
    """Sinh ba biểu đồ chính cho báo cáo Data Minimization."""
    if summary_df.empty:
        return
    plot_metric_tradeoff(
        summary_df,
        x_column="data_retained_pct",
        x_label="Training data retained (%)",
        output_path=layout.figures_dir / "ndcg_vs_data_retained.png",
    )
    plot_metric_tradeoff(
        summary_df,
        x_column="training_storage_mb",
        x_label="Training data storage (MB)",
        output_path=layout.figures_dir / "ndcg_vs_storage.png",
    )
    if summary_df["train_wall_seconds"].notna().any():
        plot_metric_tradeoff(
            summary_df,
            x_column="train_wall_seconds",
            x_label="Training wall time (seconds)",
            output_path=layout.figures_dir / "ndcg_vs_training_time.png",
        )
    print(f"[figures] {layout.figures_dir}")


def validate_requested_names(
    all_windows: list[WindowSpec],
    selected_windows: list[str] | None,
    selected_models: list[str] | None,
) -> tuple[set[str] | None, list[str]]:
    """Validate CLI selection và trả set window + danh sách model."""
    valid_windows = {window.name for window in all_windows}
    if selected_windows:
        unknown = set(selected_windows) - valid_windows
        if unknown:
            raise ValueError(f"Window không tồn tại: {sorted(unknown)}")
        window_set: set[str] | None = set(selected_windows)
    else:
        window_set = None

    valid_models = {"bpr", "content_based"}
    models = selected_models or ["bpr", "content_based"]
    unknown_models = set(models) - valid_models
    if unknown_models:
        raise ValueError(f"Model không hỗ trợ: {sorted(unknown_models)}")
    return window_set, models


def run_experiment(args: argparse.Namespace) -> None:
    """Điều phối prepare, train, evaluate, summary và figures."""
    project_root = Path(args.project_root).resolve()
    config_path = resolve_project_path(project_root, args.config)
    config = load_yaml(config_path)
    layout = build_layout(project_root, config)
    windows = parse_windows(config)
    selected_window_set, selected_models = validate_requested_names(
        windows,
        args.windows,
        args.models,
    )

    window_paths, stats_df, cutoff_time = prepare_retention_datasets(
        project_root=project_root,
        config=config,
        layout=layout,
        selected_windows=selected_window_set,
        overwrite=args.overwrite_data,
    )
    print(f"[info] cutoff timestamp = {cutoff_time}")

    if args.prepare_only:
        return

    run_rows: list[dict[str, Any]] = []
    for window in windows:
        if window.name not in window_paths:
            continue
        for model_key in selected_models:
            model_cfg = config.get("models", {}).get(model_key, {})
            if not bool(model_cfg.get("enabled", True)):
                continue
            print("=" * 72)
            print(f"[experiment] model={model_key} window={window.name}")
            run_rows.append(
                run_one_model_window(
                    project_root=project_root,
                    master_config=config,
                    layout=layout,
                    model_key=model_key,
                    window_name=window.name,
                    dataset_paths=window_paths[window.name],
                    overwrite_models=args.overwrite_models,
                    evaluation_sample_size=args.eval_sample_size,
                )
            )

    summary_df = build_summary(stats_df, run_rows, layout)
    create_figures(summary_df, layout)


def build_parser() -> argparse.ArgumentParser:
    """Tạo CLI cho thí nghiệm Data Minimization."""
    parser = argparse.ArgumentParser(description="MIND Data Minimization experiment: 1d/3d/5d/all")
    parser.add_argument(
        "--config",
        default="configs/experiment/data_minimization.yaml",
        help="Master YAML của thí nghiệm.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root; mặc định là thư mục hiện tại.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Chỉ tạo dataset retention và dataset_stats.csv.",
    )
    parser.add_argument(
        "--windows",
        nargs="+",
        default=None,
        help="Chỉ chạy một số window, ví dụ: --windows 1d 3d.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=["bpr", "content_based"],
        help="Chỉ chạy model được chọn.",
    )
    parser.add_argument(
        "--overwrite-data",
        action="store_true",
        help="Tạo lại dữ liệu 1d/3d/5d kể cả khi đã có manifest.",
    )
    parser.add_argument(
        "--overwrite-models",
        action="store_true",
        help="Train/evaluate lại checkpoint và metrics đã tồn tại.",
    )
    parser.add_argument(
        "--eval-sample-size",
        type=int,
        default=None,
        help="Giới hạn số dev impression để smoke test; final để trống.",
    )
    return parser


def main() -> None:
    """Entrypoint CLI."""
    parser = build_parser()
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
