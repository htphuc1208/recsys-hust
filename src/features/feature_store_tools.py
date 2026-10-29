"""Tạo bản Feature Store không nén và đo độ trễ nạp feature của Content-Based.

Không thay đổi Feature Store gốc hay checkpoint. Chỉ chạy với checkpoint đáng tin cậy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse


def convert_sparse_storage(source: Path, destination: Path, compressed: bool = False) -> dict:
    """Sao chép Feature Store, chỉ thay cách serialize các tệp .npz.

    Mọi ma trận sau chuyển đổi phải bằng chính xác ma trận nguồn.
    """
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_dir() or not (source / "manifest.json").is_file():
        raise FileNotFoundError(f"Không thấy Feature Store nguồn: {source}")
    if destination == source or destination.is_relative_to(source):
        raise ValueError("Output phải khác và không được nằm bên trong thư mục nguồn.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output đã có dữ liệu, không ghi đè: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    with (source / "manifest.json").open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    total_source_bytes = 0
    total_destination_bytes = 0
    for path in sorted(source.iterdir()):
        if not path.is_file():
            raise ValueError(f"Feature Store có đường dẫn con không phải file: {path}")
        output_path = destination / path.name
        if path.suffix == ".npz":
            original = sparse.load_npz(path).tocsr()
            sparse.save_npz(output_path, original, compressed=compressed)
            restored = sparse.load_npz(output_path).tocsr()
            difference = original != restored
            if original.shape != restored.shape or difference.nnz != 0:
                raise RuntimeError(f"Chuyển đổi không bảo toàn dữ liệu: {path}")
            del original, restored, difference
        else:
            shutil.copy2(path, output_path)
        total_source_bytes += path.stat().st_size
        total_destination_bytes += output_path.stat().st_size

    manifest["sparse_storage"] = {"format": "npz", "compressed": compressed}
    manifest["storage_converted_from"] = str(source)
    with (destination / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    total_destination_bytes = sum(path.stat().st_size for path in destination.iterdir())
    return {
        "source": str(source),
        "destination": str(destination),
        "sparse_compressed": compressed,
        "source_size_mb": round(total_source_bytes / 2**20, 3),
        "destination_size_mb": round(total_destination_bytes / 2**20, 3),
    }


def benchmark_feature_loading(
    checkpoint: Path,
    feature_directories: list[Path],
    repeats: int = 5,
) -> dict:
    """Đo thời gian _ensure_features_loaded và đối chiếu điểm dự đoán.

    Đo warm-cache trong một process; không phải cold-start hay request latency.
    """
    from src.models.base import BaseRecommender
    from src.models.content_based.recommender import ContentBasedRecommender

    if repeats <= 0:
        raise ValueError("repeats phải lớn hơn 0")
    model = BaseRecommender.load(checkpoint)
    if not isinstance(model, ContentBasedRecommender):
        raise TypeError("Checkpoint không phải ContentBasedRecommender")
    if not model.is_fitted:
        raise ValueError("Checkpoint chưa fit")

    original_feature_dir = model.feature_dir
    baseline_scores = None
    results = []
    try:
        for directory in feature_directories:
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            model.feature_dir = Path(directory)
            timings = []
            for _ in range(repeats):
                model._features = {}
                start = perf_counter()
                model._ensure_features_loaded()
                timings.append((perf_counter() - start) * 1000)

            # So sánh 3 ứng viên với một lịch sử xác định để bắt lỗi sai khác khi đổi storage.
            candidates = np.arange(min(3, model.num_items), dtype=np.int64)
            history = [0] if model.num_items > 0 else []
            scores = model.predict(
                user_ids=np.array([0], dtype=np.int64),
                item_ids=candidates,
                history_item_idxs=history,
            )
            if baseline_scores is None:
                baseline_scores = scores.copy()
            elif not np.array_equal(baseline_scores, scores):
                raise ValueError(f"Điểm dự đoán thay đổi sau khi đổi Feature Store: {directory}")

            results.append(
                {
                    "feature_dir": str(directory),
                    "load_times_ms": [round(t, 3) for t in timings],
                    "median_ms": round(statistics.median(timings), 3),
                    "min_ms": round(min(timings), 3),
                    "prediction_identical_to_first": True,
                }
            )
    finally:
        model._features = {}
        model.feature_dir = original_feature_dir
    return {
        "checkpoint": str(checkpoint),
        "repeats": repeats,
        "measurement": "warm-cache, feature load only; checkpoint unpickle excluded",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuyển đổi và benchmark Feature Store")
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="Tạo bản sao .npz không nén")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--compressed", action="store_true", help="Dùng nén .npz nếu cần")

    bench = subparsers.add_parser("benchmark", help="Đo thời gian nạp feature trong model")
    bench.add_argument("--checkpoint", type=Path, required=True)
    bench.add_argument("--feature-dir", action="append", type=Path, required=True)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument(
        "--output", type=Path, default=Path("reports/benchmarks/news_feature_store.json")
    )
    args = parser.parse_args()
    if args.command == "convert":
        result = convert_sparse_storage(args.source, args.output, compressed=args.compressed)
    else:
        result = benchmark_feature_loading(args.checkpoint, args.feature_dir, repeats=args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Đã lưu: {args.output}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
