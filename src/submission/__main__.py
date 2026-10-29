"""CLI: python -m src.submission --config <YAML> --split test"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.submission.runner import (
    load_config,
    resolve_impressions,
    resolve_original_behaviors,
    resolve_output_dir,
    run_submission,
)
from src.submission.validate import validate_partial_submission, validate_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Xuất prediction.txt / prediction.zip chuẩn Microsoft MIND")
    parser.add_argument("--config", required=True, help="YAML thí nghiệm BPR / Content / P3 / Hybrid")
    parser.add_argument("--split", choices=("test", "dev"), default="test")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--impressions", default=None, help="Ghi đè đường dẫn impressions Parquet")
    parser.add_argument("--checkpoint", default=None, help="Ghi đè đường dẫn checkpoint")
    parser.add_argument("--raw-behaviors", default=None, help="TSV gốc để kiểm tra thứ tự")
    parser.add_argument("--require-raw", action="store_true", help="Bắt buộc đối chiếu TSV gốc")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Lấy N mẫu đầu Dev/Test để smoke test; không đủ điều kiện nộp")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Kiểm tra submission đã xuất, không cần checkpoint")
    args = parser.parse_args()
    config = load_config(args.config)
    target = Path(args.output_dir) if args.output_dir else resolve_output_dir(config, args.split, args.limit)
    if args.validate_only:
        input_path = Path(args.impressions) if args.impressions else resolve_impressions(config, args.split)
        original = resolve_original_behaviors(
            split=args.split, raw_behaviors=args.raw_behaviors, require_raw=args.require_raw
        )
        if args.limit is None:
            result = validate_submission(
                target / "prediction.txt", input_path,
                prediction_zip=target / "prediction.zip",
                original_behaviors=original,
                batch_size=args.batch_size,
                split=args.split,
            )
        else:
            result = validate_partial_submission(
                target / "prediction.txt", input_path,
                prediction_zip=target / "prediction.zip",
                original_behaviors=original,
                batch_size=args.batch_size,
                expected_count=args.limit,
                split=args.split,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    result = run_submission(
        config_path=args.config,
        split=args.split,
        output_dir=target,
        checkpoint_path=args.checkpoint,
        impressions_path=args.impressions,
        raw_behaviors=args.raw_behaviors,
        require_raw=args.require_raw,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(f"prediction.txt: {target / 'prediction.txt'}")
    print(f"prediction.zip: {target / 'prediction.zip'}")
    print(f"manifest: {target / 'submission_manifest.json'}")
    print(f"impressions: {result['validation']['impressions']:,}")
    print(f"candidates: {result['validation']['candidates']:,}")
    print(f"submission_ready: {result['submission_ready']}")
    if not result["submission_ready"]:
        print("CHƯA CHỨNG NHẬN nộp chính thức: cần Test đầy đủ và đối chiếu behaviors.tsv gốc.")


if __name__ == "__main__":
    main()
