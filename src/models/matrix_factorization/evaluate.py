"""Matrix Factorization evaluation entrypoint delegating to standard evaluation pipeline."""

import argparse
from pathlib import Path

from src.evaluate import evaluate

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "exp03_mind_mf.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá Matrix Factorization")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Đường dẫn file cấu hình YAML",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Số lượng impressions đánh giá",
    )
    args, _ = parser.parse_known_args()
    evaluate(args.config, sample_size=args.sample_size)


if __name__ == "__main__":
    main()
