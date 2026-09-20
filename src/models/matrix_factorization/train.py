"""Matrix Factorization training entrypoint delegating to standard training pipeline."""

import argparse
from pathlib import Path

from src.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "exp03_mind_mf.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện Matrix Factorization")
    parser.add_argument(
        "--config",
        type=str,
        default=str(CONFIG_PATH),
        help="Đường dẫn file cấu hình YAML",
    )
    args, _ = parser.parse_known_args()
    train(args.config)


if __name__ == "__main__":
    main()
