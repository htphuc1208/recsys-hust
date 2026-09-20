"""BPR training entrypoint delegating to standard training pipeline."""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from src.train import train

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment" / "exp04_mind_bpr.yaml"


def calculate_bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Tính BPR loss từ điểm positive và negative."""
    return -F.logsigmoid(positive_scores - negative_scores).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện BPR")
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
