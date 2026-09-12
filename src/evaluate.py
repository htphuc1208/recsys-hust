import argparse
import yaml
from src.common.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="RecSys Offline Evaluation")
    parser.add_argument("--config", type=str, default="configs/experiment/exp01_baseline.yaml", help="Path to config file")
    args = parser.parse_args()

    logger = setup_logger("evaluate")
    logger.info(f"Loading configuration from {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    logger.info(f"Evaluating top-K: {cfg.get('top_k_list', [5, 10, 20])}")
    logger.info("Evaluation completed.")


if __name__ == "__main__":
    main()
