import argparse
import yaml
from src.common.logger import setup_logger
from src.common.seed import set_seed
from src.models.registry import get_model


def main():
    parser = argparse.ArgumentParser(description="RecSys Training Pipeline")
    parser.add_argument("--config", type=str, default="configs/experiment/exp01_baseline.yaml", help="Path to config file")
    args = parser.parse_args()

    logger = setup_logger("train")
    logger.info(f"Loading configuration from {args.config}")

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    logger.info(f"Initialized training run: {cfg.get('experiment_name', 'unnamed')}")

    model = get_model(cfg.get("model", {}).get("name", "popularity"))
    logger.info(f"Loaded model architecture: {model.__class__.__name__}")
    # Pipeline execution logic will be connected here
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
