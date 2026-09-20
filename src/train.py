import argparse
from pathlib import Path
import time
import yaml
import pyarrow.dataset as ds
import pandas as pd

from src.common.logger import setup_logger
from src.common.seed import set_seed
from src.models.registry import get_model


def load_interactions(data_path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Load interactions from a Parquet file or directory of Parquet files."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Interaction data path not found: {path}")

    dataset = ds.dataset(str(path), format="parquet")
    available_cols = [col for col in columns if col in dataset.schema.names]
    return dataset.to_table(columns=available_cols).to_pandas()


def train(config_path: str | Path) -> Path:
    """Execute training pipeline given an experiment config file."""
    logger = setup_logger("train")
    logger.info(f"Loading configuration from {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    set_seed(seed)
    exp_name = cfg.get("experiment_name", "unnamed_run")
    logger.info(f"Initialized training run: {exp_name} (seed={seed})")

    dataset_cfg = cfg.get("dataset", {})
    train_path = dataset_cfg.get(
        "train_interactions", "data/processed/mind_large/train/positive_interactions"
    )

    model_cfg = cfg.get("model", {})
    model_name = model_cfg.get("name", "popularity")
    user_col = model_cfg.get("user_col", "user_idx")
    item_col = model_cfg.get("item_col", "item_idx")
    timestamp_col = model_cfg.get("timestamp_col", "timestamp")

    cols_to_load = [user_col, item_col]
    if model_cfg.get("decay_factor") is not None:
        cols_to_load.append(timestamp_col)

    logger.info(f"Loading training data from: {train_path}")
    start_load = time.perf_counter()
    train_df = load_interactions(train_path, columns=cols_to_load)
    load_time = time.perf_counter() - start_load
    logger.info(
        f"Loaded {len(train_df):,} interaction rows in {load_time:.2f}s "
        f"({train_df[user_col].nunique():,} unique users, {train_df[item_col].nunique():,} unique items)"
    )

    model_params = {k: v for k, v in model_cfg.items() if k != "name"}
    model = get_model(model_name, **model_params)
    logger.info(f"Instantiated model architecture: {model.__class__.__name__}")

    logger.info("Starting model fitting...")
    start_fit = time.perf_counter()
    model.fit(train_df)
    fit_time = time.perf_counter() - start_fit
    logger.info(f"Model fitting completed in {fit_time:.2f}s")

    artifacts_dir = Path(cfg.get("paths", {}).get("artifacts_dir", "artifacts")) / "checkpoints"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / f"{exp_name}_{model_name}.pkl"

    logger.info(f"Saving model checkpoint to: {checkpoint_path}")
    model.save(checkpoint_path)
    logger.info("Training pipeline finished successfully.")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="RecSys Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment/exp01_mind_popularity.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
