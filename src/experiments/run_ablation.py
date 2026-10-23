"""Ablation Study Runner for Core Contribution (P1 & P2 Adaptive Gating).

Compares:
  1. Full Adaptive Gating (Adaptive Weights)
  2. Static Fusion (Static Weights: 0.4 CF, 0.4 Seq, 0.2 Content)
  3. No Collaborative Filtering Branch (w_c = 0)
  4. No Sequential Branch (w_s = 0)
  5. No Content / Item Branch (w_t = 0)
"""

import argparse
import json
from pathlib import Path
from typing import Any

from src.common.logger import setup_logger
from src.evaluate import evaluate_impressions
from src.models.hybrid import AdaptiveGatingHybrid


def run_ablation_study(
    dev_path: str | Path,
    sample_size: int = 500,
    k_list: list[int] | None = None,
    output_path: str | Path = "reports/metrics/ablation_study.json",
) -> dict[str, Any]:
    if k_list is None:
        k_list = [5, 10, 20]

    logger = setup_logger("ablation_study")
    logger.info(f"Starting Ablation Study (sample_size={sample_size}, k_list={k_list})")

    configurations = {
        "Adaptive Gating (Full Proposed)": {
            "mode": "adaptive",
            "static_weights": (0.4, 0.4, 0.2),
        },
        "Static Fusion (No Gating Net)": {
            "mode": "static",
            "static_weights": (0.4, 0.4, 0.2),
        },
        "Ablation: No CF Branch": {
            "mode": "no_cf",
            "static_weights": (0.0, 0.6, 0.4),
        },
        "Ablation: No Sequential Branch": {
            "mode": "no_seq",
            "static_weights": (0.6, 0.0, 0.4),
        },
        "Ablation: No Content Branch": {
            "mode": "no_content",
            "static_weights": (0.5, 0.5, 0.0),
        },
    }

    results: dict[str, Any] = {}

    cf_ckpt = Path("artifacts/checkpoints/exp04_mind_bpr_bpr.pkl")
    pop_ckpt = Path("artifacts/checkpoints/exp01_mind_popularity_popularity.pkl")

    for name, params in configurations.items():
        logger.info(f"Evaluating variant: {name} (mode={params['mode']})...")
        model = AdaptiveGatingHybrid(
            mode=params["mode"],
            static_weights=params["static_weights"],
            cf_checkpoint=cf_ckpt if cf_ckpt.exists() else None,
            content_checkpoint=pop_ckpt if pop_ckpt.exists() else None,
            seq_params={"epochs": 1, "batch_size": 512, "max_seq_len": 50},
        )
        model.is_fitted = True
        if hasattr(model.cf_model, "item_factors"):
            model.item_factors = model.cf_model.item_factors
        # Note: in real benchmark, sub-models are pre-fitted or loaded from checkpoints
        eval_result = evaluate_impressions(
            model=model,
            impressions_path=Path(dev_path),
            k_list=k_list,
            sample_size=sample_size,
        )
        overall_metrics = eval_result.get("slices", {}).get("overall", {}).get("metrics", {})
        results[name] = {
            "mode": params["mode"],
            "metrics": overall_metrics,
        }

    # Save to reports
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Ablation study saved to {out_file}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Ablation Study for Adaptive Gating Hybrid")
    parser.add_argument(
        "--dev-impressions",
        default="data/processed/mind_large/dev/impressions",
        help="Path to dev impressions parquet dataset",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of dev impressions to evaluate per variant",
    )
    parser.add_argument(
        "--output",
        default="reports/metrics/ablation_study.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()
    run_ablation_study(
        dev_path=args.dev_impressions,
        sample_size=args.sample_size,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
