"""Sample real dev impressions into JSONL request payloads for the serving benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from src.serving.schemas import RecommendationRequest

MAX_HISTORY = RecommendationRequest.model_fields["history_news_ids"].metadata[0].max_length


def build_payloads(
    impressions_dir: str | Path, output: str | Path, num_payloads: int, top_k: int, seed: int
) -> dict[str, int]:
    table = ds.dataset(impressions_dir, format="parquet").to_table(
        columns=["user_id", "history_news_ids"]
    )
    rng = np.random.default_rng(seed)
    # Uniform over all impressions, so warm/cold users appear at their real dev rates.
    indices = np.sort(
        rng.choice(table.num_rows, size=min(num_payloads, table.num_rows), replace=False)
    )
    rows = table.take(indices).to_pylist()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            history = list(row["history_news_ids"] or [])[-MAX_HISTORY:]
            payload = {"user_id": row["user_id"], "history_news_ids": history, "top_k": top_k}
            stream.write(json.dumps(payload) + "\n")
    return {"impressions": table.num_rows, "payloads": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impressions-dir", default="data/processed/mind_large/dev/impressions")
    parser.add_argument(
        "--output", default="data/processed/mind_large/benchmark/serving_payloads.jsonl"
    )
    parser.add_argument("--num-payloads", type=int, default=2000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_payloads(
        args.impressions_dir, args.output, args.num_payloads, args.top_k, args.seed
    )
    print(json.dumps({**summary, "output": args.output}))


if __name__ == "__main__":
    main()
