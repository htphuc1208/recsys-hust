"""Chạy thử P3 bằng checkpoint thật, chỉ xuất thứ hạng (không đánh giá)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.models.base import BaseRecommender
from src.reranking.pipeline import RerankingPipeline


def run_preview(config_path: str | Path, sample_size: int, output: str | Path | None = None) -> Path:
    """Xuất danh sách trước/sau P3 từ các impression; không đọc click labels."""
    import pyarrow.dataset as ds

    if sample_size <= 0:
        raise ValueError("sample_size phải lớn hơn 0.")
    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    rerank_cfg = config.get("reranking", {})
    if not rerank_cfg.get("enabled", False):
        raise ValueError("Bật reranking.enabled trong cấu hình P3.")
    checkpoint_value = rerank_cfg.get("base_checkpoint")
    if not checkpoint_value:
        raise ValueError("Cần khai báo reranking.base_checkpoint.")
    checkpoint_path = Path(checkpoint_value)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Chưa có checkpoint mô hình nền: {checkpoint_path}")
    model = BaseRecommender.load(checkpoint_path)
    pipeline = RerankingPipeline.from_config(config)
    impressions_path = Path(config["dataset"]["dev_impressions"])
    dataset = ds.dataset(str(impressions_path), format="parquet")
    columns = ["impression_id", "user_idx", "history_item_idxs", "candidate_item_idxs"]
    if not set(columns).issubset(dataset.schema.names):
        raise ValueError("Impressions thiếu cột cần thiết để chạy P3.")
    output_path = Path(output) if output else Path("reports/reranking") / f"{config['experiment_name']}_preview.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with output_path.open("w", encoding="utf-8") as file:
        for batch in dataset.to_batches(columns=columns, batch_size=256):
            rows = batch.to_pydict()
            for index, candidates in enumerate(rows["candidate_item_idxs"]):
                if not candidates:
                    continue
                baseline, reranked, scores = pipeline.score_and_rerank(
                    model=model,
                    user_idx=int(rows["user_idx"][index]),
                    candidate_item_idxs=candidates,
                    history_item_idxs=rows["history_item_idxs"][index] or [],
                )
                record = {
                    "impression_id": rows["impression_id"][index],
                    "user_idx": int(rows["user_idx"][index]),
                    "candidate_item_idxs": [int(item) for item in candidates],
                    "relevance_scores": scores.tolist(),
                    "baseline_ranking": baseline,
                    "reranked_ranking": reranked,
                }
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                processed += 1
                if processed >= sample_size:
                    break
            if processed >= sample_size:
                break
    print(f"Đã xử lý {processed} impressions. File: {output_path}")
    return output_path


def main() -> None:
    """Đọc tham số CLI để thử P3 trên MINDlarge."""
    parser = argparse.ArgumentParser(description="P3 reranking preview, không tính metrics")
    parser.add_argument("--config", required=True, help="Đường dẫn YAML P3")
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    run_preview(args.config, args.sample_size, args.output)


if __name__ == "__main__":
    main()
