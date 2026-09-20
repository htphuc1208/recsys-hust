import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from src.evaluation.ranking_metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from src.models.matrix_factorization.model import MatrixFactorization


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEV_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind_large"
    / "dev"
    / "impressions"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "matrix_factorization"
)

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
LEGACY_BEST_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "mf_best.pt"
)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> MatrixFactorization:
    """Tải mô hình từ checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model = MatrixFactorization(
        num_users=int(checkpoint["num_users"]),
        num_items=int(checkpoint["num_items"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        use_bias=bool(checkpoint.get("use_bias", True)),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def get_parquet_files(data_dir: Path) -> list[Path]:
    """Lấy danh sách các file Parquet."""
    files = sorted(data_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(
            f"Không tìm thấy dữ liệu trong: {data_dir}"
        )

    return files


def evaluate_candidate_group(
    scores: np.ndarray,
    item_ids: list[int],
    labels: list[int],
) -> dict[str, float]:
    """Tính các ranking metric cho một impression."""
    order = np.argsort(-scores)

    ranked_items = [
        item_ids[index]
        for index in order
    ]

    positive_items = {
        item_id
        for item_id, label in zip(
            item_ids,
            labels,
        )
        if label == 1
    }

    return {
        "mrr": mrr_at_k(
            ranked_items,
            positive_items,
            len(ranked_items),
        ),
        "hit_rate_5": hit_rate_at_k(
            ranked_items,
            positive_items,
            5,
        ),
        "hit_rate_10": hit_rate_at_k(
            ranked_items,
            positive_items,
            10,
        ),
        "recall_5": recall_at_k(
            ranked_items,
            positive_items,
            5,
        ),
        "recall_10": recall_at_k(
            ranked_items,
            positive_items,
            10,
        ),
        "ndcg_5": ndcg_at_k(
            ranked_items,
            positive_items,
            5,
        ),
        "ndcg_10": ndcg_at_k(
            ranked_items,
            positive_items,
            10,
        ),
    }


def evaluate_model(
    model: MatrixFactorization,
    data_dir: Path,
    device: torch.device,
    max_impressions: int | None = None,
    verbose: bool = True,
    impression_batch_size: int = 1000,
) -> dict[str, int | float]:
    """Đánh giá warm-start theo từng impression có nhãn."""
    model.eval()

    total_impressions = 0
    evaluated_impressions = 0
    skipped_unknown_user = 0
    skipped_no_positive = 0

    mrr_scores = []
    hit_rate_5_scores = []
    hit_rate_10_scores = []
    recall_5_scores = []
    recall_10_scores = []
    ndcg_5_scores = []
    ndcg_10_scores = []

    columns = [
        "user_idx",
        "is_known_user",
        "candidate_item_idxs",
        "candidate_seen_in_train_mask",
        "labels",
    ]

    with torch.no_grad():
        stop_evaluation = False

        for file_path in get_parquet_files(data_dir):
            parquet_file = pq.ParquetFile(file_path)

            for batch in parquet_file.iter_batches(
                batch_size=impression_batch_size,
                columns=columns,
            ):
                data = batch.to_pydict()

                # Gom nhiều impression để chấm điểm cùng một lần.
                flat_user_ids = []
                flat_item_ids = []
                groups = []

                for row_index in range(batch.num_rows):
                    if (
                        max_impressions is not None
                        and total_impressions >= max_impressions
                    ):
                        stop_evaluation = True
                        break

                    total_impressions += 1

                    user_idx = data["user_idx"][row_index]
                    is_known_user = data["is_known_user"][row_index]

                    # MF thuần chỉ đánh giá user đã xuất hiện trong train.
                    if not is_known_user or user_idx < 0:
                        skipped_unknown_user += 1
                        continue

                    item_ids = data["candidate_item_idxs"][row_index]
                    seen_mask = data[
                        "candidate_seen_in_train_mask"
                    ][row_index]
                    labels = data["labels"][row_index]

                    if labels is None:
                        raise ValueError(
                            "Tập đánh giá không có labels. "
                            "MIND test chính thức không thể tính MRR/NDCG trực tiếp."
                        )

                    valid_items = []
                    valid_labels = []

                    # MF thuần chỉ đánh giá item đã xuất hiện trong train.
                    for item_id, seen, label in zip(
                        item_ids,
                        seen_mask,
                        labels,
                    ):
                        if seen and item_id >= 0:
                            valid_items.append(item_id)
                            valid_labels.append(label)

                    # Sau khi bỏ item cold-start phải còn ít nhất một positive.
                    if not any(label == 1 for label in valid_labels):
                        skipped_no_positive += 1
                        continue

                    start = len(flat_item_ids)
                    flat_user_ids.extend(
                        [user_idx] * len(valid_items)
                    )
                    flat_item_ids.extend(valid_items)
                    end = len(flat_item_ids)

                    groups.append(
                        (
                            start,
                            end,
                            valid_items,
                            valid_labels,
                        )
                    )

                if flat_item_ids:
                    user_tensor = torch.tensor(
                        flat_user_ids,
                        dtype=torch.long,
                        device=device,
                    )

                    item_tensor = torch.tensor(
                        flat_item_ids,
                        dtype=torch.long,
                        device=device,
                    )

                    all_scores = model(
                        user_tensor,
                        item_tensor,
                    ).cpu().numpy()

                    for (
                        start,
                        end,
                        valid_items,
                        valid_labels,
                    ) in groups:
                        group_metrics = evaluate_candidate_group(
                            all_scores[start:end],
                            valid_items,
                            valid_labels,
                        )

                        mrr_scores.append(group_metrics["mrr"])
                        hit_rate_5_scores.append(
                            group_metrics["hit_rate_5"]
                        )
                        hit_rate_10_scores.append(
                            group_metrics["hit_rate_10"]
                        )
                        recall_5_scores.append(
                            group_metrics["recall_5"]
                        )
                        recall_10_scores.append(
                            group_metrics["recall_10"]
                        )
                        ndcg_5_scores.append(
                            group_metrics["ndcg_5"]
                        )
                        ndcg_10_scores.append(
                            group_metrics["ndcg_10"]
                        )
                        evaluated_impressions += 1

                if stop_evaluation:
                    break

            if stop_evaluation:
                break

    if evaluated_impressions == 0:
        raise RuntimeError(
            "Không có impression hợp lệ để đánh giá."
        )

    results: dict[str, int | float] = {
        "total_impressions": total_impressions,
        "evaluated_impressions": evaluated_impressions,
        "skipped_unknown_user": skipped_unknown_user,
        "skipped_no_positive": skipped_no_positive,
        "mrr": float(np.mean(mrr_scores)),
        "hit_rate_5": float(np.mean(hit_rate_5_scores)),
        "hit_rate_10": float(np.mean(hit_rate_10_scores)),
        "recall_5": float(np.mean(recall_5_scores)),
        "recall_10": float(np.mean(recall_10_scores)),
        "ndcg_5": float(np.mean(ndcg_5_scores)),
        "ndcg_10": float(np.mean(ndcg_10_scores)),
    }

    if verbose:
        print()
        print("===== KẾT QUẢ ĐÁNH GIÁ WARM-START =====")
        print(
            f"Số impression đã đọc: "
            f"{results['total_impressions']}"
        )
        print(
            f"Số impression được đánh giá: "
            f"{results['evaluated_impressions']}"
        )
        print(
            f"Bỏ qua do user mới: "
            f"{results['skipped_unknown_user']}"
        )
        print(
            "Bỏ qua do không còn positive đã thấy trong train: "
            f"{results['skipped_no_positive']}"
        )
        print(f"MRR: {results['mrr']:.6f}")
        print(f"HitRate@5: {results['hit_rate_5']:.6f}")
        print(f"HitRate@10: {results['hit_rate_10']:.6f}")
        print(f"Recall@5: {results['recall_5']:.6f}")
        print(f"Recall@10: {results['recall_10']:.6f}")
        print(f"NDCG@5: {results['ndcg_5']:.6f}")
        print(f"NDCG@10: {results['ndcg_10']:.6f}")

    return results


def resolve_default_checkpoint() -> Path:
    """Chọn checkpoint mặc định và hỗ trợ đường dẫn MF cũ."""
    if BEST_CHECKPOINT_PATH.exists():
        return BEST_CHECKPOINT_PATH

    if LEGACY_BEST_CHECKPOINT_PATH.exists():
        return LEGACY_BEST_CHECKPOINT_PATH

    return BEST_CHECKPOINT_PATH


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Đánh giá Matrix Factorization trên MIND dev."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint cần đánh giá. Mặc định dùng best.pt.",
    )
    parser.add_argument(
        "--max-impressions",
        type=int,
        default=None,
        help="Giới hạn số impression; bỏ qua để đánh giá toàn bộ dev.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Số impression đọc mỗi lô khi đánh giá.",
    )
    return parser.parse_args()


def main() -> None:
    """Tải checkpoint và đánh giá trên tập dev."""
    args = parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else resolve_default_checkpoint()
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    print(f"Thiết bị: {device}")
    print(f"Tải mô hình từ: {checkpoint_path}")

    model = load_model(
        checkpoint_path,
        device,
    )

    evaluate_model(
        model=model,
        data_dir=DEV_DIR,
        device=device,
        max_impressions=args.max_impressions,
        verbose=True,
        impression_batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
