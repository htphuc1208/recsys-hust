import argparse
import csv
import shutil
from pathlib import Path

import pyarrow.dataset as ds
import torch
import torch.nn.functional as F

from src.common.seed import set_seed
from src.models.bpr.data import iter_bpr_batches
from src.models.bpr.evaluate import evaluate_model
from src.models.bpr.model import BPR


PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind_large"
    / "train"
    / "pointwise"
)

DEV_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind_large"
    / "dev"
    / "impressions"
)

USER_MAPPING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind_large"
    / "mappings"
    / "users"
)

ITEM_MAPPING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "mind_large"
    / "mappings"
    / "items"
)

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "checkpoints"
    / "bpr"
)

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "best.pt"
HISTORY_PATH = CHECKPOINT_DIR / "history.csv"


DEFAULT_EMBEDDING_DIM = 64
DEFAULT_BATCH_SIZE = 4096
DEFAULT_SHUFFLE_BUFFER_SIZE = 65536
DEFAULT_READ_BATCH_SIZE = 16384
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_SEED = 42
DEFAULT_EPOCHS_TO_RUN = 3
DEFAULT_EVAL_BATCH_SIZE = 1000


def count_rows(data_dir: Path) -> int:
    """Đếm số dòng của bộ dữ liệu Parquet."""
    dataset = ds.dataset(
        data_dir,
        format="parquet",
    )
    return dataset.count_rows()


def create_model(
    num_users: int,
    num_items: int,
    embedding_dim: int,
    use_item_bias: bool,
    device: torch.device,
) -> BPR:
    """Tạo mô hình BPR."""
    model = BPR(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        use_item_bias=use_item_bias,
    )
    return model.to(device)


def create_optimizer(
    model: BPR,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Tạo bộ tối ưu Adam."""
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )


def calculate_bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> torch.Tensor:
    """Tính BPR loss từ điểm positive và negative."""
    return -F.logsigmoid(
        positive_scores - negative_scores
    ).mean()


def train_one_epoch(
    model: BPR,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_number: int,
    batch_size: int,
    shuffle_buffer_size: int,
    read_batch_size: int,
    seed: int,
) -> float:
    """Huấn luyện mô hình qua toàn bộ dữ liệu một lần."""
    model.train()

    total_loss = 0.0
    total_batches = 0

    # Epoch 1 dùng seed gốc, epoch 2 dùng seed + 1, ...
    epoch_seed = seed + epoch_number - 1

    batches = iter_bpr_batches(
        TRAIN_DIR,
        batch_size=batch_size,
        shuffle=True,
        seed=epoch_seed,
        shuffle_buffer_size=shuffle_buffer_size,
        read_batch_size=read_batch_size,
    )

    for batch_index, (
        user_ids,
        positive_item_ids,
        negative_item_ids,
    ) in enumerate(
        batches,
        start=1,
    ):
        user_ids = user_ids.to(device)
        positive_item_ids = positive_item_ids.to(device)
        negative_item_ids = negative_item_ids.to(device)

        optimizer.zero_grad()

        positive_scores = model(
            user_ids,
            positive_item_ids,
        )
        negative_scores = model(
            user_ids,
            negative_item_ids,
        )

        loss = calculate_bpr_loss(
            positive_scores,
            negative_scores,
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        if batch_index % 100 == 0:
            average_loss = total_loss / total_batches
            print(
                f"Epoch {epoch_number} | "
                f"Batch {batch_index} | "
                f"Loss trung bình: {average_loss:.6f}"
            )

    if total_batches == 0:
        raise RuntimeError("Không tạo được lô BPR pair nào từ train data.")

    return total_loss / total_batches


def read_optional_float(
    row: dict[str, str],
    key: str,
) -> float | None:
    """Đọc metric tùy chọn để tương thích history cũ."""
    value = row.get(key)

    if value in (None, ""):
        return None

    return float(value)


def read_history_until(
    max_epoch: int,
    history_path: Path,
) -> list[dict]:
    """Đọc lịch sử huấn luyện đến epoch được chọn."""
    if not history_path.exists():
        return []

    rows = []

    with history_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            epoch = int(row["epoch"])

            if epoch <= max_epoch:
                rows.append(
                    {
                        "epoch": epoch,
                        "train_loss": float(row["train_loss"]),
                        "mrr": float(row["mrr"]),
                        "hit_rate_5": read_optional_float(
                            row,
                            "hit_rate_5",
                        ),
                        "hit_rate_10": read_optional_float(
                            row,
                            "hit_rate_10",
                        ),
                        "recall_5": read_optional_float(
                            row,
                            "recall_5",
                        ),
                        "recall_10": read_optional_float(
                            row,
                            "recall_10",
                        ),
                        "ndcg_5": float(row["ndcg_5"]),
                        "ndcg_10": float(row["ndcg_10"]),
                        "total_impressions": int(row["total_impressions"]),
                        "evaluated_impressions": int(
                            row["evaluated_impressions"]
                        ),
                        "skipped_unknown_user": int(
                            row["skipped_unknown_user"]
                        ),
                        "skipped_no_positive": int(
                            row["skipped_no_positive"]
                        ),
                    }
                )

    return rows


def find_history_for_resume(
    resume_checkpoint: Path,
) -> Path | None:
    """Tìm history đi cùng checkpoint resume."""
    candidates = [
        HISTORY_PATH,
        resume_checkpoint.parent / "history.csv",
        resume_checkpoint.parent / "bpr_history.csv",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def save_history(
    history: list[dict],
) -> None:
    """Lưu kết quả từng epoch vào file CSV."""
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "mrr",
        "hit_rate_5",
        "hit_rate_10",
        "recall_5",
        "recall_10",
        "ndcg_5",
        "ndcg_10",
        "total_impressions",
        "evaluated_impressions",
        "skipped_unknown_user",
        "skipped_no_positive",
    ]

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(history)


def get_best_from_history(
    history: list[dict],
) -> tuple[int, float]:
    """Lấy epoch có NDCG@10 cao nhất trong lịch sử."""
    if not history:
        return -1, -1.0

    best_row = max(
        history,
        key=lambda row: row["ndcg_10"],
    )

    return (
        int(best_row["epoch"]),
        float(best_row["ndcg_10"]),
    )


def clear_training_outputs() -> None:
    """Xóa kết quả BPR cũ khi train lại từ đầu."""
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def checkpoint_epoch(path: Path) -> int | None:
    """Đọc số epoch từ tên checkpoint chuẩn hoặc tên cũ."""
    stem = path.stem

    if stem.startswith("epoch_"):
        suffix = stem.removeprefix("epoch_")
    elif stem.startswith("bpr_epoch_"):
        suffix = stem.removeprefix("bpr_epoch_")
    else:
        return None

    try:
        return int(suffix)
    except ValueError:
        return None


def remove_outputs_after_epoch(
    epoch: int,
) -> None:
    """Xóa checkpoint cục bộ nằm sau epoch dùng để resume."""
    if not CHECKPOINT_DIR.exists():
        return

    for checkpoint_path in CHECKPOINT_DIR.glob("*.pt"):
        epoch_number = checkpoint_epoch(checkpoint_path)

        if epoch_number is not None and epoch_number > epoch:
            checkpoint_path.unlink()


def find_epoch_checkpoint(
    epoch: int,
    resume_checkpoint: Path,
) -> Path | None:
    """Tìm file checkpoint của một epoch ở thư mục mới hoặc cũ."""
    candidates = [
        CHECKPOINT_DIR / f"epoch_{epoch:03d}.pt",
        CHECKPOINT_DIR / f"bpr_epoch_{epoch}.pt",
        resume_checkpoint.parent / f"epoch_{epoch:03d}.pt",
        resume_checkpoint.parent / f"bpr_epoch_{epoch}.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def restore_best_checkpoint(
    best_epoch: int,
    resume_checkpoint: Path,
) -> None:
    """Khôi phục best.pt khi chạy tiếp từ checkpoint."""
    if best_epoch < 1:
        return

    source = find_epoch_checkpoint(
        best_epoch,
        resume_checkpoint,
    )

    if source is None:
        print(
            "CẢNH BÁO: biết epoch tốt nhất trước đó nhưng không tìm thấy "
            "file checkpoint tương ứng để tạo best.pt."
        )
        return

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    shutil.copy2(
        source,
        BEST_CHECKPOINT_PATH,
    )


def load_resume_state(
    checkpoint_path: Path,
    device: torch.device,
    num_users: int,
    num_items: int,
) -> tuple[
    dict,
    int,
    int,
    float,
    list[dict],
]:
    """Đọc checkpoint và xác định trạng thái trước khi resume."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if int(checkpoint["num_users"]) != num_users:
        raise ValueError(
            "Số user trong checkpoint không khớp dữ liệu hiện tại."
        )

    if int(checkpoint["num_items"]) != num_items:
        raise ValueError(
            "Số item trong checkpoint không khớp dữ liệu hiện tại."
        )

    resume_epoch = int(checkpoint.get("epoch", 0))

    history_source = find_history_for_resume(
        checkpoint_path
    )

    if history_source is not None:
        history = read_history_until(
            resume_epoch,
            history_source,
        )
    else:
        history = []

    if history:
        best_epoch, best_ndcg_10 = get_best_from_history(
            history
        )
    elif (
        "best_epoch" in checkpoint
        and "best_ndcg_10" in checkpoint
    ):
        # Checkpoint mới lưu cả kết quả tốt nhất trước đó.
        best_epoch = int(checkpoint["best_epoch"])
        best_ndcg_10 = float(checkpoint["best_ndcg_10"])
    elif "ndcg_10" in checkpoint:
        # Tương thích checkpoint cũ: chỉ biết metric của chính epoch resume.
        best_epoch = resume_epoch
        best_ndcg_10 = float(checkpoint["ndcg_10"])
    else:
        best_epoch = -1
        best_ndcg_10 = -1.0

    return (
        checkpoint,
        resume_epoch,
        best_epoch,
        best_ndcg_10,
        history,
    )


def save_epoch_checkpoint(
    model: BPR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    results: dict[str, int | float],
    best_epoch: int,
    best_ndcg_10: float,
    seed: int,
    batch_size: int,
    shuffle_buffer_size: int,
    read_batch_size: int,
) -> Path:
    """Lưu đầy đủ trạng thái để có thể resume."""
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"epoch_{epoch:03d}.pt"
    )

    optimizer_group = optimizer.param_groups[0]

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_users": model.num_users,
            "num_items": model.num_items,
            "embedding_dim": model.embedding_dim,
            "use_item_bias": model.use_item_bias,
            "epoch": epoch,
            "train_loss": train_loss,
            "mrr": float(results["mrr"]),
            "hit_rate_5": float(results["hit_rate_5"]),
            "hit_rate_10": float(results["hit_rate_10"]),
            "recall_5": float(results["recall_5"]),
            "recall_10": float(results["recall_10"]),
            "ndcg_5": float(results["ndcg_5"]),
            "ndcg_10": float(results["ndcg_10"]),
            "best_epoch": best_epoch,
            "best_ndcg_10": best_ndcg_10,
            "learning_rate": float(optimizer_group["lr"]),
            "weight_decay": float(optimizer_group["weight_decay"]),
            "batch_size": batch_size,
            "shuffle_buffer_size": shuffle_buffer_size,
            "read_batch_size": read_batch_size,
            "seed": seed,
        },
        checkpoint_path,
    )

    return checkpoint_path


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện BPR trên MINDlarge."
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint để chạy tiếp. Bỏ qua để train từ đầu.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS_TO_RUN,
        help="Số epoch chạy trong lần gọi hiện tại.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=DEFAULT_EMBEDDING_DIM,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=DEFAULT_SHUFFLE_BUFFER_SIZE,
    )
    parser.add_argument(
        "--read-batch-size",
        type=int,
        default=DEFAULT_READ_BATCH_SIZE,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--eval-max-impressions",
        type=int,
        default=None,
        help="Giới hạn dev; bỏ qua để đánh giá toàn bộ dev.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    parser.add_argument(
        "--no-item-bias",
        action="store_true",
        help="Tắt item bias khi train mới.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Kiểm tra các tham số chính trước khi train."""
    if args.epochs <= 0:
        raise ValueError("--epochs phải lớn hơn 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size phải lớn hơn 0.")

    if args.shuffle_buffer_size < args.batch_size:
        raise ValueError(
            "--shuffle-buffer-size phải >= --batch-size."
        )

    if args.read_batch_size <= 0:
        raise ValueError("--read-batch-size phải lớn hơn 0.")


def main() -> None:
    """Train từ đầu hoặc chạy tiếp từ một checkpoint."""
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    num_users = count_rows(
        USER_MAPPING_DIR
    )
    num_items = count_rows(
        ITEM_MAPPING_DIR
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Số người dùng: {num_users}")
    print(f"Số bài báo: {num_items}")
    print(f"Kích thước lô BPR pair: {args.batch_size}")
    print(
        "Kích thước bộ đệm xáo trộn: "
        f"{args.shuffle_buffer_size}"
    )
    print(f"Thiết bị: {device}")

    if args.resume is None:
        print()
        print("Chế độ: TRAIN TỪ ĐẦU")

        clear_training_outputs()

        model = create_model(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=args.embedding_dim,
            use_item_bias=not args.no_item_bias,
            device=device,
        )

        optimizer = create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        start_epoch = 1
        best_epoch = -1
        best_ndcg_10 = -1.0
        history = []
        training_seed = args.seed

    else:
        resume_path = args.resume.expanduser().resolve()

        print()
        print(f"Chế độ: RESUME từ {resume_path}")

        (
            checkpoint,
            resume_epoch,
            best_epoch,
            best_ndcg_10,
            history,
        ) = load_resume_state(
            checkpoint_path=resume_path,
            device=device,
            num_users=num_users,
            num_items=num_items,
        )

        model = create_model(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=int(checkpoint["embedding_dim"]),
            use_item_bias=bool(
                checkpoint.get("use_item_bias", True)
            ),
            device=device,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer = create_optimizer(
            model=model,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )
            print("Đã tải trạng thái Adam từ checkpoint.")
        else:
            print(
                "CẢNH BÁO: checkpoint cũ không có optimizer_state_dict; "
                "Adam sẽ khởi tạo lại."
            )

        training_seed = int(
            checkpoint.get("seed", args.seed)
        )
        start_epoch = resume_epoch + 1

        CHECKPOINT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )
        remove_outputs_after_epoch(
            resume_epoch
        )
        save_history(
            history
        )
        restore_best_checkpoint(
            best_epoch,
            resume_path,
        )

        print(f"Sẽ chạy tiếp từ epoch {start_epoch}.")

        if best_epoch >= 1:
            print(
                f"Epoch tốt nhất trước khi resume: {best_epoch}"
            )
            print(
                "NDCG@10 tốt nhất trước khi resume: "
                f"{best_ndcg_10:.6f}"
            )

    end_epoch = start_epoch + args.epochs - 1

    print(
        f"Epoch sẽ chạy lần này: {start_epoch} -> {end_epoch}"
    )

    for epoch_number in range(
        start_epoch,
        end_epoch + 1,
    ):
        print()
        print(
            f"========== EPOCH {epoch_number} =========="
        )

        train_loss = train_one_epoch(
            model=model,
            optimizer=optimizer,
            device=device,
            epoch_number=epoch_number,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            read_batch_size=args.read_batch_size,
            seed=training_seed,
        )

        print()
        print(f"Loss train: {train_loss:.6f}")
        print()
        print("Đang đánh giá tập dev...")

        results = evaluate_model(
            model=model,
            data_dir=DEV_DIR,
            device=device,
            max_impressions=args.eval_max_impressions,
            verbose=True,
            impression_batch_size=args.eval_batch_size,
        )

        current_ndcg_10 = float(
            results["ndcg_10"]
        )

        is_new_best = (
            current_ndcg_10 > best_ndcg_10
        )

        if is_new_best:
            best_ndcg_10 = current_ndcg_10
            best_epoch = epoch_number

        epoch_checkpoint_path = save_epoch_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_number,
            train_loss=train_loss,
            results=results,
            best_epoch=best_epoch,
            best_ndcg_10=best_ndcg_10,
            seed=training_seed,
            batch_size=args.batch_size,
            shuffle_buffer_size=args.shuffle_buffer_size,
            read_batch_size=args.read_batch_size,
        )

        print(
            f"Đã lưu epoch {epoch_number}: "
            f"{epoch_checkpoint_path}"
        )

        history_row = {
            "epoch": epoch_number,
            "train_loss": train_loss,
            "mrr": results["mrr"],
            "hit_rate_5": results["hit_rate_5"],
            "hit_rate_10": results["hit_rate_10"],
            "recall_5": results["recall_5"],
            "recall_10": results["recall_10"],
            "ndcg_5": results["ndcg_5"],
            "ndcg_10": results["ndcg_10"],
            "total_impressions": results[
                "total_impressions"
            ],
            "evaluated_impressions": results[
                "evaluated_impressions"
            ],
            "skipped_unknown_user": results[
                "skipped_unknown_user"
            ],
            "skipped_no_positive": results[
                "skipped_no_positive"
            ],
        }

        history.append(
            history_row
        )
        save_history(
            history
        )

        if is_new_best:
            shutil.copy2(
                epoch_checkpoint_path,
                BEST_CHECKPOINT_PATH,
            )
            print()
            print("Đã cập nhật mô hình tốt nhất.")
            print(
                f"Epoch tốt nhất hiện tại: {best_epoch}"
            )
            print(
                "NDCG@10 tốt nhất hiện tại: "
                f"{best_ndcg_10:.6f}"
            )
        else:
            print()
            print(
                "Epoch hiện tại không tốt hơn mô hình tốt nhất."
            )

    print()
    print("========== HOÀN THÀNH ==========")
    print(f"Epoch cuối đã chạy: {end_epoch}")
    print(f"Epoch tốt nhất: {best_epoch}")
    print(
        f"NDCG@10 tốt nhất: {best_ndcg_10:.6f}"
    )

    if BEST_CHECKPOINT_PATH.exists():
        print(
            f"Mô hình tốt nhất: {BEST_CHECKPOINT_PATH}"
        )
    else:
        print(
            "CẢNH BÁO: chưa có best.pt cục bộ. "
            "Hãy giữ checkpoint của epoch tốt nhất khi resume."
        )

    print(f"Lịch sử huấn luyện: {HISTORY_PATH}")


if __name__ == "__main__":
    main()
