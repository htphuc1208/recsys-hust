import random
from pathlib import Path

import pyarrow.parquet as pq
import torch


def get_parquet_files(data_dir: Path) -> list[Path]:
    """Lấy danh sách các file dữ liệu Parquet."""
    files = sorted(data_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"Không tìm thấy file Parquet trong: {data_dir}")

    return files


def convert_batch_to_tensors(batch):
    """Chuyển một lô PyArrow thành tensor PyTorch."""
    user_ids = torch.tensor(
        batch.column("user_idx").to_numpy(zero_copy_only=False),
        dtype=torch.long,
    )

    item_ids = torch.tensor(
        batch.column("item_idx").to_numpy(zero_copy_only=False),
        dtype=torch.long,
    )

    labels = torch.tensor(
        batch.column("label").to_numpy(zero_copy_only=False),
        dtype=torch.float32,
    )

    return user_ids, item_ids, labels


def yield_tensor_batches(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
):
    """Chia tensor thành các lô nhỏ theo batch_size."""
    total_samples = len(labels)

    for start in range(0, total_samples, batch_size):
        end = min(start + batch_size, total_samples)

        yield (
            user_ids[start:end],
            item_ids[start:end],
            labels[start:end],
        )


def shuffle_buffer(
    user_ids: torch.Tensor,
    item_ids: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
):
    """Xáo trộn đồng thời user, item và label trong bộ đệm."""
    order = torch.randperm(
        len(labels),
        generator=generator,
    )

    return (
        user_ids[order],
        item_ids[order],
        labels[order],
    )


def iter_pointwise_batches(
    data_dir: Path,
    batch_size: int = 4096,
    shuffle: bool = True,
    seed: int = 42,
    shuffle_buffer_size: int = 65536,
    read_batch_size: int = 16384,
):
    """Đọc dữ liệu pointwise theo lô và xáo trộn bằng bộ đệm."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    if shuffle_buffer_size < batch_size:
        raise ValueError("shuffle_buffer_size phải lớn hơn hoặc bằng batch_size.")

    files = get_parquet_files(data_dir)

    # Mỗi epoch có seed khác nhau nên thứ tự file cũng khác nhau.
    random_generator = random.Random(seed)

    if shuffle:
        random_generator.shuffle(files)

    # Bộ sinh số ngẫu nhiên dùng để xáo các mẫu trong bộ đệm.
    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)

    columns = [
        "user_idx",
        "item_idx",
        "label",
    ]

    # Khi không xáo trộn, đọc và trả dữ liệu theo đúng thứ tự gốc.
    if not shuffle:
        for file_path in files:
            parquet_file = pq.ParquetFile(file_path)

            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=columns,
            ):
                yield convert_batch_to_tensors(batch)

        return

    # Bộ đệm giúp trộn mẫu giữa nhiều lô đọc liên tiếp.
    buffer_user_parts = []
    buffer_item_parts = []
    buffer_label_parts = []
    buffer_count = 0

    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(
            batch_size=read_batch_size,
            columns=columns,
        ):
            user_ids, item_ids, labels = convert_batch_to_tensors(batch)

            buffer_user_parts.append(user_ids)
            buffer_item_parts.append(item_ids)
            buffer_label_parts.append(labels)
            buffer_count += len(labels)

            if buffer_count < shuffle_buffer_size:
                continue

            buffer_users = torch.cat(buffer_user_parts, dim=0)
            buffer_items = torch.cat(buffer_item_parts, dim=0)
            buffer_labels = torch.cat(buffer_label_parts, dim=0)

            while len(buffer_labels) >= shuffle_buffer_size:
                current_users = buffer_users[:shuffle_buffer_size]
                current_items = buffer_items[:shuffle_buffer_size]
                current_labels = buffer_labels[:shuffle_buffer_size]

                (
                    current_users,
                    current_items,
                    current_labels,
                ) = shuffle_buffer(
                    current_users,
                    current_items,
                    current_labels,
                    torch_generator,
                )

                yield from yield_tensor_batches(
                    current_users,
                    current_items,
                    current_labels,
                    batch_size,
                )

                buffer_users = buffer_users[shuffle_buffer_size:]
                buffer_items = buffer_items[shuffle_buffer_size:]
                buffer_labels = buffer_labels[shuffle_buffer_size:]

            if len(buffer_labels) > 0:
                buffer_user_parts = [buffer_users]
                buffer_item_parts = [buffer_items]
                buffer_label_parts = [buffer_labels]
                buffer_count = len(buffer_labels)
            else:
                buffer_user_parts = []
                buffer_item_parts = []
                buffer_label_parts = []
                buffer_count = 0

    # Xử lý phần dữ liệu còn lại cuối epoch.
    if buffer_count > 0:
        buffer_users = torch.cat(buffer_user_parts, dim=0)
        buffer_items = torch.cat(buffer_item_parts, dim=0)
        buffer_labels = torch.cat(buffer_label_parts, dim=0)

        (
            buffer_users,
            buffer_items,
            buffer_labels,
        ) = shuffle_buffer(
            buffer_users,
            buffer_items,
            buffer_labels,
            torch_generator,
        )

        yield from yield_tensor_batches(
            buffer_users,
            buffer_items,
            buffer_labels,
            batch_size,
        )
