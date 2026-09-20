from pathlib import Path

import pyarrow.parquet as pq
import torch


def get_parquet_files(data_dir: Path) -> list[Path]:
    """Lấy danh sách các file dữ liệu Parquet theo đúng thứ tự."""
    files = sorted(data_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"Không tìm thấy file Parquet trong: {data_dir}")

    return files


def build_bpr_pairs(
    user_idx: int,
    item_ids: list[int],
    labels: list[int],
) -> list[tuple[int, int, int]]:
    """Ghép negative đều cho các positive trong cùng impression."""
    if len(item_ids) != len(labels):
        raise ValueError("item_ids và labels phải có cùng độ dài.")

    if user_idx < 0:
        return []

    positive_items = [
        item_id
        for item_id, label in zip(item_ids, labels, strict=True)
        if label == 1 and item_id >= 0
    ]
    negative_items = [
        item_id
        for item_id, label in zip(item_ids, labels, strict=True)
        if label == 0 and item_id >= 0
    ]

    if not positive_items or not negative_items:
        return []

    pairs = []

    # Mỗi negative được dùng đúng một lần và chia đều cho các positive.
    for negative_index, negative_item in enumerate(negative_items):
        positive_item = positive_items[negative_index % len(positive_items)]
        pairs.append(
            (
                user_idx,
                positive_item,
                negative_item,
            )
        )

    return pairs


def iter_impression_rows(
    data_dir: Path,
    read_batch_size: int = 16384,
):
    """Đọc pointwise và gom các dòng thuộc cùng một impression."""
    if read_batch_size <= 0:
        raise ValueError("read_batch_size phải lớn hơn 0.")

    columns = [
        "impression_id",
        "user_idx",
        "item_idx",
        "label",
    ]

    current_impression_id = None
    current_user_idx = None
    current_item_ids = []
    current_labels = []

    # Không xáo file vì một impression có thể nằm ở ranh giới hai file.
    for file_path in get_parquet_files(data_dir):
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(
            batch_size=read_batch_size,
            columns=columns,
        ):
            data = batch.to_pydict()

            for impression_id, user_idx, item_idx, label in zip(
                data["impression_id"],
                data["user_idx"],
                data["item_idx"],
                data["label"],
                strict=False,
            ):
                if current_impression_id is None:
                    current_impression_id = impression_id
                    current_user_idx = user_idx

                if impression_id != current_impression_id:
                    yield (
                        int(current_user_idx),
                        current_item_ids,
                        current_labels,
                    )

                    current_impression_id = impression_id
                    current_user_idx = user_idx
                    current_item_ids = []
                    current_labels = []

                if user_idx != current_user_idx:
                    raise ValueError("Một impression chứa nhiều user_idx khác nhau.")

                current_item_ids.append(int(item_idx))
                current_labels.append(int(label))

    if current_impression_id is not None:
        yield (
            int(current_user_idx),
            current_item_ids,
            current_labels,
        )


def convert_pairs_to_tensors(
    pairs: list[tuple[int, int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Chuyển danh sách BPR pair thành tensor PyTorch."""
    user_ids = torch.tensor(
        [pair[0] for pair in pairs],
        dtype=torch.long,
    )
    positive_item_ids = torch.tensor(
        [pair[1] for pair in pairs],
        dtype=torch.long,
    )
    negative_item_ids = torch.tensor(
        [pair[2] for pair in pairs],
        dtype=torch.long,
    )

    return (
        user_ids,
        positive_item_ids,
        negative_item_ids,
    )


def shuffle_pairs(
    pairs: list[tuple[int, int, int]],
    generator: torch.Generator,
) -> list[tuple[int, int, int]]:
    """Xáo trộn đồng thời các BPR pair trong bộ đệm."""
    if len(pairs) <= 1:
        return pairs

    order = torch.randperm(
        len(pairs),
        generator=generator,
    ).tolist()

    return [pairs[index] for index in order]


def yield_pair_batches(
    pairs: list[tuple[int, int, int]],
    batch_size: int,
):
    """Chia danh sách BPR pair thành các lô tensor."""
    for start in range(0, len(pairs), batch_size):
        end = min(start + batch_size, len(pairs))
        yield convert_pairs_to_tensors(pairs[start:end])


def iter_bpr_batches(
    data_dir: Path,
    batch_size: int = 4096,
    shuffle: bool = True,
    seed: int = 42,
    shuffle_buffer_size: int = 65536,
    read_batch_size: int = 16384,
):
    """Đọc pointwise, tạo BPR pair và trả dữ liệu theo lô."""
    if batch_size <= 0:
        raise ValueError("batch_size phải lớn hơn 0.")

    if shuffle_buffer_size < batch_size:
        raise ValueError("shuffle_buffer_size phải lớn hơn hoặc bằng batch_size.")

    if read_batch_size <= 0:
        raise ValueError("read_batch_size phải lớn hơn 0.")

    torch_generator = torch.Generator()
    torch_generator.manual_seed(seed)

    pair_buffer: list[tuple[int, int, int]] = []

    for user_idx, item_ids, labels in iter_impression_rows(
        data_dir,
        read_batch_size=read_batch_size,
    ):
        pair_buffer.extend(
            build_bpr_pairs(
                user_idx=user_idx,
                item_ids=item_ids,
                labels=labels,
            )
        )

        target_size = shuffle_buffer_size if shuffle else batch_size

        while len(pair_buffer) >= target_size:
            current_pairs = pair_buffer[:target_size]
            del pair_buffer[:target_size]

            if shuffle:
                current_pairs = shuffle_pairs(
                    current_pairs,
                    torch_generator,
                )

            yield from yield_pair_batches(
                current_pairs,
                batch_size,
            )

    # Xử lý phần dữ liệu còn lại cuối epoch.
    if pair_buffer:
        if shuffle:
            pair_buffer = shuffle_pairs(
                pair_buffer,
                torch_generator,
            )

        yield from yield_pair_batches(
            pair_buffer,
            batch_size,
        )
