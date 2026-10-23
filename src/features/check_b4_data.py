"""Kiểm tra metadata MINDlarge trước khi xây dựng mô hình B4."""

from pathlib import Path

import pyarrow.dataset as ds

DATA_ROOT = Path("data/processed/mind_large")

CONTENT_COLUMNS = [
    "news_id",
    "item_idx",
    "seen_in_train",
    "category",
    "subcategory",
    "title",
    "abstract",
    "title_entity_ids",
    "abstract_entity_ids",
]


def load_dataset(directory: Path, required_columns: list[str]):
    """Đọc Parquet và kiểm tra các cột bắt buộc."""

    if not directory.exists():
        raise FileNotFoundError(f"Không tìm thấy: {directory}")

    dataset = ds.dataset(str(directory), format="parquet")

    missing_columns = set(required_columns) - set(dataset.schema.names)

    if missing_columns:
        raise ValueError(f"Thiếu các cột trong {directory}: {missing_columns}")

    return dataset


def load_item_mapping() -> dict:
    """Đọc ánh xạ news_id sang item_idx của toàn bộ danh mục."""

    mapping_path = DATA_ROOT / "mappings" / "items"

    dataset = load_dataset(
        mapping_path,
        ["news_id", "item_idx"],
    )

    table = dataset.to_table(columns=["news_id", "item_idx"])

    mapping = {}

    for row in table.to_pylist():
        mapping[row["news_id"]] = row["item_idx"]

    print(f"Tổng số item trong mapping: {len(mapping):,}")

    return mapping


def inspect_news(split: str, item_mapping: dict):
    """Thống kê metadata và kiểm tra tính nhất quán của item_idx."""

    news_path = DATA_ROOT / split / "news"

    dataset = load_dataset(
        news_path,
        CONTENT_COLUMNS,
    )

    total_rows = 0
    missing_title = 0
    missing_abstract = 0
    missing_category = 0
    missing_subcategory = 0
    invalid_mapping = 0
    cold_items = 0
    articles_with_title_entities = 0
    total_title_entities = 0

    for batch in dataset.to_batches(
        columns=CONTENT_COLUMNS,
        batch_size=4096,
    ):
        rows = batch.to_pylist()

        for row in rows:
            total_rows += 1

            if not (row["title"] or "").strip():
                missing_title += 1

            if not (row["abstract"] or "").strip():
                missing_abstract += 1

            if not (row["category"] or "").strip():
                missing_category += 1

            if not (row["subcategory"] or "").strip():
                missing_subcategory += 1

            if item_mapping.get(row["news_id"]) != row["item_idx"]:
                invalid_mapping += 1

            if not row["seen_in_train"]:
                cold_items += 1

            title_entities = row["title_entity_ids"] or []

            if title_entities:
                articles_with_title_entities += 1

            total_title_entities += len(title_entities)

    print(f"\n===== {split.upper()} NEWS =====")
    print(f"Tổng số dòng: {total_rows:,}")
    print(f"Thiếu title: {missing_title:,}")
    print(f"Thiếu abstract: {missing_abstract:,}")
    print(f"Thiếu category: {missing_category:,}")
    print(f"Thiếu subcategory: {missing_subcategory:,}")
    print(f"Sai item mapping: {invalid_mapping:,}")
    print(f"Không xuất hiện trong train news: {cold_items:,}")
    print(f"Bài có title entities: {articles_with_title_entities:,}")

    print(f"Tổng lượt title entities: {total_title_entities:,}")

    print(f"Tỷ lệ bài có title entities: {articles_with_title_entities / total_rows * 100:.2f}%")

    sample = dataset.head(
        3,
        columns=CONTENT_COLUMNS,
    )

    print("\nMột số bài viết mẫu:")

    for row in sample.to_pylist():
        print(f"\nItem: {row['item_idx']}")
        print(f"News ID: {row['news_id']}")
        print(f"Category: {row['category']}")
        print(f"Subcategory: {row['subcategory']}")
        print(f"Title: {row['title']}")
        print(f"Abstract: {(row['abstract'] or '')[:150]}")
        print(f"Title entities: {row['title_entity_ids']}")
        print(f"Abstract entities: {row['abstract_entity_ids']}")


def inspect_impressions():
    """Kiểm tra cấu trúc lịch sử và ứng viên của tập dev."""

    impressions_path = DATA_ROOT / "dev" / "impressions"

    required_columns = [
        "user_idx",
        "history_item_idxs",
        "candidate_item_idxs",
        "labels",
    ]

    dataset = load_dataset(
        impressions_path,
        required_columns,
    )

    sample = dataset.head(
        1,
        columns=required_columns,
    ).to_pylist()

    if not sample:
        raise ValueError("Tập dev impressions không có dữ liệu.")

    row = sample[0]

    print("\n===== DEV IMPRESSION =====")
    print(f"User: {row['user_idx']}")
    print(f"Số bài trong history: {len(row['history_item_idxs'] or [])}")
    print(f"Số candidate: {len(row['candidate_item_idxs'] or [])}")
    print(f"History mẫu: {(row['history_item_idxs'] or [])[:5]}")
    print(f"Candidate mẫu: {(row['candidate_item_idxs'] or [])[:5]}")
    print(f"Labels mẫu: {(row['labels'] or [])[:5]}")


def main():
    """Chạy toàn bộ các bước kiểm tra dữ liệu B4."""

    item_mapping = load_item_mapping()

    inspect_news("train", item_mapping)
    inspect_news("dev", item_mapping)

    inspect_impressions()

    print("\nHoàn thành kiểm tra dữ liệu B4.")


if __name__ == "__main__":
    main()
