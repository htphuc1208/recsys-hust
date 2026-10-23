"""Tạo các ma trận đặc trưng nội dung cho B4 từ MINDlarge đã xử lý."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

NEWS_COLUMNS = [
    "news_id",
    "item_idx",
    "category",
    "subcategory",
    "title",
    "abstract",
    "title_entity_ids",
]
ENTITY_DIMENSION = 100


@dataclass(slots=True)
class NewsRecord:
    """Lưu metadata cần thiết của một bài viết."""

    item_idx: int
    title: str
    abstract: str
    category: str
    subcategory: str
    title_entity_ids: tuple[str, ...]


def clean_text(value: str | None) -> str:
    """Thay dữ liệu thiếu bằng chuỗi rỗng và bỏ khoảng trắng dư."""

    return (value or "").strip()


def iter_parquet_rows(directory: Path, columns: list[str]) -> Iterable[dict]:
    """Đọc từng batch Parquet để không nạp toàn bộ file vào RAM."""

    if not directory.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {directory}")

    import pyarrow.dataset as dataset_api

    dataset = dataset_api.dataset(str(directory), format="parquet")
    missing_columns = set(columns) - set(dataset.schema.names)
    if missing_columns:
        raise ValueError(f"{directory} thiếu cột: {sorted(missing_columns)}")

    for batch in dataset.to_batches(columns=columns, batch_size=4096):
        batch_data = batch.to_pydict()
        row_count = batch.num_rows
        for row_index in range(row_count):
            yield {column: batch_data[column][row_index] for column in columns}


def load_item_mapping(processed_root: Path) -> tuple[dict[str, int], list[str], list[str]]:
    """Đọc item_idx gốc, giữ nguyên thứ tự do dự án đã tạo."""

    mapping = {}
    first_seen_by_index = {}
    mapping_path = processed_root / "mappings" / "items"

    for row in iter_parquet_rows(mapping_path, ["news_id", "item_idx", "first_seen_split"]):
        news_id = row["news_id"]
        item_idx = int(row["item_idx"])
        if news_id in mapping or item_idx in first_seen_by_index:
            raise ValueError("Có news_id hoặc item_idx trùng trong mapping.")
        mapping[news_id] = item_idx
        first_seen_by_index[item_idx] = row["first_seen_split"]

    item_count = len(mapping)
    if set(first_seen_by_index) != set(range(item_count)):
        raise ValueError("item_idx không liên tục từ 0 đến N-1; không tự gán lại ID.")

    news_ids = [""] * item_count
    first_seen = [""] * item_count
    for news_id, item_idx in mapping.items():
        news_ids[item_idx] = news_id
        first_seen[item_idx] = first_seen_by_index[item_idx]

    return mapping, news_ids, first_seen


def load_news_catalog(processed_root: Path, mapping: dict[str, int]) -> list[NewsRecord | None]:
    """Gộp metadata theo thứ tự ưu tiên train, dev, test mà không đổi item_idx."""

    records: list[NewsRecord | None] = [None] * len(mapping)
    for split in ("train", "dev", "test"):
        directory = processed_root / split / "news"
        if not directory.exists():
            if split == "train":
                raise FileNotFoundError(f"Thiếu metadata train: {directory}")
            print(f"Bỏ qua split chưa có: {directory}")
            continue

        added = 0
        for row in iter_parquet_rows(directory, NEWS_COLUMNS):
            news_id = row["news_id"]
            item_idx = int(row["item_idx"])
            if mapping.get(news_id) != item_idx:
                raise ValueError(f"Mapping không khớp cho {news_id}: {item_idx}")
            if records[item_idx] is not None:
                continue

            records[item_idx] = NewsRecord(
                item_idx=item_idx,
                title=clean_text(row["title"]),
                abstract=clean_text(row["abstract"]),
                category=clean_text(row["category"]),
                subcategory=clean_text(row["subcategory"]),
                title_entity_ids=tuple(row["title_entity_ids"] or []),
            )
            added += 1
        print(f"Đã thêm {added:,} bài viết từ {split}.")

    return records


def get_train_records(records: list[NewsRecord | None], first_seen: list[str]) -> list[NewsRecord]:
    """Chỉ chọn catalog train để fit bộ từ vựng và bộ mã hóa."""

    train_records = [
        record
        for item_idx, record in enumerate(records)
        if first_seen[item_idx] == "train" and record is not None
    ]
    expected_count = sum(split == "train" for split in first_seen)
    if len(train_records) != expected_count:
        raise ValueError("Thiếu metadata của một hoặc nhiều bài viết train.")
    return train_records


def build_text_feature(
    records: list[NewsRecord | None],
    train_records: list[NewsRecord],
    field_name: str,
    max_features: int,
    min_document_frequency: int,
    batch_size: int,
) -> tuple[sparse.csr_matrix, TfidfVectorizer]:
    """Fit TF-IDF trên train, biến đổi mọi item theo batch và trả về CSR."""

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_document_frequency,
        strip_accents="unicode",
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_documents = [getattr(record, field_name) for record in train_records]
    vectorizer.fit(train_documents)

    blocks = []
    for start_index in range(0, len(records), batch_size):
        current_records = records[start_index : start_index + batch_size]
        documents = [
            getattr(record, field_name) if record is not None else "" for record in current_records
        ]
        blocks.append(vectorizer.transform(documents))

    matrix = sparse.vstack(blocks, format="csr", dtype=np.float32)
    return matrix, vectorizer


def build_category_feature(
    records: list[NewsRecord | None],
    train_records: list[NewsRecord],
    field_name: str,
) -> tuple[sparse.csr_matrix, OneHotEncoder]:
    """Fit one-hot theo category của train và bỏ qua category mới ở dev/test."""

    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)
    training_values = np.asarray(
        [[getattr(record, field_name)] for record in train_records], dtype=object
    )
    encoder.fit(training_values)

    all_values = np.asarray(
        [[getattr(record, field_name) if record is not None else ""] for record in records],
        dtype=object,
    )
    matrix = encoder.transform(all_values).tocsr()
    return matrix, encoder


def load_train_entity_embeddings(entity_path: Path, needed_ids: set[str]) -> dict[str, np.ndarray]:
    """Chỉ nạp các vector thực thể cần thiết từ file embedding train."""

    if not entity_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy: {entity_path}")

    entity_vectors = {}
    with entity_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 2 and all(part.isdigit() for part in parts):
                continue  # Bỏ qua header nếu file có header.

            entity_id = parts[0]
            if entity_id not in needed_ids or entity_id in entity_vectors:
                continue
            if len(parts) != ENTITY_DIMENSION + 1:
                raise ValueError(
                    f"Vector {entity_id} tại dòng {line_number} không có {ENTITY_DIMENSION} chiều."
                )

            vector = np.asarray([float(value) for value in parts[1:]], dtype=np.float32)
            if not np.isfinite(vector).all():
                raise ValueError(f"Vector {entity_id} chứa NaN/Inf.")
            entity_vectors[entity_id] = vector

            if len(entity_vectors) == len(needed_ids):
                break

    return entity_vectors


def build_title_entity_feature(
    records: list[NewsRecord | None], entity_vectors: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, int]]:
    """Trung bình các vector thực thể hợp lệ, chuẩn hóa L2 theo bài viết."""

    matrix = np.zeros((len(records), ENTITY_DIMENSION), dtype=np.float32)
    articles_with_entities = 0
    articles_with_matched_entities = 0
    entity_mentions = 0
    matched_entity_mentions = 0

    for item_idx, record in enumerate(records):
        if record is None:
            continue
        # Bỏ ID trùng để một thực thể không bị tính nhiều lần trong cùng tiêu đề.
        entity_ids = list(dict.fromkeys(record.title_entity_ids))
        if not entity_ids:
            continue

        articles_with_entities += 1
        entity_mentions += len(entity_ids)
        matched_vectors = [
            entity_vectors[entity_id] for entity_id in entity_ids if entity_id in entity_vectors
        ]
        matched_entity_mentions += len(matched_vectors)
        if not matched_vectors:
            continue

        average_vector = np.mean(matched_vectors, axis=0, dtype=np.float32)
        vector_norm = float(np.linalg.norm(average_vector))
        if vector_norm > 0:
            matrix[item_idx] = average_vector / vector_norm
            articles_with_matched_entities += 1

    statistics = {
        "articles_with_title_entities": articles_with_entities,
        "articles_with_matched_entities": articles_with_matched_entities,
        "unique_entity_mentions": entity_mentions,
        "matched_entity_mentions": matched_entity_mentions,
    }
    return matrix, statistics


def prepare_output_directory(output_directory: Path, overwrite: bool) -> None:
    """Không ghi đè feature cũ nếu chưa được người dùng cho phép."""

    if output_directory.exists() and any(output_directory.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Đã có dữ liệu tại {output_directory}. Dùng --overwrite nếu muốn tạo lại."
        )
    output_directory.mkdir(parents=True, exist_ok=True)


def save_feature(
    output_directory: Path,
    name: str,
    matrix: sparse.csr_matrix,
    transformer: TfidfVectorizer | OneHotEncoder,
) -> None:
    """Lưu ma trận CSR và bộ biến đổi để sử dụng lại khi có bài mới."""

    sparse.save_npz(output_directory / f"{name}.npz", matrix, compressed=True)
    joblib.dump(transformer, output_directory / f"{name}_transformer.joblib")
    print(f"{name}: shape={matrix.shape}, nnz={matrix.nnz:,}")


def build_features(
    processed_root: Path,
    entity_path: Path,
    output_directory: Path,
    title_features: int,
    abstract_features: int,
    min_document_frequency: int,
    batch_size: int,
    overwrite: bool,
) -> None:
    """Tạo đầy đủ các thành phần nội dung và xuất báo cáo thống kê."""

    if title_features <= 0 or abstract_features <= 0 or batch_size <= 0:
        raise ValueError("Các tham số kích thước và batch_size phải lớn hơn 0.")
    if min_document_frequency <= 0:
        raise ValueError("min_df phải lớn hơn 0.")
    prepare_output_directory(output_directory, overwrite)

    mapping, news_ids, first_seen = load_item_mapping(processed_root)
    records = load_news_catalog(processed_root, mapping)
    train_records = get_train_records(records, first_seen)
    print(f"Số item trong mapping: {len(mapping):,}")
    print(f"Số item dùng để fit: {len(train_records):,}")

    feature_settings = [
        ("title_tfidf", "title", title_features),
        ("abstract_tfidf", "abstract", abstract_features),
    ]
    feature_shapes = {}
    for feature_name, field_name, max_features in feature_settings:
        matrix, transformer = build_text_feature(
            records,
            train_records,
            field_name,
            max_features,
            min_document_frequency,
            batch_size,
        )
        save_feature(output_directory, feature_name, matrix, transformer)
        feature_shapes[feature_name] = list(matrix.shape)

    for feature_name, field_name in [
        ("category_onehot", "category"),
        ("subcategory_onehot", "subcategory"),
    ]:
        matrix, transformer = build_category_feature(records, train_records, field_name)
        save_feature(output_directory, feature_name, matrix, transformer)
        feature_shapes[feature_name] = list(matrix.shape)

    needed_ids = {
        entity_id
        for record in records
        if record is not None
        for entity_id in record.title_entity_ids
    }
    print(f"Số thực thể duy nhất cần tra cứu: {len(needed_ids):,}")
    entity_vectors = load_train_entity_embeddings(entity_path, needed_ids)
    print(f"Số thực thể tìm thấy trong file train: {len(entity_vectors):,}")
    entity_matrix, entity_stats = build_title_entity_feature(records, entity_vectors)
    np.save(output_directory / "title_entity_embeddings.npy", entity_matrix)
    feature_shapes["title_entity_embeddings"] = list(entity_matrix.shape)

    # Ghi chỉ mục gốc để dễ đối chiếu, không tạo lại hoặc thay đổi mapping.
    with (output_directory / "item_index.json").open("w", encoding="utf-8") as file:
        json.dump(news_ids, file, ensure_ascii=False)

    metadata = {
        "fit_scope": "train/news only",
        "transform_scope": "train, dev and available test news metadata",
        "entity_source": str(entity_path),
        "entity_dimension": ENTITY_DIMENSION,
        "total_item_mapping": len(mapping),
        "train_items": len(train_records),
        "items_with_metadata": sum(record is not None for record in records),
        "items_without_metadata": sum(record is None for record in records),
        "dev_catalog_cold_items": sum(split == "dev" for split in first_seen),
        "unique_entities_needed": len(needed_ids),
        "unique_entities_found_in_train_embedding": len(entity_vectors),
        "title_entity_statistics": entity_stats,
        "feature_shapes": feature_shapes,
        "title_max_features": title_features,
        "abstract_max_features": abstract_features,
        "min_document_frequency": min_document_frequency,
        "description": (
            "Each feature row uses the original item_idx. Sparse groups are "
            "stored separately; do not add them together without weights. "
            "Unseen categories and unmatched entities produce zero components."
        ),
    }
    with (output_directory / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Số item thiếu metadata: {metadata['items_without_metadata']:,}")
    print(
        "Bài có thực thể khớp embedding: "
        f"{entity_stats['articles_with_matched_entities']:,} / "
        f"{entity_stats['articles_with_title_entities']:,}"
    )
    print(f"Hoàn thành. Đặc trưng đã lưu tại: {output_directory}")


def main() -> None:
    """Đọc tham số từ dòng lệnh và thực hiện bước tạo đặc trưng."""

    parser = argparse.ArgumentParser(description="Xây dựng đặc trưng nội dung B4")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed/mind_large"),
    )
    parser.add_argument(
        "--entity-path",
        type=Path,
        default=Path("data/raw/mind_large/MINDlarge_train/entity_embedding.vec"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/mind_large/features/b4"),
    )
    parser.add_argument("--title-features", type=int, default=20000)
    parser.add_argument("--abstract-features", type=int, default=30000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    build_features(
        processed_root=args.processed_root,
        entity_path=args.entity_path,
        output_directory=args.output_dir,
        title_features=args.title_features,
        abstract_features=args.abstract_features,
        min_document_frequency=args.min_df,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
