"""Do coverage cua Wikidata entities tren MIND Large, khong thay doi Feature Store.

Chay tu thu muc goc repository:
    python -m src.features.entity_coverage

Cach chia nhom: first_seen_split. Moi bai chi thuoc mot trong train/dev/test,
trong khi 'overall' la tat ca item trong mapping. Metadata duoc uu tien
train -> dev -> test, giong news_content.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.features.news_content import (
    iter_parquet_rows,
    load_item_mapping,
    load_train_entity_embeddings,
)

ENTITY_COLUMNS = ["news_id", "item_idx", "title_entity_ids", "abstract_entity_ids"]
FEATURE_FIELDS = ("title_entity_ids", "abstract_entity_ids")
SPLITS = ("train", "dev", "test")
CSV_COLUMNS = [
    "split",
    "field",
    "articles",
    "articles_with_entity_ids",
    "articles_with_matched_ids",
    "articles_with_valid_embedding",
    "articles_with_ids_but_no_matching_embedding",
    "entity_mentions",
    "matched_entity_mentions",
    "unique_entity_ids",
    "matched_unique_entity_ids",
    "entity_presence_pct",
    "embedding_coverage_pct",
    "match_among_articles_with_entities_pct",
    "mention_match_pct",
]


@dataclass(frozen=True, slots=True)
class ArticleEntities:
    """Hai nhom entity cua cung item_idx."""

    title: tuple[str, ...]
    abstract: tuple[str, ...]


@dataclass(slots=True)
class CoverageCounts:
    articles: int = 0
    articles_with_entity_ids: int = 0
    articles_with_matched_ids: int = 0
    articles_with_valid_embedding: int = 0
    articles_with_ids_but_no_matching_embedding: int = 0
    entity_mentions: int = 0
    matched_entity_mentions: int = 0
    unique_entity_ids: set[str] = field(default_factory=set)
    matched_unique_entity_ids: set[str] = field(default_factory=set)

    def add(self, entity_ids: tuple[str, ...], matched_ids: tuple[str, ...], valid: bool) -> None:
        self.articles += 1
        if not entity_ids:
            return

        self.articles_with_entity_ids += 1
        self.entity_mentions += len(entity_ids)
        self.matched_entity_mentions += len(matched_ids)
        self.unique_entity_ids.update(entity_ids)
        self.matched_unique_entity_ids.update(matched_ids)

        if matched_ids:
            self.articles_with_matched_ids += 1
        else:
            self.articles_with_ids_but_no_matching_embedding += 1
        if valid:
            self.articles_with_valid_embedding += 1

    def to_row(self, split: str, field_name: str) -> dict[str, object]:
        def percent(num: int, den: int) -> float:
            return round(100.0 * num / den, 2) if den else 0.0

        return {
            "split": split,
            "field": field_name,
            "articles": self.articles,
            "articles_with_entity_ids": self.articles_with_entity_ids,
            "articles_with_matched_ids": self.articles_with_matched_ids,
            "articles_with_valid_embedding": self.articles_with_valid_embedding,
            "articles_with_ids_but_no_matching_embedding": (
                self.articles_with_ids_but_no_matching_embedding
            ),
            "entity_mentions": self.entity_mentions,
            "matched_entity_mentions": self.matched_entity_mentions,
            "unique_entity_ids": len(self.unique_entity_ids),
            "matched_unique_entity_ids": len(self.matched_unique_entity_ids),
            "entity_presence_pct": percent(self.articles_with_entity_ids, self.articles),
            "embedding_coverage_pct": percent(
                self.articles_with_valid_embedding, self.articles
            ),
            "match_among_articles_with_entities_pct": percent(
                self.articles_with_matched_ids, self.articles_with_entity_ids
            ),
            "mention_match_pct": percent(
                self.matched_entity_mentions, self.entity_mentions
            ),
        }


COMPLEMENTARITY_COLUMNS = [
    "split",
    "articles",
    "both",
    "title_only",
    "abstract_only",
    "neither",
    "both_pct",
    "title_only_pct",
    "abstract_only_pct",
    "neither_pct",
    "combined_coverage_pct",
    "gain_over_title_pp",
    "gain_over_abstract_pp",
]


@dataclass(slots=True)
class ComplementarityCounts:
    """Thong ke hai nguon embedding theo tung bai bao, tranh cong trung coverage."""

    articles: int = 0
    both: int = 0
    title_only: int = 0
    abstract_only: int = 0
    neither: int = 0

    def add(self, title_valid: bool, abstract_valid: bool) -> None:
        self.articles += 1
        if title_valid and abstract_valid:
            self.both += 1
        elif title_valid:
            self.title_only += 1
        elif abstract_valid:
            self.abstract_only += 1
        else:
            self.neither += 1

    def to_row(self, split: str) -> dict[str, object]:
        def percent(number: int) -> float:
            return round(100 * number / self.articles, 2) if self.articles else 0.0

        return {
            "split": split,
            "articles": self.articles,
            "both": self.both,
            "title_only": self.title_only,
            "abstract_only": self.abstract_only,
            "neither": self.neither,
            "both_pct": percent(self.both),
            "title_only_pct": percent(self.title_only),
            "abstract_only_pct": percent(self.abstract_only),
            "neither_pct": percent(self.neither),
            "combined_coverage_pct": percent(self.articles - self.neither),
            "gain_over_title_pp": percent(self.abstract_only),
            "gain_over_abstract_pp": percent(self.title_only),
        }


def load_entity_catalog(
    processed_root: Path, mapping: dict[str, int]
) -> tuple[list[ArticleEntities | None], list[str]]:
    """Gop metadata theo thu tu uu tien train -> dev -> test; giu item_idx."""
    records: list[ArticleEntities | None] = [None] * len(mapping)
    available_splits: list[str] = []

    for split in SPLITS:
        path = processed_root / split / "news"
        if not path.is_dir():
            if split == "train":
                raise FileNotFoundError(f"Khong tim thay metadata train: {path}")
            continue
        available_splits.append(split)

        added = 0
        for row in iter_parquet_rows(path, ENTITY_COLUMNS):
            news_id = str(row["news_id"])
            item_idx = int(row["item_idx"])
            if mapping.get(news_id) != item_idx:
                raise ValueError(f"item mapping khong khop: {news_id} / {item_idx}")
            if records[item_idx] is not None:
                continue
            records[item_idx] = ArticleEntities(
                title=tuple(row["title_entity_ids"] or []),
                abstract=tuple(row["abstract_entity_ids"] or []),
            )
            added += 1
        print(f"{split}: them {added:,} item metadata")

    return records, available_splits


def is_valid_mean_vector(
    matched_ids: tuple[str, ...], entity_vectors: dict[str, np.ndarray]
) -> bool:
    """Dung cung logic mean + L2 norm nhu build_title_entity_feature."""
    if not matched_ids:
        return False
    mean_vector = np.mean(
        [entity_vectors[entity_id] for entity_id in matched_ids],
        axis=0,
        dtype=np.float32,
    )
    return float(np.linalg.norm(mean_vector)) > 0.0


def analyze_coverage(processed_root: Path, entity_path: Path) -> dict[str, object]:
    """Doc du lieu, thong ke coverage theo first_seen_split va overall."""
    mapping, _, first_seen = load_item_mapping(processed_root)
    records, available_splits = load_entity_catalog(processed_root, mapping)

    needed_ids = {
        entity_id
        for record in records
        if record is not None
        for entity_id in (*record.title, *record.abstract)
    }
    print(f"So entity ID duy nhat can tim: {len(needed_ids):,}")
    entity_vectors = load_train_entity_embeddings(entity_path, needed_ids)
    print(f"So entity ID tim thay trong embedding train: {len(entity_vectors):,}")

    groups: dict[tuple[str, str], CoverageCounts] = {
        (split, field_name): CoverageCounts()
        for split in ("overall", *SPLITS)
        for field_name in ("title", "abstract")
    }
    complementary_groups = {
        split: ComplementarityCounts() for split in ("overall", *SPLITS)
    }
    missing_metadata = 0
    for item_idx, record in enumerate(records):
        if record is None:
            missing_metadata += 1
            continue
        split = first_seen[item_idx]
        if split not in SPLITS:
            raise ValueError(f"Split la: {split!r}, item_idx={item_idx}")

        validity_by_field: dict[str, bool] = {}
        for field_name in ("title", "abstract"):
            # Giong builder hien tai: moi Wikidata ID chi tinh mot lan / bai.
            entity_ids = tuple(dict.fromkeys(getattr(record, field_name)))
            matched_ids = tuple(
                entity_id for entity_id in entity_ids if entity_id in entity_vectors
            )
            valid = is_valid_mean_vector(matched_ids, entity_vectors)
            validity_by_field[field_name] = valid
            groups[("overall", field_name)].add(entity_ids, matched_ids, valid)
            groups[(split, field_name)].add(entity_ids, matched_ids, valid)

        title_valid = validity_by_field["title"]
        abstract_valid = validity_by_field["abstract"]
        complementary_groups["overall"].add(title_valid, abstract_valid)
        complementary_groups[split].add(title_valid, abstract_valid)

    rows = [
        groups[(split, field_name)].to_row(split, field_name)
        for split in ("overall", *SPLITS)
        for field_name in ("title", "abstract")
        if groups[(split, field_name)].articles
    ]
    return {
        "metadata": {
            "processed_root": str(processed_root),
            "entity_path": str(entity_path),
            "grouping": "first_seen_split (disjoint catalog item groups)",
            "catalog_priority": "train -> dev -> test",
            "available_metadata_splits": available_splits,
            "total_mapping_items": len(mapping),
            "items_without_metadata": missing_metadata,
            "unique_entity_ids_needed": len(needed_ids),
            "unique_entity_ids_found": len(entity_vectors),
            "note": (
                "Embedding lay tu file train; khong doc click labels. "
                "Neu khong co test/news thi khong co dong test. "
                "Cac cot unique_entity_ids la ID duy nhat, con entity_mentions "
                "la tong so mention sau khi bo ID lap trong moi bai."
            ),
        },
        "rows": rows,
        "complementarity_rows": [
            complementary_groups[split].to_row(split)
            for split in ("overall", *SPLITS)
            if complementary_groups[split].articles
        ],
    }


def save_report(result: dict[str, object], csv_path: Path, json_path: Path) -> None:
    """Ghi CSV va JSON, khong sua Feature Store."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(result["rows"])
    # Ghi them bang bo sung khong lam thay doi schema CSV coverage cu.
    complementarity_path = csv_path.parent / "news_entity_complementarity.csv"
    with complementarity_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPLEMENTARITY_COLUMNS)
        writer.writeheader()
        writer.writerows(result["complementarity_rows"])

    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(f"Da ghi CSV: {csv_path}")
    print(f"Da ghi CSV bo sung: {complementarity_path}")
    print(f"Da ghi JSON: {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Wikidata Entity Coverage")
    parser.add_argument(
        "--processed-root", type=Path, default=Path("data/processed/mind_large")
    )
    parser.add_argument(
        "--entity-path",
        type=Path,
        default=Path("data/raw/mind_large/MINDlarge_train/entity_embedding.vec"),
    )
    parser.add_argument(
        "--csv", type=Path, default=Path("reports/tables/news_entity_coverage.csv")
    )
    parser.add_argument(
        "--json", type=Path, default=Path("reports/metrics/news_entity_coverage.json")
    )
    args = parser.parse_args()
    result = analyze_coverage(args.processed_root, args.entity_path)
    save_report(result, args.csv, args.json)

    print("\nTONG HOP COVERAGE (%):")
    for row in result["rows"]:
        print(
            f"  {row['split']:7} {row['field']:8} "
            f"n={row['articles']:>7,} "
            f"co_entity={row['entity_presence_pct']:>6.2f}% "
            f"co_vector={row['embedding_coverage_pct']:>6.2f}%"
        )

    print("\nBO SUNG COVERAGE (% DIEM SO VOI TITLE):")
    for row in result["complementarity_rows"]:
        print(
            f"  {row['split']:7} both={row['both']:>7,} "
            f"title_only={row['title_only']:>7,} "
            f"abstract_only={row['abstract_only']:>7,} "
            f"neither={row['neither']:>7,} "
            f"combined={row['combined_coverage_pct']:>6.2f}% "
            f"gain_vs_title=+{row['gain_over_title_pp']:.2f}pp"
        )


if __name__ == "__main__":
    main()
