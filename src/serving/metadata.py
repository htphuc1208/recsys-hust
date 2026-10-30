"""Fast metadata and sample user provider for the RecSys serving UI."""

from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

_NEWS_CACHE: dict[str, dict[str, Any]] = {}
_COMBINED_NEWS_DATASET: ds.Dataset | None = None


def _get_news_dataset() -> ds.Dataset | None:
    global _COMBINED_NEWS_DATASET
    if _COMBINED_NEWS_DATASET is not None:
        return _COMBINED_NEWS_DATASET

    datasets = []
    for rel_path in ("data/processed/mind_large/dev/news", "data/processed/mind_large/train/news"):
        path = Path(rel_path)
        if path.exists():
            datasets.append(ds.dataset(str(path), format="parquet"))

    if datasets:
        _COMBINED_NEWS_DATASET = ds.dataset(datasets)
    return _COMBINED_NEWS_DATASET


def get_news_metadata_batch(news_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Retrieve metadata (title, category, subcategory, abstract, url) for news IDs."""
    if not news_ids:
        return {}

    missing_ids = [nid for nid in set(news_ids) if nid not in _NEWS_CACHE]
    dataset = _get_news_dataset()

    if missing_ids and dataset is not None:
        try:
            table = dataset.to_table(
                columns=["news_id", "title", "category", "subcategory", "abstract", "url"],
                filter=ds.field("news_id").isin(missing_ids),
            )
            pyd = table.to_pydict()
            for i in range(len(pyd["news_id"])):
                nid = pyd["news_id"][i]
                if nid not in _NEWS_CACHE:
                    _NEWS_CACHE[nid] = {
                        "news_id": nid,
                        "title": pyd["title"][i] or "Tin tức tổng hợp",
                        "category": pyd["category"][i] or "news",
                        "subcategory": pyd["subcategory"][i] or "",
                        "abstract": pyd["abstract"][i]
                        or "Không có tóm tắt chi tiết cho bài viết này.",
                        "url": pyd["url"][i] or "#",
                    }
        except Exception:
            pass

    # Ensure all requested IDs return at least a fallback representation
    results: dict[str, dict[str, Any]] = {}
    for nid in news_ids:
        if nid in _NEWS_CACHE:
            results[nid] = _NEWS_CACHE[nid]
        else:
            results[nid] = {
                "news_id": nid,
                "title": f"Bản tin {nid}",
                "category": "general",
                "subcategory": "top",
                "abstract": "Thông tin bài viết từ kho dữ liệu Microsoft News (MIND).",
                "url": f"https://assets.msn.com/labs/mind/{nid}.html",
            }
    return results


def get_sample_personas() -> list[dict[str, Any]]:
    """Return pre-configured reader personas for UI demonstration."""
    sample_news_ids = [
        "N88753",
        "N23144",
        "N98178",
        "N122287",
        "N94572",
        "N16403",
        "N32154",
        "N86255",
    ]
    meta = get_news_metadata_batch(sample_news_ids)

    return [
        {
            "id": "U134050",
            "name": "Nguyễn Minh Tuấn",
            "role": "Active Sports & Travel Reader",
            "avatar": "🏃‍♂️",
            "category_pref": "Thể thao & Khám phá",
            "history_news_ids": ["N98178", "N88753", "N23144"],
            "history": [
                meta.get(
                    "N98178",
                    {
                        "news_id": "N98178",
                        "title": "Charles Rogers, NFL star dead at 38",
                        "category": "sports",
                    },
                ),
                meta.get(
                    "N88753",
                    {
                        "news_id": "N88753",
                        "title": "The Brands Queen Elizabeth Swears By",
                        "category": "lifestyle",
                    },
                ),
                meta.get(
                    "N23144",
                    {
                        "news_id": "N23144",
                        "title": "50 Worst Habits For Belly Fat",
                        "category": "health",
                    },
                ),
            ],
            "description": "Người đọc thường xuyên theo dõi các tin thể thao NFL, chế độ dinh dưỡng và lối sống.",
        },
        {
            "id": "U107107",
            "name": "Lê Mai Phương",
            "role": "Pop Culture & Food Enthusiast",
            "avatar": "🍕",
            "category_pref": "Ẩm thực & Giải trí",
            "history_news_ids": ["N88753", "N94572", "N23144"],
            "history": [
                meta.get(
                    "N88753",
                    {
                        "news_id": "N88753",
                        "title": "The Brands Queen Elizabeth Swears By",
                        "category": "lifestyle",
                    },
                ),
                meta.get(
                    "N94572",
                    {
                        "news_id": "N94572",
                        "title": "Sarah Palin says she is fighting to repair marriage",
                        "category": "news",
                    },
                ),
                meta.get(
                    "N23144",
                    {
                        "news_id": "N23144",
                        "title": "50 Worst Habits For Belly Fat",
                        "category": "health",
                    },
                ),
            ],
            "description": "Quan tâm đến xu hướng ẩm thực, giới nghệ sĩ và các bài viết phong cách sống.",
        },
        {
            "id": "U254959",
            "name": "Trần Quốc Hưng",
            "role": "Current Affairs & Sports Fan",
            "avatar": "📰",
            "category_pref": "Thời sự & Tin nóng",
            "history_news_ids": ["N98178", "N94572"],
            "history": [
                meta.get(
                    "N98178",
                    {
                        "news_id": "N98178",
                        "title": "Charles Rogers, Detroit Lions star, dead at 38",
                        "category": "sports",
                    },
                ),
                meta.get(
                    "N94572",
                    {
                        "news_id": "N94572",
                        "title": "Sarah Palin says she is fighting to repair marriage",
                        "category": "news",
                    },
                ),
            ],
            "description": "Theo dõi các sự kiện xã hội trong ngày và tin tức bóng bầu dục.",
        },
        {
            "id": "NEW_USER_COLD",
            "name": "Độc giả Mới (Cold User)",
            "role": "Lần đầu truy cập (Zero History)",
            "avatar": "❄️",
            "category_pref": "Chưa xác định (Popularity Fallback)",
            "history_news_ids": [],
            "history": [],
            "description": "Chưa có lịch sử click. Hệ thống sẽ kích hoạt Circuit Breaker hạ cấp an toàn sang Popularity.",
        },
    ]
