"""Fast metadata and sample user provider for the RecSys serving UI."""

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pyarrow.dataset as ds

_NEWS_CACHE: dict[str, dict[str, Any]] = {}
_COMBINED_NEWS_DATASET: ds.Dataset | None = None

# Display names are aliases; IDs and reading histories come from MIND.
_DEMO_READERS = (
    ("U134050", "Nguyễn Minh Tuấn", "🏃‍♂️"),
    ("U107107", "Lê Mai Phương", "🍕"),
    ("U254959", "Trần Quốc Hưng", "📰"),
    ("U375076", "Phạm Đức Anh", "⚽"),
    ("U655197", "Đặng Thu Hà", "📈"),
    ("U470220", "Vũ Ngọc Lan", "💚"),
    ("U670770", "Hoàng Bảo Ngọc", "🍲"),
    ("U623970", "Đỗ Hải Nam", "✈️"),
    ("U122329", "Bùi Gia Huy", "🚗"),
    ("U102682", "Trịnh Khánh Linh", "🎬"),
)


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


@lru_cache(maxsize=1)
def _sample_histories() -> dict[str, list[str]]:
    """Use a real impression history for each demo user, preferring dev."""
    user_ids = [uid for uid, _name, _avatar in _DEMO_READERS]
    histories: dict[str, list[str]] = {}
    for split in ("dev", "train"):
        remaining = [uid for uid in user_ids if uid not in histories]
        path = Path(f"data/processed/mind_large/{split}/impressions")
        if not remaining or not path.exists():
            continue
        table = ds.dataset(str(path), format="parquet").to_table(
            columns=["user_id", "history_news_ids"],
            filter=ds.field("user_id").isin(remaining),
        )
        for row in table.to_pylist():
            uid = row["user_id"]
            history = list(row["history_news_ids"] or [])[-500:]
            if uid not in histories and history:
                histories[uid] = history
    return histories


def get_sample_personas() -> list[dict[str, Any]]:
    """Show real user histories and derive interest labels from those articles."""
    category_names = {
        "sports": "Thể thao", "news": "Thời sự", "finance": "Tài chính",
        "health": "Sức khỏe", "lifestyle": "Đời sống", "foodanddrink": "Ẩm thực",
        "travel": "Du lịch", "tv": "Truyền hình", "entertainment": "Giải trí",
        "movies": "Điện ảnh", "music": "Âm nhạc", "autos": "Xe",
        "weather": "Thời tiết", "video": "Video", "kids": "Gia đình",
    }
    histories = _sample_histories()
    meta = get_news_metadata_batch([nid for history in histories.values() for nid in history])
    personas = []
    for uid, name, avatar in _DEMO_READERS:
        history_ids = list(histories.get(uid, []))
        articles = [meta[nid] for nid in history_ids]
        categories = Counter(a["category"] for a in articles if a["category"] != "general")
        preference = " & ".join(
            category_names.get(category, category) for category, _ in categories.most_common(2)
        ) or "Chưa có lịch sử"
        personas.append({
            "id": uid,
            "name": name,
            "role": "Độc giả từ dữ liệu MIND",
            "avatar": avatar,
            "category_pref": preference,
            "history_news_ids": history_ids,
            "history": articles,
            "description": f"Chuyên mục thường đọc trong lịch sử: {preference}.",
        })
    personas.append({
        "id": "NEW_USER_COLD",
        "name": "Độc giả Mới (Cold User)",
        "role": "Lần đầu truy cập (Zero History)",
        "avatar": "❄️",
        "category_pref": "Chưa xác định (Popularity Fallback)",
        "history_news_ids": [],
        "history": [],
        "description": "Chưa có lịch sử đọc. Hệ thống gợi ý các bài phổ biến.",
    })
    return personas
