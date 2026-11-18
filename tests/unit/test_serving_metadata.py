import pyarrow as pa
import pyarrow.parquet as pq

from src.serving import metadata


def test_demo_personas_use_real_histories_and_matching_interest_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for split, rows in (
        ("dev", {"user_id": ["U107107", "U375076"], "history_news_ids": [["Nfood", "Nfood2"], ["Nsport"]]}),
        ("train", {"user_id": ["U107107", "U134050"], "history_news_ids": [["Nother"], ["Nsport"]]}),
    ):
        path = tmp_path / "data/processed/mind_large" / split / "impressions"
        path.mkdir(parents=True)
        pq.write_table(pa.table(rows), path / "part.parquet")
    monkeypatch.setattr(metadata, "get_news_metadata_batch", lambda ids: {
        nid: {"news_id": nid, "category": "foodanddrink" if nid.startswith("Nfood") else "sports"}
        for nid in ids
    })
    metadata._sample_histories.cache_clear()
    try:
        personas = {p["id"]: p for p in metadata.get_sample_personas()}
    finally:
        metadata._sample_histories.cache_clear()

    assert personas["U107107"]["history_news_ids"] == ["Nfood", "Nfood2"]
    assert personas["U107107"]["category_pref"] == "Ẩm thực"
    assert personas["U134050"]["history_news_ids"] == ["Nsport"]
    assert personas["U134050"]["category_pref"] == "Thể thao"
    assert personas["U254959"]["history_news_ids"] == []
    assert len(personas) == 11
    assert personas["U375076"]["history_news_ids"] == ["Nsport"]
    assert personas["U375076"]["category_pref"] == "Thể thao"
    for persona in personas.values():
        assert [article["news_id"] for article in persona["history"]] == persona["history_news_ids"]
