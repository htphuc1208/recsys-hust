from pathlib import Path

import pytest

from src.ingest.mind import _iter_tsv, parse_entity_ids, parse_history, parse_impressions


def test_parse_history_handles_empty_and_ordered_history() -> None:
    assert parse_history("") == []
    assert parse_history("N1 N2 N3") == ["N1", "N2", "N3"]


def test_parse_labeled_impressions() -> None:
    news_ids, labels = parse_impressions("N1-0 N2-1 N3-0", require_labels=True)

    assert news_ids == ["N1", "N2", "N3"]
    assert labels == [0, 1, 0]


def test_parse_unlabeled_test_impressions() -> None:
    news_ids, labels = parse_impressions("N1 N2", require_labels=False)

    assert news_ids == ["N1", "N2"]
    assert labels is None


@pytest.mark.parametrize(
    ("raw_impressions", "require_labels"),
    [("N1 N2", True), ("N1-0 N2-1", False), ("N1-0 N2", True)],
)
def test_parse_impressions_rejects_wrong_schema(raw_impressions: str, require_labels: bool) -> None:
    with pytest.raises(ValueError):
        parse_impressions(raw_impressions, require_labels=require_labels)


def test_parse_entity_ids() -> None:
    raw = '[{"Label":"A","WikidataId":"Q1"},{"Label":"missing id"},{"Label":"B","WikidataId":"Q2"}]'

    assert parse_entity_ids(raw) == ["Q1", "Q2"]


def test_iter_tsv_treats_unmatched_quotes_as_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "news.tsv"
    path.write_text(
        'N1\tcategory\t"unmatched quote\nN2\tcategory\tordinary text\n',
        encoding="utf-8",
    )

    assert list(_iter_tsv(path, expected_columns=3)) == [
        ["N1", "category", '"unmatched quote'],
        ["N2", "category", "ordinary text"],
    ]
