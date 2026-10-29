"""Kiểm thử contract nộp Microsoft MIND, có mock dataset để chạy khi thiếu pyarrow."""

from zipfile import ZipFile

import numpy as np
import pytest

from src.submission.inference import infer_ranks
from src.submission.ranking import (
    format_prediction_line,
    parse_prediction_line,
    positions_to_ranks,
    scores_to_ranks,
)
from src.submission import runner, validate


class DummyModel:
    def predict(self, user_ids, item_ids, **kwargs):
        assert len(user_ids) == 1
        # Score theo ID để tạo kết quả cố định, kể cả cold user.
        return np.asarray(item_ids, dtype=np.float64)


class DummyP3:
    def rerank(self, candidates, scores):
        # Trả đủ các IDs và đảo thứ tự baseline.
        return list(reversed(candidates))


ROWS = [
    {
        "impression_id": 24481, "user_idx": -1,
        "candidate_news_ids": ["N1", "N2", "N3", "N4"],
        "candidate_item_idxs": [0, 3, 1, 2],
        "history_item_idxs": [], "labels": None,
    },
    {
        "impression_id": 24482, "user_idx": 3,
        "candidate_news_ids": ["N8", "N9"],
        "candidate_item_idxs": [-1, 4],
        "history_item_idxs": [1], "labels": None,
    },
]


def test_official_rank_format_original_candidate_positions():
    # N2 > N4 > N3 > N1.
    assert scores_to_ranks([0.1, 0.9, 0.4, 0.8]) == [4, 1, 3, 2]
    assert format_prediction_line(24481, [4, 1, 3, 2]) == "24481 [4,1,3,2]\n"
    assert parse_prediction_line("24481 [4,1,3,2]\n") == (24481, [4, 1, 3, 2])
    assert scores_to_ranks([0.5, 0.5, 0.8]) == [2, 3, 1]
    assert positions_to_ranks([2, 0, 1], 3) == [2, 3, 1]


@pytest.mark.parametrize("line", ["24481 [1,1]", "24481 [0,1]", "24481 [3,1]", "24481 1,2", "N24481 [1,2]", "24481 []"])
def test_invalid_output_rejected(line):
    with pytest.raises(ValueError):
        parse_prediction_line(line)


def test_cold_user_unknown_item_and_p3():
    base = infer_ranks(DummyModel(), user_idx=-1, candidate_item_idxs=[0, 3, 1, 2], history_item_idxs=[])
    assert base.ranks == [4, 1, 3, 2]
    reranked = infer_ranks(DummyModel(), user_idx=-1, candidate_item_idxs=[0, 3, 1, 2], history_item_idxs=[], reranker=DummyP3())
    assert reranked.ranks == [4, 3, 2, 1]
    unknown = infer_ranks(DummyModel(), user_idx=-1, candidate_item_idxs=[-1, 4], history_item_idxs=[])
    assert unknown.ranks == [2, 1] and unknown.unknown_candidates == 1
    duplicate = infer_ranks(DummyModel(), user_idx=1, candidate_item_idxs=[2, 2], history_item_idxs=[], reranker=DummyP3())
    assert duplicate.ranks == [1, 2] and duplicate.p3_fallback_reason


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def __call__(self, path, batch_size=2048):
        yield from self.rows


def test_full_test_txt_zip_and_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(ROWS))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(ROWS))
    config = tmp_path / "model.yaml"
    config.write_text("experiment_name: unit_test\nmodel: {name: content_based}\n", encoding="utf8")
    raw = tmp_path / "behaviors.tsv"
    raw.write_text(
        "24481\tU1\t11/20/2019 01:00:00 PM\t\tN1 N2 N3 N4\n"
        "24482\tU2\t11/20/2019 01:00:00 PM\tN1\tN8 N9\n", encoding="utf8"
    )
    manifest = runner.run_submission(
        config_path=config, model=DummyModel(),
        output_dir=tmp_path / "output", raw_behaviors=raw, require_raw=True,
    )
    folder = tmp_path / "output"
    assert (folder / "prediction.txt").read_text(encoding="utf8") == (
        "24481 [4,1,3,2]\n24482 [2,1]\n"
    )
    with ZipFile(folder / "prediction.zip") as archive:
        assert archive.namelist() == ["prediction.txt"]
        assert archive.read("prediction.txt").decode() == (folder / "prediction.txt").read_text(encoding="utf8")
    assert manifest["submission_ready"] is True
    assert manifest["validation"]["original_order_verified"] is True
    assert manifest["unknown_candidates"] == 1
    assert manifest["validation"]["impressions"] == 2
    with pytest.raises(FileExistsError):
        runner.run_submission(config_path=config, model=DummyModel(), output_dir=folder, raw_behaviors=raw)


def test_mismatch_original_candidate_order_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(ROWS))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(ROWS))
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf8")
    raw = tmp_path / "bad.tsv"
    raw.write_text("24481\tU1\tT\t\tN2 N1 N3 N4\n", encoding="utf8")
    with pytest.raises(ValueError, match="candidate order"):
        runner.run_submission(config_path=cfg, model=DummyModel(), output_dir=tmp_path / "out", raw_behaviors=raw)
    assert not (tmp_path / "out" / "prediction.txt").exists()


def test_test_split_must_be_unlabeled(tmp_path, monkeypatch):
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf8")
    bad = [{**ROWS[0], "labels": [1, 0, 0, 0]}]
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(bad))
    with pytest.raises(ValueError, match="không được có nhãn"):
        runner.run_submission(config_path=cfg, model=DummyModel(), output_dir=tmp_path / "out", require_raw=False)


def test_model_must_return_finite_and_exact_scores():
    class BadModel:
        def predict(self, *args, **kwargs):
            return [np.nan]
    with pytest.raises(ValueError, match="NaN/Inf"):
        infer_ranks(BadModel(), user_idx=1, candidate_item_idxs=[1], history_item_idxs=[])


def test_validate_corrupted_zip_and_prediction(tmp_path, monkeypatch):
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(ROWS))
    path = tmp_path / "prediction.txt"
    path.write_text("24481 [1,1,2,3]\n24482 [2,1]\n", encoding="utf8")
    with pytest.raises(ValueError, match="rank"):
        validate.validate_submission(path, "unused")
    path.write_text("24481 [4,1,3,2]\n24482 [2,1]\n", encoding="utf8")
    zip_path = tmp_path / "prediction.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("folder/prediction.txt", path.read_bytes())
    with pytest.raises(ValueError, match="ZIP"):
        validate.validate_submission(path, "unused", prediction_zip=zip_path)


def test_dev_sample_reports_not_ready(tmp_path, monkeypatch):
    # Dev trong MIND dùng nhãn -0/-1. Tạo TSV giả lập để test độc lập
    # với dữ liệu MIND thực có thể đang hiện diện trên máy chạy pytest.
    dev_rows = [{**ROWS[0], "labels": [0, 1, 0, 0]}]
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(dev_rows))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(dev_rows))
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf8")
    raw = tmp_path / "dev_behaviors.tsv"
    raw.write_text(
        "24481\tU1\t11/15/2019 01:00:00 PM\t\tN1-0 N2-1 N3-0 N4-0\n",
        encoding="utf8",
    )
    manifest = runner.run_submission(
        config_path=cfg, model=DummyModel(), split="dev", limit=1,
        output_dir=tmp_path / "out", raw_behaviors=raw, require_raw=True,
    )
    assert manifest["submission_ready"] is False
    assert manifest["validation"]["partial"] is True
    assert manifest["validation"]["original_order_verified"] is True
    assert manifest["validation"]["impressions"] == 1


def test_raw_behaviors_split_strictly_handles_labels(tmp_path):
    from src.submission.io import iter_original_behaviors

    dev = tmp_path / "dev.tsv"
    dev.write_text("1\tU1\tT\t\tN1-0 N2-1\n", encoding="utf8")
    test = tmp_path / "test.tsv"
    test.write_text("2\tU1\tT\t\tN1 N2\n", encoding="utf8")

    assert list(iter_original_behaviors(dev, split="dev")) == [(1, ["N1", "N2"])]
    assert list(iter_original_behaviors(test, split="test")) == [(2, ["N1", "N2"])]
    with pytest.raises(ValueError, match="Expected unlabeled"):
        list(iter_original_behaviors(dev, split="test"))
    with pytest.raises(ValueError, match="Expected click labels"):
        list(iter_original_behaviors(test, split="dev"))
    with pytest.raises(ValueError, match="Split không hỗ trợ"):
        list(iter_original_behaviors(test, split="train"))


def test_full_dev_with_labeled_raw_behaviors(tmp_path, monkeypatch):
    """Full Dev có nhãn vẫn phải đối chiếu thứ tự TSV và hoàn tất validate."""
    dev_rows = [
        {**ROWS[0], "labels": [0, 1, 0, 0]},
        {**ROWS[1], "labels": [1, 0]},
    ]
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(dev_rows))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(dev_rows))
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf8")
    raw = tmp_path / "dev.tsv"
    raw.write_text(
        "24481\tU1\tT\t\tN1-0 N2-1 N3-0 N4-0\n"
        "24482\tU2\tT\t\tN8-1 N9-0\n",
        encoding="utf8",
    )
    result = runner.run_submission(
        config_path=cfg, model=DummyModel(), split="dev",
        output_dir=tmp_path / "full_dev", raw_behaviors=raw, require_raw=True,
    )
    assert result["submission_ready"] is False  # Dev không phải Test nộp leaderboard
    assert result["validation"]["original_order_verified"] is True
    assert result["validation"]["zip_valid"] is True
    assert result["validation"]["impressions"] == 2
    with ZipFile(tmp_path / "full_dev" / "prediction.zip") as archive:
        assert archive.namelist() == ["prediction.txt"]


def test_test_sample_exports_prefix_without_being_submittable(tmp_path, monkeypatch):
    """Test --limit sinh TXT/ZIP chuẩn cho prefix nhưng phải đánh dấu không được nộp."""
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(ROWS))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(ROWS))
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "conf.yaml"
    config.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf-8")
    raw = tmp_path / "test_behaviors.tsv"
    raw.write_text(
        "24481\tU1\tT\t\tN1 N2 N3 N4\n"
        "24482\tU2\tT\t\tN8 N9\n", encoding="utf-8",
    )
    manifest = runner.run_submission(
        config_path=config, split="test", limit=1, model=DummyModel(),
        raw_behaviors=raw, require_raw=True,
    )
    folder = tmp_path / "reports/submissions/mind_large/test_samples/xx/first_1"
    assert (folder / "prediction.txt").read_text(encoding="utf-8") == "24481 [4,1,3,2]\n"
    with ZipFile(folder / "prediction.zip") as archive:
        assert archive.namelist() == ["prediction.txt"]
    assert manifest["submission_ready"] is False
    assert manifest["complete_dataset"] is False
    assert manifest["sample_limit"] == 1
    assert manifest["validation"]["partial"] is True
    assert manifest["validation"]["original_order_verified"] is True
    assert manifest["validation"]["zip_valid"] is True


def test_test_sample_rejects_insufficient_impressions(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(ROWS))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(ROWS))
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf-8")
    # Dùng TSV giả lập khớp FakeDataset, không tự tìm Test thật của người chạy.
    raw = tmp_path / "test_behaviors.tsv"
    raw.write_text(
        "24481\tU1\tT\t\tN1 N2 N3 N4\n"
        "24482\tU2\tT\t\tN8 N9\n",
        encoding="utf-8",
    )
    target = tmp_path / "out"
    with pytest.raises(ValueError, match="Yêu cầu 5000 impressions"):
        runner.run_submission(
            config_path=cfg, split="test", limit=5000,
            model=DummyModel(), output_dir=target,
            raw_behaviors=raw, require_raw=True,
        )
    assert not (target / "prediction.txt").exists()
    assert not (target / "prediction.txt.part").exists()


def test_test_sample_validate_only_uses_partial_validator(tmp_path, monkeypatch, capsys):
    import sys
    from src.submission import __main__ as cli

    monkeypatch.setattr(runner, "iter_impressions", FakeDataset(ROWS))
    monkeypatch.setattr(validate, "iter_impressions", FakeDataset(ROWS))
    cfg = tmp_path / "conf.yaml"
    cfg.write_text("experiment_name: xx\nmodel: {name: bpr}\n", encoding="utf-8")
    raw = tmp_path / "test_behaviors.tsv"
    raw.write_text("24481\tU1\tT\t\tN1 N2 N3 N4\n24482\tU2\tT\t\tN8 N9\n", encoding="utf8")
    target = tmp_path / "out"
    runner.run_submission(
        config_path=cfg, split="test", limit=1, model=DummyModel(),
        output_dir=target, raw_behaviors=raw,
    )
    monkeypatch.setattr(sys, "argv", [
        "submission", "--config", str(cfg), "--split", "test",
        "--limit", "1", "--raw-behaviors", str(raw),
        "--output-dir", str(target), "--validate-only",
    ])
    cli.main()
    assert '"partial": true' in capsys.readouterr().out
