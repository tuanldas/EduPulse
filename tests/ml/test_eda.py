"""EDA: định dạng số kiểu Việt, log-odds, và số đo khớp nghiên cứu mà không lộ câu gốc."""

import collections
import json
import random

import pytest

from ml import eda
from ml.common import load_processed


def test_vietnamese_number_format():
    assert eda.vn(54.123, 1) == "54,1"
    assert eda.vn(11426, 0) == "11.426"
    assert eda.vn(0.018, 3) == "0,018"
    assert eda.vn(1234567.5, 1) == "1.234.567,5"


def test_check_helper_compares_lists_exactly_and_numbers_within_tolerance():
    assert eda.check("a", [1, 2], [1, 2])["ok"]
    assert not eda.check("a", [1, 2], [1, 3])["ok"]
    assert eda.check("b", 0.3562, 0.356, 0.001)["ok"]
    assert not eda.check("b", 0.36, 0.356, 0.001)["ok"]


def test_log_odds_puts_distinctive_terms_first_and_common_ones_last():
    cell = collections.Counter({"máy_chiếu": 40, "phòng": 50, "thầy": 60})
    rest = collections.Counter({"máy_chiếu": 2, "phòng": 40, "thầy": 900})
    prior = cell + rest
    table = eda.log_odds(cell, rest, prior)
    assert table["ngram"].tolist()[0] == "máy_chiếu"
    assert table["ngram"].tolist()[-1] == "thầy"
    assert table["z"].is_monotonic_decreasing


@pytest.mark.data
@pytest.mark.slow
def test_full_run_matches_research_numbers_and_leaks_no_sentences(tmp_path, monkeypatch):
    monkeypatch.setattr(eda, "RESULTS", tmp_path)
    summary = eda.run()
    failed = [c for c in summary["checks"] if not c["ok"]]
    assert failed == []

    figures = sorted(p.name for p in (tmp_path / "figures").glob("eda_*.png"))
    assert figures == ["eda_label_dist.png", "eda_length.png", "eda_topic_sentiment.png"]
    tables = sorted(p.name for p in (tmp_path / "tables").glob("eda_*.csv"))
    assert len(tables) >= 6
    assert (tmp_path / "metrics" / "data_summary.json").is_file()

    assert summary["rows"] == {"train": 11426, "validation": 1583, "test": 3166}
    assert summary["inter_annotator_agreement"] == {"sentiment": 0.91, "topic": 0.71}
    assert summary["old_style_tone"]["sentences_affected"] == 284

    # Bảng và JSON chỉ có số đếm hoặc từ/cụm ngắn: không câu gốc nào xuất hiện nguyên văn.
    text = "\n".join(
        p.read_text(encoding="utf-8")
        for p in [*(tmp_path / "tables").glob("*.csv"), *(tmp_path / "metrics").glob("*.json")]
    )
    pool = [
        s
        for split in ("train", "validation")
        for s in load_processed(split)["sentence"]
        if len(s.split()) >= 4
    ]
    assert all(s not in text for s in random.Random(0).sample(pool, 300))
    json.loads((tmp_path / "metrics" / "data_summary.json").read_text("utf-8"))
