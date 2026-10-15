"""N-gram dùng chung cho đếm nền và insight: đơn vị và parity với bộ đếm nền đã lưu."""

import json
from collections import Counter

import pandas as pd
import pytest

from edupulse.config import load_settings
from edupulse.text.ngrams import count_ngrams, ngrams


def test_unigrams_then_bigrams_keep_underscore_words():
    assert ngrams("giảng_viên dạy hay") == [
        "giảng_viên",
        "dạy",
        "hay",
        "giảng_viên dạy",
        "dạy hay",
    ]


def test_punctuation_tokens_are_dropped_and_bigrams_do_not_cross_them():
    assert ngrams("hay . phòng nóng , quạt") == ["hay", "phòng", "nóng", "quạt", "phòng nóng"]


def test_name_tokens_are_dropped():
    assert ngrams("thầy wzjwz12 dạy hay") == ["thầy", "dạy", "hay", "dạy hay"]


def test_special_and_numeric_tokens_are_kept():
    assert "colonsad" in ngrams("phòng nóng colonsad")
    assert "50%" in ngrams("đề 50% lý thuyết")


def test_empty_and_punctuation_only_inputs():
    assert ngrams("") == []
    assert ngrams(" . , ! ") == []
    assert ngrams("hay") == ["hay"]


def test_repeated_ngrams_are_kept_for_counting():
    assert ngrams("hay hay").count("hay") == 2


def test_count_ngrams_accumulates_across_sentences():
    counts = count_ngrams(["phòng máy hỏng", "phòng máy nóng"])
    assert counts["phòng máy"] == 2
    assert counts["hỏng"] == 1
    assert isinstance(counts, Counter)


@pytest.mark.data
def test_stored_background_matches_a_fresh_count_of_train():
    settings = load_settings()
    train = pd.read_parquet(settings.data_dir / "processed" / "train.parquet")
    stored = json.loads(
        (settings.outputs_dir / "data" / "insight_background.json").read_text("utf-8")
    )
    fresh = count_ngrams(train["text_rdr"])
    min_count = stored["min_count"]
    assert stored["n_sentences"] == len(train)
    assert stored["total_ngrams"] == sum(fresh.values())
    assert stored["ngram_counts"] == {g: c for g, c in fresh.items() if c >= min_count}
