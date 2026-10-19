"""Quét câu UIT-VSFC trong tệp: khớp n-gram từ, bỏ dấu câu, không bắt cụm ngắn thông dụng."""

import unicodedata

import pandas as pd
import pytest

from ml import audit_sentences as audit

SENTENCE = "thầy giảng bài rất dễ hiểu và luôn nhiệt tình giúp đỡ sinh viên ."
SHORT = "em rất hài lòng ."


@pytest.fixture()
def corpus(monkeypatch):
    """Nguồn câu thu nhỏ cùng cột với dữ liệu thật; logic quét vẫn chạy thật."""
    frames = {
        "train": pd.DataFrame({"sentence": [SENTENCE, SHORT]}),
        "validation": pd.DataFrame({"sentence": ["phòng máy hay hỏng , wifi yếu và rất chậm ."]}),
        "test": pd.DataFrame({"sentence": []}),
    }
    monkeypatch.setattr(audit, "load_processed", lambda split, allow_test=False: frames[split])


def hits(text, n=8):
    return audit.scan_text(text, audit.corpus_grams(n), n)


def test_words_drop_punctuation_and_case_and_normalise_unicode():
    decomposed = unicodedata.normalize("NFD", "Thầy")  # dạng tổ hợp, không phải NFC
    assert decomposed != "Thầy"
    assert audit.words(f"{decomposed}, GIẢNG bài!") == ["thầy", "giảng", "bài"]


def test_a_whole_sentence_or_any_eight_words_of_it_are_found(corpus):
    assert hits(SENTENCE) == [(1, "thầy giảng bài rất dễ hiểu và luôn")]
    assert hits("trích: ... luôn nhiệt tình giúp đỡ sinh viên ... hết") == []  # chỉ 7 từ liên tiếp
    assert hits("luôn nhiệt tình giúp đỡ sinh viên") == []  # 7 từ
    assert [h[0] for h in hits("và luôn nhiệt tình giúp đỡ sinh viên")] == [1]  # đúng 8 từ


def test_punctuation_case_and_spacing_do_not_hide_a_match(corpus):
    assert hits("THẦY giảng, bài rất; dễ hiểu (và) luôn nhiệt-tình!")
    assert hits("\n\n  Thầy giảng bài rất dễ hiểu và luôn nhiệt tình giúp đỡ")


def test_the_line_number_points_at_the_line_with_the_match(corpus):
    text = "dòng một\nkhông liên quan\n" + SENTENCE + "\ncuối"
    assert [number for number, _ in hits(text)] == [3]


def test_short_generic_sentences_are_not_counted(corpus):
    assert hits(SHORT) == []
    assert hits(f"có người viết {SHORT} nhiều lần") == []


def test_a_different_ngram_size_changes_what_counts(corpus):
    assert audit.scan_text("em rất hài lòng", audit.corpus_grams(4), 4)
    assert audit.scan_text("em rất hài lòng", audit.corpus_grams(5), 5) == []


def test_audit_walks_directories_and_skips_generated_ones(corpus, tmp_path):
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "bad.csv").write_text(f"a,{SENTENCE}\n", encoding="utf-8")
    (tmp_path / "tables" / "good.md").write_text("chỉ có số liệu 12,5%\n", encoding="utf-8")
    (tmp_path / "tables" / "image.png").write_bytes(SENTENCE.encode("utf-8"))  # không phải văn bản
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "x.json").write_text(SENTENCE, encoding="utf-8")
    grams = audit.corpus_grams()
    report = audit.audit([tmp_path], grams)
    assert list(report) == [tmp_path / "tables" / "bad.csv"]
    # Truyền thẳng một tệp thì quét tệp đó dù đuôi gì.
    assert audit.audit([tmp_path / "tables" / "image.png"], grams)


def test_main_exit_codes_and_output(corpus, tmp_path, capsys):
    clean = tmp_path / "clean.md"
    clean.write_text("không có câu nào\n", encoding="utf-8")
    assert audit.main([str(clean)]) == 0
    assert "0 tệp" in capsys.readouterr().out

    dirty = tmp_path / "dirty.csv"
    dirty.write_text("x\n" + SENTENCE + "\n", encoding="utf-8")
    assert audit.main([str(clean), str(dirty), "--show"]) == 1
    out = capsys.readouterr().out
    assert f"{dirty}:2  …thầy giảng bài rất dễ hiểu và luôn…" in out
    assert "1 tệp, 1 dòng" in out

    assert audit.main([str(clean), "--ngram", "1"]) == 2


def test_main_reports_when_the_corpus_cannot_be_read(monkeypatch, tmp_path, capsys):
    def missing(split, allow_test=False):
        raise RuntimeError("chưa có dữ liệu — chạy `make data`")

    monkeypatch.setattr(audit, "load_processed", missing)
    assert audit.main([str(tmp_path)]) == 2
    assert "make data" in capsys.readouterr().err


@pytest.mark.data
def test_a_real_training_sentence_is_found_and_an_invented_one_is_not():
    from ml.common import load_processed

    train = load_processed("train")
    real = next(s for s in train["sentence"] if len(s.split()) >= 10)
    grams = audit.corpus_grams()
    assert audit.scan_text(real, grams)
    assert audit.scan_text("Hôm nay trời đẹp nên mình đi dạo quanh hồ Gươm cùng bạn", grams) == []
