"""Pipeline tiền xử lý đầy đủ (ẩn PII, chuẩn hoá, tách từ). Cần JRE; vài ca cần data/processed."""

import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from edupulse.config import load_settings
from edupulse.text import NORMALIZER_VERSION, Prepared, prepare, prepare_many
from edupulse.text.normalize import diacritic_ratio
from edupulse.text.pipeline import MAX_INPUT_CHARS

pytestmark = pytest.mark.java

# Token đặc biệt của dữ liệu -> dạng người dùng thật sự gõ.
_EMOTICON_RAW = {
    "colonsmilesmile": ":))",
    "colonsmile": ":)",
    "colonsad": ":(",
    "colonbigsmile": ":D",
    "colonlove": "<3",
    "coloncontemn": ":v",
    "colonhihi": ":3",
    "colonsmallsmile": ":>",
    "colonsurprise": ":o",
}


def to_raw(sentence: str) -> str:
    """Đưa câu của dữ liệu về dạng gõ tay: viết hoa đầu câu, dấu câu dính chữ, ":)", ":", "v.v"."""
    text = re.sub(r"(?<=\d)doubledot(?=\d)", ":", sentence)
    text = re.sub(r"\bdoubledot\b", ":", text)
    text = re.sub(r"\bvdotv\b", "v.v", text)
    text = re.sub(r"\bdotdotdot\b", "...", text)
    text = re.sub(r"\s+([.,!?;:)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    for token, raw in _EMOTICON_RAW.items():
        text = re.sub(rf"\b{token}\b", raw, text)
    return text[:1].upper() + text[1:]


def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


@pytest.fixture(scope="module")
def validation() -> pd.DataFrame:
    return pd.read_parquet(load_settings().data_dir / "processed" / "validation.parquet")


# --- Hành vi cơ bản ---------------------------------------------------------------------------


def test_prepare_returns_both_views_and_flags():
    result = prepare("Thầy Nam dạy rất nhiệt tình :) liên hệ 0912 345 678.", use_ner=False)
    assert isinstance(result, Prepared)
    assert result.display_text == "Thầy [TÊN] dạy rất nhiệt tình :) liên hệ [SĐT]."
    assert result.model_text == "thầy wzjwz1 dạy rất nhiệt_tình colonsmile liên_hệ ."
    assert result.pii_types == ("name", "phone")
    assert result.no_diacritics is False
    assert result.n_words == len(result.model_text.split())


def test_emoticon_is_not_split_into_two_sentences():
    tokens = prepare("Dạy hay :) Thầy tốt", use_ner=False).model_text.split()
    assert "colonsmile" in tokens
    assert ")" not in tokens and ":" not in tokens


def test_old_style_tone_marks_are_normalised():
    text = prepare("Khóa học hòa đồng", use_ner=False).model_text
    assert text == "khoá học hoà_đồng"
    assert "khóa" not in text and "hòa" not in text


def test_text_with_only_pii_gives_empty_model_text():
    result = prepare("0912345678", use_ner=False)
    assert result.model_text == ""
    assert result.n_words == 0
    assert result.display_text == "[SĐT]"


def test_empty_text():
    result = prepare("", use_ner=False)
    assert (result.display_text, result.model_text, result.n_words) == ("", "", 0)
    assert result.pii_types == ()


def test_type_and_size_are_validated():
    with pytest.raises(TypeError):
        prepare(123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tối đa"):
        prepare("a" * (MAX_INPUT_CHARS + 1))


def test_unaccented_text_is_detected():
    assert prepare("giang vien day rat hay va nhiet tinh", use_ner=False).no_diacritics
    assert not prepare("giảng viên dạy rất hay", use_ner=False).no_diacritics


def test_ner_setting_is_read_from_environment(monkeypatch):
    text = "Em thấy Nguyễn Văn An dạy tốt"
    monkeypatch.setenv("EDUPULSE_NER", "0")
    assert prepare(text).pii_types == ()
    monkeypatch.setenv("EDUPULSE_NER", "1")
    assert prepare(text).pii_types == ("name",)


def test_prepare_many_matches_prepare():
    texts = ["Thầy dạy hay :)", "phòng máy nóng quá!!!", "", "Cô Lan 0912345678"]
    assert prepare_many(texts, use_ner=False) == [prepare(t, use_ner=False) for t in texts]


def test_prepare_many_is_fast_after_jvm_is_loaded():
    base = ["Thầy dạy rất nhiệt tình :) Bài giảng hay.", "Phòng máy hay hỏng, wifi yếu!!!"]
    texts = [f"{base[i % 2]} Lớp {i}" for i in range(1000)]
    prepare_many(texts[:20], use_ner=False)
    start = time.perf_counter()
    prepare_many(texts, use_ner=False)
    assert time.perf_counter() - start < 2.0


def test_threads_get_the_same_results_as_a_single_thread():
    texts = [f"Thầy {name} dạy hay :) quá! Lớp {i}" for i, name in enumerate(["Nam", "Lan"] * 100)]
    sequential = prepare_many(texts, use_ner=False)
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(lambda t: prepare(t, use_ner=False), texts))
    assert parallel == sequential


# --- Parity với dữ liệu thật (cần data/processed) ---------------------------------------------


@pytest.mark.data
def test_dataset_sentences_reproduce_the_cached_segmentation(validation):
    # `prepare` là đường duy nhất: câu của dữ liệu phải ra đúng `text_rdr` đã ghi.
    sample = validation.sample(200, random_state=0)
    prepared = prepare_many(sample["sentence"], use_ner=False)
    assert [p.model_text for p in prepared] == sample["text_rdr"].tolist()


@pytest.mark.data
def test_raw_typing_variants_give_the_same_model_text(validation):
    # 200 câu val gõ lại như người dùng: mọi câu có token đặc biệt + phần còn lại lấy ngẫu nhiên.
    special = validation["text_rdr"].str.contains(r"\b(?:doubledot|vdotv|dotdotdot|colon\w+)\b")
    rng = np.random.default_rng(0)
    others = np.flatnonzero(~special.to_numpy())
    chosen = list(np.flatnonzero(special.to_numpy()))
    chosen += list(rng.choice(others, size=200 - len(chosen), replace=False))
    mismatches = []
    for i in chosen:
        sentence = validation["sentence"].iloc[i]
        if prepare(to_raw(sentence), use_ner=False).model_text != validation["text_rdr"].iloc[i]:
            mismatches.append(i)
    # Mục tiêu >= 98% (196/200); hiện đạt 100% nên không cho phép lệch.
    assert len(chosen) == 200
    assert len(mismatches) == 0, f"{len(mismatches)} câu lệch, chỉ số: {mismatches}"


@pytest.mark.data
@pytest.mark.slow
def test_raw_typing_variants_hold_on_the_whole_validation_set(validation):
    wrong = sum(
        prepare(to_raw(s), use_ner=False).model_text != r
        for s, r in zip(validation["sentence"], validation["text_rdr"], strict=True)
    )
    assert wrong / len(validation) <= 0.02


@pytest.mark.data
def test_no_diacritics_detector_has_no_false_alarm_on_real_validation_text(validation):
    assert not validation["no_diacritics"].any()


@pytest.mark.data
def test_no_diacritics_detector_catches_stripped_validation_text(validation):
    eligible = [s for s in validation["sentence"] if diacritic_ratio(s)[1] >= 3]
    hits = sum(prepare(strip_accents(s), use_ner=False).no_diacritics for s in eligible)
    assert len(eligible) > 1000
    assert hits / len(eligible) >= 0.95


@pytest.mark.data
def test_cached_data_was_built_by_this_normalizer_version():
    from ml.common import assert_normalizer

    assert assert_normalizer()["normalizer_version"] == NORMALIZER_VERSION
