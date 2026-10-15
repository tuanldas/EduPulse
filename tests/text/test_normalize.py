"""Quy tắc chuẩn hoá thuần Python (không cần JVM): mỗi quy tắc một ca vàng."""

import unicodedata

import pytest

from edupulse.text.normalize import (
    TONE_MAP,
    clean_text,
    collapse_repeats,
    diacritic_ratio,
    is_no_diacritics,
    map_special_tokens,
    normalize_punctuation,
    normalize_text,
    normalize_tone,
    space_punctuation,
)


def test_clean_text_composes_unicode_and_collapses_whitespace():
    composed = "Thầy dạy"
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert clean_text(decomposed) == composed
    assert clean_text("  a \t\n b c  ") == "a b c"


def test_clean_text_drops_invisible_and_control_characters():
    assert clean_text("hay​ tốt﻿\x00 lắm\x07") == "hay tốt lắm"


@pytest.mark.parametrize(("old", "new"), sorted(TONE_MAP.items()))
def test_tone_map_pairs(old, new):
    assert normalize_tone(f"x{old}y") == f"x{new}y"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("hòa đồng", "hoà đồng"),
        ("khóa học", "khoá học"),
        ("thủy điện", "thuỷ điện"),
        ("khỏe", "khoẻ"),
        ("hoà đồng", "hoà đồng"),  # kiểu mới giữ nguyên
        ("quý thầy cô", "quý thầy cô"),  # "ý" không nằm trong bảng
    ],
)
def test_normalize_tone_words(raw, expected):
    assert normalize_tone(raw) == expected
    assert normalize_tone(normalize_tone(raw)) == expected  # lũy đẳng


def test_punctuation_to_ascii():
    assert normalize_punctuation("“hay” – ‘tốt’ ！？，") == "\"hay\" - 'tốt' !?,"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tốt :))", "tốt colonsmilesmile"),
        ("tốt :)))))", "tốt colonsmilesmile"),
        ("tốt :)", "tốt colonsmile"),
        ("tốt:)", "tốt colonsmile"),
        ("hay =)", "hay colonsmile"),
        ("hay ;)", "hay colonsmile"),
        ("buồn :(", "buồn colonsad"),
        ("buồn :(((", "buồn colonsad"),
        ("vui :d", "vui colonbigsmile"),
        ("yêu <3", "yêu colonlove"),
        ("yêu :*", "yêu colonlove"),
        ("chán :v", "chán coloncontemn"),
        ("ôi :o", "ôi colonsurprise"),
        ("hihi :3", "hihi colonhihi"),
        ("hihi :>", "hihi colonsmallsmile"),
    ],
)
def test_emoticons_become_dataset_tokens(raw, expected):
    assert map_special_tokens(raw) == expected


def test_emoticons_are_mapped_before_the_colon_rule():
    # Nếu ":" bị đổi trước thì ":)" thành "doubledot )" và RDR cắt câu.
    assert "doubledot" not in map_special_tokens("tốt :) nhé")


def test_colon_is_not_an_emoticon_inside_words_or_lists():
    assert map_special_tokens("ví dụ:(1) và 12:30") == "ví dụ doubledot (1) và 12doubledot30"
    assert map_special_tokens("tỉ lệ a:v") == "tỉ lệ a doubledot v"


def test_emoji_and_non_bmp_characters_are_removed():
    assert map_special_tokens("hay 😀👍 lắm ❤️ nhé") == "hay lắm nhé"
    assert map_special_tokens("𝐡𝐚𝐲 tốt") == "tốt"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("lúc 11:55 giờ", "lúc 11doubledot55 giờ"),
        ("ví dụ: các bạn", "ví dụ doubledot các bạn"),
        ("bài tập v.v", "bài tập vdotv"),
        ("bài tập v.v.", "bài tập vdotv ."),
        ("thầy thì...", "thầy thì dotdotdot"),
        ("thầy thì…", "thầy thì dotdotdot"),
        ("thầy thì..", "thầy thì dotdotdot"),
        ("điểm 9.5 và 3.0 ghz", "điểm 9dot5 và 3dot0 ghz"),
        ("xem uit.edu.vn.", "xem uitdotedudotvn."),  # dấu chấm cuối câu để bộ tách từ xử lý
        ("lớp it003.e21 nhé", "lớp it003dote21 nhé"),
        ("học .net nhé", "học dotnet nhé"),
        ("trên turnitin.com đó", "trên turnitindotcom đó"),
        ("tốt.em thấy", "tốt . em thấy"),
        ("tot.em thay", "tot . em thay"),
    ],
)
def test_special_token_rules(raw, expected):
    assert map_special_tokens(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("quáaaa", "quá"),
        ("quáááá", "quá"),
        ("đẹpppp", "đẹp"),
        ("hayyyyy", "hay"),
        ("đẹp quá", "đẹp quá"),
        ("good", "good"),  # lặp 2 lần là bình thường
        ("năm 2000 và 1111", "năm 2000 và 1111"),  # số không bị gộp
        ("hay!!!", "hay!"),
        ("sao???", "sao?"),
        ("vâng,, ạ", "vâng, ạ"),
        ("hay ?!", "hay ?!"),
    ],
)
def test_collapse_repeats(raw, expected):
    assert collapse_repeats(raw) == expected


def test_normalize_text_full_order():
    raw = "Cô giảng HÒA đồng, :)) quáaaa!!! lúc 11:55 v.v..."
    expected = "cô giảng hoà đồng , colonsmilesmile quá ! lúc 11doubledot55 vdotv dotdotdot"
    assert normalize_text(raw) == expected


@pytest.mark.parametrize(
    "dataset_style",
    [
        "thầy dạy rất nhiệt tình colonsmile .",
        "ví dụ [đã gỡ câu UIT-VSFC] ) .",
        "[đã gỡ câu UIT-VSFC] , thầy wzjwz209 dạy thay .",
        "[đã gỡ câu UIT-VSFC] .",
        "điểm 3dot0 ghz và 11doubledot55 pm , uitdotedudotvn .",
    ],
)
def test_dataset_format_is_a_fixed_point(dataset_style):
    assert normalize_text(dataset_style) == dataset_style


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("tốt.", "tốt ."),
        ("tốt, hay!", "tốt , hay !"),
        ("(wzjwz12)", "( wzjwz12 )"),
        ('nói "abc" đó', 'nói " abc " đó'),
        ("điểm 9dot5.", "điểm 9dot5 ."),
        ("50% và e-mail", "50% và e-mail"),  # % và gạch ngang trong từ giữ nguyên
        ("  a   .  b ", "a . b"),
    ],
)
def test_space_punctuation(raw, expected):
    assert space_punctuation(raw) == expected


def test_normalize_text_keeps_alphanumeric_tokens_whole_next_to_punctuation():
    assert normalize_text("Thầy wzjwz155. Lớp it003.e21! (9.5)") == (
        "thầy wzjwz155 . lớp it003dote21 ! ( 9dot5 )"
    )


@pytest.mark.parametrize("filler", ["a", "a.", "a.b.", "1.2.", ":)", "!", "aáâ", "x-", "..", "1:"])
def test_pathological_long_inputs_are_normalised_quickly(filler):
    import time

    text = filler * (5000 // len(filler))
    start = time.perf_counter()
    normalize_text(clean_text(text))
    assert time.perf_counter() - start < 1.0


def test_diacritic_ratio_counts_only_plain_alphabetic_syllables():
    assert diacritic_ratio("thầy dạy hay") == (2, 3)
    # Bỏ qua tên thay thế, mã lớp và số.
    assert diacritic_ratio("thầy wzjwz12 it001 2000 dạy") == (2, 3 - 1)


def test_is_no_diacritics_detects_stripped_text():
    assert is_no_diacritics("giang vien day rat hay va nhiet tinh")
    assert not is_no_diacritics("giảng viên dạy rất hay và nhiệt tình")


def test_is_no_diacritics_needs_enough_syllables():
    assert is_no_diacritics("thay day hay")  # 3 âm tiết đủ để kết luận
    assert not is_no_diacritics("ok tot")  # chỉ 2 âm tiết: không đủ căn cứ


def test_is_no_diacritics_threshold_is_adjustable():
    text = "thầy day hay lam"  # 1/4 âm tiết có dấu
    assert not is_no_diacritics(text)
    assert is_no_diacritics(text, threshold=0.5)
