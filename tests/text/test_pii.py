"""Ẩn PII: regex kiểm bằng chuỗi tổng hợp (dữ liệu thật không có PII); luật tên có giới hạn rõ."""

import pandas as pd
import pytest

from edupulse.config import load_settings
from edupulse.text.pii import find_spans, mask

# 34 chuỗi PII tổng hợp: mọi chuỗi phải bị bắt (12 chuỗi cuối thêm ở bản chuẩn hoá 2).
POSITIVE = [
    "0912345678",
    "0912 345 678",
    "091.234.5678",
    "091-234-5678",
    "+84 912 345 678",
    "+84912345678",
    "84912345678",
    "(+84) 912 345 678",
    "0084912345678",
    "0386 123 456",
    "0789.123.456",
    "028 3725 2002",
    "0243 8686 868",
    "02838252002",
    "abc.def@gmail.com",
    "mssv 18520001",
    "mã số sinh viên: 20521234",
    "msv: 2151010001",
    "cccd 001095012345",
    "cmnd 079123456",
    "em là 18520001 lớp it001",
    "mssv 1852001",
    # Bản chuẩn hoá 2: SĐT bàn có ngoặc, MSSV chữ + số, URL không có http, @tên tài khoản.
    "(028) 3812 3456",
    "(024)38686868",
    "(0243) 8686 868",
    "mã sinh viên B20DCCN123",
    "B21DCCN1234",
    "instagram.com/nguyenvana",
    "github.com/x/y",
    "linkedin.com/in/ten-ban",
    "youtu.be/abcdef",
    "t.me/tenban",
    "@handle",
    "@nguyen_van.a",
]

# 36 chuỗi trông giống PII nhưng không phải: không được bắt (11 chuỗi cuối thêm ở bản 2).
NEGATIVE = [
    "điểm 8.5",
    "9/10",
    "học phí 2.500.000đ",
    "10.500.000 đồng",
    "năm 2013",
    "2023 - 2024",
    "12/10/2023",
    "7h30",
    "it001.o11",
    "cs101",
    "c2.04",
    "100.000 đồng",
    "1.000.000.000 đồng",
    "0.5 điểm",
    "điểm 10",
    "tuần 12",
    "khóa 2018",
    "còn 1234567 lỗi",
    "wifi 5ghz",
    "visual studio 2013",
    "4.0",
    "85%",
    "90 phút",
    "120 tiết",
    "0 điểm",
    # Bản chuẩn hoá 2: các mẫu mới không được ăn vào chuỗi thường gặp.
    "(0.5 điểm)",
    "(2020) cũ",
    "github.com",  # tên miền trần, không có đường dẫn
    "facebook.com",
    "B20DCCN12",  # chỉ 2 chữ số cuối
    "b20dccn123",  # chữ thường: luật MSSV chữ + số chỉ bắt chữ HOA (giới hạn đã biết)
    "IT001",
    "A12BCD12345678",  # quá nhiều chữ số cuối
    "gửi @ cho em",
    "email@",
    "mã 1712345",  # 7 chữ số không kèm từ gợi ý: dễ nhầm mã môn, mã lớp (giới hạn đã biết)
]


def test_fixture_sizes():
    assert (len(POSITIVE), len(NEGATIVE)) == (34, 36)


@pytest.mark.parametrize("text", POSITIVE)
def test_synthetic_pii_is_detected(text):
    result = mask(text)
    assert result.types, text
    # PII khác tên bị xoá khỏi đầu vào mô hình; phần chữ còn lại giữ nguyên.
    assert not any(ch.isdigit() for ch in result.model.replace("it001", "")), result.model
    assert "@" not in result.model


@pytest.mark.parametrize("text", NEGATIVE)
def test_number_like_text_is_not_pii(text):
    result = mask(text)
    assert result.types == ()
    assert result.display == result.model == text


@pytest.mark.parametrize(
    ("text", "placeholder", "pii_type", "model_view"),
    [
        ("gọi 0912345678 nhé", "[SĐT]", "phone", "gọi nhé"),
        ("gọi 028 3725 2002 nhé", "[SĐT]", "phone", "gọi nhé"),
        ("gửi abc.def@gmail.com nhé", "[EMAIL]", "email", "gửi nhé"),
        ("xem https://fb.com/abc nhé", "[URL]", "url", "xem nhé"),
        ("xem www.uit.edu.vn nhé", "[URL]", "url", "xem nhé"),
        ("mssv 18520001 nhé", "[MSSV]", "student_id", "nhé"),
        ("số 18520001 nhé", "[MSSV]", "student_id", "số nhé"),
        ("cccd 001095012345 nhé", "[CCCD]", "national_id", "nhé"),
        ("cmnd 079123456 nhé", "[CCCD]", "national_id", "nhé"),
        ("gọi (028) 3812 3456 nhé", "[SĐT]", "phone", "gọi nhé"),
        ("mã sinh viên B20DCCN123 nhé", "[MSSV]", "student_id", "mã sinh viên nhé"),
        ("xem github.com/x/y nhé", "[URL]", "url", "xem nhé"),
        ("theo dõi @handle nhé", "[URL]", "url", "theo dõi nhé"),
    ],
)
def test_display_view_uses_placeholders_and_model_view_drops_them(
    text, placeholder, pii_type, model_view
):
    result = mask(text)
    assert result.types == (pii_type,)
    assert placeholder in result.display
    assert result.model == model_view


def test_model_view_removes_non_name_pii_without_gluing_words():
    result = mask("liên hệ thầy qua 0912 345 678 hoặc abc@gmail.com nhé")
    assert result.display == "liên hệ thầy qua [SĐT] hoặc [EMAIL] nhé"
    assert result.model == "liên hệ thầy qua hoặc nhé"
    assert result.types == ("phone", "email")


@pytest.mark.parametrize("filler", ["a", "1", "a.", "aa@", "0912 "])
def test_pathological_long_inputs_are_handled_quickly(filler):
    import time

    text = filler * (5000 // len(filler))
    start = time.perf_counter()
    mask(text)
    assert time.perf_counter() - start < 1.0


def test_plain_domain_is_not_a_url():
    # "uit.edu.vn" có trong dữ liệu thật như một từ (uitdotedudotvn), không phải liên kết cá nhân.
    assert mask("vào uit.edu.vn nhé").types == ()


# --- Tên người --------------------------------------------------------------------------------


def test_name_after_honorific_is_masked_in_both_views():
    result = mask("Thầy Nam dạy hay")
    assert result.display == "Thầy [TÊN] dạy hay"
    assert result.model == "Thầy wzjwz1 dạy hay"
    assert result.types == ("name",)


def test_lowercase_name_is_not_masked_known_limitation():
    result = mask("thầy nam dạy hay")
    assert result.types == ()
    assert result.display == result.model == "thầy nam dạy hay"


def test_multi_syllable_names_and_titles():
    assert mask("Giảng viên Nguyễn Văn An dạy tốt").display == "Giảng viên [TÊN] dạy tốt"
    assert mask("ThS. Trần Thị Mai hướng dẫn").display == "ThS. [TÊN] hướng dẫn"
    assert mask("cô Nguyễn Thị Thu Hà nói nhanh").display == "cô [TÊN] nói nhanh"
    assert mask("em cảm ơn anh Huy").display == "em cảm ơn anh [TÊN]"


def test_same_name_gets_same_token_and_different_names_get_different_ones():
    result = mask("Thầy Nam và cô Lan, thầy Nam vui tính")
    assert result.model == "Thầy wzjwz1 và cô wzjwz2 , thầy wzjwz1 vui tính"
    assert result.display.count("[TÊN]") == 3


def test_name_matching_ignores_case_of_repeated_names():
    assert mask("Thầy Nam và thầy NAM").model == "Thầy wzjwz1 và thầy wzjwz1"


@pytest.mark.parametrize(
    "text",
    [
        "Thầy Cô rất nhiệt tình",  # "Cô" là từ xưng hô, không phải tên
        "Cô Giáo dạy tốt",
        "Thầy Hiệu Trưởng nói nhanh",
        "Cảm ơn thầy. Môn này hay",  # dấu chấm kết câu: "Môn" không thuộc tên
        "Em thấy thầy dạy hay",
        "THẦY NAM DẠY RẤT HAY",  # viết hoa toàn câu: chữ hoa không báo hiệu tên
        "thầy dạy hay lắm",
    ],
)
def test_capitalised_common_words_are_not_names(text):
    assert mask(text).types == ()
    assert mask(text).display == text


def test_name_never_overlaps_other_pii():
    result = mask("Thầy Nam, mssv 18520001, 0912345678")
    assert result.display == "Thầy [TÊN], [MSSV], [SĐT]"
    assert result.types == ("name", "student_id", "phone")


def test_find_spans_are_sorted_and_disjoint():
    text = "Thầy Nam gọi 0912345678 và cô Lan gửi abc@x.vn"
    spans = find_spans(text)
    assert [s.type for s in spans] == ["name", "phone", "name", "email"]
    assert all(a.end <= b.start for a, b in zip(spans, spans[1:], strict=False))


# --- Bản chuẩn hoá 2 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "display", "model"),
    [
        ("Thầy giáo Minh dạy hay", "Thầy giáo [TÊN] dạy hay", "Thầy giáo wzjwz1 dạy hay"),
        ("Cô giáo Lan rất tốt", "Cô giáo [TÊN] rất tốt", "Cô giáo wzjwz1 rất tốt"),
        (
            "thầy hiệu trưởng Hùng phát biểu",
            "thầy hiệu trưởng [TÊN] phát biểu",
            "thầy hiệu trưởng wzjwz1 phát biểu",
        ),
        (
            "Thầy Hiệu Trưởng Hùng phát biểu",
            "Thầy Hiệu Trưởng [TÊN] phát biểu",
            "Thầy Hiệu Trưởng wzjwz1 phát biểu",
        ),
        (
            "Hiệu trưởng Nguyễn Văn An phát biểu",
            "Hiệu trưởng [TÊN] phát biểu",
            "Hiệu trưởng wzjwz1 phát biểu",
        ),
        (
            "cô phó hiệu trưởng Lan nói",
            "cô phó hiệu trưởng [TÊN] nói",
            "cô phó hiệu trưởng wzjwz1 nói",
        ),
        ("cô trưởng khoa Lan nói", "cô trưởng khoa [TÊN] nói", "cô trưởng khoa wzjwz1 nói"),
        (
            "Trưởng bộ môn Trần Văn Bình họp",
            "Trưởng bộ môn [TÊN] họp",
            "Trưởng bộ môn wzjwz1 họp",
        ),
        (
            "thầy trợ giảng Minh hướng dẫn",
            "thầy trợ giảng [TÊN] hướng dẫn",
            "thầy trợ giảng wzjwz1 hướng dẫn",
        ),
        ("Trợ giảng Minh hướng dẫn", "Trợ giảng [TÊN] hướng dẫn", "Trợ giảng wzjwz1 hướng dẫn"),
        (
            "cố vấn học tập Hà nhắc",
            "cố vấn học tập [TÊN] nhắc",
            "cố vấn học tập wzjwz1 nhắc",
        ),
        (
            "thầy cố vấn học tập Hà nhắc",
            "thầy cố vấn học tập [TÊN] nhắc",
            "thầy cố vấn học tập wzjwz1 nhắc",
        ),
        (
            "thầy cô giáo Hà Nam hướng dẫn",
            "thầy cô giáo [TÊN] hướng dẫn",
            "thầy cô giáo wzjwz1 hướng dẫn",
        ),
        (
            "cô giáo chủ nhiệm Hạnh rất tốt",
            "cô giáo chủ nhiệm [TÊN] rất tốt",
            "cô giáo chủ nhiệm wzjwz1 rất tốt",
        ),
        ("cô giáo vụ Lan hỗ trợ", "cô giáo vụ [TÊN] hỗ trợ", "cô giáo vụ wzjwz1 hỗ trợ"),
        (
            "Thầy giáo Minh và thầy Minh",
            "Thầy giáo [TÊN] và thầy [TÊN]",
            "Thầy giáo wzjwz1 và thầy wzjwz1",
        ),
    ],
)
def test_names_after_teaching_titles_are_masked(text, display, model):
    result = mask(text)
    assert result.display == display
    assert result.model == model
    assert result.types == ("name",)


@pytest.mark.parametrize(
    ("text", "display"),
    [
        ("Thầy Nam dạy hay", "Thầy [TÊN] dạy hay"),
        ("Cô Nguyễn Thị Hương nói nhanh", "Cô [TÊN] nói nhanh"),
        ("TS. Lê Văn Tám hướng dẫn", "TS. [TÊN] hướng dẫn"),
        ("GV Nam dạy tốt", "GV [TÊN] dạy tốt"),
        ("bạn Lan giúp em", "bạn [TÊN] giúp em"),
        ("em cảm ơn anh Tuấn", "em cảm ơn anh [TÊN]"),
        ("liên hệ abc.def@gmail.com", "liên hệ [EMAIL]"),
        ("xem https://fb.com/abc", "xem [URL]"),
        ("xem www.uit.edu.vn", "xem [URL]"),
        ("mssv 18520001", "[MSSV]"),  # từ gợi ý nằm trong đoạn bị thay, như trước đây
        ("cccd 001095012345", "[CCCD]"),
        ("cmnd 079123456", "[CCCD]"),
    ],
)
def test_what_was_already_masked_is_still_masked_the_same_way(text, display):
    assert mask(text).display == display


@pytest.mark.parametrize(
    "text",
    [
        "Thầy giáo dạy tốt",
        "Cô giáo rất nhiệt tình",
        "Thầy cô giáo rất nhiệt tình",
        "Hiệu trưởng nói nhanh",
        "Thầy giáo Hiệu Trưởng nói nhanh",
        "Cô Giáo dạy tốt",
        "Em học Toán rất tốt",
        "Em thích môn Toán lắm",
        "Em Khoa Học Máy Tính",  # chức danh phải là cụm đủ nghĩa; "khoa" lẻ vẫn bị chặn như cũ
        "Trợ giảng nhiệt tình",
        "Cố vấn học tập hay nhắc",
    ],
)
def test_titles_alone_and_common_capitalised_words_are_not_names(text):
    assert mask(text).types == ()
    assert mask(text).display == text


def test_emails_are_not_split_into_a_handle():
    result = mask("gửi abc@gmail.com và @tenban nhé")
    assert result.display == "gửi [EMAIL] và [URL] nhé"
    assert result.types == ("email", "url")


def test_names_still_need_a_capital_letter_after_the_new_cues():
    assert mask("thầy giáo minh dạy hay").types == ()
    assert mask("cô hiệu trưởng lan nói").types == ()


@pytest.mark.xfail(
    strict=True,
    reason="mã 7 chữ số không kèm từ gợi ý: dễ nhầm mã môn, mã lớp; giới hạn đã biết (thẻ mô hình)",
)
def test_known_gap_seven_digit_codes_are_not_masked():
    assert mask("mã 1712345").types == ("student_id",)


ADVERSARIAL = {
    "cue-repeated": "thầy " * 100,
    "title-chain": "Thầy " + "Hiệu Trưởng " * 40,
    "title-chain-lowercase": "thầy " + "hiệu trưởng " * 40 + "Hùng",
    "cue-giao-repeated": "cô giáo " * 62,
    "long-blank-run": "thầy" + " " * 496,
    "parens": "(" * 500,
    "zero-parens": "(0" * 250,
    "zeros": "0" * 500,
    "landline-repeated": "(028) " * 80,
    "alnum-id-repeated": "B20DCCN123 " * 45,
    "handles": "@a" * 250,
    "ats": "@" * 500,
    "dots-then-at": "a." * 249 + "@x",
    "url-repeated": "github.com/" * 45,
    "scheme-repeated": "http://" * 70,
}


@pytest.mark.parametrize("name", ADVERSARIAL)
def test_adversarial_inputs_of_500_characters_are_masked_in_a_few_milliseconds(name):
    import time

    text = ADVERSARIAL[name]
    assert len(text) <= 520
    start = time.perf_counter()
    mask(text)
    assert time.perf_counter() - start < 0.05, name


@pytest.mark.slow
def test_ner_finds_capitalised_person_names_only():
    text = "Em thấy Nguyễn Văn An dạy rất nhiệt tình"
    assert mask(text, use_ner=True).display == "Em thấy [TÊN] dạy rất nhiệt tình"
    # Không có chữ hoa thì NER không chạy và không bắt gì.
    assert mask(text.lower(), use_ner=True).types == ()
    # Mặc định tắt NER.
    assert mask(text).types == ()


@pytest.mark.data
def test_real_sentences_have_no_pii_hits():
    processed = load_settings().data_dir / "processed"
    total = hits = 0
    for split in ("train", "validation", "test"):
        for sentence in pd.read_parquet(processed / f"{split}.parquet")["sentence"]:
            total += 1
            hits += bool(mask(sentence).types)
    assert total == 16175
    assert hits == 0
