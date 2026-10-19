"""CheckList: phép biến đổi chuỗi, cách tính từng dòng bằng bộ dự đoán giả có đáp án tính tay,
bảng/JSON kết quả, và một lượt chạy thật nhỏ trên mô hình triển khai."""

import json
import random
import re
import unicodedata
from functools import partial

import numpy as np
import pytest

from edupulse.bundle_schema import ChecklistRow
from edupulse.text.normalize import is_no_diacritics, normalize_text
from edupulse.text.pii import mask
from ml import checklist
from ml.common import best_run, load_processed

# Xác suất mẫu theo thứ tự nhãn: cảm xúc (negative, neutral, positive); chủ đề (lecturer,
# training_program, facility, others).
NEG, NEU, POS = (0.8, 0.1, 0.1), (0.1, 0.8, 0.1), (0.1, 0.1, 0.8)
LEC, PRO, FAC, OTH = (
    (0.7, 0.1, 0.1, 0.1),
    (0.1, 0.7, 0.1, 0.1),
    (0.1, 0.1, 0.7, 0.1),
    (0.1, 0.1, 0.1, 0.7),
)


def prediction(sentiment, topic, flags=None):
    n = len(sentiment)
    return checklist.Prediction(
        probs={
            "sentiment": np.asarray(sentiment, dtype=float).reshape(n, 3),
            "topic": np.asarray(topic, dtype=float).reshape(n, 4),
        },
        no_diacritics=np.zeros(n, dtype=bool) if flags is None else np.asarray(flags, dtype=bool),
    )


def table_predictor(table):
    """Bộ dự đoán giả tra bảng {văn bản: (xác suất cảm xúc, xác suất chủ đề[, cờ không dấu])}.

    Hỏi văn bản ngoài bảng thì KeyError, nên một biến thể ngoài dự kiến làm test hỏng ngay.
    """
    calls = []

    def predict(texts):
        calls.append(list(texts))
        entries = [table[text] for text in texts]
        return prediction(
            [entry[0] for entry in entries],
            [entry[1] for entry in entries],
            [entry[2] if len(entry) > 2 else False for entry in entries],
        )

    predict.calls = calls
    return predict


def constant_predictor():
    """Mô hình giả luôn trả "tích cực, giảng viên, không cờ"; ghi lại mọi văn bản được hỏi."""
    asked = []

    def predict(texts):
        asked.extend(texts)
        return prediction([POS] * len(texts), [LEC] * len(texts))

    predict.asked = asked
    return predict


# --- Biến đổi chuỗi ----------------------------------------------------------------------------


def test_case_transformations_change_only_the_letters_case():
    assert checklist.to_upper("thầy dạy hay") == "THẦY DẠY HAY"
    assert checklist.to_title_case("thầy dạy hay") == "Thầy Dạy Hay"
    assert checklist.to_title_case("thầy  dạy hay") == "Thầy  Dạy Hay"  # giữ khoảng trắng
    assert checklist.to_title_case("wzjwz12 và 3d") == "Wzjwz12 Và 3d"
    assert checklist.to_upper("123 .") == "123 ."  # không có chữ: không đổi


def test_punctuation_and_whitespace_transformations():
    assert checklist.append_suffix("thầy dạy hay", " .") == "thầy dạy hay ."
    assert checklist.append_suffix("thầy dạy hay", " !!") == "thầy dạy hay !!"
    assert checklist.double_spaces("a b c") == "a  b  c"
    assert checklist.double_spaces("abc") == "abc"  # không có dấu cách: không đổi
    assert checklist.pad_spaces("a b") == " a b "


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("thầy dạy rất hay", "cô dạy rất hay"),
        ("cô dạy rất hay", "thầy dạy rất hay"),
        ("thầy giảng , thầy hỏi", "cô giảng , cô hỏi"),  # mọi lần xuất hiện cùng đổi
        ("thầy cô đều tốt", "thầy cô đều tốt"),  # có cả hai token: giữ nguyên
        ("phòng học nóng", "phòng học nóng"),  # không có token nào: giữ nguyên
        ("thầy_giáo dạy hay", "thầy_giáo dạy hay"),  # không phải token "thầy"
        ("thầy, dạy hay", "thầy, dạy hay"),  # dấu câu dính liền: không phải token "thầy"
        ("", ""),
    ],
)
def test_swap_teacher_title_only_when_exactly_one_title_is_present(sentence, expected):
    assert checklist.swap_teacher_title(sentence) == expected


def test_swap_variants_count_each_direction_separately():
    to_co = checklist.SWAP_VARIANTS["thầy→cô"]
    to_thay = checklist.SWAP_VARIANTS["cô→thầy"]
    assert to_co("thầy dạy") == "cô dạy" and to_co("cô dạy") == "cô dạy"
    assert to_thay("cô dạy") == "thầy dạy" and to_thay("thầy dạy") == "thầy dạy"
    assert to_co("thầy cô dạy") == "thầy cô dạy"


def test_reindex_names_keeps_the_same_name_the_same_and_different_names_different():
    sentence = "thầy wzjwz12 hơn thầy wzjwz7 và wzjwz12 vẫn tốt"
    changed = checklist.reindex_names(sentence, random.Random(checklist.SEED))
    ids = [int(k) for k in re.findall(r"wzjwz(\d+)", changed)]
    assert len(ids) == 3
    assert ids[0] == ids[2] and ids[0] != ids[1]
    assert not {ids[0], ids[1]} & {12, 7}  # không chỉ số mới nào trùng chỉ số cũ
    assert all(1 <= k <= checklist.MAX_NAME_ID for k in ids)
    # Ngoài chỉ số, câu giữ nguyên từng ký tự.
    assert re.sub(r"wzjwz\d+", "N", changed) == re.sub(r"wzjwz\d+", "N", sentence)
    # Cùng hạt giống thì cùng kết quả (tái lập).
    assert changed == checklist.reindex_names(sentence, random.Random(checklist.SEED))


@pytest.mark.parametrize(
    "sentence",
    ["thầy dạy hay", "awzjwz12 dạy", "wzjwz dạy", "wzjwz12abc dạy", "wzjwz12_x dạy", ""],
)
def test_reindex_names_leaves_sentences_without_a_name_token_alone(sentence):
    assert checklist.reindex_names(sentence, random.Random(0)) == sentence


def test_reindex_names_gives_up_quietly_when_no_free_id_is_left():
    every_name = " ".join(f"wzjwz{k}" for k in range(1, checklist.MAX_NAME_ID + 1))
    assert checklist.reindex_names(every_name, random.Random(0)) == every_name


def test_insert_middle_places_the_snippet_after_half_of_the_tokens():
    assert checklist.insert_middle("a b c d", "X") == "a b X c d"
    assert checklist.insert_middle("a b c", "X") == "a X b c"
    assert checklist.insert_middle("a", "X") == "X a"
    assert checklist.insert_middle("", "X") == "X"


@pytest.mark.parametrize(
    ("label", "pii_type"), [("SĐT", "phone"), ("email", "email"), ("MSSV", "student_id")]
)
def test_inserted_personal_data_is_recognised_and_removed_by_the_pii_module(label, pii_type):
    original = "thầy dạy rất hay và nhiệt tình ."
    snippet = checklist.PII_SNIPPETS[label]
    inserted = checklist.insert_middle(original, snippet)
    assert snippet in inserted and inserted != original
    result = mask(inserted)
    assert result.types == (pii_type,)
    assert result.model == original  # phần đưa vào mô hình giống hệt câu gốc


def test_negation_and_sad_emoticon_wrap_the_sentence():
    assert checklist.add_negation("thầy dạy hay .") == "không thầy dạy hay ."
    assert checklist.add_sad_emoticon("thầy dạy hay .") == "thầy dạy hay . :("
    # Chuẩn hoá của hệ thống nhận ":(" thành token cảm xúc buồn của dữ liệu gốc.
    assert normalize_text(checklist.add_sad_emoticon("thầy dạy hay .")).split()[-1] == "colonsad"


def test_strip_diacritics_removes_marks_and_maps_d_stroke_to_d():
    assert (
        checklist.strip_diacritics("Đường đi thật tuyệt vời, ở đâu cũng được")
        == "Duong di that tuyet voi, o dau cung duoc"
    )
    assert checklist.strip_diacritics("ok 123 .") == "ok 123 ."
    assert checklist.strip_diacritics("") == ""
    decomposed = unicodedata.normalize("NFD", "thầy dạy")
    assert checklist.strip_diacritics(decomposed) == "thay day"


def test_stripped_long_sentence_is_flagged_by_the_system_but_the_original_is_not():
    long_sentence = (
        "em thấy môn học này có nhiều nội dung quan trọng nhưng thầy trình bày quá nhanh và "
        "tài liệu tham khảo không đầy đủ nên em mong khoa xem xét điều chỉnh lại lịch học"
    )
    stripped = checklist.strip_diacritics(long_sentence)
    assert stripped.isascii() and stripped != long_sentence
    assert is_no_diacritics(stripped) is True
    assert is_no_diacritics(long_sentence) is False


# Câu rỗng, chỉ có khoảng trắng, emoji, dấu kết hợp đứng riêng, chữ đổi độ dài khi viết hoa.
ODD_SENTENCES = ["", " ", "\n\t", "😀", "e" + chr(0x301) + " " + chr(0x301), "ß ǆ ﬃ", "wzjwz"]


@pytest.mark.parametrize(
    "transform",
    [
        checklist.to_upper,
        checklist.to_title_case,
        checklist.double_spaces,
        checklist.pad_spaces,
        checklist.swap_teacher_title,
        checklist.add_negation,
        checklist.add_sad_emoticon,
        checklist.strip_diacritics,
        partial(checklist.append_suffix, suffix=" ."),
        partial(checklist.insert_middle, snippet="0912345678"),
        partial(checklist.reindex_names, rng=random.Random(0)),
        *checklist.PUNCTUATION_VARIANTS.values(),
        *checklist.SWAP_VARIANTS.values(),
    ],
)
@pytest.mark.parametrize("sentence", ODD_SENTENCES)
def test_every_transformation_survives_empty_and_odd_sentences(transform, sentence):
    assert isinstance(transform(sentence), str)


# --- Dự đoán -----------------------------------------------------------------------------------


def test_prediction_rejects_inconsistent_shapes_and_missing_tasks():
    flags = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="topic"):
        checklist.Prediction(
            {"sentiment": np.zeros((2, 3)), "topic": np.zeros((3, 4))}, no_diacritics=flags
        )
    with pytest.raises(ValueError, match="nhiệm vụ"):
        checklist.Prediction({"sentiment": np.zeros((2, 3))}, no_diacritics=flags)


def test_prediction_exposes_labels_and_label_probabilities():
    result = prediction([NEG, POS], [FAC, LEC])
    assert result.labels("sentiment").tolist() == [0, 2]
    assert result.labels("topic").tolist() == [2, 0]
    assert result.prob("sentiment", "positive").tolist() == [0.1, 0.8]
    assert len(result) == 2


def test_a_predictor_that_drops_rows_is_rejected():
    def broken(texts):
        return prediction([NEG], [LEC])

    with pytest.raises(ValueError, match="trả 1 dòng cho 2 câu"):
        checklist.compute_rows(["aa bb", "cc dd"], {"sentiment": [0, 0], "topic": [0, 0]}, broken)


# --- Tính từng dòng bằng bộ dự đoán giả --------------------------------------------------------


def test_invariance_counts_only_changed_sentences_and_needs_both_labels_unchanged():
    sentences = ["aa bb", "cc dd", "ee ff", "gg hh", ""]
    base = prediction([NEG, POS, NEG, NEU, NEU], [LEC, LEC, PRO, FAC, OTH])
    predict = table_predictor(
        {
            "aa bb .": (NEG, LEC),  # giữ cả hai nhãn: đạt
            "cc dd .": (NEG, LEC),  # cảm xúc đổi: hỏng
            "ee ff .": (NEG, LEC),  # chủ đề đổi: hỏng
            "gg hh .": (NEU, FAC),  # giữ cả hai nhãn: đạt
        }
    )
    row = checklist.invariance_row(
        "thử", sentences, base, predict, {"": partial(checklist.append_suffix, suffix=" .")}
    )
    assert (row.name, row.kind, row.n, row.pass_rate) == ("thử", "invariance", 4, 0.5)
    assert "hỏng 2/4 câu" in row.note
    # Câu rỗng không đủ điều kiện nên không bao giờ được đưa vào mô hình.
    assert predict.calls == [["aa bb .", "cc dd .", "ee ff .", "gg hh ."]]


def test_a_sentence_passes_only_if_every_applicable_variant_passes():
    sentences = ["aa bb", "cc dd", "ee"]
    base = prediction([NEG, POS, NEU], [LEC, LEC, OTH])
    predict = table_predictor(
        {
            "aa bb .": (POS, LEC),  # cảm xúc đổi: hỏng
            "cc dd .": (POS, LEC),  # đạt
            "ee .": (NEU, OTH),  # đạt
            "aa  bb": (NEG, PRO),  # chủ đề đổi: hỏng
            "cc  dd": (NEG, LEC),  # cảm xúc đổi: hỏng
        }
    )
    variants = {
        "chấm": partial(checklist.append_suffix, suffix=" ."),
        "đôi": checklist.double_spaces,
    }
    row = checklist.invariance_row("thử", sentences, base, predict, variants)
    # "ee" không có dấu cách nên biến thể "đôi" không áp dụng; câu 0 và 1 hỏng ở biến thể "đôi",
    # câu 0 hỏng cả hai: hợp lại hỏng 2/3 câu.
    assert row.n == 3 and row.pass_rate == pytest.approx(1 / 3)
    assert "hỏng 2/3 câu (chấm 1, đôi 2/2)" in row.note
    assert predict.calls == [["aa bb .", "cc dd .", "ee ."], ["aa  bb", "cc  dd"]]


def test_a_row_without_eligible_sentences_has_no_pass_rate_and_never_calls_the_model():
    predict = table_predictor({})
    base = prediction([NEG, POS], [LEC, LEC])
    # Câu một từ không có dấu cách để nhân đôi nên biến thể không đổi gì.
    row = checklist.invariance_row(
        "thử", ["aa", "bb"], base, predict, {"": checklist.double_spaces}
    )
    assert (row.n, row.pass_rate) == (0, None)
    assert "không có câu đủ điều kiện" in row.note
    assert predict.calls == []


def test_negation_row_only_looks_at_confident_positive_sentences():
    sentences = ["s0", "s1", "s2", "s3", "s4", "s5"]
    base = prediction(
        [
            (0.05, 0.05, 0.90),  # tích cực chắc chắn
            (0.20, 0.30, 0.50),  # tích cực đúng 0,5: vẫn đủ điều kiện
            (0.30, 0.25, 0.45),  # nhãn tích cực nhưng P < 0,5: loại
            (0.70, 0.20, 0.10),  # tiêu cực: loại
            (0.10, 0.10, 0.80),  # tích cực
            (0.10, 0.80, 0.10),  # trung tính: loại
        ],
        [LEC] * 6,
    )
    predict = table_predictor(
        {
            "không s0": ((0.30, 0.30, 0.40), LEC),  # P(tích cực) 0,90 -> 0,40: giảm, đạt
            "không s1": ((0.25, 0.25, 0.50), LEC),  # giữ nguyên 0,50: không giảm, hỏng
            "không s4": ((0.05, 0.10, 0.85), LEC),  # 0,80 -> 0,85: tăng, hỏng
        }
    )
    row = checklist.negation_row(sentences, base, predict)
    assert (row.kind, row.n) == ("directional", 3)
    assert row.pass_rate == pytest.approx(1 / 3)
    assert "hỏng 2/3 câu" in row.note
    assert predict.calls == [["không s0", "không s1", "không s4"]]


def test_negation_row_without_confident_positive_sentences_has_no_pass_rate():
    predict = table_predictor({})
    row = checklist.negation_row(["a", "b"], prediction([NEG, NEU], [LEC, LEC]), predict)
    assert (row.kind, row.n, row.pass_rate) == ("directional", 0, None)
    assert predict.calls == []


def test_sad_emoticon_row_allows_a_small_drop_of_negative_probability():
    sentences = ["s0", "s1", "s2", "s3", ""]
    base = prediction(
        [
            (0.50, 0.30, 0.20),
            (0.50, 0.30, 0.20),
            (0.50, 0.30, 0.20),
            (0.30, 0.40, 0.30),
            NEU,
        ],
        [LEC] * 5,
    )
    predict = table_predictor(
        {
            "s0 :(": ((0.60, 0.20, 0.20), LEC),  # P(tiêu cực) tăng: đạt
            "s1 :(": ((0.495, 0.30, 0.205), LEC),  # giảm 0,005: trong dung sai, đạt
            "s2 :(": ((0.48, 0.30, 0.22), LEC),  # giảm 0,02: quá dung sai, hỏng
            "s3 :(": ((0.30, 0.40, 0.30), LEC),  # không đổi: đạt
        }
    )
    row = checklist.sad_emoticon_row(sentences, base, predict)
    assert checklist.DIRECTIONAL_TOLERANCE == 0.01
    assert (row.kind, row.n, row.pass_rate) == ("directional", 4, 0.75)  # câu rỗng bị loại
    assert "hỏng 1/4 câu" in row.note and "0,01" in row.note


def test_sad_emoticon_tolerance_boundary_is_inclusive():
    base = prediction([(0.75, 0.125, 0.125)] * 2, [LEC] * 2)
    predict = table_predictor(
        {
            "s0 :(": ((0.5, 0.25, 0.25), LEC),  # giảm đúng 0,25: đạt
            "s1 :(": ((0.4375, 0.28125, 0.28125), LEC),  # giảm 0,3125: hỏng
        }
    )
    row = checklist.sad_emoticon_row(["s0", "s1"], base, predict, tolerance=0.25)
    assert (row.n, row.pass_rate) == (2, 0.5)


def test_no_diacritics_row_reports_flag_rate_and_accuracy_drop_on_the_same_sentences():
    sentences = ["thầy dạy rất hay", "cô giảng bài khó hiểu", "phòng học nóng quá", "ok"]
    targets = {"sentiment": [2, 0, 0, 1], "topic": [0, 0, 2, 3]}
    base = prediction([POS, NEG, NEG, NEU], [LEC, LEC, FAC, OTH])
    predict = table_predictor(
        {
            "thay day rat hay": (NEG, LEC, True),  # cảm xúc sai, chủ đề đúng; bị gắn cờ
            "co giang bai kho hieu": (NEG, PRO, True),  # cảm xúc đúng, chủ đề sai; bị gắn cờ
            "phong hoc nong qua": (POS, FAC, False),  # cảm xúc sai, chủ đề đúng; lọt cờ
        }
    )
    row = checklist.no_diacritics_row(sentences, targets, base, predict)
    assert (row.kind, row.n) == ("robustness", 3)  # "ok" không có dấu nên không đủ điều kiện
    assert row.pass_rate == pytest.approx(2 / 3)
    assert "lọt cờ 1/3 câu" in row.note
    assert "cảm xúc giảm 66,7 điểm (100,0% → 33,3%)" in row.note
    assert "chủ đề giảm 33,3 điểm (100,0% → 66,7%)" in row.note


def test_no_diacritics_row_words_an_accuracy_gain_as_an_increase():
    base = prediction([NEG], [LEC])
    predict = table_predictor({"thay": (POS, LEC, True)})
    row = checklist.no_diacritics_row(["thầy"], {"sentiment": [2], "topic": [0]}, base, predict)
    assert "cảm xúc tăng 100,0 điểm (0,0% → 100,0%)" in row.note
    assert "chủ đề giảm 0,0 điểm (100,0% → 100,0%)" in row.note


def test_no_diacritics_row_without_marked_letters_has_no_pass_rate():
    predict = table_predictor({})
    row = checklist.no_diacritics_row(
        ["ok", "hi"],
        {"sentiment": [1, 1], "topic": [3, 3]},
        prediction([NEU, NEU], [OTH, OTH]),
        predict,
    )
    assert (row.kind, row.n, row.pass_rate) == ("robustness", 0, None)
    assert predict.calls == []


# --- Ghép các dòng -----------------------------------------------------------------------------

SENTENCES = [
    "thầy dạy rất hay .",
    "cô giảng bài khó hiểu quá .",
    "thầy cô wzjwz12 nhiệt tình lắm .",
    "phòng học nóng quá .",
    "ok",
]
TARGETS = {"sentiment": [2, 0, 2, 0, 1], "topic": [0, 0, 0, 2, 3]}
EXPECTED_ROWS = [
    # tên, loại, n, tỉ lệ đạt với mô hình giả luôn trả cùng một kết quả
    ("Viết HOA toàn câu", "invariance", 5, 1.0),
    ("Viết Hoa Chữ Đầu Mỗi Từ", "invariance", 5, 1.0),
    ("Dấu câu và khoảng trắng thừa", "invariance", 5, 1.0),
    ("Đổi thầy ↔ cô", "invariance", 2, 1.0),  # chỉ câu 0 (thầy) và câu 1 (cô)
    ("Đổi chỉ số tên ẩn (wzjwz + số)", "invariance", 1, 1.0),  # chỉ câu 2
    ("Chèn thông tin cá nhân", "invariance", 5, 1.0),
    ('Thêm "không" phủ định', "directional", 5, 0.0),  # P(tích cực) không đổi nên không giảm
    ('Thêm ":(" cuối câu', "directional", 5, 1.0),
    ("Bỏ dấu toàn câu", "robustness", 4, 0.0),  # "ok" không đổi; mô hình giả không gắn cờ
]


def test_compute_rows_builds_every_row_from_the_original_sentences():
    predict = constant_predictor()
    rows = checklist.compute_rows(SENTENCES, TARGETS, predict)
    assert [(r.name, r.kind, r.n, r.pass_rate) for r in rows] == EXPECTED_ROWS
    assert predict.asked[: len(SENTENCES)] == SENTENCES  # dự đoán gốc chạy lại đầu tiên

    asked = set(predict.asked)
    for expected in (
        "THẦY DẠY RẤT HAY .",
        "Thầy Dạy Rất Hay .",
        "thầy dạy rất hay . .",
        "thầy dạy rất hay . !!",
        "thầy  dạy  rất  hay  .",
        " thầy dạy rất hay . ",
        "cô dạy rất hay .",
        "thầy giảng bài khó hiểu quá .",
        "thầy dạy 0912345678 rất hay .",
        "thầy dạy ten.ho@example.com rất hay .",
        "thầy dạy mssv 18520001 rất hay .",
        "không thầy dạy rất hay .",
        "thầy dạy rất hay . :(",
        "thay day rat hay .",
    ):
        assert expected in asked, expected
    renamed = [
        int(m.group(1))
        for text in predict.asked
        if (m := re.fullmatch(r"thầy cô wzjwz(\d+) nhiệt tình lắm \.", text))
    ]
    assert renamed.count(12) == 1  # chính câu gốc
    (new_id,) = [k for k in renamed if k != 12]
    assert 1 <= new_id <= checklist.MAX_NAME_ID


def test_compute_rows_is_reproducible_with_the_fixed_seed():
    first, second = constant_predictor(), constant_predictor()
    checklist.compute_rows(SENTENCES, TARGETS, first)
    checklist.compute_rows(SENTENCES, TARGETS, second)
    assert first.asked == second.asked


def test_compute_rows_rejects_targets_that_do_not_match_the_sentences():
    with pytest.raises(ValueError, match="topic"):
        checklist.compute_rows(
            SENTENCES, {"sentiment": [0] * 5, "topic": [0] * 4}, constant_predictor()
        )


def test_compute_rows_reports_progress_through_the_log_callback():
    lines = []
    checklist.compute_rows(SENTENCES, TARGETS, constant_predictor(), log=lines.append)
    assert len(lines) == len(EXPECTED_ROWS)
    assert "Viết HOA toàn câu" in lines[0] and "100,0%" in lines[0]


def test_rows_validate_against_the_bundle_schema():
    rows = checklist.compute_rows(SENTENCES, TARGETS, constant_predictor())
    payload = checklist.build_payload(rows, run="mtl-s42", n_sentences=len(SENTENCES))
    assert set(payload) == {"split", "run", "n_sentences", "seed", "rows"}
    assert payload["split"] == "validation" and payload["seed"] == 0
    validated = [ChecklistRow.model_validate(row) for row in payload["rows"]]
    assert [row.kind for row in validated] == ["invariance"] * 6 + ["directional"] * 2 + [
        "robustness"
    ]
    for row in validated:
        assert row.pass_rate is None or 0.0 <= row.pass_rate <= 1.0
        assert (row.pass_rate is None) == (row.n == 0)


# --- Ghi kết quả -------------------------------------------------------------------------------


def test_write_outputs_creates_table_and_json_with_counts_only(tmp_path):
    rows = checklist.compute_rows(SENTENCES, TARGETS, constant_predictor())
    rows.append(ChecklistRow(name="Không có|câu", kind="invariance", n=0, note="a|b"))
    table_path, metrics_path = checklist.write_outputs(
        rows,
        run="mtl-s42",
        n_sentences=len(SENTENCES),
        temperatures={"sentiment": 1.0, "topic": 1.5},
        results_dir=tmp_path,
    )
    assert table_path == tmp_path / "tables" / "checklist.md"
    assert metrics_path == tmp_path / "metrics" / "checklist.json"

    document = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(document) == {"split", "run", "n_sentences", "seed", "rows"}
    assert (document["split"], document["run"], document["n_sentences"], document["seed"]) == (
        "validation",
        "mtl-s42",
        5,
        0,
    )
    assert len(document["rows"]) == len(EXPECTED_ROWS) + 1
    assert document["rows"][-1]["pass_rate"] is None

    table = table_path.read_text(encoding="utf-8")
    assert "n = 5 câu" in table and "`mtl-s42`" in table and "seed 0" in table
    assert "int8" in table and "cảm xúc 1,00" in table and "chủ đề 1,50" in table
    assert "Không có\\|câu" in table and "a\\|b" in table  # dấu | không làm vỡ bảng
    assert "| 0 | – |" in table  # dòng không có câu đủ điều kiện
    assert "Chạy thử" not in table
    for sentence in SENTENCES[:4]:
        assert sentence not in table and sentence not in json.dumps(document, ensure_ascii=False)


def test_partial_runs_are_labelled_in_the_table(tmp_path):
    rows = checklist.compute_rows(SENTENCES, TARGETS, constant_predictor())
    table_path, _ = checklist.write_outputs(
        rows,
        run="mtl-s42",
        n_sentences=5,
        temperatures={"sentiment": 1.0, "topic": 1.0},
        results_dir=tmp_path,
        partial_run=True,
    )
    assert "Chạy thử" in table_path.read_text(encoding="utf-8")


def test_default_results_dir_is_the_repository_results_folder():
    from edupulse.config import REPO_ROOT

    assert checklist.default_results_dir() == REPO_ROOT / "results"


# --- Dòng lệnh ---------------------------------------------------------------------------------


def fake_run_folder(tmp_path, monkeypatch):
    run_dir = tmp_path / "models" / "phobert" / "mtl-s1"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    return run_dir


def test_main_asks_for_export_when_the_onnx_model_is_missing(monkeypatch, tmp_path, capsys):
    fake_run_folder(tmp_path, monkeypatch)
    assert checklist.main(["--run", "mtl-s1"]) == 2
    assert "make export" in capsys.readouterr().err


def test_main_asks_for_calibration_when_only_the_onnx_model_exists(monkeypatch, tmp_path, capsys):
    fake_run_folder(tmp_path, monkeypatch)
    model = tmp_path / "outputs" / "onnx" / "mtl-s1" / "model.int8.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"")
    assert checklist.main(["--run", "mtl-s1"]) == 2
    assert "make calibrate" in capsys.readouterr().err


def test_parser_defaults_and_limit_validation():
    args = checklist.build_parser().parse_args([])
    assert args.run == "best" and args.limit is None
    assert checklist.build_parser().parse_args(["--run", "mtl-s43", "--limit", "5"]).limit == 5
    for bad in ("0", "-3", "abc"):
        with pytest.raises(SystemExit):
            checklist.build_parser().parse_args(["--limit", bad])


def test_run_rejects_a_non_positive_limit():
    with pytest.raises(ValueError, match="limit"):
        checklist.run("best", limit=0)


# --- Chạy thật trên mô hình triển khai ---------------------------------------------------------


@pytest.fixture(scope="module")
def deployed_run():
    """Run triển khai cùng ONNX int8 và calibration.json; thiếu thì bỏ qua."""
    try:
        run_dir = best_run()
        checklist.load_deployment(run_dir)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    return run_dir


@pytest.mark.data
@pytest.mark.java
@pytest.mark.train
@pytest.mark.slow
def test_run_on_the_deployed_model_writes_valid_results_without_any_sentence(
    deployed_run, tmp_path
):
    payload = checklist.run("best", limit=60, results_dir=tmp_path, log=lambda _message: None)
    assert (payload["split"], payload["n_sentences"], payload["seed"]) == ("validation", 60, 0)
    assert payload["run"] == deployed_run.name

    rows = [ChecklistRow.model_validate(row) for row in payload["rows"]]
    assert [(row.name, row.kind) for row in rows] == [
        (name, kind) for name, kind, _, _ in EXPECTED_ROWS
    ]
    for row in rows:
        assert 0 <= row.n <= 60
        assert (row.pass_rate is None) == (row.n == 0)
        assert row.pass_rate is None or 0.0 <= row.pass_rate <= 1.0
    # Thông tin cá nhân bị xoá trước khi tới mô hình nên nhãn không bao giờ đổi.
    pii = next(row for row in rows if row.name == "Chèn thông tin cá nhân")
    assert pii.n == 60 and pii.pass_rate == 1.0

    table = (tmp_path / "tables" / "checklist.md").read_text(encoding="utf-8")
    metrics = (tmp_path / "metrics" / "checklist.json").read_text(encoding="utf-8")
    assert json.loads(metrics) == payload
    assert "Chạy thử" in table  # chỉ 60/1.583 câu
    for sentence in load_processed("validation")["sentence"].tolist()[:60]:
        if len(sentence) >= 20:
            assert sentence not in table and sentence not in metrics
