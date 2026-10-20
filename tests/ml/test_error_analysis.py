"""Phân tích lỗi và thiên lệch: khung dữ liệu và dự đoán tổng hợp nhỏ, đáp án tính tay."""

import csv
import json
import unicodedata

import numpy as np
import pandas as pd
import pytest

from edupulse.labels import LABELS, TASKS
from ml import error_analysis as ea
from ml import preds as preds_lib
from ml import preprocess
from ml.common import prediction_metrics, write_json
from ml.preds import DEPLOY_KEY, DONE_FILE, META_FILE, EvalContext, Preds, save_preds

REFERENCE = ea.REFERENCE_KEY


@pytest.fixture(autouse=True)
def no_real_data(monkeypatch):
    """Không test nào được đọc dữ liệu thật; test cần khung dữ liệu thì thay bằng `stage`."""

    def refuse(split, *, allow_test=False):
        raise AssertionError(f"không được đọc dữ liệu thật ({split})")

    monkeypatch.setattr(preds_lib, "load_processed", refuse)


def make_preds(key, y, pred, conf=None):
    """Dự đoán tổng hợp: lớp được đoán nhận xác suất `conf`, phần còn lại chia đều.

    logits = log xác suất nên softmax trả lại đúng `conf` (mặc định 0,9).
    """
    logits = {}
    for task in TASKS:
        k, guess = len(LABELS[task]), np.asarray(pred[task])
        level = np.full(len(guess), 0.9) if conf is None else np.asarray(conf[task], dtype=float)
        probs = np.tile(((1 - level) / (k - 1))[:, None], (1, k))
        probs[np.arange(len(guess)), guess] = level
        logits[task] = np.log(probs)
    return Preds(
        key,
        y={task: np.asarray(y[task]) for task in TASKS},
        pred={task: np.asarray(pred[task]) for task in TASKS},
        logits=logits,
    )


# Mười hai câu có đáp án tính tay. Chủ đề thật 0 giảng viên, 1 chương trình, 2 cơ sở vật chất,
# 3 khác; cảm xúc thật 0 tiêu cực, 1 trung tính, 2 tích cực.
BASE = {
    "sentence": [
        "thầy dạy rất hay",
        "thầy giảng dễ hiểu",
        "cô chấm bài quá chậm",
        "giảng viên dạy bình thường",
        "chương trình học nặng quá",
        "bài tập côngg việc quá nhiều",
        "cơ sở vật chất xuống cấp",
        "máy chiếu hay hỏng",
        "thầy cô và sinh viên cùng góp ý",
        "chúc nhà trường nhiều sức khỏe",
        "cô đăng ký môn học bị lỗi",
        "thầy giảng quá nhanh",
    ],
    "topic": [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 3, 0],
    "sentiment": [2, 2, 0, 1, 0, 0, 0, 0, 1, 2, 0, 0],
    "n_words": [10, 45, 8, 6, 12, 41, 7, 9, 5, 40, 4, 3],
    "main_sentiment": [2, 1, 0, 2, 0, 0, 0, 2, 1, 2, 0, 0],  # sai ở hàng 1, 3, 7
    "main_topic": [0, 0, 0, 0, 3, 1, 2, 2, 3, 0, 3, 0],  # sai ở hàng 4, 9
    "ref_sentiment": [2, 2, 1, 0, 0, 2, 0, 0, 1, 2, 1, 0],  # sai ở hàng 2, 3, 5, 10
    "ref_topic": [0, 3, 0, 0, 3, 1, 2, 1, 3, 3, 3, 0],  # sai ở hàng 1, 4, 7
}
SPECIAL_TEXT = {3: "giảng_viên dạy bình_thường wzjwz5", 10: "cô đăng_ký môn_học bị lỗi colonsmile"}
NO_MARKS_ROWS = {6, 7}  # chữ không dấu


def base_frame():
    return pd.DataFrame(
        {
            "sentence": BASE["sentence"],
            "text_rdr": [SPECIAL_TEXT.get(i, s) for i, s in enumerate(BASE["sentence"])],
            "sentiment": BASE["sentiment"],
            "topic": BASE["topic"],
            "n_words": BASE["n_words"],
            "no_diacritics": [i in NO_MARKS_ROWS for i in range(len(BASE["sentence"]))],
        }
    )


def base_case():
    y = {"sentiment": BASE["sentiment"], "topic": BASE["topic"]}
    deployed = make_preds(
        DEPLOY_KEY, y, {"sentiment": BASE["main_sentiment"], "topic": BASE["main_topic"]}
    )
    reference = make_preds(
        REFERENCE, y, {"sentiment": BASE["ref_sentiment"], "topic": BASE["ref_topic"]}
    )
    return base_frame(), deployed, reference


def counts(entry):
    """(n, int8 sai cảm xúc, int8 sai chủ đề, linearsvc sai cảm xúc, linearsvc sai chủ đề)."""
    return (
        entry["n"],
        entry[DEPLOY_KEY]["sentiment"]["errors"],
        entry[DEPLOY_KEY]["topic"]["errors"],
        entry[REFERENCE]["sentiment"]["errors"],
        entry[REFERENCE]["topic"]["errors"],
    )


# --- chỉ số nhỏ -----------------------------------------------------------------------------


def test_scores_average_only_over_classes_present_in_the_slice():
    # Lớp 1 và 2 vắng ở cả nhãn thật lẫn dự đoán: bị bỏ, không kéo macro-F1 xuống 1/3.
    assert ea.scores(np.array([0, 0]), np.array([0, 0]))["macro_f1"] == pytest.approx(1.0)
    # Lớp 1 có trong nhãn thật nhưng không được đoán: F1 = 0 vẫn tính, như bảng chính.
    found = ea.scores(np.array([0, 1]), np.array([0, 0]))
    assert found["macro_f1"] == pytest.approx((2 / 3 + 0.0) / 2)
    assert (found["n"], found["errors"], found["accuracy"]) == (2, 1, 0.5)


def test_scores_of_an_empty_slice_are_none_not_a_division_error():
    found = ea.scores(np.array([], dtype=int), np.array([], dtype=int))
    assert found == {"n": 0, "errors": 0, "accuracy": None, "macro_f1": None}


def test_scores_on_a_full_set_equal_the_main_table_metrics():
    frame, deployed, _ = base_case()
    expected = prediction_metrics(BASE["sentiment"], BASE["main_sentiment"], LABELS["sentiment"])
    found = ea.scores(frame["sentiment"].to_numpy(), deployed.pred["sentiment"])
    assert found["accuracy"] == pytest.approx(expected["accuracy"])
    assert found["macro_f1"] == pytest.approx(expected["macro_f1"])


# --- kiểm tra đầu vào -------------------------------------------------------------------------


def test_inputs_must_match_the_frame_row_for_row():
    frame, deployed, reference = base_case()
    models = {DEPLOY_KEY: deployed, REFERENCE: reference}
    ea.check_inputs(frame, models)  # khớp thì không báo gì

    with pytest.raises(ea.InputError, match="thiếu cột"):
        ea.check_inputs(frame.drop(columns=["text_rdr"]), models)
    with pytest.raises(ea.InputError, match="rỗng"):
        ea.check_inputs(frame.iloc[:0], models)
    with pytest.raises(ea.InputError, match="12 dòng dự đoán nhưng khung dữ liệu có 11"):
        ea.check_inputs(frame.iloc[:-1], models)
    reversed_frame = frame.iloc[::-1].reset_index(drop=True)
    with pytest.raises(ea.InputError, match="nhãn thật khác khung dữ liệu"):
        ea.check_inputs(reversed_frame, models)


def test_predicted_labels_outside_the_label_range_are_refused():
    frame, deployed, reference = base_case()
    shifted = {**deployed.pred, "sentiment": deployed.pred["sentiment"] + 1}  # có nhãn 3
    broken = Preds(DEPLOY_KEY, y=deployed.y, pred=shifted, logits=deployed.logits)
    with pytest.raises(ea.InputError, match="nhãn dự đoán nằm ngoài"):
        ea.check_inputs(frame, {DEPLOY_KEY: broken, REFERENCE: reference})


def test_a_model_without_one_of_the_tasks_is_refused():
    frame, deployed, reference = base_case()
    single = Preds(
        REFERENCE,
        y={"sentiment": reference.y["sentiment"]},
        pred={"sentiment": reference.pred["sentiment"]},
        logits={"sentiment": reference.logits["sentiment"]},
    )
    with pytest.raises(ea.InputError, match="không có dự đoán cho tác vụ topic"):
        ea.check_inputs(frame, {DEPLOY_KEY: deployed, REFERENCE: single})


# --- 1. cặp nhầm lẫn --------------------------------------------------------------------------


def test_confusion_pairs_are_sorted_by_error_count_with_share_and_class_recall():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    pred = np.array([0, 0, 1, 2, 1, 0, 0, 2, 2, 0])
    pairs = ea.confusion_pairs(y, pred, LABELS["sentiment"])
    # 5 lỗi: trung tính -> tiêu cực x2 (đứng đầu); ba cặp còn lại 1 lỗi, hoà thì theo thứ tự nhãn.
    assert [(p["true"], p["pred"], p["count"]) for p in pairs] == [
        ("neutral", "negative", 2),
        ("negative", "neutral", 1),
        ("negative", "positive", 1),
        ("positive", "negative", 1),
    ]
    assert [p["share_of_errors"] for p in pairs] == pytest.approx([0.4, 0.2, 0.2, 0.2])
    assert [p["true_support"] for p in pairs] == [3, 4, 4, 3]
    # recall lớp thật: trung tính 1/3, tiêu cực 2/4, tích cực 2/3
    assert [p["true_recall"] for p in pairs] == pytest.approx([1 / 3, 1 / 2, 1 / 2, 2 / 3])


def test_confusion_pairs_of_a_model_without_errors_is_empty():
    labels = LABELS["sentiment"]
    assert ea.confusion_pairs(np.array([0, 1, 2]), np.array([0, 1, 2]), labels) == []


def test_confusion_pairs_of_the_base_case_per_task():
    _, deployed, _ = base_case()
    sentiment = ea.confusion_pairs(
        deployed.y["sentiment"], deployed.pred["sentiment"], LABELS["sentiment"]
    )
    assert [(p["true"], p["pred"]) for p in sentiment] == [
        ("negative", "positive"),
        ("neutral", "positive"),
        ("positive", "neutral"),
    ]
    assert [p["true_support"] for p in sentiment] == [7, 2, 3]
    assert [p["true_recall"] for p in sentiment] == pytest.approx([6 / 7, 1 / 2, 2 / 3])
    topic = ea.confusion_pairs(deployed.y["topic"], deployed.pred["topic"], LABELS["topic"])
    assert [(p["true"], p["pred"]) for p in topic] == [
        ("training_program", "others"),
        ("others", "lecturer"),
    ]
    assert [p["share_of_errors"] for p in topic] == [0.5, 0.5]


# --- 2. lỗi tự tin nhất ------------------------------------------------------------------------


def test_confident_errors_take_the_highest_confidence_errors_of_each_task():
    sentences = [f"câu mẫu số {i:02d}" for i in range(10)]
    frame = pd.DataFrame({"sentence": sentences})
    y = {"sentiment": [0, 0, 0, 0, 2, 1, 1, 1, 2, 2], "topic": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]}
    pred = {"sentiment": [0, 1, 1, 0, 1, 1, 0, 2, 2, 0], "topic": [0, 1, 2, 3, 1, 0, 3, 3, 3, 1]}
    # Sai cảm xúc ở hàng 1, 2, 4, 6, 7, 9; hàng 2 và 4 hoà 0,95. Hàng 0 và 8 đúng nhưng tự tin hơn.
    conf = {
        "sentiment": [0.99, 0.80, 0.95, 0.70, 0.95, 0.60, 0.85, 0.90, 0.97, 0.55],
        "topic": [0.9, 0.9, 0.9, 0.9, 0.60, 0.99, 0.80, 0.9, 0.75, 0.9],
    }
    deployed = make_preds(DEPLOY_KEY, y, pred, conf)
    rows = ea.confident_errors(frame, deployed, "test", per_task=3)

    assert [(r["task"], r["row"]) for r in rows] == [
        ("sentiment", 2),
        ("sentiment", 4),  # hoà độ tin cậy: hàng nhỏ hơn đứng trước
        ("sentiment", 7),
        ("topic", 5),
        ("topic", 6),
        ("topic", 8),  # hàng 4 (0,60) bị cắt
    ]
    assert [r["sentence"] for r in rows] == [sentences[i] for i in (2, 4, 7, 5, 6, 8)]
    assert [(r["true"], r["pred"]) for r in rows] == [
        ("negative", "neutral"),
        ("positive", "neutral"),
        ("neutral", "positive"),
        ("training_program", "lecturer"),
        ("facility", "others"),
        ("lecturer", "others"),
    ]
    assert [r["confidence"] for r in rows] == pytest.approx(
        [0.95, 0.95, 0.90, 0.99, 0.80, 0.75], abs=1e-6
    )
    assert {r["split"] for r in rows} == {"test"}
    assert list(rows[0]) == list(ea.FULL_SAMPLE_COLUMNS)


def test_a_hundred_errors_are_cut_to_fifteen_per_task_and_thirty_rows():
    n = 100
    frame = pd.DataFrame({"sentence": [f"câu mẫu số {i:03d}" for i in range(n)]})
    y = {"sentiment": np.zeros(n, dtype=int), "topic": np.zeros(n, dtype=int)}
    pred = {"sentiment": np.ones(n, dtype=int), "topic": np.ones(n, dtype=int)}  # sai hết
    rising = 0.5 + 0.004 * np.arange(n)
    conf = {"sentiment": rising, "topic": rising[::-1]}
    rows = ea.confident_errors(frame, make_preds(DEPLOY_KEY, y, pred, conf), "validation")

    assert len(rows) == ea.MAX_SAMPLE_ROWS == 30
    assert [r["row"] for r in rows if r["task"] == "sentiment"] == list(range(99, 84, -1))
    assert [r["row"] for r in rows if r["task"] == "topic"] == list(range(15))


def test_error_samples_writer_refuses_more_than_thirty_rows(tmp_path):
    row = {column: "x" for column in ea.SAMPLE_COLUMNS}
    ea.write_error_samples(tmp_path / "ok.csv", [row] * 30)
    with pytest.raises(ValueError, match="tối đa 30 dòng"):
        ea.write_error_samples(tmp_path / "too_many.csv", [row] * 31)
    assert not (tmp_path / "too_many.csv").exists()


AWKWARD_ROW = {
    "split": "validation",
    "row": 4,
    "task": "topic",
    "true": "others",
    "pred": "lecturer",
    "confidence": 0.987654,
    "sentence": 'thầy nói "ổn", nhưng\nchưa hết',
}


def test_the_full_error_samples_csv_roundtrips_commas_quotes_and_newlines(tmp_path):
    ea.write_error_samples_full(tmp_path / "full.csv", [AWKWARD_ROW])
    with (tmp_path / "full.csv").open(encoding="utf-8", newline="") as handle:
        back = list(csv.DictReader(handle))
    assert back == [{key: str(value) for key, value in AWKWARD_ROW.items()}]


def test_the_error_samples_csv_for_the_repo_never_holds_the_sentence(tmp_path):
    ea.write_error_samples(tmp_path / "samples.csv", [AWKWARD_ROW])  # dòng còn mang câu gốc
    text = (tmp_path / "samples.csv").read_text(encoding="utf-8")
    assert "sentence" not in text and "thầy nói" not in text
    with (tmp_path / "samples.csv").open(encoding="utf-8", newline="") as handle:
        back = list(csv.DictReader(handle))
    assert (
        list(back[0])
        == list(ea.SAMPLE_COLUMNS)
        == [
            "split",
            "row",
            "task",
            "true",
            "pred",
            "confidence",
        ]
    )
    assert back[0]["row"] == "4" and back[0]["confidence"] == "0.987654"


# --- 3. lỗi theo lát ---------------------------------------------------------------------------


def test_error_slices_by_true_topic():
    frame, deployed, reference = base_case()
    slices = ea.error_slices(frame, {DEPLOY_KEY: deployed, REFERENCE: reference})
    assert counts(slices["overall"]) == (12, 3, 2, 4, 3)
    by_topic = slices["by_true_topic"]
    assert counts(by_topic["lecturer"]) == (5, 2, 0, 2, 1)
    assert counts(by_topic["training_program"]) == (2, 0, 1, 1, 1)
    assert counts(by_topic["facility"]) == (2, 1, 0, 0, 1)
    assert counts(by_topic["others"]) == (3, 0, 1, 1, 0)
    lecturer = by_topic["lecturer"][DEPLOY_KEY]["sentiment"]
    assert lecturer["error_rate"] == pytest.approx(2 / 5)


def test_error_slices_by_length_cut_at_forty_words_exclusive():
    frame, deployed, reference = base_case()
    by_length = ea.error_slices(frame, {DEPLOY_KEY: deployed, REFERENCE: reference})["by_length"]
    # 41 và 45 từ là dài; đúng 40 từ (hàng 9) vẫn là ngắn.
    assert counts(by_length["long"]) == (2, 1, 0, 1, 1)
    assert counts(by_length["short"]) == (10, 2, 2, 3, 2)


def test_error_slices_by_special_token_use_the_preprocess_pattern():
    assert ea.SPECIAL_TOKEN is preprocess._SPECIAL
    frame, deployed, reference = base_case()
    by_special = ea.error_slices(frame, {DEPLOY_KEY: deployed, REFERENCE: reference})[
        "by_special_token"
    ]
    # hàng 3 có wzjwz5, hàng 10 có colonsmile
    assert counts(by_special["with"]) == (2, 1, 0, 2, 0)
    assert counts(by_special["without"]) == (10, 2, 2, 2, 3)


def test_error_slices_by_diacritics():
    frame, deployed, reference = base_case()
    by_marks = ea.error_slices(frame, {DEPLOY_KEY: deployed, REFERENCE: reference})["by_diacritics"]
    assert counts(by_marks["no_diacritics"]) == (2, 1, 0, 0, 1)
    assert counts(by_marks["with_diacritics"]) == (10, 2, 2, 4, 2)


def test_an_empty_slice_reports_zero_rows_and_a_note_instead_of_a_rate():
    frame, deployed, reference = base_case()
    frame["no_diacritics"] = False  # không câu nào gõ không dấu
    by_marks = ea.error_slices(frame, {DEPLOY_KEY: deployed, REFERENCE: reference})["by_diacritics"]
    empty = by_marks["no_diacritics"]
    assert empty["n"] == 0
    assert empty[DEPLOY_KEY]["sentiment"] == {"errors": 0, "error_rate": None}
    assert "n = 0" in empty["note"]
    assert "note" not in by_marks["with_diacritics"]
    assert by_marks["with_diacritics"]["n"] == 12


# --- 4. giao lỗi -------------------------------------------------------------------------------


def test_error_overlap_counts_and_share_of_main_errors_shared_with_the_baseline():
    _, deployed, reference = base_case()
    overlap = ea.error_overlap(deployed, reference)
    # cảm xúc: int8 sai {1,3,7}, linearsvc sai {2,3,5,10}
    sentiment = overlap["sentiment"]
    assert (sentiment["both_wrong"], sentiment["only_main_wrong"]) == (1, 2)
    assert (sentiment["only_reference_wrong"], sentiment["both_right"]) == (3, 6)
    assert (sentiment["main_errors"], sentiment["reference_errors"]) == (3, 4)
    assert sentiment["main_errors_also_reference_wrong"] == pytest.approx(1 / 3)
    # chủ đề: int8 sai {4,9}, linearsvc sai {1,4,7}
    topic = overlap["topic"]
    assert (topic["both_wrong"], topic["only_main_wrong"]) == (1, 1)
    assert (topic["only_reference_wrong"], topic["both_right"]) == (2, 8)
    assert topic["main_errors_also_reference_wrong"] == pytest.approx(1 / 2)
    for task in TASKS:
        parts = ("both_wrong", "only_main_wrong", "only_reference_wrong", "both_right")
        assert sum(overlap[task][part] for part in parts) == overlap[task]["n"] == 12


def test_error_overlap_share_is_none_when_the_main_model_has_no_errors():
    y = {"sentiment": BASE["sentiment"], "topic": BASE["topic"]}
    perfect = make_preds(DEPLOY_KEY, y, y)
    _, _, reference = base_case()
    overlap = ea.error_overlap(perfect, reference)
    assert overlap["sentiment"]["main_errors"] == 0
    assert overlap["sentiment"]["main_errors_also_reference_wrong"] is None
    assert overlap["sentiment"]["only_reference_wrong"] == 4  # linearsvc sai {2,3,5,10}


# --- 5a. ô chủ đề x cảm xúc ----------------------------------------------------------------------


def test_topic_sentiment_cells_of_the_base_case():
    frame, deployed, _ = base_case()
    cells = ea.topic_sentiment_cells(frame, deployed)
    assert [(c["topic"], c["sentiment"], c["n"], c["recall"]) for c in cells] == [
        ("lecturer", "negative", 2, 1.0),
        ("lecturer", "neutral", 1, 0.0),
        ("lecturer", "positive", 2, 0.5),
        ("training_program", "negative", 2, 1.0),
        ("training_program", "neutral", 0, None),
        ("training_program", "positive", 0, None),
        ("facility", "negative", 2, 0.5),
        ("facility", "neutral", 0, None),
        ("facility", "positive", 0, None),
        ("others", "negative", 1, 1.0),
        ("others", "neutral", 1, 1.0),
        ("others", "positive", 1, 1.0),
    ]
    assert {c["task"] for c in cells} == {"sentiment"}
    assert all(c["small_sample"] for c in cells)  # mọi ô đều dưới 10 câu


def test_topic_sentiment_cells_flag_small_samples_at_the_ten_row_boundary():
    spec = [
        # (chủ đề, cảm xúc, số câu đoán đúng, số câu đoán sai)
        (0, 2, 8, 2),  # giảng viên x tích cực: n = 10, đủ lớn
        (0, 0, 6, 3),  # giảng viên x tiêu cực: n = 9, nhỏ
        (3, 2, 0, 1),  # khác x tích cực: n = 1, recall 0
    ]
    topic, sentiment, pred = [], [], []
    for topic_id, sentiment_id, right, wrong in spec:
        topic += [topic_id] * (right + wrong)
        sentiment += [sentiment_id] * (right + wrong)
        pred += [sentiment_id] * right + [(sentiment_id + 1) % 3] * wrong
    frame = pd.DataFrame({"topic": topic, "sentiment": sentiment})
    y = {"sentiment": sentiment, "topic": topic}
    deployed = make_preds(DEPLOY_KEY, y, {"sentiment": pred, "topic": topic})

    cells = {(c["topic"], c["sentiment"]): c for c in ea.topic_sentiment_cells(frame, deployed)}
    assert sum(c["n"] for c in cells.values()) == 20 and len(cells) == 12
    assert cells[("lecturer", "positive")]["n"] == 10
    assert cells[("lecturer", "positive")]["recall"] == pytest.approx(0.8)
    assert cells[("lecturer", "positive")]["small_sample"] is False
    assert cells[("lecturer", "negative")]["n"] == 9
    assert cells[("lecturer", "negative")]["recall"] == pytest.approx(6 / 9)
    assert cells[("lecturer", "negative")]["small_sample"] is True
    assert cells[("others", "positive")]["recall"] == 0.0
    assert cells[("facility", "neutral")]["recall"] is None
    assert cells[("facility", "neutral")]["small_sample"] is True


def test_bias_cells_csv_has_the_documented_columns_and_blank_recall_for_empty_cells(tmp_path):
    frame, deployed, _ = base_case()
    ea.write_bias_cells(tmp_path / "bias.csv", ea.topic_sentiment_cells(frame, deployed))
    with (tmp_path / "bias.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["task", "topic", "sentiment", "n", "recall", "small_sample"]
    assert len(rows) == 12
    assert rows[0] == {
        "task": "sentiment",
        "topic": "lecturer",
        "sentiment": "negative",
        "n": "2",
        "recall": "1.0",
        "small_sample": "True",
    }
    empty = next(r for r in rows if (r["topic"], r["sentiment"]) == ("facility", "positive"))
    assert (empty["n"], empty["recall"], empty["small_sample"]) == ("0", "", "True")


# --- 5b. baseline chỉ-dùng-chủ-đề ---------------------------------------------------------------


def test_topic_only_baseline_and_prior_cells():
    frame, deployed, reference = base_case()
    result = ea.topic_only_baseline(frame, {DEPLOY_KEY: deployed, REFERENCE: reference})
    # Đoán: giảng viên -> tích cực, còn lại -> tiêu cực; đúng ở hàng 0,1,4,5,6,7,10 (7/12).
    assert result["topic_only"]["accuracy"] == pytest.approx(7 / 12)
    # F1 tiêu cực 5/7, trung tính 0 (không bao giờ được đoán), tích cực 1/2 -> macro 17/42.
    assert result["topic_only"]["macro_f1"] == pytest.approx(17 / 42)
    assert result["cells"] == {"on_prior": 7, "against_prior": 5}

    main = result["models"][DEPLOY_KEY]
    assert main["overall"]["accuracy"] == pytest.approx(9 / 12)
    assert main["overall"]["macro_f1"] == pytest.approx((12 / 13 + 1 / 2 + 4 / 7) / 3)
    assert (main["on_prior"]["n"], main["against_prior"]["n"]) == (7, 5)
    assert main["on_prior"]["accuracy"] == pytest.approx(5 / 7)
    assert main["against_prior"]["accuracy"] == pytest.approx(4 / 5)
    assert main["gap"] == pytest.approx(5 / 7 - 4 / 5)

    other = result["models"][REFERENCE]
    assert other["overall"]["accuracy"] == pytest.approx(8 / 12)
    assert other["overall"]["macro_f1"] == pytest.approx((2 / 3 + 2 / 5 + 6 / 7) / 3)
    assert other["on_prior"]["accuracy"] == pytest.approx(5 / 7)
    assert other["against_prior"]["accuracy"] == pytest.approx(3 / 5)
    assert other["gap"] == pytest.approx(5 / 7 - 3 / 5)


def test_topic_only_baseline_has_no_gap_when_every_row_follows_the_prior():
    frame = pd.DataFrame({"topic": [0, 1, 2], "sentiment": [2, 0, 0]})  # đều theo prior
    y = {"sentiment": [2, 0, 0], "topic": [0, 1, 2]}
    deployed = make_preds(DEPLOY_KEY, y, y)
    result = ea.topic_only_baseline(frame, {DEPLOY_KEY: deployed})
    assert result["cells"] == {"on_prior": 3, "against_prior": 0}
    assert result["models"][DEPLOY_KEY]["against_prior"]["accuracy"] is None
    assert result["models"][DEPLOY_KEY]["gap"] is None


# --- 5c. thầy / cô ----------------------------------------------------------------------------


def test_teacher_slices_of_the_base_case():
    frame, deployed, _ = base_case()
    result = ea.teacher_slices(frame, deployed)
    assert result["n_both"] == 1  # hàng 8 có cả "thầy" lẫn "cô": bỏ khỏi cả hai lát

    thay, co = result["thay"], result["co"]  # thầy: hàng 0, 1, 11; cô: hàng 2, 10
    assert (thay["n"], co["n"]) == (3, 2)
    assert thay["negative_rate_true"] == pytest.approx(1 / 3)
    assert thay["negative_rate_pred"] == pytest.approx(1 / 3)
    assert thay["accuracy"] == pytest.approx(2 / 3)
    # Nhãn thật [2,2,0], dự đoán [2,1,0]: F1 tiêu cực 1, tích cực 2/3, và trung tính 0 (không có
    # câu thật nhưng bị đoán nhầm một câu, nên lớp vẫn có mặt trong lát).
    assert thay["macro_f1"] == pytest.approx((1 + 0 + 2 / 3) / 3)
    assert (co["negative_rate_true"], co["negative_rate_pred"]) == (1.0, 1.0)
    assert (co["accuracy"], co["macro_f1"]) == (1.0, 1.0)  # chỉ có lớp tiêu cực trong lát
    assert thay["small_sample"] and co["small_sample"]

    difference = result["difference"]
    assert difference["negative_rate_true"] == pytest.approx(1 / 3 - 1)
    assert difference["negative_rate_pred"] == pytest.approx(1 / 3 - 1)
    assert difference["accuracy"] == pytest.approx(2 / 3 - 1)
    assert difference["macro_f1"] == pytest.approx(5 / 9 - 1)


def test_teacher_slices_match_whole_tokens_not_substrings():
    sentences = [
        "thầy dạy hay",  # thầy
        "cô dạy hay",  # cô
        "cơ sở vật chất tốt",  # "cơ" khác "cô"
        "côngg việc nhiều",  # "côngg" chỉ chứa chuỗi "cô"
        "thầy cô cùng dạy",  # cả hai: bỏ khỏi cả hai lát
        "Thầy dạy chậm",  # viết hoa vẫn là token thầy
        unicodedata.normalize("NFD", "thầy dạy rõ"),  # dấu tổ hợp rời vẫn là token thầy
        unicodedata.normalize("NFD", "cô giáo hiền"),
        "thầyy dạy",  # "thầyy" chỉ bắt đầu bằng "thầy"
    ]
    zeros = {"sentiment": [0] * len(sentences), "topic": [0] * len(sentences)}
    frame = pd.DataFrame({"sentence": sentences, **zeros})
    result = ea.teacher_slices(frame, make_preds(DEPLOY_KEY, zeros, zeros))
    assert (result["thay"]["n"], result["co"]["n"], result["n_both"]) == (3, 2, 1)


def test_teacher_slices_are_empty_without_any_teacher_token():
    frame = pd.DataFrame({"sentence": ["giờ học muộn", "phòng nóng"], "sentiment": [0, 0]})
    y = {"sentiment": [0, 0], "topic": [0, 0]}
    result = ea.teacher_slices(frame, make_preds(DEPLOY_KEY, y, y))
    assert (result["thay"]["n"], result["co"]["n"], result["n_both"]) == (0, 0, 0)
    assert result["thay"]["accuracy"] is None and result["co"]["macro_f1"] is None
    assert set(result["difference"].values()) == {None}


# --- analyze và báo cáo markdown ----------------------------------------------------------------


def test_analyze_is_plain_json_and_holds_no_sentence():
    frame, deployed, reference = base_case()
    result = ea.analyze(frame, deployed, reference, split="validation")
    text = json.dumps(result, ensure_ascii=False)  # numpy lọt vào sẽ làm dòng này lỗi
    assert all(sentence not in text for sentence in frame["sentence"])
    samples = result["error_samples"]
    assert samples["n_rows"] == 5  # 3 lỗi cảm xúc + 2 lỗi chủ đề của int8, đều dưới 15
    assert all("sentence" not in row for row in samples["rows"])
    assert result["models"] == {"main": DEPLOY_KEY, "reference": REFERENCE}
    assert set(result["overall"]) == {DEPLOY_KEY, REFERENCE}


def test_analyze_refuses_a_frame_that_does_not_match_the_predictions():
    frame, deployed, reference = base_case()
    with pytest.raises(ea.InputError):
        ea.analyze(frame.iloc[::-1].reset_index(drop=True), deployed, reference, split="test")


def test_markdown_has_every_section_vietnamese_numbers_and_no_sentence():
    frame, deployed, reference = base_case()
    result = ea.analyze(frame, deployed, reference, split="validation")
    text = ea.render_markdown(result)
    for heading in (
        "## 1. Cặp nhầm lẫn",
        "## 2. Lỗi tự tin nhất",
        "## 3. Lỗi theo lát",
        "## 4. Giao lỗi",
        "## 5. Thiên lệch",
        "### 5.1",
        "### 5.2",
        "### 5.3",
    ):
        assert heading in text
    assert "error_samples.csv" in text
    assert "validation (diễn tập, KHÔNG phải test)" in text
    assert "33,3%" in text and "33.3%" not in text  # mỗi cặp nhầm của int8 là 1/3 số lỗi cảm xúc
    assert "mẫu quá ít để kết luận" in text
    assert "Chương trình → Khác" in text  # nhãn tiếng Việt và mũi tên ở bảng cặp nhầm lẫn
    assert all(sentence not in text for sentence in frame["sentence"])


def test_markdown_of_a_model_without_errors_says_so():
    frame = base_frame()
    y = {"sentiment": BASE["sentiment"], "topic": BASE["topic"]}
    result = ea.analyze(
        frame, make_preds(DEPLOY_KEY, y, y), make_preds(REFERENCE, y, y), split="test"
    )
    text = ea.render_markdown(result)
    assert text.count("Không có lỗi nào.") == 4  # hai mô hình x hai tác vụ
    assert result["error_samples"]["n_rows"] == 0
    assert "Tập test, n = 12" in text and "diễn tập" not in text


# --- chạy end-to-end ---------------------------------------------------------------------------


def big_case(n=120, seed=7):
    """120 câu với ~30% dự đoán sai: đủ để cả hai tác vụ vượt 15 lỗi."""
    rng = np.random.default_rng(seed)
    y = {"sentiment": rng.integers(0, 3, n), "topic": rng.integers(0, 4, n)}

    def noisy(task, rate):
        k = len(LABELS[task])
        wrong = rng.random(n) < rate
        return np.where(wrong, (y[task] + rng.integers(1, k, n)) % k, y[task])

    conf = {task: rng.uniform(0.5, 0.99, n) for task in TASKS}
    deployed = make_preds(DEPLOY_KEY, y, {task: noisy(task, 0.3) for task in TASKS}, conf)
    reference = make_preds(REFERENCE, y, {task: noisy(task, 0.4) for task in TASKS})
    who = ["thầy", "cô", "lớp"]
    frame = pd.DataFrame(
        {
            "sentence": [f"câu tổng hợp số {i:03d} {who[i % 3]} giảng bài" for i in range(n)],
            "text_rdr": [("wzjwz1 " if i % 10 == 0 else "") + "giảng_bài" for i in range(n)],
            "sentiment": y["sentiment"],
            "topic": y["topic"],
            "n_words": rng.integers(3, 60, n),
            "no_diacritics": [i % 25 == 0 for i in range(n)],
        }
    )
    return frame, deployed, reference


def stage(tmp_path, monkeypatch, frame, deployed, reference, *, done=True, meta=True):
    """Dựng thư mục dự đoán tổng hợp và thay `context` cùng khung dữ liệu của split.

    `load_processed` giả từ chối tập test; trả về danh sách các lần mở để test kiểm tra.
    """
    preds_dir = tmp_path / "preds"
    save_preds(preds_dir, deployed)
    if reference is not None:
        save_preds(preds_dir, reference)
    if meta:
        info = {"split": "validation", "n": len(frame), "created_at": "2026-10-19T10:00:00+0700"}
        write_json(preds_dir / META_FILE, {**info, "deploy_run": "mtl-s42", "git_commit": "abc123"})
    if done:
        (preds_dir / DONE_FILE).write_text("xong\n", encoding="utf-8")
    ctx = EvalContext("validation", preds_dir, tmp_path / "results", rehearsal=True)

    opened = []

    def fake_load(split, *, allow_test=False):
        assert split != "test" and not allow_test, "test không được mở trong test đơn vị"
        opened.append(split)
        return frame

    monkeypatch.setattr(preds_lib, "load_processed", fake_load)
    monkeypatch.setattr(ea, "context", lambda rehearsal=False: ctx)
    return ctx, opened


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_main_end_to_end_writes_four_files_in_results_and_the_sentences_outside_the_repo(
    tmp_path, monkeypatch, capsys
):
    frame, deployed, reference = big_case()
    ctx, opened = stage(tmp_path, monkeypatch, frame, deployed, reference)

    assert ea.main(["--rehearsal"]) == 0
    assert opened == ["validation"]  # chỉ mở đúng split của ngữ cảnh, một lần

    written = sorted(p.relative_to(ctx.results_dir).as_posix() for p in ctx.results_dir.rglob("*"))
    assert [name for name in written if "." in name] == [
        "metrics/error_analysis.json",
        "tables/bias_cells.csv",
        "tables/error_analysis.md",
        "tables/error_samples.csv",
    ]
    out = capsys.readouterr().out
    assert all(name in out for name in ("error_analysis.json", "error_analysis.md"))
    assert "error_samples.csv" in out and "bias_cells.csv" in out
    assert "error_samples_full.csv" in out

    # error_samples.csv: đúng 15 lỗi tự tin nhất mỗi tác vụ, tổng 30, KHÔNG có cột câu gốc.
    repo_rows = read_rows(ctx.table_path("error_samples.csv"))
    assert list(repo_rows[0]) == list(ea.SAMPLE_COLUMNS) and len(repo_rows) == 30
    # Bản có câu gốc nằm cạnh thư mục dự đoán, ngoài results/.
    full_path = ctx.preds_dir.parent / "error_samples_full.csv"
    assert not full_path.is_relative_to(ctx.results_dir)
    samples = read_rows(full_path)
    assert list(samples[0]) == list(ea.FULL_SAMPLE_COLUMNS) and len(samples) == 30
    assert [{k: r[k] for k in ea.SAMPLE_COLUMNS} for r in samples] == repo_rows
    reloaded = preds_lib.load_preds(ctx.preds_dir, DEPLOY_KEY)
    for task in TASKS:
        logits = reloaded.logits[task].astype(np.float64)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        conf = (probs / probs.sum(axis=1, keepdims=True)).max(axis=1)
        wrong = np.flatnonzero(reloaded.pred[task] != reloaded.y[task])
        assert len(wrong) > 15
        expected = wrong[np.argsort(-conf[wrong], kind="stable")][:15].tolist()
        assert [int(r["row"]) for r in samples if r["task"] == task] == expected
    for row in samples:
        i = int(row["row"])
        assert row["sentence"] == frame["sentence"][i] and row["split"] == "validation"
        assert row["true"] == LABELS[row["task"]][reloaded.y[row["task"]][i]]
        assert row["pred"] == LABELS[row["task"]][reloaded.pred[row["task"]][i]]
        assert row["true"] != row["pred"]

    # Câu gốc chỉ nằm ở tệp ngoài repo, và chỉ những câu được chọn (một câu có thể bị chọn ở cả hai
    # tác vụ nên đếm theo câu khác nhau); không tệp nào trong results/ chứa câu nào.
    full_text = full_path.read_text(encoding="utf-8")
    chosen = {row["sentence"] for row in samples}
    assert sum(sentence in full_text for sentence in frame["sentence"]) == len(chosen) <= 30
    for other in (
        "metrics/error_analysis.json",
        "tables/error_analysis.md",
        "tables/error_samples.csv",
        "tables/bias_cells.csv",
    ):
        text = (ctx.results_dir / other).read_text(encoding="utf-8")
        assert all(sentence not in text for sentence in frame["sentence"]), other

    cells = read_rows(ctx.table_path("bias_cells.csv"))
    assert len(cells) == 12 and sum(int(c["n"]) for c in cells) == len(frame)
    result = json.loads(ctx.metrics_path("error_analysis.json").read_text(encoding="utf-8"))
    assert result["n"] == 120 and result["split"] == "validation"
    assert result["source"] == {
        "rehearsal": True,
        "deploy_run": "mtl-s42",
        "evaluated_at": "2026-10-19T10:00:00+0700",
        "git_commit": "abc123",
    }
    assert result["error_samples"]["n_rows"] == 30
    report = ctx.table_path("error_analysis.md").read_text(encoding="utf-8")
    assert "run `mtl-s42`" in report and "error_samples.csv" in report


def test_main_exits_2_without_the_done_marker_and_reads_nothing(tmp_path, monkeypatch, capsys):
    frame, deployed, reference = base_case()
    ctx, opened = stage(tmp_path, monkeypatch, frame, deployed, reference, done=False)
    assert ea.main(["--rehearsal"]) == 2
    err = capsys.readouterr().err
    assert "chưa có" in err and DONE_FILE in err and "ml.evaluate --rehearsal" in err
    assert opened == [] and not ctx.results_dir.exists()


def test_main_exits_2_when_the_predictions_directory_does_not_exist(tmp_path, monkeypatch, capsys):
    ctx = EvalContext("validation", tmp_path / "chua-co", tmp_path / "results", rehearsal=True)
    monkeypatch.setattr(ea, "context", lambda rehearsal=False: ctx)
    assert ea.main(["--rehearsal"]) == 2
    assert "chưa chấm xong" in capsys.readouterr().err


def test_rehearsal_flag_chooses_the_context_and_the_real_hint(tmp_path, monkeypatch, capsys):
    seen = []

    def fake_context(rehearsal=False):
        seen.append(rehearsal)
        split = "validation" if rehearsal else "test"
        return EvalContext(split, tmp_path / "none", tmp_path / "res", rehearsal=rehearsal)

    monkeypatch.setattr(ea, "context", fake_context)
    assert ea.main(["--rehearsal"]) == 2
    assert "ml.evaluate --rehearsal" in capsys.readouterr().err
    assert ea.main([]) == 2
    assert "--split test --final" in capsys.readouterr().err
    assert seen == [True, False]  # thiếu .done nên chưa hề đọc split nào


def test_main_exits_2_when_the_meta_file_is_missing(tmp_path, monkeypatch, capsys):
    frame, deployed, reference = base_case()
    ctx, _ = stage(tmp_path, monkeypatch, frame, deployed, reference, meta=False)
    assert ea.main(["--rehearsal"]) == 2
    assert META_FILE in capsys.readouterr().err
    assert not ctx.results_dir.exists()


def test_main_exits_2_when_a_model_has_no_predictions(tmp_path, monkeypatch, capsys):
    frame, deployed, _ = base_case()
    ctx, _ = stage(tmp_path, monkeypatch, frame, deployed, None)  # thiếu linearsvc.npz
    assert ea.main(["--rehearsal"]) == 2
    assert "chưa có dự đoán" in capsys.readouterr().err
    assert not ctx.results_dir.exists()


def test_main_exits_2_when_the_frame_is_not_the_split_the_predictions_came_from(
    tmp_path, monkeypatch, capsys
):
    frame, deployed, reference = base_case()
    wrong_split = frame.assign(sentiment=(frame["sentiment"] + 1) % 3)
    ctx, _ = stage(tmp_path, monkeypatch, wrong_split, deployed, reference)
    assert ea.main(["--rehearsal"]) == 2
    assert "nhãn thật khác khung dữ liệu" in capsys.readouterr().err
    assert not ctx.results_dir.exists()  # không ghi số tính trên hàng lệch
