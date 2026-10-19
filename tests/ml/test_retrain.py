"""Huấn luyện lại từ nhãn đã duyệt: đọc CSV, làm sạch, baseline, quy tắc nhận, PhoBERT nhỏ, CLI.

Mọi dữ liệu là câu tự viết hoặc DataFrame tổng hợp; hàm đọc dữ liệu thật được thay bằng hàm trả
bảng tổng hợp, nên không test nào chạm train/validation/test thật.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edupulse.labels import LABELS, TASKS
from edupulse.text import NORMALIZER_VERSION
from ml import retrain
from ml.common import write_json
from ml.train_baseline import GRIDS, make_classifier, make_features

HEADER = [
    "id_hash",
    "text_masked",
    "text_model",
    "model_ver",
    "pred_s",
    "conf_s",
    "pred_t",
    "conf_t",
    "label_s",
    "label_t",
    "source",
    "reviewed",
]
HP = {"sentiment": {"C": 1.0, "class_weight": None}, "topic": {"C": 1.0, "class_weight": None}}

# 4 chủ đề x 3 từ chủ đề x 3 cảm xúc x 3 từ ý kiến = 108 câu tự viết, mỗi câu một tổ hợp riêng.
TOPIC_WORDS = {
    0: ("thầy", "cô", "giảng_viên"),
    1: ("chương_trình", "giáo_trình", "môn_học"),
    2: ("phòng_học", "máy_chiếu", "thư_viện"),
    3: ("căng_tin", "bãi_xe", "học_phí"),
}
OPINIONS = {
    0: ("tệ", "kém", "chán"),
    1: ("bình_thường", "tạm_được", "vừa_phải"),
    2: ("tốt", "tuyệt_vời", "hay"),
}
# Năm câu phản hồi mới (có từ chưa từng thấy trong tập huấn luyện tổng hợp).
OWN_SENTENCES = [
    ("wifi ở ký_túc_xá chậm quá .", 0, 2),
    ("ký_túc_xá có wifi ổn_định .", 2, 2),
    ("thầy hướng_dẫn đồ_án tận_tâm .", 2, 0),
    ("học_phí tăng nhưng chưa rõ lý_do .", 0, 3),
    ("lịch thi dày , đề vừa_phải .", 1, 1),
]
# Năm câu coi như đã nằm trong validation thật (chỉ dùng băm của chúng).
KNOWN_SENTENCES = [
    "giảng_viên giảng bài chậm .",
    "máy_chiếu mờ , micro rè .",
    "đề_cương môn_học rất chi_tiết .",
    "căng_tin bán đồ ăn ngon .",
    "thư_viện mở cửa muộn .",
]


def synthetic_corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (72 câu) và phần giữ riêng (36 câu): tổ hợp (i + j) chia hết cho 3 để riêng."""
    train_rows, held_rows = [], []
    for topic, words in TOPIC_WORDS.items():
        for i, word in enumerate(words):
            for sentiment, opinions in OPINIONS.items():
                for j, opinion in enumerate(opinions):
                    row = {
                        "text_rdr": f"{word} {opinion} .",
                        "sentiment": sentiment,
                        "topic": topic,
                    }
                    (held_rows if (i + j) % 3 == 0 else train_rows).append(row)
    return pd.DataFrame(train_rows), pd.DataFrame(held_rows)


def feedback_frame(rows: list[tuple[str, int, int]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["text_model", "label_s", "label_t"])
    return frame.astype({"label_s": "int64", "label_t": "int64"})


def csv_row(text: str, label_s, label_t, *, reviewed="true", source="flag") -> list[str]:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return [digest, text, text, "v1.0", "0", "0.5", "0", "0.5", label_s, label_t, source, reviewed]


def write_csv(path: Path, rows: list[list], header: list[str] = HEADER) -> Path:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


# --- Đọc phản hồi ---------------------------------------------------------------------------------


def test_read_feedback_accepts_bom_numeric_and_named_labels(tmp_path):
    path = write_csv(
        tmp_path / "fb.csv",
        [
            csv_row("thầy dạy rất tốt .", 2, 0),
            csv_row("phòng_học nóng quá .", "negative", "facility"),
            csv_row("giáo_trình cũ .", "1", "1.0"),
            csv_row("học_phí hợp_lý .", " Positive ", "Training Program"),
        ],
    )
    assert path.read_bytes().startswith(b"\xef\xbb\xbf")  # tệp có BOM như bản API xuất
    frame = retrain.read_feedback([path])
    assert frame["text_model"].tolist() == [
        "thầy dạy rất tốt .",
        "phòng_học nóng quá .",
        "giáo_trình cũ .",
        "học_phí hợp_lý .",
    ]
    assert frame["label_s"].tolist() == [2, 0, 1, 2]
    assert frame["label_t"].tolist() == [0, 2, 1, 1]
    assert frame["label_s"].dtype == np.int64 and frame["label_t"].dtype == np.int64
    assert set(frame["file"]) == {"fb.csv"}
    assert frame["line"].tolist() == [2, 3, 4, 5]  # dòng 1 là tiêu đề


@pytest.mark.parametrize(
    ("in_csv", "expected"),
    [
        ("'=2+2 thầy tốt", "=2+2 thầy tốt"),
        ("'+84 điểm danh", "+84 điểm danh"),
        ("'-5 điểm cuối kỳ", "-5 điểm cuối kỳ"),
        ("'@thầy dạy tốt", "@thầy dạy tốt"),
        (f"'{chr(0xFF1D)}abc thầy", f"{chr(0xFF1D)}abc thầy"),
        (f"'{chr(0xFF0B)}abc thầy", f"{chr(0xFF0B)}abc thầy"),
        (f"'{chr(0xFF0D)}abc thầy", f"{chr(0xFF0D)}abc thầy"),
        (f"'{chr(0xFF20)}abc thầy", f"{chr(0xFF20)}abc thầy"),
        ("'\tphòng học nóng", "phòng học nóng"),  # sau khi bỏ nháy, TAB đầu bị strip
        ("'\rphòng học nóng", "phòng học nóng"),
        ("'chào thầy", "'chào thầy"),  # nháy đầu câu bình thường: giữ nguyên
        ("=2+2 chưa có nháy", "=2+2 chưa có nháy"),  # không tự thêm hay bỏ gì khác
    ],
)
def test_read_feedback_strips_the_injection_guard_only_before_dangerous_characters(
    tmp_path, in_csv, expected
):
    path = write_csv(tmp_path / "fb.csv", [csv_row(in_csv, 1, 1)])
    assert retrain.read_feedback(path)["text_model"].tolist() == [expected]


def test_read_feedback_strips_the_guard_in_label_cells(tmp_path):
    path = write_csv(tmp_path / "fb.csv", [csv_row("thầy dạy tốt .", "'+1", "'+0")])
    frame = retrain.read_feedback(path)
    assert frame[["label_s", "label_t"]].values.tolist() == [[1, 0]]


def test_read_feedback_keeps_only_reviewed_rows(tmp_path):
    flags = ["true", "TRUE", "1", "yes", "false", "0", "", "no"]
    rows = [csv_row(f"câu số {i} .", 1, 1, reviewed=flag) for i, flag in enumerate(flags)]
    tally: Counter[str] = Counter()
    frame = retrain.read_feedback(write_csv(tmp_path / "fb.csv", rows), tally)
    assert frame["text_model"].tolist() == [f"câu số {i} ." for i in range(4)]
    assert tally[retrain.READ_ROWS] == 8 and tally[retrain.SKIP_UNREVIEWED] == 4


def test_read_feedback_without_a_reviewed_column_treats_every_row_as_reviewed(tmp_path):
    header = [name for name in HEADER if name != "reviewed"]
    rows = [csv_row(f"câu số {i} .", 0, 0)[:-1] for i in range(3)]
    frame = retrain.read_feedback(write_csv(tmp_path / "fb.csv", rows, header))
    assert len(frame) == 3


def test_read_feedback_drops_rows_missing_text_or_labels(tmp_path):
    rows = [
        csv_row("", 1, 1),
        csv_row("   ", 1, 1),
        csv_row("thiếu cảm xúc .", "", 1),
        csv_row("thiếu chủ đề .", 1, ""),
        csv_row("đủ cả .", 1, 1),
    ]
    tally: Counter[str] = Counter()
    frame = retrain.read_feedback(write_csv(tmp_path / "fb.csv", rows), tally)
    assert frame["text_model"].tolist() == ["đủ cả ."]
    assert tally[retrain.SKIP_MISSING] == 4


@pytest.mark.parametrize(
    ("label_s", "label_t", "column"),
    [(3, 0, "label_s"), (-1, 0, "label_s"), (0, 4, "label_t"), ("great", 0, "label_s"),
     (0, "lecturers", "label_t"), (1.5, 0, "label_s")],
)  # fmt: skip
def test_read_feedback_names_the_line_of_an_out_of_range_label(tmp_path, label_s, label_t, column):
    rows = [csv_row("câu đúng .", 1, 1), csv_row("câu có nhãn lỗi .", label_s, label_t)]
    with pytest.raises(retrain.FeedbackError, match=rf"fb\.csv, dòng 3: cột {column}"):
        retrain.read_feedback(write_csv(tmp_path / "fb.csv", rows))


def test_an_unreviewed_row_with_a_bad_label_is_not_an_error(tmp_path):
    rows = [csv_row("chưa duyệt .", 9, 9, reviewed="false"), csv_row("đã duyệt .", 1, 1)]
    frame = retrain.read_feedback(write_csv(tmp_path / "fb.csv", rows))
    assert frame["text_model"].tolist() == ["đã duyệt ."]


def test_read_feedback_requires_the_three_columns(tmp_path):
    header = [name for name in HEADER if name != "label_t"]
    rows = [csv_row("câu .", 1, 1)[:9] + csv_row("câu .", 1, 1)[10:]]
    path = write_csv(tmp_path / "thieu.csv", rows, header)
    with pytest.raises(retrain.FeedbackError, match=r"thieu\.csv: thiếu cột bắt buộc label_t"):
        retrain.read_feedback(path)


def test_read_feedback_reports_missing_and_unreadable_files(tmp_path):
    with pytest.raises(retrain.FeedbackError, match="không thấy tệp"):
        retrain.read_feedback(tmp_path / "khong-co.csv")
    broken = tmp_path / "hong.csv"
    broken.write_bytes(b"text_model,label_s,label_t\n\xff\xfe\x80,1,1\n")
    with pytest.raises(retrain.FeedbackError, match=r"hong\.csv"):
        retrain.read_feedback(broken)
    with pytest.raises(retrain.FeedbackError, match="chưa chỉ định"):
        retrain.read_feedback([])


def test_read_feedback_concatenates_files_in_order_and_tags_the_source_file(tmp_path):
    first = write_csv(tmp_path / "a.csv", [csv_row("câu một .", 0, 0, source="flag")])
    second = write_csv(
        tmp_path / "b.csv",
        [csv_row("câu hai .", 1, 1, source="audit"), csv_row("câu ba .", 2, 2)],
    )
    tally: Counter[str] = Counter()
    frame = retrain.read_feedback([first, second], tally)
    assert frame["text_model"].tolist() == ["câu một .", "câu hai .", "câu ba ."]
    assert frame["file"].tolist() == ["a.csv", "b.csv", "b.csv"]
    assert frame["source"].tolist() == ["flag", "audit", "flag"]
    assert tally[retrain.READ_ROWS] == 3
    assert frame.index.tolist() == [0, 1, 2]


# --- Làm sạch -------------------------------------------------------------------------------------


def test_text_hash_ignores_surrounding_whitespace_only():
    assert retrain.text_hash("  thầy tốt .\n") == retrain.text_hash("thầy tốt .")
    assert retrain.text_hash("thầy tốt .") != retrain.text_hash("thầy tốt")
    assert retrain.text_hash("thầy tốt .") == hashlib.sha256("thầy tốt .".encode()).hexdigest()


def test_clean_feedback_keeps_own_sentences_and_drops_ones_already_in_the_data():
    known = {retrain.text_hash(text) for text in KNOWN_SENTENCES}
    rows = [(text, 1, 1) for text, _, _ in OWN_SENTENCES] + [(t, 0, 0) for t in KNOWN_SENTENCES]
    cleaned, reasons = retrain.clean_feedback(feedback_frame(rows), known)
    assert cleaned["text_model"].tolist() == [text for text, _, _ in OWN_SENTENCES]
    assert reasons == Counter({retrain.REASON_KNOWN: 5})
    assert cleaned.index.tolist() == list(range(5))


def test_clean_feedback_dedups_by_hash_keeping_the_first_row_and_counts_every_reason():
    rows = [
        ("thầy dạy tốt .", 2, 0),
        ("  thầy dạy tốt .  ", 0, 1),  # trùng sau khi strip: bỏ, giữ dòng đầu
        ("phòng_học nóng .", 0, 2),
        ("câu đã có trong validation .", 1, 1),
        ("câu đã có trong validation .", 1, 1),
        ("", 1, 1),
        ("câu nhãn lỗi .", 5, 1),
        ("câu nhãn lỗi khác .", 1, -1),
        ("phòng_học nóng .", 0, 2),
    ]
    known = {retrain.text_hash("câu đã có trong validation .")}
    cleaned, reasons = retrain.clean_feedback(feedback_frame(rows), known)
    assert cleaned["text_model"].tolist() == ["thầy dạy tốt .", "phòng_học nóng ."]
    assert cleaned[["label_s", "label_t"]].values.tolist() == [[2, 0], [0, 2]]
    assert reasons == {
        retrain.REASON_DUPLICATE: 2,
        retrain.REASON_KNOWN: 2,
        retrain.REASON_EMPTY: 1,
        retrain.REASON_BAD_LABEL: 2,
    }


def test_clean_feedback_on_an_empty_frame_returns_an_empty_frame():
    cleaned, reasons = retrain.clean_feedback(feedback_frame([]), set())
    assert cleaned.empty and not reasons
    assert list(cleaned.columns) == ["text_model", "label_s", "label_t"]


class FakeData:
    """Thay `load_processed`: trả bảng tổng hợp và ghi lại split nào được yêu cầu."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []
        train, held = synthetic_corpus()
        self.frames = {
            "train": train,
            "validation": held,
            "test": pd.DataFrame(
                {"text_rdr": KNOWN_SENTENCES, "sentiment": [1] * 5, "topic": [1] * 5}
            ),
        }

    def __call__(self, split: str, *, allow_test: bool = False) -> pd.DataFrame:
        self.calls.append((split, allow_test))
        if split == "test" and not allow_test:
            raise PermissionError("tập test chỉ mở khi allow_test=True")
        return self.frames[split].copy()


def test_known_text_hashes_opens_test_only_to_hash_it(monkeypatch):
    fake = FakeData()
    monkeypatch.setattr(retrain, "load_processed", fake)
    hashes = retrain.known_text_hashes()
    assert fake.calls == [("train", False), ("validation", False), ("test", True)]
    expected = {
        retrain.text_hash(text) for frame in fake.frames.values() for text in frame["text_rdr"]
    }
    assert hashes == expected
    assert all(len(h) == 64 and set(h) <= set("0123456789abcdef") for h in hashes)  # chỉ băm


def test_known_text_hashes_can_leave_test_out(monkeypatch):
    fake = FakeData()
    monkeypatch.setattr(retrain, "load_processed", fake)
    hashes = retrain.known_text_hashes(splits=("train", "validation"))
    assert [split for split, _ in fake.calls] == ["train", "validation"]
    assert retrain.text_hash(KNOWN_SENTENCES[0]) not in hashes


# --- Baseline -------------------------------------------------------------------------------------


def accuracy(models, frame: pd.DataFrame) -> dict[str, float]:
    texts = frame["text_rdr"].tolist()
    return {t: float((models[t].predict(texts) == frame[t].to_numpy()).mean()) for t in TASKS}


def test_retrain_baseline_runs_end_to_end_and_predicts_held_out_sentences():
    train, held = synthetic_corpus()
    models = retrain.retrain_baseline(train, feedback_frame(OWN_SENTENCES), HP)
    assert set(models) == set(TASKS)
    for task in TASKS:
        assert [name for name, _ in models[task].steps] == ["features", "clf"]
        assert set(models[task].predict(held["text_rdr"].tolist())) <= set(range(len(LABELS[task])))
    assert models["sentiment"].named_steps["features"] is models["topic"].named_steps["features"]
    scores = accuracy(models, held)
    assert scores["sentiment"] >= 0.95 and scores["topic"] >= 0.95
    own = pd.DataFrame(
        {"text_rdr": [t for t, _, _ in OWN_SENTENCES]}
        | {"sentiment": [s for _, s, _ in OWN_SENTENCES], "topic": [t for _, _, t in OWN_SENTENCES]}
    )
    assert all(value == 1.0 for value in accuracy(models, own).values())  # học được phản hồi


def test_retrain_baseline_with_empty_feedback_equals_fitting_on_train_alone():
    train, held = synthetic_corpus()
    texts = train["text_rdr"].tolist()
    features = make_features().fit(texts)
    matrix = features.transform(texts)
    models = retrain.retrain_baseline(train, feedback_frame([]), HP)
    for task in TASKS:
        direct = make_classifier("linearsvc", HP[task]).fit(matrix, train[task].to_numpy())
        got = models[task].decision_function(held["text_rdr"].tolist())
        want = direct.decision_function(features.transform(held["text_rdr"].tolist()))
        assert np.array_equal(got, want)


def test_feedback_labels_override_the_training_labels_they_contradict():
    train, _ = synthetic_corpus()
    probe = "bãi_xe tốt ."
    before = retrain.retrain_baseline(train, feedback_frame([]), HP)
    assert before["topic"].predict([probe])[0] == 3  # bãi_xe thuộc chủ đề «khác» trong train
    contradicting = feedback_frame([(f"bãi_xe {w} .", 1, 2) for w in ("tốt", "tệ", "hay", "kém")])
    after = retrain.retrain_baseline(train, contradicting, HP, weight=3)
    assert after["topic"].predict([probe])[0] == 2


def test_feedback_weight_repeats_feedback_rows_before_fitting_the_vocabulary():
    train, _ = synthetic_corpus()
    feedback = feedback_frame([("ký_túc_xá ồn_ào .", 0, 2)])
    once = retrain.retrain_baseline(train, feedback, HP, weight=1)
    twice = retrain.retrain_baseline(train, feedback, HP, weight=2)

    def vocabulary(models) -> dict[str, int]:
        return models["topic"].named_steps["features"].transformer_list[0][1].vocabulary_

    assert "ký_túc_xá" not in vocabulary(once)  # chỉ xuất hiện ở 1 câu: dưới ngưỡng min_df
    assert "ký_túc_xá" in vocabulary(twice)  # lặp 2 lần: 2 tài liệu


def test_retrain_baseline_rejects_bad_arguments():
    train, _ = synthetic_corpus()
    with pytest.raises(retrain.InputError, match="feedback-weight"):
        retrain.retrain_baseline(train, feedback_frame([]), HP, weight=0)
    with pytest.raises(retrain.InputError, match="thiếu cột"):
        retrain.retrain_baseline(train, pd.DataFrame({"text_model": ["a"]}), HP)
    with pytest.raises(retrain.InputError, match="topic"):
        retrain.retrain_baseline(train, feedback_frame([]), {"sentiment": HP["sentiment"]})


def test_hyperparameters_are_shown_in_short_vietnamese_form():
    assert retrain.hp_text({"C": 3.0, "class_weight": None}) == "C = 3"
    assert retrain.hp_text({"C": 0.3, "class_weight": None}) == "C = 0,3"
    balanced = retrain.hp_text({"C": 1.0, "class_weight": "balanced"})
    assert balanced == "C = 1, class_weight = balanced"


def test_deployed_hyperparameters_are_read_from_the_frozen_baseline_table():
    hp = retrain.load_deployed_baseline_hp()
    assert set(hp) == set(TASKS)
    grid = GRIDS["linearsvc"]
    assert all(hp[task] in grid for task in TASKS)


# --- Quy tắc nhận mô hình mới ---------------------------------------------------------------------


def test_accept_requires_every_task_to_hold_its_macro_f1():
    before = {"sentiment": 0.78, "topic": 0.76}
    ok, reason = retrain.accept(before, {"sentiment": 0.79, "topic": 0.77})
    assert ok and "không tác vụ nào tụt" in reason
    ok, reason = retrain.accept(before, {"sentiment": 0.80, "topic": 0.75})
    assert not ok and "chủ đề tụt" in reason and "cảm xúc" not in reason.split("vượt")[0]
    ok, reason = retrain.accept(before, {"sentiment": 0.70, "topic": 0.70})
    assert not ok and "cảm xúc tụt" in reason and "chủ đề tụt" in reason


def test_accept_takes_a_model_that_is_exactly_as_good():
    scores = {"sentiment": 0.7805215668300903, "topic": 0.7638285072281606}
    ok, _ = retrain.accept(scores, dict(scores))
    assert ok
    ok, _ = retrain.accept(scores, {task: value - 1e-15 for task, value in scores.items()})
    assert ok  # nhiễu dấu phẩy động không đủ để từ chối


def test_accept_tolerance_is_zero_and_honoured_when_changed(monkeypatch):
    assert retrain.ACCEPT_TOLERANCE == 0.0
    before, after = {"sentiment": 0.78}, {"sentiment": 0.775}
    assert not retrain.accept(before, after)[0]
    monkeypatch.setattr(retrain, "ACCEPT_TOLERANCE", 0.01)
    assert retrain.accept(before, after)[0]
    assert not retrain.accept(before, {"sentiment": 0.76})[0]


def test_accept_rejects_mismatched_task_sets():
    with pytest.raises(ValueError, match="cùng tập tác vụ"):
        retrain.accept({"sentiment": 0.7}, {"topic": 0.7})


# --- Thư mục ghi ----------------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    monkeypatch.delenv("EDUPULSE_ARTIFACTS_DIR", raising=False)
    return tmp_path


def test_output_directory_may_not_touch_models_or_the_bundle(home):
    from edupulse.config import load_settings

    run = home / "models" / "phobert" / "mtl-s1"
    run.mkdir(parents=True)
    bundle = load_settings().bundle_dir
    for forbidden in (
        run,
        run / "epoch2",
        home / "models" / "phobert",
        home / "models" / "baseline",
        home / "models",
        bundle,
        bundle / "extra",
    ):
        with pytest.raises(retrain.InputError, match="đang triển khai"):
            retrain.assert_safe_output(forbidden, run)


def test_output_directory_must_be_new_or_empty(home):
    target = home / "outputs" / "retrain" / "x"
    assert retrain.assert_safe_output(target) == target.resolve()
    target.mkdir(parents=True)
    assert retrain.assert_safe_output(target) == target.resolve()  # rỗng thì dùng được
    (target / "da-co.txt").write_text("giữ nguyên", encoding="utf-8")
    with pytest.raises(retrain.InputError, match="không ghi đè"):
        retrain.assert_safe_output(target)


def test_default_output_lives_under_outputs_retrain(home):
    out = retrain.default_out("phobert")
    assert out.parent == home / "outputs" / "retrain"
    assert out.name.startswith("phobert-")


def test_phobert_retrain_refuses_every_device_but_cpu():
    train, _ = synthetic_corpus()
    with pytest.raises(retrain.InputError, match="chỉ chạy trên CPU.*mps"):
        retrain.retrain_phobert("best", feedback_frame(OWN_SENTENCES), train, device="mps")


def test_phobert_retrain_needs_feedback_and_sane_arguments():
    train, _ = synthetic_corpus()
    with pytest.raises(retrain.InputError, match="không có câu phản hồi"):
        retrain.retrain_phobert("best", feedback_frame([]), train)
    with pytest.raises(retrain.InputError, match="steps"):
        retrain.retrain_phobert("best", feedback_frame(OWN_SENTENCES), train, steps=0)
    with pytest.raises(retrain.InputError, match="replay"):
        retrain.retrain_phobert("best", feedback_frame(OWN_SENTENCES), train, replay=1.0)


def test_phobert_retrain_refuses_to_write_inside_models(home):
    run = home / "models" / "phobert" / "mtl-s1"
    run.mkdir(parents=True)
    (run / "config.json").write_text("{}", encoding="utf-8")
    train, _ = synthetic_corpus()
    with pytest.raises(retrain.InputError, match="đang triển khai"):
        retrain.retrain_phobert(
            run, feedback_frame(OWN_SENTENCES), train, out=home / "models" / "phobert" / "moi"
        )


# --- PhoBERT trên checkpoint nhỏ ------------------------------------------------------------------


def snapshot(root: Path) -> dict[str, str]:
    """Tên tệp tương đối -> sha256 của mọi tệp dưới `root`."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """Run PhoBERT nhỏ (2 lớp, ẩn 32) nhưng từ vựng đủ lớn cho id của tokenizer thật."""
    torch = pytest.importorskip("torch")
    from ml.phobert_model import MultiHeadConfig, MultiHeadPhoBERT, load_tokenizer

    try:
        load_tokenizer()
    except Exception as error:
        pytest.skip(f"không nạp được tokenizer PhoBERT (cần cache Hub hoặc mạng): {error}")
    run = tmp_path_factory.mktemp("tiny") / "mtl-s1"
    torch.manual_seed(0)
    config = MultiHeadConfig(
        task_heads={"sentiment": 3, "topic": 4},
        vocab_size=64001,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=132,
        type_vocab_size=1,
        pad_token_id=1,
    )
    MultiHeadPhoBERT(config).save_pretrained(run)
    write_json(
        run / "history.json",
        {"normalizer_version": NORMALIZER_VERSION, "finished": True, "best_score": 0.5},
    )
    return run


@pytest.mark.train
@pytest.mark.slow
def test_phobert_retrain_runs_the_requested_steps_and_leaves_models_untouched(
    tiny_checkpoint, home, monkeypatch
):
    import torch

    from ml.phobert_model import MultiHeadPhoBERT

    run = home / "models" / "phobert" / "mtl-s1"
    shutil.copytree(tiny_checkpoint, run)
    untouched = snapshot(home / "models")
    train, _ = synthetic_corpus()

    optimizer_steps = []
    original_step = torch.optim.AdamW.step

    def counting_step(self, *args, **kwargs):
        optimizer_steps.append(1)
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", counting_step)
    record = retrain.retrain_phobert(
        "best", feedback_frame(OWN_SENTENCES), train, steps=3, batch_size=4, seed=0
    )

    assert len(optimizer_steps) == 3 == record["steps"] == len(record["losses"])
    assert all(np.isfinite(record["losses"]))
    assert snapshot(home / "models") == untouched  # models/phobert/* không bị ghi
    out = Path(record["out"])
    assert out.parent == home / "outputs" / "retrain" and out.name.startswith("phobert-")
    assert record["base_run"] == "mtl-s1" and record["device"] == "cpu"
    assert (record["batch_size"], record["feedback_per_batch"], record["replay_per_batch"]) == (
        4,
        2,
        2,
    )
    assert record["feedback_size"] == 5 and record["replay_pool"] == len(train)
    assert record["seconds"] >= 0
    assert json.loads((out / "retrain.json").read_text("utf-8")) == record

    base = MultiHeadPhoBERT.from_pretrained(run)
    new = MultiHeadPhoBERT.from_pretrained(out)  # checkpoint mới nạp lại được
    assert dict(new.config.task_heads) == {"sentiment": 3, "topic": 4}
    assert not torch.equal(base.heads["sentiment"].weight, new.heads["sentiment"].weight)
    assert not torch.equal(
        base.roberta.encoder.layer[0].output.dense.weight,
        new.roberta.encoder.layer[0].output.dense.weight,
    )


@pytest.mark.train
@pytest.mark.slow
def test_phobert_retrain_is_reproducible_and_never_overwrites_an_existing_output(
    tiny_checkpoint, home
):
    import torch

    from ml.phobert_model import MultiHeadPhoBERT

    train, _ = synthetic_corpus()
    feedback = feedback_frame(OWN_SENTENCES)
    first = retrain.retrain_phobert(
        tiny_checkpoint, feedback, train, steps=2, batch_size=4, seed=3, out=home / "a"
    )
    second = retrain.retrain_phobert(
        tiny_checkpoint, feedback, train, steps=2, batch_size=4, seed=3, out=home / "b"
    )
    assert first["losses"] == second["losses"]
    weights = [MultiHeadPhoBERT.from_pretrained(home / d).heads["topic"].weight for d in "ab"]
    assert torch.equal(*weights)
    with pytest.raises(retrain.InputError, match="không ghi đè"):
        retrain.retrain_phobert(
            tiny_checkpoint, feedback, train, steps=2, batch_size=4, out=home / "a"
        )


@pytest.mark.train
@pytest.mark.slow
def test_phobert_retrain_refuses_a_run_trained_with_another_normalizer(tiny_checkpoint, home):
    stale = home / "stale"
    shutil.copytree(tiny_checkpoint, stale)
    write_json(stale / "history.json", {"normalizer_version": "0", "finished": True})
    train, _ = synthetic_corpus()
    with pytest.raises(retrain.InputError, match="bản chuẩn hoá"):
        retrain.retrain_phobert(stale, feedback_frame(OWN_SENTENCES), train, out=home / "x")


@pytest.mark.train
@pytest.mark.slow
def test_predict_phobert_returns_logits_in_input_order(tiny_checkpoint):
    texts = ["thầy dạy rất tốt .", "phòng_học nóng quá .", "học_phí cao ."]
    logits = retrain.predict_phobert(tiny_checkpoint, texts)
    assert logits["sentiment"].shape == (3, 3) and logits["topic"].shape == (3, 4)
    assert np.isfinite(logits["topic"]).all()
    reversed_logits = retrain.predict_phobert(tiny_checkpoint, texts[::-1])
    for task in TASKS:
        assert np.allclose(reversed_logits[task], logits[task][::-1], atol=1e-4)


# --- Dòng lệnh ------------------------------------------------------------------------------------


@pytest.fixture
def cli(home, monkeypatch):
    """Dữ liệu tổng hợp thay cho train/validation/test thật; ghi lại split nào bị đọc."""
    fake = FakeData()
    monkeypatch.setattr(retrain, "load_processed", fake)
    monkeypatch.setattr(retrain, "load_deployed_baseline_hp", lambda: HP)
    return fake


def good_feedback(tmp_path: Path) -> Path:
    rows = [csv_row(text, s, t) for text, s, t in OWN_SENTENCES]
    rows += [csv_row(text, 1, 1) for text in KNOWN_SENTENCES[:2]]  # đã có trong test thật
    rows.append(csv_row("câu chưa duyệt .", 0, 0, reviewed="false"))
    return write_csv(tmp_path / "fb.csv", rows)


def test_cli_baseline_accepts_a_model_that_holds_validation_and_writes_it_to_out(cli, home, capsys):
    from ml.train_baseline import load_model

    out = home / "moi"
    code = retrain.main(["--feedback", str(good_feedback(home)), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0, printed
    assert "đọc 8 dòng" in printed and "chưa duyệt 1" in printed
    assert "giữ 5 câu" in printed and "trùng dữ liệu gốc 2" in printed
    assert "NHẬN" in printed
    assert [call for call in cli.calls if call[0] == "test"] == [("test", True)]  # chỉ để băm
    for task in TASKS:
        model = load_model(out / f"{task}_linearsvc.skops")
        assert model.classes_.tolist() == list(range(len(LABELS[task])))
        probs = model.predict_proba(["thầy dạy tốt ."])
        assert probs.shape == (1, len(LABELS[task])) and abs(probs.sum() - 1) < 1e-6
    record = json.loads((out / "retrain.json").read_text("utf-8"))
    assert record["decision"]["accepted"] and record["feedback_size"] == 5
    assert set(record["val_macro_f1"]) == {"before", "after"}


def test_cli_baseline_rejects_a_model_that_gets_worse_and_writes_nothing(cli, home, capsys):
    # Phản hồi cố ý sai: đảo nhãn cảm xúc của mọi từ ý kiến và gán mọi câu nói về thầy cô vào chủ
    # đề «khác»; lặp 6 lần. Câu ghép hai từ chủ đề là câu mới so với dữ liệu gốc nên không bị loại.
    rows = []
    pairs = [("thầy", "cô"), ("cô", "thầy"), ("thầy", "giảng_viên"), ("giảng_viên", "cô")]
    for opinion in ("tốt", "tuyệt_vời", "hay", "tệ", "kém", "chán"):
        flipped = 0 if opinion in ("tốt", "tuyệt_vời", "hay") else 2
        rows += [csv_row(f"{a} và {b} {opinion} .", flipped, 3) for a, b in pairs]
    path = write_csv(home / "xau.csv", rows)
    out = home / "khong-duoc-ghi"
    code = retrain.main(["--feedback", str(path), "--feedback-weight", "6", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 3, printed
    assert "TỪ CHỐI" in printed and "tụt" in printed
    assert not out.exists()


def test_cli_exits_with_2_on_bad_input(cli, home, capsys):
    assert retrain.main(["--feedback", str(home / "khong-co.csv")]) == 2
    assert "Lỗi đầu vào" in capsys.readouterr().err

    rows = [csv_row("câu .", 7, 1)]
    assert retrain.main(["--feedback", str(write_csv(home / "sai.csv", rows))]) == 2
    assert "dòng 2" in capsys.readouterr().err

    only_known = write_csv(home / "da-co.csv", [csv_row(KNOWN_SENTENCES[0], 1, 1)])
    assert retrain.main(["--feedback", str(only_known)]) == 2
    assert "không còn câu phản hồi hợp lệ" in capsys.readouterr().err


def test_cli_refuses_other_devices_and_protected_output_before_reading_any_data(cli, home, capsys):
    path = good_feedback(home)
    assert retrain.main(["--feedback", str(path), "--model", "phobert", "--device", "mps"]) == 2
    assert "--device cpu" in capsys.readouterr().err
    assert retrain.main(["--feedback", str(path), "--out", str(home / "models" / "baseline")]) == 2
    assert "đang triển khai" in capsys.readouterr().err
    assert retrain.main(["--feedback", str(path), "--max-steps", "0"]) == 2
    assert cli.calls == []  # chưa đọc dữ liệu nào


def test_cli_never_asks_for_test_outside_the_hashing_step(cli, home):
    retrain.main(["--feedback", str(good_feedback(home)), "--out", str(home / "moi")])
    assert [split for split, _ in cli.calls] == [
        "train",
        "validation",
        "test",
        "train",
        "validation",
    ]
    assert all(allowed == (split == "test") for split, allowed in cli.calls[:3])
    assert cli.calls[3:] == [("train", False), ("validation", False)]


@pytest.mark.train
@pytest.mark.slow
@pytest.mark.parametrize(("tolerance", "code"), [(2.0, 0), (-2.0, 3)])
def test_cli_phobert_path_keeps_or_discards_the_new_checkpoint(
    tiny_checkpoint, cli, home, monkeypatch, capsys, tolerance, code
):
    """Dung sai cực rộng thì luôn nhận, cực hẹp thì luôn từ chối: kiểm cả hai nhánh ghi/xoá."""
    run = home / "models" / "phobert" / "mtl-s1"
    shutil.copytree(tiny_checkpoint, run)
    untouched = snapshot(home / "models")
    monkeypatch.setattr(retrain, "ACCEPT_TOLERANCE", tolerance)
    out = home / "outputs" / "retrain" / "phobert-moi"
    argv = ["--feedback", str(good_feedback(home)), "--model", "phobert", "--max-steps", "2"]
    assert retrain.main([*argv, "--out", str(out)]) == code, capsys.readouterr()
    assert snapshot(home / "models") == untouched
    leftovers = sorted(path.name for path in (home / "outputs" / "retrain").iterdir())
    if code == 0:
        assert leftovers == ["phobert-moi"]  # không còn thư mục tạm
        record = json.loads((out / "retrain.json").read_text("utf-8"))
        assert record["out"] == str(out) and record["decision"]["accepted"]
        assert record["steps"] == 2 and (out / "model.safetensors").is_file()
    else:
        assert leftovers == []  # bị từ chối: không để lại gì
        assert "TỪ CHỐI" in capsys.readouterr().out


def test_phobert_cli_waits_for_no_one_when_the_machine_lock_is_held(cli, home, capsys):
    """Khoá toàn máy đang do job khác giữ: PhoBERT từ chối ngay (mã 4); baseline vẫn chạy được."""
    from ml.common import mps_lock

    path = good_feedback(home)
    with mps_lock(note="job khác"):
        code = retrain.main(
            ["--feedback", str(path), "--model", "phobert", "--out", str(home / "a")]
        )
        assert code == 4
        assert "job khác" in capsys.readouterr().err
        assert not (home / "a").exists()
        assert retrain.main(["--feedback", str(path), "--out", str(home / "b")]) == 0
