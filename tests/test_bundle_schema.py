"""Schema bundle: bản smoke (số val) và bản release (số test) hợp lệ; thiếu/sai trường thì lỗi."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from edupulse.bundle_schema import (
    ArchitectureRow,
    BaselineRow,
    BudgetEfficiency,
    BudgetRow,
    Calibration,
    ChecklistRow,
    ClassMetrics,
    DemoManifest,
    DemoResults,
    DemoRow,
    GoldenItem,
    GoldenPredictions,
    HeadCalibration,
    InsightBackground,
    Manifest,
    ModelCard,
    ModelCardNotes,
    ReliabilityBin,
    SeedStat,
    SplitMetrics,
    TaskMetrics,
)
from edupulse.config import BUDGETS, DEFAULT_BUDGET
from edupulse.labels import LABELS
from edupulse.scoring import CDF_POINTS

SHA = "a" * 64


def manifest_data(**override):
    data = {
        "bundle_version": "v0-smoke",
        "stage": "smoke",
        "model_version": "mtl-s42",
        "created_at": datetime(2026, 10, 16, 9, 30, tzinfo=UTC),
        "git_commit": "abc1234",
        "normalizer_version": "1",
        "max_len": 128,
        "labels": {task: list(names) for task, names in LABELS.items()},
        "onnx": {"quantization": "int8 dynamic, MatMul, per-channel"},
        "files": {"model.int8.onnx": SHA, "baseline/sentiment_linearsvc.skops": "b" * 64},
        "dataset": {
            "name": "uitnlp/vietnamese_students_feedback",
            "revision": "2d76906157232f87e9883fa37ebfc91ffdee1f83",
            "splits": {"train": 11426, "validation": 1583, "test": 3166},
            "license_note": "học thuật, phi thương mại",
        },
        "base_model": {
            "name": "vinai/phobert-base-v2",
            "revision": "86cd7fd4",
            "license": "AGPL-3.0",
        },
    }
    data.update(override)
    return data


def head(temperature=1.1):
    return HeadCalibration(
        temperature=temperature, cdf=[0.4 + 0.006 * i for i in range(CDF_POINTS)]
    )


def efficiency(share=0.1):
    return BudgetEfficiency(
        flagged_share=share,
        caught_sentiment_errors=0.5,
        caught_topic_errors=0.4,
        accept_accuracy_both=0.9,
        all_accuracy_both=0.86,
    )


def calibration_data(**override):
    data = {
        "n": 1583,
        "sentiment": head(),
        "topic": head(1.0),
        "tau": {b: 0.05 * b for b in BUDGETS},
        "efficiency": {b: efficiency(b / 100) for b in BUDGETS},
    }
    data.update(override)
    return data


def task_metrics(labels):
    k = len(labels)
    return TaskMetrics(
        accuracy=0.95,
        macro_f1=0.85,
        weighted_f1=0.94,
        ece=0.02,
        per_class=[
            ClassMetrics(label=name, precision=0.9, recall=0.9, f1=0.9, support=100)
            for name in labels
        ],
        confusion=[[1.0 if i == j else 0.0 for j in range(k)] for i in range(k)],
        reliability=[
            ReliabilityBin(lower=0.0, upper=0.5, n=0, confidence=None, accuracy=None),
            ReliabilityBin(lower=0.5, upper=1.0, n=10, confidence=0.9, accuracy=0.88),
        ],
    )


def split_metrics(split):
    return SplitMetrics(
        split=split,
        n=1583,
        sentiment=task_metrics(LABELS["sentiment"]),
        topic=task_metrics(LABELS["topic"]),
    )


def model_card_data(**override):
    data = {
        "model_version": "mtl-s42",
        "bundle_version": "v0-smoke",
        "stage": "smoke",
        "metrics": split_metrics("val"),
        "intended_use": "Giúp đọc nhanh góp ý của sinh viên.",
        "out_of_scope": "Không dùng để đánh giá cá nhân giảng viên.",
        "limitations": ["Lớp trung tính yếu."],
        "notes": ModelCardNotes(x86="Chưa đo trên x86"),
    }
    data.update(override)
    return data


# --- Manifest ---------------------------------------------------------------------------------


def test_manifest_round_trips_through_json():
    manifest = Manifest(**manifest_data())
    again = Manifest.model_validate_json(manifest.model_dump_json())
    assert again == manifest
    assert again.stage == "smoke"
    assert again.onnx.output_names == ("sentiment_logits", "topic_logits")
    assert again.onnx.pad_id == 1


def test_manifest_release_stage_is_valid_too():
    assert Manifest(**manifest_data(bundle_version="v1.0", stage="release")).stage == "release"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "beta", "stage"),
        ("max_len", 0, "max_len"),
        (
            "labels",
            {"sentiment": ["positive", "neutral", "negative"], "topic": list(LABELS["topic"])},
            "nhãn",
        ),
        ("files", {"../model.onnx": SHA}, "không an toàn"),
        ("files", {"/etc/passwd": SHA}, "không an toàn"),
        ("files", {"a//b": SHA}, "không an toàn"),
        ("files", {"model.onnx": "xyz"}, "sha256"),
        ("files", {"model.onnx": "A" * 64}, "sha256"),
    ],
)
def test_manifest_rejects_bad_values(field, value, message):
    with pytest.raises(ValidationError, match=message):
        Manifest(**manifest_data(**{field: value}))


def test_manifest_requires_every_field_and_rejects_unknown_ones():
    data = manifest_data()
    del data["normalizer_version"]
    with pytest.raises(ValidationError, match="normalizer_version"):
        Manifest(**data)
    with pytest.raises(ValidationError, match="extra"):
        Manifest(**manifest_data(surprise=1))


# --- Calibration ------------------------------------------------------------------------------


def test_calibration_round_trips_with_integer_budget_keys():
    calibration = Calibration(**calibration_data())
    again = Calibration.model_validate_json(calibration.model_dump_json())
    assert again == calibration
    assert set(again.tau) == set(BUDGETS)
    assert DEFAULT_BUDGET in again.efficiency


def test_calibration_records_that_logits_were_run_one_sentence_at_a_time():
    calibration = Calibration(**calibration_data())
    assert calibration.logits_batch_size == 1
    assert '"logits_batch_size":1' in calibration.model_dump_json()
    with pytest.raises(ValidationError):
        Calibration(**calibration_data(logits_batch_size=32))


def test_calibration_rejects_bad_cdf():
    with pytest.raises(ValidationError):
        HeadCalibration(temperature=1.0, cdf=[0.5] * 100)
    with pytest.raises(ValidationError, match="không giảm"):
        HeadCalibration(temperature=1.0, cdf=[0.9 - 0.001 * i for i in range(CDF_POINTS)])
    with pytest.raises(ValidationError):
        HeadCalibration(temperature=0.0, cdf=[0.5] * CDF_POINTS)
    with pytest.raises(ValidationError, match=r"\[0, 1\]"):
        HeadCalibration(temperature=1.0, cdf=[1.5] * CDF_POINTS)


def test_calibration_requires_all_budgets_and_matching_tables():
    tau = {b: 0.1 for b in BUDGETS if b != DEFAULT_BUDGET}
    eff = {b: efficiency() for b in tau}
    with pytest.raises(ValidationError, match="thiếu ngân sách"):
        Calibration(**calibration_data(tau=tau, efficiency=eff))
    with pytest.raises(ValidationError, match="cùng tập ngân sách"):
        Calibration(**calibration_data(efficiency={b: efficiency() for b in BUDGETS if b != 5}))


# --- Model card -------------------------------------------------------------------------------


def test_smoke_model_card_has_validation_numbers_only():
    card = ModelCard(**model_card_data())
    assert card.metrics.split == "val"
    assert card.review_budget == []
    assert ModelCard.model_validate_json(card.model_dump_json()) == card


def test_smoke_model_card_rejects_test_numbers():
    with pytest.raises(ValidationError, match="val"):
        ModelCard(**model_card_data(metrics=split_metrics("test")))


def release_card_data(**override):
    data = model_card_data(
        stage="release",
        bundle_version="v1.0",
        metrics=split_metrics("test"),
        baselines=[
            BaselineRow(
                name="TF-IDF + LinearSVC",
                sentiment_accuracy=0.918,
                sentiment_macro_f1=0.796,
                topic_accuracy=0.875,
                topic_macro_f1=0.772,
            )
        ],
        architectures=[
            ArchitectureRow(
                name="Đa nhiệm",
                seeds=[42, 43, 44],
                sentiment_macro_f1=SeedStat(mean=0.85, std=0.01, n=3),
                topic_macro_f1=SeedStat(mean=0.82, std=0.01, n=3),
            )
        ],
        review_budget=[
            BudgetRow(budget=b, default=(b == DEFAULT_BUDGET), efficiency=efficiency(b / 100))
            for b in BUDGETS
        ],
        checklist=[
            ChecklistRow(name="Đổi thầy thành cô", kind="invariance", n=200, pass_rate=0.98)
        ],
    )
    data.update(override)
    return data


def test_release_model_card_needs_test_numbers_and_the_default_budget():
    card = ModelCard(**release_card_data())
    assert card.metrics.split == "test"
    assert [row.budget for row in card.review_budget if row.default] == [DEFAULT_BUDGET]
    with pytest.raises(ValidationError, match="test"):
        ModelCard(**release_card_data(metrics=split_metrics("val")))
    with pytest.raises(ValidationError, match="mặc định"):
        ModelCard(**release_card_data(review_budget=[]))


def test_probabilities_are_bounded():
    with pytest.raises(ValidationError):
        ClassMetrics(label="x", precision=1.2, recall=0.5, f1=0.5, support=1)


# --- Golden, insight, demo --------------------------------------------------------------------


def test_golden_predictions_check_vector_lengths():
    ok = GoldenItem(
        id="g01",
        model_text="thầy dạy hay .",
        sentiment="positive",
        topic="lecturer",
        p_sentiment=[0.05, 0.05, 0.9],
        p_topic=[0.9, 0.05, 0.03, 0.02],
    )
    golden = GoldenPredictions(normalizer_version="1", items=[ok])
    assert GoldenPredictions.model_validate_json(golden.model_dump_json()) == golden
    with pytest.raises(ValidationError):
        GoldenItem(**{**ok.model_dump(), "p_topic": [0.5, 0.5]})


def background_data(**override):
    data = {
        "normalizer_version": "1",
        "alpha0": 1000.0,
        "min_count": 2,
        "n_sentences": 11426,
        "total_ngrams": 199244,
        "ngram_counts": {"phòng": 321, "máy chiếu": 12},
        "sentiment_share": dict(zip(LABELS["sentiment"], (0.466, 0.040, 0.494), strict=True)),
        "topic_share": dict(zip(LABELS["topic"], (0.715, 0.193, 0.043, 0.049), strict=True)),
        "negative_rate_by_topic": dict(
            zip(LABELS["topic"], (0.356, 0.780, 0.954, 0.402), strict=True)
        ),
    }
    data.update(override)
    return data


def test_insight_background_validates_keys_and_counts():
    background = InsightBackground(**background_data())
    assert InsightBackground.model_validate_json(background.model_dump_json()) == background
    with pytest.raises(ValidationError, match="nhãn"):
        InsightBackground(**background_data(sentiment_share={"negative": 1.0}))
    with pytest.raises(ValidationError, match="min_count"):
        InsightBackground(**background_data(ngram_counts={"hiếm": 1}))


def test_demo_manifest_requires_a_reason_when_unavailable():
    assert DemoManifest(ok=True, dataset_revision="2d769061", bundle_version="v1.0", n_rows=3166).ok
    with pytest.raises(ValidationError, match="reason"):
        DemoManifest(ok=False)
    assert DemoManifest(ok=False, reason="không tải được dữ liệu").reason


def test_demo_rows_allow_extra_fields_from_the_analyzer():
    results = DemoResults(
        bundle_version="v1.0",
        rows=[
            DemoRow(display_text="Thầy dạy hay", sentiment="positive", topic="lecturer", score=0.93)
        ],
    )
    again = DemoResults.model_validate_json(results.model_dump_json())
    assert again.rows[0].model_extra == {"score": 0.93}
