"""Thẻ mô hình phát hành: số trên test từ kết quả đã chấm, kiểm nguồn gốc, đúng schema."""

import json

import numpy as np
import pytest

from edupulse.bundle_schema import ModelCard
from edupulse.config import BUDGETS, DEFAULT_BUDGET
from ml import build_artifacts as builder
from ml import evaluate, preds, release_card, review_report
from ml.calibrate import calibrate
from ml.common import settings, sha256_file, write_json

SIZES = {"sentiment": 3, "topic": 4}
KEYS = ("linearsvc", "logreg", "complementnb", "mtl-s42", "mtl-int8")
ACCURACY = (0.80, 0.78, 0.70, 0.92, 0.91)
FREEZE_SHA = "f" * 64
CHECKLIST_ROWS = [
    {"name": "Đổi hoa/thường", "kind": "invariance", "n": 1583, "pass_rate": 1.0, "note": "0 hỏng"},
    {
        "name": "Thêm 'không'",
        "kind": "directional",
        "n": 900,
        "pass_rate": 0.71,
        "note": "261 hỏng",
    },
    {
        "name": "Bỏ dấu toàn câu",
        "kind": "robustness",
        "n": 1583,
        "pass_rate": 0.97,
        "note": (
            "Đạt khi hệ thống gắn cờ không dấu (lọt cờ 47/1.583 câu). Accuracy: cảm xúc giảm "
            "12,5 điểm (94,8% → 82,3%), chủ đề giảm 15,0 điểm (89,9% → 74,9%)."
        ),
    },
]


def make_preds(key: str, y: dict, accuracy: float, seed: int) -> preds.Preds:
    rng = np.random.default_rng(seed)
    pred, logits = {}, {}
    for task, truth in y.items():
        wrong = (truth + rng.integers(1, SIZES[task], size=len(truth))) % SIZES[task]
        pred[task] = np.where(rng.random(len(truth)) < accuracy, truth, wrong)
        noise = rng.normal(scale=0.7, size=(len(truth), SIZES[task]))
        logits[task] = (3.0 * np.eye(SIZES[task])[pred[task]] + noise).astype(np.float32)
        pred[task] = logits[task].argmax(axis=1)
    return preds.Preds(key, y, pred, logits)


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Thư mục results/ tổng hợp đúng như `make report` tạo ra, bằng chính mã của dự án."""
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path / "home"))
    rng = np.random.default_rng(0)
    y = {task: rng.integers(0, size, size=500) for task, size in SIZES.items()}
    directory = tmp_path / "preds"
    for index, (key, accuracy) in enumerate(zip(KEYS, ACCURACY, strict=True)):
        preds.save_preds(directory, make_preds(key, y, accuracy, seed=index + 20))
    val_y = {task: rng.integers(0, size, size=300) for task, size in SIZES.items()}
    calibration, _ = calibrate(make_preds("mtl-int8", val_y, 0.9, seed=77).logits, val_y)
    calibration_path = settings().outputs_dir / "calibration" / "mtl-s42" / "calibration.json"
    write_json(calibration_path, calibration.model_dump(mode="json"))
    meta = {
        "split": "test",
        "n": 500,
        "created_at": "2026-10-19T10:00:00+0700",
        "deploy_run": "mtl-s42",
        "freeze_sha256": FREEZE_SHA,
    }
    write_json(directory / preds.META_FILE, meta)
    ctx = preds.EvalContext("test", directory, tmp_path / "results", rehearsal=False)
    evaluate.render_outputs(ctx)
    report = review_report.build_report(ctx, calibration, "mtl-s42")
    review_report.write_outputs(ctx, report, calibration)
    write_json(ctx.metrics_path("checklist.json"), {"split": "validation", "rows": CHECKLIST_ROWS})
    gate = {"int8": {"sentiment": {"macro_f1": 0.8531}, "topic": {"macro_f1": 0.8096}}}
    write_json(ctx.metrics_path("onnx_gate_v1.json"), gate)
    return ctx


def card(results_ctx, **overrides) -> ModelCard:
    options = {
        "bundle_version": "v1.0",
        "model_version": "edupulse-mtl-s42",
        "run": "mtl-s42",
        "freeze_sha256": FREEZE_SHA,
        "results_dir": results_ctx.results_dir,
        "intended_use": "dùng để đọc nhanh",
        "out_of_scope": "không dùng để kỷ luật",
        "static_limitations": ["Mỉa mai dễ bị hiểu sai."],
        "notes": builder.ModelCardNotes(x86=builder.NOTE_X86, demo=builder.NOTE_DEMO),
    }
    return release_card.build_release_card(**{**options, **overrides})


def test_evaluation_notes_lead_the_limitations_and_static_ones_come_last(results):
    note = "Số test lấy từ bản trước."
    limits = card(results, evaluation_notes=[note]).limitations
    assert limits[0] == note
    assert limits[-1] == "Mỉa mai dễ bị hiểu sai."
    assert card(results).limitations == limits[1:]  # không ghi chú thì danh sách như cũ


def test_the_card_is_a_valid_release_card_with_test_metrics_of_the_deployed_model(results):
    model_card = card(results)
    assert model_card.stage == "release" and model_card.metrics.split == "test"
    assert model_card.metrics.n == 500 and model_card.bundle_version == "v1.0"
    test_all = json.loads(results.metrics_path("test_all.json").read_text("utf-8"))
    deployed = test_all["models"]["mtl-int8"]["sentiment"]
    assert model_card.metrics.sentiment.accuracy == pytest.approx(deployed["accuracy"])
    assert model_card.metrics.sentiment.ece == pytest.approx(deployed["ece"])
    assert len(model_card.metrics.topic.reliability) == 15
    assert len(model_card.metrics.topic.confusion) == 4
    again = ModelCard.model_validate_json(model_card.model_dump_json())
    assert again == model_card


def test_baselines_are_listed_with_test_numbers_and_the_int8_build_is_the_deployed_row(results):
    rows = card(results).baselines
    assert [r.name for r in rows] == [
        "TF-IDF + LinearSVC",
        "TF-IDF + LogisticRegression",
        "TF-IDF + ComplementNB",
        "PhoBERT đa nhiệm · FP32 (seed 42)",
        "PhoBERT đa nhiệm · int8 (bản triển khai)",
    ]
    assert [r.deployed for r in rows] == [False, False, False, False, True]
    test_all = json.loads(results.metrics_path("test_all.json").read_text("utf-8"))
    assert rows[0].topic_macro_f1 == pytest.approx(
        test_all["models"]["linearsvc"]["topic"]["macro_f1"]
    )


def test_review_budget_rows_come_from_the_test_report_with_ten_percent_as_default(results):
    rows = card(results).review_budget
    assert [r.budget for r in rows] == list(BUDGETS)
    assert [r.budget for r in rows if r.default] == [DEFAULT_BUDGET]
    report = json.loads(results.metrics_path("review_budget_test.json").read_text("utf-8"))
    assert rows[1].efficiency.caught_topic_errors == pytest.approx(
        report["budgets"]["10"]["rank_batch"]["caught_topic_errors"]
    )


def test_architecture_rows_are_the_research_experiment_and_say_so(results):
    rows = card(results).architectures
    assert len(rows) == 2 and all(r.seeds == [42] for r in rows)
    assert all("sơ bộ" in r.name for r in rows)
    assert all(r.sentiment_macro_f1.n == 1 and r.sentiment_macro_f1.std == 0.0 for r in rows)
    stl, mtl = rows
    assert stl.sentiment_macro_f1.mean == pytest.approx(0.8701)
    assert mtl.topic_macro_f1.mean == pytest.approx(0.8272)


def test_checklist_rows_are_carried_over_unchanged(results):
    rows = card(results).checklist
    assert [(r.name, r.kind, r.n, r.pass_rate) for r in rows] == [
        (row["name"], row["kind"], row["n"], row["pass_rate"]) for row in CHECKLIST_ROWS
    ]


def test_limitations_carry_measured_numbers_then_the_static_ones(results):
    items = card(results).limitations
    assert items[0].startswith("Lớp Trung tính và chủ đề Khác yếu: trên test, recall")
    diacritics = next(item for item in items if item.startswith("Câu gõ không dấu"))
    assert "cảm xúc giảm 12,5 điểm" in diacritics and "chủ đề giảm 15,0 điểm" in diacritics
    assert "Đạt khi" not in diacritics and "97,0% số câu thử bị gắn cờ" in diacritics
    interval = next(item for item in items if item.startswith("Khoảng tin cậy 95%"))
    assert "[" in interval and "2.000 vòng" in interval
    gap = next(item for item in items if item.startswith("Macro-F1 của bản int8 trên test và"))
    assert "85,3 trên validation" in gap and "81,0 trên validation" in gap  # số trong gate
    assert gap.count("(chênh ") == 2
    assert any("một run huấn luyện" in item and "McNemar p = 0,31" in item for item in items)
    assert items[-1] == "Mỉa mai dễ bị hiểu sai."


def test_without_the_gate_file_the_validation_gap_sentence_is_left_out(results):
    results.metrics_path("onnx_gate_v1.json").unlink()
    items = card(results).limitations
    assert not any(item.startswith("Macro-F1 của bản int8 trên test và") for item in items)


def test_checklist_rows_below_ninety_percent_are_named_in_the_limitations(results):
    items = card(results).limitations
    weak = next(item for item in items if item.startswith("CheckList trên validation"))
    assert "Thêm 'không' 71,0% (n = 900)" in weak and "Đổi hoa/thường" not in weak


def test_accuracy_drop_keeps_only_the_accuracy_part_of_the_note_or_the_whole_note():
    note = "Đạt khi cờ (lọt cờ 1/9 câu). Accuracy: cảm xúc giảm 13,6 điểm, chủ đề giảm 13,0 điểm."
    assert release_card.accuracy_drop(note) == "cảm xúc giảm 13,6 điểm, chủ đề giảm 13,0 điểm"
    assert release_card.accuracy_drop("giảm nhiều.") == "giảm nhiều"


def test_a_clean_checklist_adds_no_weak_row_sentence(results):
    rows = [{**row, "pass_rate": 0.97} for row in CHECKLIST_ROWS]
    write_json(results.metrics_path("checklist.json"), {"split": "validation", "rows": rows})
    assert not any(i.startswith("CheckList trên validation") for i in card(results).limitations)


def test_without_a_robustness_row_the_no_diacritics_sentence_is_left_out(results):
    rows = [row for row in CHECKLIST_ROWS if row["kind"] != "robustness"]
    write_json(results.metrics_path("checklist.json"), {"split": "validation", "rows": rows})
    assert not any(item.startswith("Câu gõ không dấu") for item in card(results).limitations)


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda d: d.update(split="validation"), "không phải kết quả chấm test thật"),
        (lambda d: d.update(rehearsal=True), "không phải kết quả chấm test thật"),
        (lambda d: d.update(deploy_run="mtl-s43"), "là của run 'mtl-s43'"),
        (lambda d: d.update(freeze_sha256="0" * 64), "freeze_sha256 không khớp"),
    ],
)
def test_results_from_another_run_a_rehearsal_or_another_freeze_are_refused(results, edit, message):
    path = results.metrics_path("test_all.json")
    document = json.loads(path.read_text("utf-8"))
    edit(document)
    write_json(path, document)
    with pytest.raises(release_card.ReleaseError, match=message):
        card(results)


def test_missing_report_files_are_listed(results):
    results.metrics_path("checklist.json").unlink()
    with pytest.raises(release_card.ReleaseError, match="checklist.json"):
        card(results)


# --- ghép với bundle ---------------------------------------------------------------------------


def frozen_document(bundle):
    files = {
        frozen: sha256_file(bundle / name) for frozen, name in builder.FROZEN_IN_BUNDLE.items()
    }
    return {"deploy": {"run": "mtl-s42"}, "files": files}


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    (directory / "baseline").mkdir(parents=True)
    for index, name in enumerate(builder.FROZEN_IN_BUNDLE.values()):
        (directory / name).write_bytes(f"nội dung {index}".encode())
    return directory


def test_bundle_files_that_match_the_freeze_pass(bundle):
    builder.assert_matches_freeze(bundle, frozen_document(bundle))


@pytest.mark.parametrize(
    ("frozen_name", "message"),
    [
        ("onnx/model.int8.onnx", "model.int8.onnx khác bản đã đóng băng"),
        ("calibration/calibration.json", "calibration.json khác bản đã đóng băng"),
        ("baseline/topic_linearsvc.skops", "topic_linearsvc.skops khác bản đã đóng băng"),
    ],
)
def test_a_bundle_file_that_differs_from_the_frozen_one_is_refused(bundle, frozen_name, message):
    document = frozen_document(bundle)
    document["files"][frozen_name] = "0" * 64
    with pytest.raises(builder.BundleError, match=message):
        builder.assert_matches_freeze(bundle, document)


def test_release_build_needs_a_clean_tree_and_the_freeze_of_the_same_run(monkeypatch, tmp_path):
    from ml import freeze

    class Run:
        name = "mtl-s42"

    monkeypatch.setattr(builder, "git_commit", lambda: "abc1234567+dirty")
    with pytest.raises(builder.BundleError, match="cây git sạch"):
        builder.load_release_freeze(Run())

    monkeypatch.setattr(builder, "git_commit", lambda: "abc1234567")
    monkeypatch.setattr(freeze, "FREEZE_PATH", tmp_path / "freeze.json")
    with pytest.raises(builder.BundleError, match="chưa có"):
        builder.load_release_freeze(Run())
    write_json(freeze.FREEZE_PATH, {"deploy": {"run": "mtl-s43"}})
    with pytest.raises(builder.BundleError, match="freeze.json là của run 'mtl-s43'"):
        builder.load_release_freeze(Run())
    write_json(freeze.FREEZE_PATH, {"deploy": {"run": "mtl-s42"}})
    assert builder.load_release_freeze(Run())["deploy"]["run"] == "mtl-s42"
