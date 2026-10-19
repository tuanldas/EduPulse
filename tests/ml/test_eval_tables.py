"""Bảng, hình và test_all.json sinh từ dự đoán đã lưu: số đúng, khoảng tin cậy, đủ tệp."""

import csv
import json

import numpy as np
import pytest

from edupulse.labels import LABELS
from ml import eval_tables, evaluate, preds
from ml.calibrate import calibrate
from ml.common import prediction_metrics, settings, write_json

SIZES = {"sentiment": 3, "topic": 4}
QUALITY = {
    "linearsvc": 0.80,
    "logreg": 0.78,
    "complementnb": 0.70,
    "mtl-s42": 0.92,
    "mtl-int8": 0.91,
}


def make_preds(key: str, y: dict, accuracy: float, seed: int) -> preds.Preds:
    rng = np.random.default_rng(seed)
    pred, logits = {}, {}
    for task, truth in y.items():
        wrong = (truth + rng.integers(1, SIZES[task], size=len(truth))) % SIZES[task]
        pred[task] = np.where(rng.random(len(truth)) < accuracy, truth, wrong)
        sharp = 3.0 * np.eye(SIZES[task])[pred[task]] + rng.normal(
            scale=0.7, size=(len(truth), SIZES[task])
        )
        logits[task] = sharp.astype(np.float32)
        pred[task] = logits[task].argmax(axis=1)
    return preds.Preds(key, y, pred, logits)


@pytest.fixture(scope="module")
def synthetic_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    y = {task: rng.integers(0, size, size=600) for task, size in SIZES.items()}
    directory = tmp_path_factory.mktemp("preds")
    for index, (key, accuracy) in enumerate(QUALITY.items()):
        preds.save_preds(directory, make_preds(key, y, accuracy, seed=index + 10))
    return directory


TEMPERATURES = {"sentiment": 1.0, "topic": 1.0}


def test_collect_metrics_has_every_model_with_intervals_that_contain_the_estimate(synthetic_dir):
    collected = eval_tables.collect_metrics(synthetic_dir, TEMPERATURES)
    assert list(collected["models"]) == list(QUALITY)  # thứ tự chuẩn
    assert collected["n"] == 600 and collected["bootstrap"]["rounds"] == 2000
    for key, by_task in collected["models"].items():
        for task, metrics in by_task.items():
            ci = metrics["ci95"]
            for name in ("accuracy", "macro_f1", "weighted_f1"):
                assert ci[name]["low"] <= metrics[name] <= ci[name]["high"], (key, task, name)
            assert len(ci["per_class_f1"]) == len(LABELS[task])
            assert ("reliability" in metrics) == (key in eval_tables.RELIABILITY_KEYS)


def test_metrics_match_the_shared_prediction_metrics_function(synthetic_dir):
    preds_loaded = preds.load_preds(synthetic_dir, "linearsvc")
    direct = prediction_metrics(
        preds_loaded.y["topic"], preds_loaded.pred["topic"], LABELS["topic"]
    )
    metrics = eval_tables.preds_metrics(preds_loaded, "topic")
    for name in ("accuracy", "macro_f1", "weighted_f1", "confusion"):
        assert metrics[name] == direct[name]
    assert 0 <= metrics["ece"] <= 1 and metrics["nll"] > 0 and metrics["n"] == 600


def test_temperature_only_changes_the_probability_based_metrics(synthetic_dir):
    loaded = preds.load_preds(synthetic_dir, "mtl-int8")
    cold = eval_tables.preds_metrics(loaded, "sentiment", 1.0)
    hot = eval_tables.preds_metrics(loaded, "sentiment", 3.0)
    assert cold["macro_f1"] == hot["macro_f1"] and cold["accuracy"] == hot["accuracy"]
    assert cold["ece"] != hot["ece"] and cold["nll"] != hot["nll"]


def test_main_rows_cover_each_model_per_task_with_confidence_intervals(synthetic_dir):
    rows = eval_tables.main_rows(eval_tables.collect_metrics(synthetic_dir, TEMPERATURES))
    assert len(rows) == 5 * 2
    assert [r["model"] for r in rows if r["task"] == "sentiment"] == list(QUALITY)
    first = rows[0]
    assert {"low", "high", "value"} <= set(first["macro_f1"])
    assert len(first["per_class_f1"]) == 3 and first["display"] == "TF-IDF + LinearSVC"
    assert rows[4]["display"].startswith("PhoBERT đa nhiệm · int8")


def test_markdown_shows_intervals_and_vietnamese_decimals(synthetic_dir):
    rows = eval_tables.main_rows(eval_tables.collect_metrics(synthetic_dir, TEMPERATURES))
    text = eval_tables.main_markdown(rows, n=600, split_label="test", note="ghi chú")
    assert "### Cảm xúc" in text and "### Chủ đề" in text and text.rstrip().endswith("ghi chú")
    assert "[" in text and "–" in text and "(n = 600)" in text
    assert "F1 Trung tính" in text and "F1 Khác" in text
    for line in text.splitlines():
        if line.startswith("| TF-IDF + LinearSVC"):
            assert "," in line and "." not in line.split("|")[2]  # dấu phẩy thập phân kiểu Việt


def test_main_csv_has_one_row_per_model_and_task_with_interval_columns(synthetic_dir, tmp_path):
    rows = eval_tables.main_rows(eval_tables.collect_metrics(synthetic_dir, TEMPERATURES))
    path = tmp_path / "test_main.csv"
    eval_tables.write_main_csv(rows, path)
    with path.open(encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    assert len(data) == 10
    first = data[0]
    assert first["model"] == "linearsvc" and first["task"] == "sentiment"
    assert float(first["macro_f1_low"]) <= float(first["macro_f1"]) <= float(first["macro_f1_high"])
    assert first["f1_neutral"] != "" and first["f1_lecturer"] == ""  # cột của tác vụ khác để trống


def test_per_class_csv_lists_every_class_with_support_and_interval(synthetic_dir, tmp_path):
    collected = eval_tables.collect_metrics(synthetic_dir, TEMPERATURES)
    path = tmp_path / "per_class.csv"
    eval_tables.write_per_class_csv(collected["models"], path)
    with path.open(encoding="utf-8") as handle:
        data = list(csv.DictReader(handle))
    assert len(data) == 5 * (3 + 4)
    support = {
        r["label"]: int(r["support"])
        for r in data
        if r["model"] == "mtl-int8" and r["task"] == "topic"
    }
    assert sum(support.values()) == 600 and set(support) == set(LABELS["topic"])
    assert all(float(r["f1_low"]) <= float(r["f1"]) <= float(r["f1_high"]) for r in data)


def test_model_display_names(synthetic_dir):
    assert eval_tables.model_display("complementnb") == "TF-IDF + ComplementNB"
    assert eval_tables.model_display("mtl-s42") == "PhoBERT đa nhiệm · FP32 (seed 42)"
    assert eval_tables.model_display("mtl-int8").endswith("(bản triển khai)")
    with pytest.raises(ValueError):
        eval_tables.model_display("stl-topic-s42")


def test_confusion_and_reliability_figures_are_written(synthetic_dir, tmp_path):
    collected = eval_tables.collect_metrics(synthetic_dir, TEMPERATURES)
    models = collected["models"]
    cm = eval_tables.plot_confusion(
        tmp_path / "cm.png", models["mtl-int8"]["topic"]["confusion"], "topic", "Chủ đề"
    )
    series = {
        "PhoBERT": models["mtl-int8"]["sentiment"],
        "TF-IDF": models["linearsvc"]["sentiment"],
    }
    rel = eval_tables.plot_reliability(tmp_path / "rel.png", series, "sentiment", "Độ tin cậy")
    assert cm.stat().st_size > 5000 and rel.stat().st_size > 5000


@pytest.fixture
def rendered(synthetic_dir, tmp_path, monkeypatch):
    """`render_outputs` đầy đủ: calibration đóng băng của run triển khai + META do chấm cuối ghi."""
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path / "home"))
    rng = np.random.default_rng(5)
    val_y = {task: rng.integers(0, size, size=400) for task, size in SIZES.items()}
    val = make_preds("mtl-int8", val_y, 0.9, seed=99)
    calibration, _ = calibrate(val.logits, val_y)
    calibration_path = (
        tmp_path / "home" / "outputs" / "calibration" / "mtl-s42" / "calibration.json"
    )
    write_json(calibration_path, calibration.model_dump(mode="json"))
    meta = {
        "split": "test",
        "n": 600,
        "created_at": "2026-10-19T09:00:00+0700",
        "deploy_run": "mtl-s42",
    }
    write_json(synthetic_dir / preds.META_FILE, meta)
    ctx = preds.EvalContext("test", synthetic_dir, tmp_path / "results", rehearsal=False)
    return ctx, evaluate.render_outputs(ctx)


def test_render_outputs_writes_every_documented_file(rendered):
    ctx, written = rendered
    names = {path.name for path in written}
    assert {"test_all.json", "test_main.md", "test_main.csv", "test_per_class.csv"} <= names
    assert {"reliability_sentiment.png", "reliability_topic.png"} <= names
    for task in SIZES:
        for key in QUALITY:
            assert f"cm_{task}_{key}.png" in names
    assert all(path.is_file() for path in written)
    assert len(written) == 4 + 2 + 10


def test_test_all_json_records_the_provenance_and_the_frozen_temperatures(rendered):
    ctx, _ = rendered
    document = json.loads(ctx.metrics_path("test_all.json").read_text("utf-8"))
    assert document["split"] == "test" and document["deploy_run"] == "mtl-s42"
    assert document["evaluated_at"] == "2026-10-19T09:00:00+0700"
    frozen_path = settings().outputs_dir / "calibration" / "mtl-s42" / "calibration.json"
    frozen = json.loads(frozen_path.read_text("utf-8"))
    assert document["temperatures"] == {t: frozen[t]["temperature"] for t in SIZES}
    assert (
        document["models"]["mtl-int8"]["topic"]["ece"]
        != document["models"]["mtl-s42"]["topic"]["ece"]
    )
    assert document["models"]["mtl-int8"]["topic"]["ci95"]["macro_f1"]["low"] > 0
    assert "reliability" in document["models"]["linearsvc"]["sentiment"]


def test_rehearsal_outputs_are_labelled_so_they_cannot_pass_as_test(
    synthetic_dir, rendered, tmp_path
):
    ctx, _ = rendered
    rehearsal = preds.EvalContext(
        "validation", synthetic_dir, tmp_path / "rehearsal", rehearsal=True
    )
    evaluate.render_outputs(rehearsal)
    text = rehearsal.table_path("test_main.md").read_text("utf-8")
    assert "diễn tập" in text and "KHÔNG phải test" in text
    document = json.loads(rehearsal.metrics_path("test_all.json").read_text("utf-8"))
    assert document["rehearsal"] is True and document["split"] == "validation"
    assert "diễn tập" not in ctx.table_path("test_main.md").read_text("utf-8")
