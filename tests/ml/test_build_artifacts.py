"""Đóng gói bundle: whitelist, schema, đồng nhất normalizer, và guard không lọt câu gốc."""

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline

from edupulse.bundle_schema import (
    BudgetEfficiency,
    Calibration,
    GoldenPredictions,
    HeadCalibration,
    InsightBackground,
    Manifest,
)
from edupulse.config import BUDGETS
from edupulse.labels import LABELS
from edupulse.scoring import CDF_POINTS
from edupulse.text import NORMALIZER_VERSION
from ml import build_artifacts as builder
from ml.common import load_processed, sha256_file, write_json
from ml.train_baseline import GRIDS, make_classifier, make_features, save_model

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

TEXTS = [
    "thầy dạy hay",
    "phòng máy nóng",
    "chương_trình nặng",
    "thầy nhiệt_tình",
    "máy_chiếu mờ",
] * 6
LABEL_IDS = np.array([2, 0, 0, 2, 0] * 6)


BASE_ROWS, TOTAL_ROWS, UNK = 10, 20, 3  # bảng embedding nhỏ: 10 hàng gốc, nới tới 20, <unk> = 3


def tiny_onnx(
    path: Path, output_names=("sentiment_logits", "topic_logits"), extra_rows_match_unk=True
) -> None:
    """ONNX nhỏ mang đúng tên vào/ra của bundle thật và có bảng embedding được nới như bản thật.

    Logits = trung bình có mặt nạ của embedding nhân trọng số cố định. Các hàng từ BASE_ROWS trở đi
    sao chép hàng UNK (hoặc khác đi nếu `extra_rows_match_unk=False`, để thử bộ kiểm tra).
    """
    ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    mask = helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "seq"])
    outputs = [
        helper.make_tensor_value_info(name, TensorProto.FLOAT, ["batch", size])
        for name, size in zip(output_names, (3, 4), strict=True)
    ]
    table = np.arange(TOTAL_ROWS * 4, dtype=np.float32).reshape(TOTAL_ROWS, 4) / 10.0
    table[BASE_ROWS:] = table[UNK] if extra_rows_match_unk else table[UNK] + 1.0
    nodes = [
        helper.make_node("Gather", ["table", "input_ids"], ["emb"]),
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["mask_f", "axes"], ["mask3"]),
        helper.make_node("Mul", ["emb", "mask3"], ["masked"]),
        helper.make_node("ReduceMean", ["masked"], ["mean"], axes=[1], keepdims=0),
        helper.make_node("MatMul", ["mean", "w_s"], [output_names[0]]),
        helper.make_node("MatMul", ["mean", "w_t"], [output_names[1]]),
    ]
    weights = [
        numpy_helper.from_array(table, "table"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axes"),
        numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), "w_s"),
        numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), "w_t"),
    ]
    graph = helper.make_graph(nodes, "tiny", [ids, mask], outputs, weights)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


def toy_baseline():
    features = make_features().fit(TEXTS)
    clf = make_classifier("linearsvc", GRIDS["linearsvc"][0]).fit(
        features.transform(TEXTS), LABEL_IDS
    )
    pipeline = Pipeline([("features", features), ("clf", clf)])
    return CalibratedClassifierCV(FrozenEstimator(pipeline), method="temperature").fit(
        TEXTS, LABEL_IDS
    )


def calibration() -> Calibration:
    head = HeadCalibration(temperature=1.0, cdf=[0.4 + 0.006 * i for i in range(CDF_POINTS)])
    efficiency = BudgetEfficiency(
        flagged_share=0.1,
        caught_sentiment_errors=0.5,
        caught_topic_errors=0.4,
        accept_accuracy_both=0.9,
        all_accuracy_both=0.86,
    )
    return Calibration(
        n=10,
        sentiment=head,
        topic=head,
        tau={b: 0.1 for b in BUDGETS},
        efficiency={b: efficiency for b in BUDGETS},
    )


def background(version=NORMALIZER_VERSION) -> InsightBackground:
    return InsightBackground(
        normalizer_version=version,
        alpha0=1000.0,
        min_count=2,
        n_sentences=10,
        total_ngrams=100,
        ngram_counts={"phòng": 5},
        sentiment_share=dict.fromkeys(LABELS["sentiment"], 1 / 3),
        topic_share=dict.fromkeys(LABELS["topic"], 0.25),
        negative_rate_by_topic=dict.fromkeys(LABELS["topic"], 0.5),
    )


def card_json() -> dict:
    task = {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "weighted_f1": 0.9,
        "ece": 0.02,
        "per_class": [],
        "confusion": [],
        "reliability": [],
    }
    return {
        "model_version": "m",
        "bundle_version": "v-test",
        "stage": "smoke",
        "metrics": {"split": "val", "n": 10, "sentiment": task, "topic": task},
        "intended_use": "x",
        "out_of_scope": "y",
        "limitations": [],
        "notes": {"x86": "chưa đo"},
    }


@pytest.fixture()
def bundle(tmp_path) -> Path:
    """Bundle tối thiểu nhưng đúng cấu trúc: đủ tệp, đúng schema, ONNX có đúng tên vào/ra."""
    root = tmp_path / "bundle"
    (root / "baseline").mkdir(parents=True)
    tiny_onnx(root / "model.int8.onnx")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    write_json(root / "calibration.json", calibration().model_dump(mode="json"))
    write_json(root / "model_card.json", card_json())
    write_json(root / "insight_background.json", background().model_dump(mode="json"))
    golden = GoldenPredictions(normalizer_version=NORMALIZER_VERSION, items=[])
    write_json(root / "golden_predictions.json", golden.model_dump(mode="json"))
    for name in builder.BASELINE_FILES:
        save_model(toy_baseline(), root / name)
    manifest = Manifest(
        bundle_version="v-test",
        stage="smoke",
        model_version="m",
        created_at=datetime.now(UTC),
        git_commit="abc",
        normalizer_version=NORMALIZER_VERSION,
        max_len=128,
        labels={task: list(names) for task, names in LABELS.items()},
        onnx={
            "quantization": "test",
            "base_vocab_size": BASE_ROWS,
            "unk_id": UNK,
            "embedding_rows": TOTAL_ROWS,
        },
        files={name: sha256_file(root / name) for name in builder.BUNDLE_FILES},
        dataset={"name": "d", "revision": "r", "splits": {"train": 1}, "license_note": "n"},
        base_model={"name": "b", "revision": "r", "license": "l"},
    )
    write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    return root


def test_bundle_file_list_is_flat_or_one_level_and_unique():
    assert len(set(builder.BUNDLE_FILES)) == len(builder.BUNDLE_FILES)
    for name in builder.BUNDLE_FILES:
        assert ".." not in name and not name.startswith("/")
        assert name.count("/") <= 1


def test_valid_bundle_passes_verification(bundle):
    manifest = builder.verify_bundle(bundle)
    assert manifest.bundle_version == "v-test"
    assert set(manifest.files) == set(builder.BUNDLE_FILES)


def test_tampered_file_is_caught_by_sha256(bundle):
    (bundle / "tokenizer.json").write_text('{"x": 1}', encoding="utf-8")
    with pytest.raises(builder.BundleError, match="tokenizer.json.*sha256"):
        builder.verify_bundle(bundle)


def test_file_outside_the_whitelist_is_refused(bundle):
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["extra.bin"] = "a" * 64
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(builder.BundleError, match="danh sách cho phép"):
        builder.verify_bundle(bundle)


def test_invalid_json_content_fails_the_schema(bundle):
    broken = json.loads((bundle / "calibration.json").read_text("utf-8"))
    broken["tau"].pop("10")
    write_json(bundle / "calibration.json", broken)
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["calibration.json"] = sha256_file(bundle / "calibration.json")
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(ValueError, match="ngân sách"):
        builder.verify_bundle(bundle)


def test_wrong_onnx_output_names_are_caught(bundle):
    tiny_onnx(bundle / "model.int8.onnx", output_names=("a", "b"))
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["model.int8.onnx"] = sha256_file(bundle / "model.int8.onnx")
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(builder.BundleError, match="đầu ra ONNX"):
        builder.verify_bundle(bundle)


def test_expanded_embedding_rows_that_do_not_behave_as_unk_are_caught(bundle):
    tiny_onnx(bundle / "model.int8.onnx", extra_rows_match_unk=False)
    manifest = json.loads((bundle / "manifest.json").read_text("utf-8"))
    manifest["files"]["model.int8.onnx"] = sha256_file(bundle / "model.int8.onnx")
    write_json(bundle / "manifest.json", manifest)
    with pytest.raises(builder.BundleError, match="không hành xử như <unk>"):
        builder.verify_bundle(bundle)


def test_manifest_records_the_embedding_expansion(bundle):
    spec = builder.verify_bundle(bundle).onnx
    assert (spec.base_vocab_size, spec.unk_id, spec.embedding_rows) == (BASE_ROWS, UNK, TOTAL_ROWS)


def passing_gate(run_name="mtl-s1", passed=True):
    return {
        "run": run_name,
        "normalizer_version": NORMALIZER_VERSION,
        "gate": {"passed": passed},
        "embedding": {"base_vocab_size": 64001, "unk_id": 3, "embedding_rows": 66119},
    }


def test_gate_must_have_passed_for_the_same_run(tmp_path):
    run = tmp_path / "mtl-s1"
    assert (
        builder.require_passed_gate(passing_gate(), run, allow_failed=False)["embedding_rows"]
        == 66119
    )
    with pytest.raises(builder.BundleError, match="cổng int8 chưa qua"):
        builder.require_passed_gate(passing_gate(passed=False), run, allow_failed=False)
    assert builder.require_passed_gate(passing_gate(passed=False), run, allow_failed=True)
    with pytest.raises(builder.BundleError, match="không phải"):
        builder.require_passed_gate(passing_gate("mtl-s2"), run, allow_failed=False)
    stale = {**passing_gate(), "normalizer_version": "0"}
    with pytest.raises(builder.BundleError, match="bản chuẩn hoá khác"):
        builder.require_passed_gate(stale, run, allow_failed=False)


def test_different_normalizer_versions_are_refused():
    history = {"normalizer_version": NORMALIZER_VERSION}
    golden = GoldenPredictions(normalizer_version=NORMALIZER_VERSION, items=[])
    stale = background(version="0")
    with pytest.raises(builder.BundleError, match="không đồng nhất"):
        builder.assert_same_normalizer(history, stale, golden)


@pytest.mark.data
def test_guard_rejects_a_bundle_that_contains_a_source_sentence(tmp_path):
    train = load_processed("train")
    long_sentences = [s for s in train["sentence"] if len(s.split()) >= builder.GUARD_MIN_TOKENS]
    leaked = tmp_path / "leaky"
    leaked.mkdir()
    write_json(leaked / "model_card.json", {"quote": long_sentences})
    with pytest.raises(builder.BundleError, match="câu gốc"):
        builder.assert_no_source_sentences(leaked)


@pytest.mark.data
def test_guard_accepts_a_bundle_without_source_sentences(bundle, tmp_path):
    clean = tmp_path / "clean"
    shutil.copytree(bundle, clean)
    assert builder.assert_no_source_sentences(clean) == builder.GUARD_SAMPLES
