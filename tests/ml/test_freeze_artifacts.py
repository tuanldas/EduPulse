"""Freeze trên artifact thật của máy: băm đủ tệp, mô tả đúng run triển khai, bắt tệp đã đổi.

Chỉ đọc MANIFEST của dữ liệu (băm của tệp test), không mở test.
"""

import pytest

pytestmark = [pytest.mark.data, pytest.mark.train, pytest.mark.slow]

pytest.importorskip("torch")

from ml import freeze  # noqa: E402
from ml.common import best_run, processed_manifest  # noqa: E402


@pytest.fixture(scope="module")
def document():
    try:
        return freeze.build_freeze()
    except (FileNotFoundError, freeze.GuardError) as error:
        pytest.skip(f"chưa đủ artifact để đóng băng: {error}")


def test_freeze_names_the_deployed_run_and_the_rule_that_chose_it(document):
    assert document["deploy"]["run"] == best_run().name
    assert document["deploy"]["seed"] == 42 or document["deploy"]["run"].startswith("mtl-s")
    assert "Run PhoBERT đa nhiệm duy nhất" in document["deploy"]["rule"]
    assert document["scope"].startswith("Không huấn luyện thêm")
    assert document["reruns"] == []


def test_freeze_hashes_the_checkpoint_the_int8_model_the_calibration_and_the_baselines(document):
    names = set(document["files"])
    run = document["deploy"]["run"]
    assert {f"phobert/{run}/model.safetensors", f"phobert/{run}/config.json"} <= names
    assert {"onnx/model.int8.onnx", "calibration/calibration.json", "tokenizer.json"} <= names
    baselines = {n for n in names if n.startswith("baseline/")}
    assert len(baselines) == 6 and all(n.endswith(".skops") for n in baselines)
    assert all(len(digest) == 64 for digest in document["files"].values())


def test_freeze_records_the_frozen_calibration_quantisation_and_baseline_settings(document):
    assert set(document["calibration"]["temperature"]) == {"sentiment", "topic"}
    assert set(document["calibration"]["tau"]) == {"5", "10", "20", "30"}
    assert document["quantization"]["gate_passed"] is True
    assert document["quantization"]["config"]
    assert set(document["baselines"]) == {"sentiment", "topic"}
    assert set(document["baselines"]["topic"]) == {"linearsvc", "logreg", "complementnb"}


def test_freeze_pins_the_test_file_by_the_hash_in_the_data_manifest(document):
    manifest = processed_manifest()
    assert document["data"]["test_sha256"] == manifest["files"]["test"]["sha256"]
    assert document["data"]["rows"]["test"] == manifest["files"]["test"]["rows"]
    assert document["data"]["normalizer_version"] == manifest["normalizer_version"]


def test_a_fresh_freeze_has_no_file_problems(document):
    assert freeze.file_problems(document) == []


def test_a_changed_model_file_is_reported_by_name(document):
    tampered = {**document, "files": {**document["files"], "onnx/model.int8.onnx": "0" * 64}}
    assert freeze.file_problems(tampered) == ["onnx/model.int8.onnx: sha256 khác freeze"]


def test_a_changed_deploy_run_or_test_hash_is_reported(document):
    other = {**document, "deploy": {**document["deploy"], "run": "mtl-s99"}}
    assert any("run triển khai theo luật" in p for p in freeze.file_problems(other))
    moved = {**document, "data": {**document["data"], "test_sha256": "1" * 64}}
    assert "dữ liệu test đã đổi so với freeze" in freeze.file_problems(moved)
