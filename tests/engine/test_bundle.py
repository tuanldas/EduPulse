"""Nạp bundle: tệp thiếu, hỏng, sai sha256 hoặc lệch chuẩn hoá đều dừng với thông báo rõ.

Các test lỗi dựng thư mục bundle nhỏ (tệp mô hình và tokenizer chỉ là byte giữ chỗ, vì
`load_bundle` kiểm tra tính toàn vẹn chứ không đọc ONNX). Test với bundle thật gắn marker `model`.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from edupulse.bundle_schema import Calibration
from edupulse.config import BUDGETS, Settings
from edupulse.engine.bundle import (
    CALIBRATION,
    MODEL,
    TOKENIZER,
    BundleError,
    load_bundle,
)
from edupulse.labels import LABELS
from edupulse.scoring import CDF_POINTS
from edupulse.text import NORMALIZER_VERSION

PLACEHOLDERS = {MODEL: b"khong phai onnx", TOKENIZER: b"{}"}


def _calibration() -> Calibration:
    efficiency = {
        "flagged_share": 0.1,
        "caught_sentiment_errors": 0.4,
        "caught_topic_errors": 0.3,
        "accept_accuracy_both": 0.9,
        "all_accuracy_both": 0.85,
    }
    cdf = [0.4 + 0.6 * i / (CDF_POINTS - 1) for i in range(CDF_POINTS)]
    return Calibration.model_validate(
        {
            "n": 100,
            "sentiment": {"temperature": 1.2, "cdf": cdf},
            "topic": {"temperature": 1.1, "cdf": cdf},
            "tau": {budget: budget / 100 for budget in BUDGETS},
            "efficiency": {budget: efficiency for budget in BUDGETS},
        }
    )


def write_bundle(root: Path, **manifest_override) -> Path:
    """Thư mục bundle hợp lệ; `manifest_override` ghi đè trường manifest (kể cả để làm nó sai)."""
    root.mkdir(parents=True, exist_ok=True)
    for name, content in PLACEHOLDERS.items():
        (root / name).write_bytes(content)
    (root / CALIBRATION).write_text(_calibration().model_dump_json(), encoding="utf-8")
    files = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (MODEL, TOKENIZER, CALIBRATION)
    }
    files.update(manifest_override.pop("files", {}))
    manifest = {
        "bundle_version": "v-test",
        "stage": "smoke",
        "model_version": "mtl-s42",
        "created_at": datetime(2026, 10, 16, tzinfo=UTC).isoformat(),
        "git_commit": "abc1234",
        "normalizer_version": NORMALIZER_VERSION,
        "max_len": 128,
        "labels": {task: list(names) for task, names in LABELS.items()},
        "onnx": {"quantization": "int8"},
        "files": files,
        "dataset": {"name": "d", "revision": "r", "splits": {"train": 1}, "license_note": "n"},
        "base_model": {"name": "b", "revision": "r", "license": "AGPL-3.0"},
    }
    (root / "manifest.json").write_text(json.dumps(manifest | manifest_override), encoding="utf-8")
    return root


def test_valid_bundle_loads_manifest_and_calibration(tmp_path):
    bundle = load_bundle(write_bundle(tmp_path / "v"))
    assert bundle.manifest.bundle_version == "v-test"
    assert bundle.calibration.sentiment.temperature == pytest.approx(1.2)
    assert bundle.calibration.tau[10] == pytest.approx(0.10)
    assert bundle.model_path == tmp_path / "v" / MODEL
    assert bundle.tokenizer_path == tmp_path / "v" / TOKENIZER


def test_missing_bundle_directory_says_where_and_how_to_get_one(tmp_path):
    with pytest.raises(BundleError) as failure:
        load_bundle(tmp_path / "chua-co")
    message = str(failure.value)
    assert "chua-co" in message and "manifest.json" in message
    assert "make artifacts" in message and "make docker-build" in message


def test_empty_directory_is_a_missing_bundle(tmp_path):
    with pytest.raises(BundleError, match="manifest.json"):
        load_bundle(tmp_path)


def test_unreadable_manifest_reports_the_fields_but_not_their_content(tmp_path):
    root = write_bundle(tmp_path / "v")
    broken = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    broken["max_len"] = "NOI-DUNG-LA-1"
    del broken["files"]
    (root / "manifest.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(BundleError) as failure:
        load_bundle(root)
    message = str(failure.value)
    assert "max_len" in message and "files" in message
    assert "NOI-DUNG-LA-1" not in message


def test_manifest_that_is_not_json_is_rejected(tmp_path):
    root = write_bundle(tmp_path / "v")
    (root / "manifest.json").write_text("{không phải json", encoding="utf-8")
    with pytest.raises(BundleError, match="manifest.json"):
        load_bundle(root)


def test_label_order_that_differs_from_the_code_is_rejected(tmp_path):
    swapped = {"sentiment": ["positive", "neutral", "negative"], "topic": list(LABELS["topic"])}
    root = write_bundle(tmp_path / "v", labels=swapped)
    with pytest.raises(BundleError, match="labels"):
        load_bundle(root)


def test_tampered_model_file_fails_the_sha256_check(tmp_path):
    root = write_bundle(tmp_path / "v")
    (root / MODEL).write_bytes(b"da bi sua")
    with pytest.raises(BundleError) as failure:
        load_bundle(root)
    assert MODEL in str(failure.value) and "sha256" in str(failure.value)


def test_verification_can_be_skipped_for_trusted_local_runs(tmp_path):
    root = write_bundle(tmp_path / "v")
    (root / MODEL).write_bytes(b"da bi sua")
    assert load_bundle(root, verify_files=False).manifest.bundle_version == "v-test"


def test_file_listed_in_the_manifest_but_absent_is_reported(tmp_path):
    root = write_bundle(tmp_path / "v", files={"baseline/svm.skops": "a" * 64})
    with pytest.raises(BundleError, match=r"baseline/svm\.skops"):
        load_bundle(root)


def test_bundle_built_with_another_normalizer_version_is_refused(tmp_path):
    root = write_bundle(tmp_path / "v", normalizer_version="0")
    with pytest.raises(BundleError) as failure:
        load_bundle(root)
    message = str(failure.value)
    assert "chuẩn hoá" in message and "0" in message and NORMALIZER_VERSION in message


@pytest.mark.parametrize("missing", [MODEL, TOKENIZER, CALIBRATION])
def test_manifest_must_list_the_files_the_engine_needs(tmp_path, missing):
    root = write_bundle(tmp_path / "v")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    del manifest["files"][missing]
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match=missing):
        load_bundle(root)


def test_invalid_calibration_is_rejected(tmp_path):
    root = write_bundle(tmp_path / "v")
    (root / CALIBRATION).write_text('{"n": 1}', encoding="utf-8")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["files"][CALIBRATION] = hashlib.sha256((root / CALIBRATION).read_bytes()).hexdigest()
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BundleError, match="calibration.json"):
        load_bundle(root)


@pytest.mark.model
def test_real_bundle_passes_every_check(settings: Settings):
    bundle = load_bundle(settings.bundle_dir)
    assert bundle.manifest.normalizer_version == NORMALIZER_VERSION
    assert bundle.manifest.labels == {task: list(names) for task, names in LABELS.items()}
    assert bundle.model_path.is_file() and bundle.tokenizer_path.is_file()
    assert bundle.calibration.tau[10] > 0
