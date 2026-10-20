"""Bundle tối thiểu nhưng đúng cấu trúc, dùng chung cho các test đóng gói, đẩy lên Hub và tải về.

ONNX nhỏ mang đúng tên vào/ra và bảng embedding được nới như bản thật, baseline skops thật, mọi
JSON đúng schema; chỉ có số liệu là giả định. `stage="release"` thêm thẻ mô hình có số trên test
(đủ bảng để dựng README) và `frozen_document` dựng bản freeze khớp. Phần cuối dựng một "Hub" giả
là thư mục và chạy scripts/fetch-artifacts.sh qua file://, nên không test nào cần mạng.
"""

import json
import os
import shutil
import subprocess
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
from edupulse.config import BUDGETS, DEFAULT_BUDGET, REPO_ROOT
from edupulse.labels import LABELS
from edupulse.scoring import CDF_POINTS
from edupulse.text import NORMALIZER_VERSION
from ml import build_artifacts as builder
from ml.common import sha256_file, write_json
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
BUNDLE_VERSION, MODEL_VERSION = "v-test", "m"
DATASET_NAME, BASE_MODEL_NAME = "uitnlp/vietnamese_students_feedback", "vinai/phobert-base-v2"


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


def efficiency() -> BudgetEfficiency:
    return BudgetEfficiency(
        flagged_share=0.1,
        caught_sentiment_errors=0.5,
        caught_topic_errors=0.4,
        accept_accuracy_both=0.9,
        all_accuracy_both=0.86,
    )


def calibration() -> Calibration:
    head = HeadCalibration(temperature=1.0, cdf=[0.4 + 0.006 * i for i in range(CDF_POINTS)])
    return Calibration(
        n=10,
        sentiment=head,
        topic=head,
        tau={b: 0.1 for b in BUDGETS},
        efficiency={b: efficiency() for b in BUDGETS},
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


def task_json() -> dict:
    return {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "weighted_f1": 0.9,
        "ece": 0.02,
        "per_class": [],
        "confusion": [],
        "reliability": [],
    }


def card_json() -> dict:
    """Thẻ smoke tối thiểu (số validation, không bảng)."""
    return {
        "model_version": MODEL_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "stage": "smoke",
        "metrics": {"split": "val", "n": 10, "sentiment": task_json(), "topic": task_json()},
        "intended_use": "x",
        "out_of_scope": "y",
        "limitations": [],
        "notes": {"x86": "chưa đo"},
    }


def per_class(task: str) -> list[dict]:
    return [
        {
            "label": label,
            "precision": 0.8 + 0.01 * i,
            "recall": 0.7 + 0.01 * i,
            "f1": 0.75 + 0.01 * i,
            "support": 100 * (i + 1),
        }
        for i, label in enumerate(LABELS[task])
    ]


def release_task(task: str, accuracy: float, macro_f1: float, ece: float) -> dict:
    return {
        **task_json(),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "ece": ece,
        "per_class": per_class(task),
    }


def release_card_json() -> dict:
    """Thẻ release: số trên test, đủ bảng baseline, kiến trúc, ngân sách duyệt và CheckList.

    Số liệu chọn để làm tròn không rơi vào ranh giới (82,13 chứ không phải 82,05).
    """
    card = card_json()
    card["stage"] = "release"
    card["metrics"] = {
        "split": "test",
        "n": 3166,
        "sentiment": release_task("sentiment", 0.9399, 0.8213, 0.0138),
        "topic": release_task("topic", 0.8998, 0.8131, 0.0232),
    }
    card["baselines"] = [
        {
            "name": "TF-IDF + LinearSVC",
            "sentiment_accuracy": 0.8942,
            "sentiment_macro_f1": 0.7345,
            "topic_accuracy": 0.8632,
            "topic_macro_f1": 0.7324,
            "sentiment_ece": 0.0107,
        },
        {
            "name": "PhoBERT đa nhiệm · int8 (bản triển khai)",
            "deployed": True,
            "sentiment_accuracy": 0.9399,
            "sentiment_macro_f1": 0.8213,
            "topic_accuracy": 0.8998,
            "topic_macro_f1": 0.8131,
        },
    ]
    card["architectures"] = [
        {
            "name": "Đa nhiệm (MTL), sơ bộ: validation, 1 seed",
            "seeds": [42],
            "sentiment_macro_f1": {"mean": 0.8555, "std": 0.0, "n": 1},
            "topic_macro_f1": {"mean": 0.8272, "std": 0.0, "n": 1},
        }
    ]
    card["review_budget"] = [
        {
            "budget": b,
            "default": b == DEFAULT_BUDGET,
            "efficiency": efficiency().model_dump(mode="json"),
        }
        for b in BUDGETS
    ]
    card["checklist"] = [
        {"name": "Viết HOA toàn câu", "kind": "invariance", "n": 1583, "pass_rate": 1.0},
        {"name": "Bỏ dấu toàn câu", "kind": "robustness", "n": 1579, "pass_rate": None},
    ]
    card["intended_use"] = "Giúp cán bộ đọc nhanh góp ý của sinh viên."
    card["out_of_scope"] = "Đánh giá, xếp loại hay kỷ luật cá nhân."
    card["limitations"] = ["Mỉa mai, châm biếm dễ bị hiểu sai.", "Dữ liệu từ một trường."]
    card["notes"] = {"x86": "Chưa đo trên x86.", "demo": "Dữ liệu demo tính lại lúc build image."}
    return card


def manifest_for(files: dict[str, str], *, stage: str, git_commit: str) -> Manifest:
    return Manifest(
        bundle_version=BUNDLE_VERSION,
        stage=stage,
        model_version=MODEL_VERSION,
        created_at=datetime.now(UTC),
        git_commit=git_commit,
        normalizer_version=NORMALIZER_VERSION,
        max_len=128,
        labels={task: list(names) for task, names in LABELS.items()},
        onnx={
            "quantization": "test",
            "base_vocab_size": BASE_ROWS,
            "unk_id": UNK,
            "embedding_rows": TOTAL_ROWS,
        },
        files=files,
        dataset={
            "name": DATASET_NAME,
            "revision": "d" * 40,
            "splits": {"train": 11426, "validation": 1583, "test": 3166},
            "license_note": "Học thuật, phi thương mại.",
        },
        base_model={"name": BASE_MODEL_NAME, "revision": "b" * 40, "license": "AGPL-3.0"},
    )


def frozen_document(root: Path) -> dict:
    """Bản freeze khớp bundle: sha256 của đúng các tệp build_artifacts đối chiếu khi phát hành."""
    return {
        "files": {
            frozen: sha256_file(root / name) for frozen, name in builder.FROZEN_IN_BUNDLE.items()
        }
    }


def write_bundle(root: Path, *, stage: str = "smoke", git_commit: str = "abc") -> Path:
    """Ghi một bundle đầy đủ vào `root` (thư mục con baseline/ được tạo) và trả về `root`."""
    (root / "baseline").mkdir(parents=True)
    tiny_onnx(root / "model.int8.onnx")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    write_json(root / "calibration.json", calibration().model_dump(mode="json"))
    write_json(root / "model_card.json", release_card_json() if stage == "release" else card_json())
    write_json(root / "insight_background.json", background().model_dump(mode="json"))
    golden = GoldenPredictions(normalizer_version=NORMALIZER_VERSION, items=[])
    write_json(root / "golden_predictions.json", golden.model_dump(mode="json"))
    for name in builder.BASELINE_FILES:
        save_model(toy_baseline(), root / name)
    files = {name: sha256_file(root / name) for name in builder.BUNDLE_FILES}
    manifest = manifest_for(files, stage=stage, git_commit=git_commit)
    write_json(root / "manifest.json", manifest.model_dump(mode="json"))
    return root


def refresh_manifest(root: Path) -> None:
    """Ghi lại sha256 của mọi tệp vào manifest sau khi test sửa một tệp của bundle."""
    manifest = json.loads((root / "manifest.json").read_text("utf-8"))
    manifest["files"] = {name: sha256_file(root / name) for name in manifest["files"]}
    write_json(root / "manifest.json", manifest)


# --- "Hub" giả là một thư mục, và script tải ----------------------------------------------------

SCRIPT = REPO_ROOT / "scripts" / "fetch-artifacts.sh"
HUB_REPO = "owner/edupulse-phobert-mtl"
REVISION = "0123456789abcdef0123456789abcdef01234567"


def publish_to_hub(directory: Path, hub: Path, *, repo: str = HUB_REPO) -> Path:
    """Đặt `directory` vào đúng đường dẫn `<repo>/resolve/<revision>/` mà curl sẽ gọi."""
    shutil.copytree(directory, hub / repo / "resolve" / REVISION)
    return hub


def fetch_with_script(
    hub: Path, target: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    """Chạy scripts/fetch-artifacts.sh với "Hub" là thư mục (file://), chỉ dùng biến môi trường.

    `env` ghi đè hoặc bổ sung biến; giá trị rỗng coi như chưa đặt, như trong .env.example.
    """
    variables = {
        "PATH": os.environ["PATH"],
        "HOME": str(target.parent),
        "EDUPULSE_ENV_FILE": str(target.parent / "absent.env"),
        "HF_ENDPOINT": hub.as_uri(),
        "EDUPULSE_HF_REPO": HUB_REPO,
        "EDUPULSE_HF_REVISION": REVISION,
        "BUNDLE_VERSION": BUNDLE_VERSION,
        "EDUPULSE_ARTIFACTS_DIR": str(target),
        **(env or {}),
    }
    return subprocess.run(
        ["bash", str(SCRIPT)],
        env=variables,
        capture_output=True,
        encoding="utf-8",
        timeout=120,
        check=False,
    )
