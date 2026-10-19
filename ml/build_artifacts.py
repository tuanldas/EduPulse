"""Gom bundle triển khai `$EDUPULSE_HOME/artifacts/<version>/` từ kết quả huấn luyện và export.

    python -m ml.build_artifacts --version v0-smoke      # make artifacts BUNDLE_VERSION=v0-smoke
    python -m ml.build_artifacts --version v1.1          # bản phát hành: thẻ mô hình trên test

Bundle chỉ gồm mô hình và cấu hình (model.int8.onnx, tokenizer.json, calibration.json,
model_card.json, insight_background.json, golden_predictions.json, baseline/*.skops, manifest.json).
Trước khi ghi: mọi JSON phải qua schema `edupulse.bundle_schema`; `normalizer_version` của mã, dữ
liệu đã xử lý, run huấn luyện, đếm nền và dự đoán vàng phải trùng nhau; và 200 câu ngẫu nhiên của
UIT-VSFC không được xuất hiện nguyên văn trong bất kỳ JSON nào (bundle không chứa câu gốc).

Bản phát hành (stage release) đòi thêm: cây git sạch; thẻ mô hình lấy số trên TEST từ kết quả của
đúng lần chấm cuối (`results/metrics/test_all.json`, cùng `freeze.json`); các tệp mô hình, cấu
hình và baseline trong bundle có sha256 khớp `freeze.json`, tức đúng thứ đã được chấm trên test.
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from edupulse.bundle_schema import (
    BaselineRow,
    BaseModelInfo,
    BudgetRow,
    Calibration,
    DatasetInfo,
    GoldenPredictions,
    InsightBackground,
    Manifest,
    ModelCard,
    ModelCardNotes,
    OnnxSpec,
    SplitMetrics,
    TaskMetrics,
)
from edupulse.config import BUDGETS, DEFAULT_BUDGET, REPO_ROOT
from edupulse.labels import LABELS, TASKS
from edupulse.text import NORMALIZER_VERSION
from ml import freeze as freeze_lib
from ml.common import (
    SPLITS,
    assert_normalizer,
    load_processed,
    normalizer_compatible,
    read_json,
    resolve_run,
    settings,
    sha256_file,
    task_metrics,
    write_json,
)
from ml.download_data import DATASET_REPO, load_raw_manifest
from ml.export_onnx import INPUT_NAMES, OUTPUT_NAMES, unk_alias_gap
from ml.phobert_model import BASE_MODEL, BASE_REVISION, MAX_LEN, PAD_ID, tokenizer_path
from ml.release_card import ReleaseError, build_release_card
from ml.train_baseline import load_model

LOG_PREFIX = "build_artifacts"
GUARD_SAMPLES = 200
GUARD_MIN_TOKENS = 4  # câu ngắn hơn có thể trùng một n-gram trong đếm nền
FULL_SCAN_MIN_TOKENS = 8  # quét hết mọi câu: ngắn hơn thì trùng ngẫu nhiên với câu tự viết
GUARD_TEXT_SUFFIXES = (".json", ".md")
BASELINE_FILES = tuple(f"baseline/{task}_linearsvc.skops" for task in TASKS)
BUNDLE_FILES = (
    "model.int8.onnx",
    "tokenizer.json",
    "calibration.json",
    "model_card.json",
    "insight_background.json",
    "golden_predictions.json",
    *BASELINE_FILES,
)
JSON_SCHEMAS = {
    "calibration.json": Calibration,
    "model_card.json": ModelCard,
    "insight_background.json": InsightBackground,
    "golden_predictions.json": GoldenPredictions,
}

INTENDED_USE = (
    "Giúp cán bộ khoa và phòng đảm bảo chất lượng đọc nhanh góp ý của sinh viên: phân loại cảm xúc "
    "và chủ đề từng câu, gắn cờ câu cần người duyệt, tổng hợp xu hướng theo môn."
)
OUT_OF_SCOPE = (
    "Đánh giá, xếp loại hay kỷ luật cá nhân giảng viên; truy vết hay xếp hạng sinh viên; ra "
    "quyết định tự động khi không có người duyệt."
)
# Giới hạn không gắn với một con số đo được; bản phát hành thêm các giới hạn kèm số đo trên test.
STATIC_LIMITATIONS = [
    "Mỉa mai, châm biếm dễ bị hiểu sai.",
    "Câu viết hoa chữ đầu mọi từ dễ bị luật ẩn tên coi nhiều từ là tên người, làm đổi nhãn.",
    "Dữ liệu từ một trường, giai đoạn 2013-2016; cách viết và môn học hiện nay có thể khác.",
    "Cơ sở vật chất gần như chỉ có câu tiêu cực trong dữ liệu nên mô hình hay gán CSVC tiêu cực.",
    "Tên người chỉ ẩn được khi viết hoa sau từ xưng hô (best effort); tên viết thường thì sót.",
    "Tỉ lệ trên dashboard tính từ nhãn dự đoán nên có thể lệch vài điểm so với thực tế.",
    "Mô hình int8 cho xác suất lệch nhẹ giữa phân tích một câu và phân tích hàng loạt.",
]
LIMITATIONS = [
    "Lớp Trung tính và chủ đề Khác yếu (recall khoảng 53%).",
    "Câu gõ không dấu: độ chính xác giảm mạnh (khoảng 13 điểm), hệ thống cảnh báo và gắn cờ.",
    *STATIC_LIMITATIONS,
]
NOTE_X86 = "Chưa đo trên x86: số đo tốc độ và RAM lấy từ máy Apple M1 (arm64)."
NOTE_DEMO = (
    "Dữ liệu demo tính lại lúc build image trên kiến trúc máy build, nên có thể lệch nhẹ so với "
    "số đo trên M1."
)


class BundleError(RuntimeError):
    pass


def git_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return git("rev-parse", "--short=10", "HEAD") + (
            "+dirty" if git("status", "--porcelain") else ""
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def assert_same_normalizer(
    history: dict[str, Any], background: InsightBackground, golden: GoldenPredictions
) -> None:
    """Mã, dữ liệu đã xử lý, đếm nền và dự đoán vàng phải cùng một bản chuẩn hoá. Run huấn luyện
    có thể do bản cũ hơn nếu bằng chứng bất biến chứng minh hai bản cho model_text giống hệt."""
    versions = {
        "mã nguồn": NORMALIZER_VERSION,
        "dữ liệu đã xử lý": assert_normalizer()["normalizer_version"],
        "đếm nền insight": background.normalizer_version,
        "dự đoán vàng": golden.normalizer_version,
    }
    trained_with = history.get("normalizer_version")
    if len(set(versions.values())) != 1 or not normalizer_compatible(trained_with):
        raise BundleError(
            f"normalizer_version không đồng nhất: {versions}, run huấn luyện {trained_with!r} "
            "(khác bản của mã thì cần results/metrics/normalizer_v*_invariance.json còn hiệu lực)"
        )


def guard_texts(directory: Path) -> dict[str, str]:
    """Tệp văn bản ở gốc thư mục mà guard quét (JSON, Markdown); `.skops` là nhị phân nên bỏ qua."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix in GUARD_TEXT_SUFFIXES
    }


def corpus_sentences(column: str, min_tokens: int) -> list[str]:
    """Câu của cả ba split ở dạng `column` (`sentence` gốc hoặc `text_rdr` đã tách từ)."""
    return [
        sentence
        for split in SPLITS
        # Chỉ tìm chuỗi, không tính chỉ số nào trên test.
        for sentence in load_processed(split, allow_test=True)[column]
        if len(sentence.split()) >= min_tokens
    ]


def first_file_with(texts: dict[str, str], sentences: Iterable[str]) -> str | None:
    for sentence in sentences:
        for name, text in texts.items():
            if sentence in text:
                return name
    return None


def assert_no_source_sentences(bundle_dir: Path, samples: int = GUARD_SAMPLES) -> int:
    """Không câu gốc UIT-VSFC nào (mẫu ngẫu nhiên) xuất hiện nguyên văn trong JSON của bundle."""
    pool = corpus_sentences("sentence", GUARD_MIN_TOKENS)
    leaked = first_file_with(guard_texts(bundle_dir), random.Random(0).sample(pool, samples))
    if leaked:
        raise BundleError(f"bundle chứa câu gốc của dữ liệu trong {leaked}")
    return samples


def assert_no_source_sentences_anywhere(directory: Path) -> int:
    """Như trên nhưng quét MỌI câu dài (dạng gốc và dạng đã tách từ): guard trước khi công khai.

    Chỉ tính câu từ `FULL_SCAN_MIN_TOKENS` tiếng, vì quét hết thì câu ngắn dễ trùng ngẫu nhiên với
    câu tự viết (vd. "em rất hài_lòng ." trong dự đoán vàng). Trả về số câu đã đối chiếu.
    """
    pool = dict.fromkeys(
        [
            *corpus_sentences("sentence", FULL_SCAN_MIN_TOKENS),
            *corpus_sentences("text_rdr", FULL_SCAN_MIN_TOKENS),
        ]
    )
    leaked = first_file_with(guard_texts(directory), pool)
    if leaked:
        raise BundleError(f"thư mục chứa câu gốc của dữ liệu trong {leaked}")
    return len(pool)


def model_card(
    *,
    bundle_version: str,
    model_version: str,
    stage: str,
    val_logits: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    calibration: Calibration,
) -> ModelCard:
    if stage != "smoke":
        raise BundleError("thẻ smoke chỉ có số validation; bản release dùng ml.release_card")
    temperatures = {
        "sentiment": calibration.sentiment.temperature,
        "topic": calibration.topic.temperature,
    }
    metrics = {
        task: TaskMetrics(
            **task_metrics(val_logits[task], targets[task], LABELS[task], temperatures[task])
        )
        for task in TASKS
    }
    deployed = BaselineRow(
        name="PhoBERT đa nhiệm · int8",
        deployed=True,
        sentiment_accuracy=metrics["sentiment"].accuracy,
        sentiment_macro_f1=metrics["sentiment"].macro_f1,
        topic_accuracy=metrics["topic"].accuracy,
        topic_macro_f1=metrics["topic"].macro_f1,
        sentiment_ece=metrics["sentiment"].ece,
    )
    return ModelCard(
        model_version=model_version,
        bundle_version=bundle_version,
        stage="smoke",
        metrics=SplitMetrics(
            split="val",
            n=len(targets["sentiment"]),
            sentiment=metrics["sentiment"],
            topic=metrics["topic"],
        ),
        baselines=[*baseline_rows(), deployed],
        review_budget=[
            BudgetRow(budget=b, default=(b == DEFAULT_BUDGET), efficiency=calibration.efficiency[b])
            for b in BUDGETS
        ],
        intended_use=INTENDED_USE,
        out_of_scope=OUT_OF_SCOPE,
        limitations=LIMITATIONS,
        notes=ModelCardNotes(x86=NOTE_X86, demo=NOTE_DEMO),
    )


def baseline_rows() -> list[BaselineRow]:
    path = REPO_ROOT / "results" / "metrics" / "val_baselines.json"
    if not path.is_file():
        raise BundleError(f"chưa có {path} — chạy `make baseline`")
    data = read_json(path)
    if not normalizer_compatible(data["normalizer_version"]):
        raise BundleError(
            "val_baselines.json sinh bởi bản chuẩn hoá khác — chạy lại `make baseline`"
        )
    rows = []
    for name in ("complementnb", "logreg", "linearsvc"):
        sentiment = data["tasks"]["sentiment"][name]
        topic = data["tasks"]["topic"][name]
        rows.append(
            BaselineRow(
                name=sentiment["display"],
                sentiment_accuracy=sentiment["raw"]["accuracy"],
                sentiment_macro_f1=sentiment["raw"]["macro_f1"],
                topic_accuracy=topic["raw"]["accuracy"],
                topic_macro_f1=topic["raw"]["macro_f1"],
                sentiment_ece=sentiment["calibrated"]["ece"],
            )
        )
    return rows


def verify_bundle(bundle_dir: Path) -> Manifest:
    """Đọc lại bundle như bên chạy thật: sha256, schema, nhãn, tên vào/ra ONNX, nạp baseline."""
    manifest = Manifest.model_validate_json((bundle_dir / "manifest.json").read_text("utf-8"))
    if manifest.normalizer_version != NORMALIZER_VERSION:
        raise BundleError("manifest.normalizer_version lệch mã nguồn")
    if set(manifest.files) != set(BUNDLE_FILES):
        raise BundleError(f"tập tệp lệch danh sách cho phép: {sorted(manifest.files)}")
    for name, digest in manifest.files.items():
        if sha256_file(bundle_dir / name) != digest:
            raise BundleError(f"{name}: sha256 không khớp manifest")
    for name, schema in JSON_SCHEMAS.items():
        schema.model_validate_json((bundle_dir / name).read_text("utf-8"))
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(bundle_dir / "model.int8.onnx"), providers=["CPUExecutionProvider"]
    )
    if tuple(i.name for i in session.get_inputs()) != tuple(manifest.onnx.input_names):
        raise BundleError("tên đầu vào ONNX lệch manifest")
    if tuple(o.name for o in session.get_outputs()) != tuple(manifest.onnx.output_names):
        raise BundleError("tên đầu ra ONNX lệch manifest")
    spec = manifest.onnx
    if spec.embedding_rows > spec.base_vocab_size:
        # Hàng embedding thừa phải hành xử y như <unk>: id cuối bảng cho logits giống id <unk>.
        gap = unk_alias_gap(session, spec.embedding_rows, spec.unk_id)
        if gap > 1e-5:
            raise BundleError(
                f"id {spec.embedding_rows - 1} không hành xử như <unk> (lệch {gap:.2e})"
            )
    for name in BASELINE_FILES:
        load_model(bundle_dir / name)
    return manifest


def require_passed_gate(gate: dict[str, Any], run: Path, allow_failed: bool) -> dict[str, Any]:
    """Cổng int8 của đúng run này phải qua trước khi đóng gói; trả về mục `embedding`."""
    if gate.get("run") != run.name:
        raise BundleError(f"onnx_gate.json là của run {gate.get('run')!r}, không phải {run.name!r}")
    if not normalizer_compatible(gate.get("normalizer_version")):
        raise BundleError("onnx_gate.json sinh bởi bản chuẩn hoá khác — chạy lại `make export`")
    if not gate["gate"]["passed"] and not allow_failed:
        raise BundleError(
            "cổng int8 chưa qua (xem onnx_gate.json); chỉ dùng --allow-failed-gate cho bundle thử "
            "đường ống, không để báo số"
        )
    return gate["embedding"]


# Tên trong freeze.json -> tên trong bundle (mô hình, cấu hình và baseline đã được chấm trên test).
FROZEN_IN_BUNDLE = {
    "onnx/model.int8.onnx": "model.int8.onnx",
    "calibration/calibration.json": "calibration.json",
    "tokenizer.json": "tokenizer.json",
    **{f"baseline/{task}_linearsvc.skops": f"baseline/{task}_linearsvc.skops" for task in TASKS},
}


def load_release_freeze(run: Path) -> dict[str, Any]:
    """Freeze của đúng run này; bản phát hành chỉ dựng từ cây git sạch (manifest ghi commit)."""
    if git_commit().endswith("+dirty"):
        raise BundleError("bản phát hành phải dựng từ cây git sạch: commit mọi thay đổi trước")
    try:
        document = freeze_lib.load_freeze()
    except freeze_lib.GuardError as error:
        raise BundleError(str(error)) from error
    if document["deploy"]["run"] != run.name:
        raise BundleError(
            f"freeze.json là của run {document['deploy']['run']!r}, không phải {run.name!r}"
        )
    return document


def assert_matches_freeze(bundle_dir: Path, document: dict[str, Any]) -> None:
    """Mô hình, cấu hình và baseline trong bundle phải đúng thứ đã được chấm trên test."""
    for frozen_name, bundle_name in FROZEN_IN_BUNDLE.items():
        if sha256_file(bundle_dir / bundle_name) != document["files"][frozen_name]:
            raise BundleError(
                f"{bundle_name} khác bản đã đóng băng và chấm test ({frozen_name} trong freeze)"
            )


def build(
    version: str, run: Path, stage: str, force: bool, allow_failed_gate: bool = False
) -> Path:
    target = settings().artifacts_dir(version)
    if target.exists() and not force:
        raise BundleError(
            f"{target} đã có (bundle theo phiên bản không ghi đè); dùng --force nếu cố ý"
        )
    onnx_dir = settings().outputs_dir / "onnx" / run.name
    calibration_path = settings().outputs_dir / "calibration" / run.name / "calibration.json"
    for needed in (
        onnx_dir / "model.int8.onnx",
        onnx_dir / "onnx_gate.json",
        calibration_path,
        onnx_dir / "val_logits_int8.npz",
    ):
        if not needed.is_file():
            raise BundleError(f"thiếu {needed} — chạy `make export calibrate`")
    gate = read_json(onnx_dir / "onnx_gate.json")
    embedding = require_passed_gate(gate, run, allow_failed_gate)

    history = read_json(run / "history.json")
    calibration = Calibration.model_validate_json(calibration_path.read_text("utf-8"))
    golden = GoldenPredictions.model_validate_json(
        (onnx_dir / "golden_predictions.json").read_text("utf-8")
    )
    background_path = settings().outputs_dir / "data" / "insight_background.json"
    background = InsightBackground.model_validate_json(background_path.read_text("utf-8"))
    assert_same_normalizer(history, background, golden)

    model_version = f"edupulse-{run.name}"
    frozen = load_release_freeze(run) if stage == "release" else None
    if frozen is not None:
        try:
            card = build_release_card(
                bundle_version=version,
                model_version=model_version,
                run=run.name,
                freeze_sha256=sha256_file(freeze_lib.FREEZE_PATH),
                results_dir=REPO_ROOT / "results",
                intended_use=INTENDED_USE,
                out_of_scope=OUT_OF_SCOPE,
                static_limitations=STATIC_LIMITATIONS,
                notes=ModelCardNotes(x86=NOTE_X86, demo=NOTE_DEMO),
            )
        except ReleaseError as error:
            raise BundleError(str(error)) from error
    else:
        logits = np.load(onnx_dir / "val_logits_int8.npz")
        card = model_card(
            bundle_version=version,
            model_version=model_version,
            stage=stage,
            val_logits={task: logits[task] for task in TASKS},
            targets={task: logits[f"y_{task}"] for task in TASKS},
            calibration=calibration,
        )

    staging = target.with_name(f"{target.name}.building")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "baseline").mkdir(parents=True)
    try:
        shutil.copyfile(onnx_dir / "model.int8.onnx", staging / "model.int8.onnx")
        shutil.copyfile(tokenizer_path(), staging / "tokenizer.json")
        # Chép nguyên tệp calibration đã kiểm schema (không ghi lại) để giữ đúng byte đã đóng băng.
        shutil.copyfile(calibration_path, staging / "calibration.json")
        write_json(staging / "model_card.json", card.model_dump(mode="json"))
        write_json(staging / "insight_background.json", background.model_dump(mode="json"))
        write_json(staging / "golden_predictions.json", golden.model_dump(mode="json"))
        for name in BASELINE_FILES:
            shutil.copyfile(settings().models_dir / name, staging / name)
        if frozen is not None:
            assert_matches_freeze(staging, frozen)

        raw = load_raw_manifest()
        manifest = Manifest(
            bundle_version=version,
            stage=stage,
            model_version=model_version,
            created_at=datetime.now(UTC),
            git_commit=git_commit(),
            normalizer_version=NORMALIZER_VERSION,
            max_len=MAX_LEN,
            labels={task: list(LABELS[task]) for task in TASKS},
            onnx=OnnxSpec(
                input_names=INPUT_NAMES,
                output_names=tuple(OUTPUT_NAMES[t] for t in TASKS),
                pad_id=PAD_ID,
                quantization=gate["quantization"],
                base_vocab_size=embedding["base_vocab_size"],
                unk_id=embedding["unk_id"],
                embedding_rows=embedding["embedding_rows"],
            ),
            files={name: sha256_file(staging / name) for name in BUNDLE_FILES},
            dataset=DatasetInfo(
                name=DATASET_REPO,
                revision=raw["revision"],
                splits={split: info["rows"] for split, info in raw["files"].items()},
                license_note="Học thuật, phi thương mại; bundle không chứa câu gốc của bộ dữ liệu.",
            ),
            base_model=BaseModelInfo(name=BASE_MODEL, revision=BASE_REVISION, license="AGPL-3.0"),
        )
        write_json(staging / "manifest.json", manifest.model_dump(mode="json"))
        verify_bundle(staging)
        checked = assert_no_source_sentences(staging)
        print(f"{LOG_PREFIX}: guard câu gốc: {checked} câu ngẫu nhiên, 0 xuất hiện trong bundle")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="vd. v0-smoke hoặc v1.1")
    parser.add_argument("--run", default="best", help="'best', tên run hoặc thư mục checkpoint")
    parser.add_argument(
        "--stage", choices=["smoke", "release"], help="mặc định: smoke nếu tên có 'smoke'"
    )
    parser.add_argument("--force", action="store_true", help="ghi đè bundle cùng phiên bản")
    parser.add_argument(
        "--allow-failed-gate",
        action="store_true",
        help="đóng gói dù cổng int8 chưa qua (chỉ để thử đường ống, không báo số)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage or ("smoke" if "smoke" in args.version else "release")
    try:
        target = build(
            args.version, resolve_run(args.run), stage, args.force, args.allow_failed_gate
        )
    except BundleError as error:
        print(f"{LOG_PREFIX}: {error}", file=sys.stderr)
        return 1
    manifest = read_json(target / "manifest.json")
    print(f"{LOG_PREFIX}: {target} · stage={manifest['stage']} · {len(manifest['files'])} tệp")
    return 0


if __name__ == "__main__":
    sys.exit(main())
