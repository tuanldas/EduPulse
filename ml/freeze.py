"""Đóng băng cấu hình trước khi chấm test, và các chốt chặn của lần chấm cuối.

`results/freeze.json` ghi lại mọi thứ đã chọn bằng validation (run được triển khai, nhiệt độ T,
ngưỡng τ_B, siêu tham số baseline, sha256 của mọi mô hình sẽ được chấm). Nó phải được COMMIT trước
khi `evaluate --split test --final` chạy; lần chấm cuối từ chối chạy khi:

- chưa có `freeze.json`, hoặc nó chưa được commit nguyên vẹn (bằng chứng thứ tự: commit freeze
  có trước dự đoán test);
- cây làm việc còn thay đổi chưa commit, hoặc commit lúc đóng băng không phải tổ tiên của HEAD;
- một mô hình hay tệp cấu hình đã đổi so với sha256 trong freeze, hay seed triển khai theo luật
  không còn khớp;
- test đã được chấm (`.done` có mặt) mà chưa ghi lý do chạy lại bằng `--record-rerun`.
"""

from __future__ import annotations

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from edupulse.config import BUDGETS, DEFAULT_BUDGET, REPO_ROOT
from edupulse.labels import TASKS
from edupulse.text import NORMALIZER_VERSION
from ml.common import (
    best_run,
    phobert_runs_dir,
    processed_manifest,
    read_json,
    settings,
    sha256_file,
    write_json,
)
from ml.preds import BASELINES

SCHEMA_VERSION = 1
FREEZE_PATH = REPO_ROOT / "results" / "freeze.json"
POLICY = (
    "Tập test chỉ được chấm một lần, sau khi tệp này đã commit. Mọi lựa chọn (run triển khai, "
    "nhiệt độ, ngưỡng cờ, siêu tham số baseline) lấy từ validation. Phân tích sau đó (kiểm định, "
    "lỗi, ngân sách duyệt, thẻ mô hình) chỉ đọc dự đoán lưu ở outputs/test_preds/."
)
SCOPE = (
    "Không huấn luyện thêm: bản triển khai là run PhoBERT đa nhiệm duy nhất (seed 42) đã train ở "
    "lần chạy đầu; so sánh với đơn nhiệm chỉ có thí nghiệm sơ bộ của nghiên cứu (validation)."
)
DEPLOY_RULE = "Run PhoBERT đa nhiệm duy nhất đã huấn luyện (best_run: điểm validation cao nhất)."


class GuardError(RuntimeError):
    """Một chốt chặn của lần chấm cuối không thoả; thông điệp nói cần làm gì."""


# --- git ------------------------------------------------------------------------------------------


def git(
    *args: str, repo: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Chạy git ở `repo` (mặc định REPO_ROOT, tra lúc gọi để test đổi được)."""
    return subprocess.run(
        ["git", *args],
        cwd=repo or REPO_ROOT,
        capture_output=True,
        text=True,
        check=check,
        timeout=60,
    )


def git_head(repo: Path | None = None) -> str:
    return git("rev-parse", "HEAD", repo=repo).stdout.strip()


def git_dirty(repo: Path | None = None) -> list[str]:
    """Các dòng `git status --porcelain`: thay đổi chưa commit và tệp chưa theo dõi."""
    return [line for line in git("status", "--porcelain", repo=repo).stdout.splitlines() if line]


def git_is_tracked(path: Path, repo: Path | None = None) -> bool:
    root = (repo or REPO_ROOT).resolve()
    relative = str(path.resolve().relative_to(root))
    return git("ls-files", "--error-unmatch", relative, repo=root, check=False).returncode == 0


def git_is_ancestor(ancestor: str, descendant: str = "HEAD", repo: Path | None = None) -> bool:
    result = git("merge-base", "--is-ancestor", ancestor, descendant, repo=repo, check=False)
    return result.returncode == 0


# --- những gì được băm ----------------------------------------------------------------------------


def calibration_path(deploy_run: str) -> Path:
    return settings().outputs_dir / "calibration" / deploy_run / "calibration.json"


def frozen_files(deploy_run: str) -> dict[str, Path]:
    """Tên logic -> tệp được băm: checkpoint, ONNX int8, calibration, tokenizer, 6 baseline."""
    from ml.phobert_model import tokenizer_path
    from ml.predictors import baseline_path, deploy_onnx_path

    files: dict[str, Path] = {}
    for name in ("model.safetensors", "config.json"):
        files[f"phobert/{deploy_run}/{name}"] = phobert_runs_dir() / deploy_run / name
    files["onnx/model.int8.onnx"] = deploy_onnx_path(deploy_run)
    files["calibration/calibration.json"] = calibration_path(deploy_run)
    files["tokenizer.json"] = tokenizer_path()
    for task in TASKS:
        for name in BASELINES:
            files[f"baseline/{task}_{name}.skops"] = baseline_path(task, name)
    return files


def hash_files(files: dict[str, Path]) -> dict[str, str]:
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise GuardError(f"thiếu tệp cần băm: {missing}")
    with ThreadPoolExecutor(max_workers=4) as pool:  # hashlib nhả GIL nên chạy song song được
        digests = list(pool.map(sha256_file, files.values()))
    return dict(zip(files, digests, strict=True))


# --- tạo và kiểm freeze ---------------------------------------------------------------------------


def deploy_summary(deploy_run: str) -> dict[str, Any]:
    """Run được triển khai và điểm validation của nó (đọc từ history.json của run)."""
    history = read_json(phobert_runs_dir() / deploy_run / "history.json")
    if not history.get("finished"):
        raise GuardError(f"run {deploy_run} chưa hoàn tất")
    return {
        "run": deploy_run,
        "rule": DEPLOY_RULE,
        "seed": history["args"]["seed"],
        "epochs": history["args"]["epochs"],
        "best_epoch": history["best_epoch"],
        "val_mean_macro_f1": history["best_score"],
    }


def build_freeze(deploy_run: str | None = None, repo: Path | None = None) -> dict[str, Any]:
    """Nội dung freeze cho run được triển khai (mặc định theo `best_run`)."""
    repo = repo or REPO_ROOT
    deploy_run = deploy_run or best_run().name
    gate_path = settings().outputs_dir / "onnx" / deploy_run / "onnx_gate.json"
    for needed in (gate_path, calibration_path(deploy_run)):
        if not needed.is_file():
            raise GuardError(f"thiếu {needed}: chạy `make export calibrate` cho {deploy_run}")
    gate = read_json(gate_path)
    if gate.get("run") != deploy_run or not gate["gate"]["passed"]:
        raise GuardError(
            f"cổng int8 của {deploy_run} chưa qua hoặc là của run khác: dừng, báo lead "
            "(không tìm cấu hình lượng tử hoá mới)"
        )
    if gate.get("normalizer_version") != NORMALIZER_VERSION:
        raise GuardError("onnx_gate.json sinh bởi bản chuẩn hoá khác: chạy lại `make export`")
    calibration = read_json(calibration_path(deploy_run))
    manifest = processed_manifest()
    baselines_path = repo / "results" / "metrics" / "val_baselines.json"
    baselines = read_json(baselines_path)
    if baselines["normalizer_version"] != NORMALIZER_VERSION:
        raise GuardError("val_baselines.json sinh bởi bản chuẩn hoá khác: chạy lại `make baseline`")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git": {"commit": git_head(repo), "dirty": git_dirty(repo)},
        "policy": POLICY,
        "scope": SCOPE,
        "deploy": deploy_summary(deploy_run),
        "quantization": {
            "config": gate["quant_config"],
            "heads_fp32": gate.get("heads_fp32"),
            "gate_passed": gate["gate"]["passed"],
            "delta_macro_f1_points": gate["delta_macro_f1_points"],
            "label_flips": gate["label_flips"],
        },
        "calibration": {
            "temperature": {t: calibration[t]["temperature"] for t in TASKS},
            "tau": calibration["tau"],
            "default_budget": DEFAULT_BUDGET,
            "budgets": list(BUDGETS),
        },
        "baselines": {
            task: {name: baselines["tasks"][task][name]["hp"] for name in BASELINES}
            for task in TASKS
        },
        "data": {
            "dataset_revision": manifest["dataset_revision"],
            "normalizer_version": manifest["normalizer_version"],
            "test_sha256": manifest["files"]["test"]["sha256"],
            "rows": {split: info["rows"] for split, info in manifest["files"].items()},
        },
        "files": hash_files(frozen_files(deploy_run)),
        "reruns": [],
    }


def write_freeze(freeze: dict[str, Any], path: Path | None = None) -> None:
    write_json(path or FREEZE_PATH, freeze)


def load_freeze(path: Path | None = None) -> dict[str, Any]:
    path = path or FREEZE_PATH
    if not path.is_file():
        raise GuardError(
            f"chưa có {path}: chạy `python -m ml.evaluate --freeze` rồi commit trước khi chấm test"
        )
    return read_json(path)


def file_problems(freeze: dict[str, Any]) -> list[str]:
    """Những khác biệt giữa freeze và trạng thái hiện tại của mô hình, cấu hình và dữ liệu."""
    problems: list[str] = []
    deploy_run = freeze["deploy"]["run"]
    now = best_run().name
    if now != deploy_run:
        problems.append(f"run triển khai theo luật hiện là {now}, freeze ghi {deploy_run}")
    manifest = processed_manifest()
    if manifest["files"]["test"]["sha256"] != freeze["data"]["test_sha256"]:
        problems.append("dữ liệu test đã đổi so với freeze")
    if manifest["normalizer_version"] != freeze["data"]["normalizer_version"]:
        problems.append("normalizer_version của dữ liệu đã đổi so với freeze")
    try:
        current = hash_files(frozen_files(deploy_run))
    except GuardError as error:
        return [*problems, str(error)]
    for name, digest in freeze["files"].items():
        if current.get(name) != digest:
            problems.append(f"{name}: sha256 khác freeze")
    return problems


def assert_final_allowed(
    *,
    preds_dir: Path,
    repo: Path | None = None,
    freeze_path: Path | None = None,
    check_files: bool = True,
) -> dict[str, Any]:
    """Mọi chốt chặn của `--final`; trả về freeze nếu được phép, nếu không ném GuardError."""
    freeze_path = freeze_path or FREEZE_PATH
    freeze = load_freeze(freeze_path)
    if preds_dir.exists():
        raise GuardError(
            f"{preds_dir} đã tồn tại: test đã được chấm. Chấm lại chỉ khi phát hiện lỗi mã và đã "
            'ghi lý do bằng `python -m ml.evaluate --record-rerun "lý do"` (nêu trong báo cáo)'
        )
    if not git_is_tracked(freeze_path, repo):
        raise GuardError(f"{freeze_path.name} chưa được commit: commit nó trước khi chấm test")
    dirty = git_dirty(repo)
    if dirty:
        raise GuardError(
            "cây làm việc còn thay đổi chưa commit (commit hết trước khi chấm test để có bằng "
            f"chứng thứ tự): {dirty[:5]}"
        )
    frozen_at = freeze["git"]["commit"]
    if not git_is_ancestor(frozen_at, "HEAD", repo):
        raise GuardError(f"commit lúc đóng băng {frozen_at[:10]} không phải tổ tiên của HEAD")
    if check_files:
        problems = file_problems(freeze)
        if problems:
            raise GuardError(
                "mô hình hoặc cấu hình đã đổi sau khi đóng băng: " + "; ".join(problems)
            )
    return freeze


def record_rerun(reason: str, preds_dir: Path, path: Path | None = None) -> Path:
    """Ghi lý do chạy lại vào freeze.json rồi cất bộ dự đoán cũ sang `<preds_dir>.superseded-N`."""
    path = path or FREEZE_PATH
    reason = reason.strip()
    if len(reason) < 10:
        raise GuardError("lý do chạy lại phải mô tả lỗi mã đã sửa (>= 10 ký tự)")
    freeze = load_freeze(path)
    if not preds_dir.exists():
        raise GuardError(f"{preds_dir} chưa có: chưa chấm test nên không cần ghi lần chạy lại")
    n = len(freeze["reruns"]) + 1
    archive = preds_dir.with_name(f"{preds_dir.name}.superseded-{n}")
    preds_dir.rename(archive)
    freeze["reruns"].append(
        {"n": n, "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "reason": reason}
    )
    write_freeze(freeze, path)
    return archive
