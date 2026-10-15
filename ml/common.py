"""Tiện ích dùng chung cho các script `ml/`: đường dẫn, JSON, dữ liệu, khoá MPS, seed, chỉ số.

Mọi đường dẫn đi qua `edupulse.config` (theo EDUPULSE_HOME) nên script không ghi cứng `data/`,
`models/`, `outputs/`.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from edupulse.config import Settings, load_settings
from edupulse.scoring import softmax_t

if TYPE_CHECKING:
    import pandas as pd

SPLITS = ("train", "validation", "test")
ECE_BINS = 15


def settings() -> Settings:
    return load_settings()


def raw_dir() -> Path:
    return settings().data_dir / "raw"


def processed_dir() -> Path:
    return settings().data_dir / "processed"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any, *, sort_keys: bool = False) -> None:
    """Ghi JSON UTF-8 qua tệp tạm rồi đổi tên, để tiến trình khác không bao giờ đọc nửa tệp."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=sort_keys)
            handle.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def processed_manifest() -> dict[str, Any]:
    """MANIFEST của dữ liệu đã tiền xử lý; báo lỗi có hướng dẫn nếu chưa chạy `make data`."""
    path = processed_dir() / "MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(f"chưa có {path} — chạy `make data`")
    return read_json(path)


def assert_normalizer() -> dict[str, Any]:
    """Dữ liệu đã tiền xử lý phải do đúng bản chuẩn hoá đang có trong mã; trả về MANIFEST."""
    from edupulse.text import NORMALIZER_VERSION

    manifest = processed_manifest()
    found = manifest.get("normalizer_version")
    if found != NORMALIZER_VERSION:
        raise RuntimeError(
            f"dữ liệu đã tiền xử lý bằng bản chuẩn hoá {found!r} nhưng mã đang là "
            f"{NORMALIZER_VERSION!r} — chạy lại `make data` rồi huấn luyện lại"
        )
    return manifest


def load_processed(split: str, *, allow_test: bool = False) -> pd.DataFrame:
    """Đọc một split đã tiền xử lý, sau khi kiểm bản chuẩn hoá và sha256 so với MANIFEST.

    Tập test chỉ mở khi gọi rõ `allow_test=True` (thống kê mô tả ở EDA, đánh giá cuối ở một lệnh
    duy nhất), để huấn luyện và chọn mô hình không thể chạm vào nó do sơ suất.
    """
    import pandas as pd

    if split not in SPLITS:
        raise ValueError(f"split phải thuộc {SPLITS}, nhận {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("tập test chỉ dùng cho EDA mô tả và đánh giá cuối (allow_test=True)")
    manifest = assert_normalizer()
    info = manifest["files"][split]
    path = processed_dir() / info["file"]
    if sha256_file(path) != info["sha256"]:
        raise RuntimeError(f"{path.name} không khớp MANIFEST — chạy lại `make data`")
    return pd.read_parquet(path)


# --- Môi trường chạy --------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # group train không có (vd. môi trường chỉ chạy baseline)
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name: str = "auto") -> str:
    """'auto' ưu tiên mps, rồi cuda, rồi cpu; tên cụ thể phải khả dụng."""
    import torch

    available = {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if name == "auto":
        return next(device for device in ("mps", "cuda", "cpu") if available[device])
    if not available.get(name, False):
        raise RuntimeError(f"thiết bị {name!r} không khả dụng trên máy này")
    return name


def docker_running() -> bool:
    """Docker Desktop đang chạy khi `docker info` thành công. Không có docker thì coi như tắt."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


class MpsBusyError(RuntimeError):
    """Khoá MPS đang do tiến trình khác giữ."""


def _holder_note(fd: int) -> str:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        text = os.read(fd, 4096).decode("utf-8", "replace").strip()
    except OSError:
        return "không rõ"
    return text or "không rõ"


@contextmanager
def mps_lock(path: Path | None = None, note: str = "") -> Iterator[Path]:
    """Khoá toàn máy cho GPU MPS: chỉ một job huấn luyện mỗi lúc, và không chờ.

    `fcntl.flock(LOCK_EX | LOCK_NB)` trên `$EDUPULSE_HOME/.mps.lock`, giữ tới khi thoát. Trên macOS
    flock và lockf tác động lẫn nhau, nên bên kiểm tra dùng kiểu nào cũng thấy khoá đang giữ.
    Nội dung tệp (pid, lệnh, giờ bắt đầu) chỉ để chẩn đoán.
    """
    lock_path = path or settings().mps_lock_path
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise MpsBusyError(
                f"khoá MPS {lock_path} đang do job khác giữ ({_holder_note(fd)}); "
                "mỗi lúc chỉ chạy một job huấn luyện, chờ job đó xong"
            ) from None
        os.ftruncate(fd, 0)
        info = {"pid": os.getpid(), "note": note, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        os.write(fd, json.dumps(info, ensure_ascii=False).encode("utf-8"))
        yield lock_path
    finally:
        os.close(fd)  # đóng fd là nhả khoá


def mps_lock_is_held(path: Path | None = None) -> bool:
    """Thử lấy khoá không chờ: lấy được nghĩa là rảnh (nhả ngay), không thì đang có job giữ."""
    lock_path = path or settings().mps_lock_path
    if not lock_path.exists():
        return False
    fd = os.open(lock_path, os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


# --- Chỉ số -----------------------------------------------------------------------------------


def _as_array(values: Any, dtype: Any = None) -> np.ndarray:
    return np.asarray(values, dtype=dtype)


def confusion_normalized(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: int
) -> list[list[float]]:
    """Ma trận nhầm lẫn chuẩn hoá theo hàng (hàng = nhãn thật); hàng rỗng giữ 0."""
    counts = np.zeros((n_classes, n_classes), dtype=np.float64)
    for truth, pred in zip(y_true, y_pred, strict=True):
        counts[truth, pred] += 1
    sums = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, sums, out=np.zeros_like(counts), where=sums > 0).tolist()


def reliability_bins(probs: np.ndarray, y: Sequence[int], n_bins: int = ECE_BINS) -> list[dict]:
    """Độ tin cậy (xác suất lớn nhất) so với tỉ lệ đúng trong các khoảng đều nhau trên [0, 1]."""
    probs = _as_array(probs, np.float64)
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == _as_array(y)
    index = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = index == b
        n = int(mask.sum())
        bins.append(
            {
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "n": n,
                "confidence": float(conf[mask].mean()) if n else None,
                "accuracy": float(correct[mask].mean()) if n else None,
            }
        )
    return bins


def expected_calibration_error(
    probs: np.ndarray, y: Sequence[int], n_bins: int = ECE_BINS
) -> float:
    total = len(y)
    return float(
        sum(
            b["n"] / total * abs(b["accuracy"] - b["confidence"])
            for b in reliability_bins(probs, y, n_bins)
            if b["n"]
        )
    )


def negative_log_likelihood(probs: np.ndarray, y: Sequence[int]) -> float:
    probs = _as_array(probs, np.float64)
    picked = probs[np.arange(len(probs)), _as_array(y)]
    return float(-np.log(np.clip(picked, 1e-12, 1.0)).mean())


def prediction_metrics(
    y: Sequence[int], pred: Sequence[int], labels: Sequence[str]
) -> dict[str, Any]:
    """Chỉ số từ nhãn dự đoán cứng: acc, macro/weighted-F1, từng lớp, CM chuẩn hoá hàng."""
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    truth = _as_array(y, np.int64)
    pred = _as_array(pred, np.int64)
    ids = list(range(len(labels)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=ids, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, labels=ids, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(truth, pred, labels=ids, average="weighted", zero_division=0)
        ),
        "per_class": [
            {
                "label": labels[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in ids
        ],
        "confusion": confusion_normalized(truth, pred, len(labels)),
    }


def task_metrics(
    logits: np.ndarray, y: Sequence[int], labels: Sequence[str], temperature: float = 1.0
) -> dict[str, Any]:
    """Bộ chỉ số của một head từ logits: `prediction_metrics` cộng ECE, NLL, biểu đồ độ tin cậy.

    Khớp trường của `bundle_schema.TaskMetrics`.
    """
    truth = _as_array(y, np.int64)
    probs = softmax_t(_as_array(logits, np.float64), temperature)
    return {
        **prediction_metrics(truth, probs.argmax(axis=1), labels),
        "ece": expected_calibration_error(probs, truth),
        "nll": negative_log_likelihood(probs, truth),
        "reliability": reliability_bins(probs, truth),
    }


def macro_f1(logits: np.ndarray, y: Sequence[int], n_classes: int) -> float:
    """Macro-F1 nhanh (dùng để chọn epoch); cùng định nghĩa với `task_metrics`."""
    from sklearn.metrics import f1_score

    pred = _as_array(logits).argmax(axis=1)
    return float(
        f1_score(
            _as_array(y), pred, labels=list(range(n_classes)), average="macro", zero_division=0
        )
    )


# --- Chọn run --------------------------------------------------------------------------------


def phobert_runs_dir() -> Path:
    return settings().models_dir / "phobert"


def finished_runs(prefix: str = "mtl-s") -> list[Path]:
    """Các run PhoBERT đã hoàn tất (history.finished), tên bắt đầu bằng `prefix`."""
    runs = []
    for history_path in sorted(phobert_runs_dir().glob(f"{prefix}*/history.json")):
        if read_json(history_path).get("finished"):
            runs.append(history_path.parent)
    return runs


def best_run(prefix: str = "mtl-s") -> Path:
    """Bản triển khai: run đa nhiệm đã xong có điểm val cao nhất; hoà thì lấy seed nhỏ hơn."""
    runs = finished_runs(prefix)
    if not runs:
        raise FileNotFoundError(
            f"chưa có run {prefix}* nào hoàn tất trong {phobert_runs_dir()} — chạy `make train`"
        )

    def rank(run: Path) -> tuple[float, int]:
        return read_json(run / "history.json")["best_score"], -int(run.name.rsplit("-s", 1)[1])

    return max(runs, key=rank)


def resolve_run(spec: str) -> Path:
    """'best' | tên run dưới models/phobert | đường dẫn thư mục run."""
    if spec == "best":
        return best_run()
    candidate = Path(spec)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = phobert_runs_dir() / spec
    if not (candidate / "config.json").is_file():
        raise FileNotFoundError(f"không thấy checkpoint ở {candidate}")
    return candidate.resolve()
