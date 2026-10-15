"""Tách từ tiếng Việt bằng VnCoreNLP RDRSegmenter (py_vncorenlp chạy trên JVM qua pyjnius).

Ba bẫy của py_vncorenlp được xử lý ở đây:
- JVM chỉ khởi động được một lần mỗi process, nên `Segmenter` là singleton và mọi lời gọi đi qua
  một khoá.
- `VnCoreNLP()` gọi `os.chdir(save_dir)` vĩnh viễn; cwd được lưu và khôi phục.
- `download_model()` gọi `wget` và tải cả NER/parse; không dùng. Ba tệp cần cho tách từ được
  vendor sẵn, ghim commit (xem `vendor/vncorenlp/SHA256SUMS`).
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import threading
from pathlib import Path

from edupulse.config import load_settings

DEFAULT_MAX_HEAP = "-Xmx512m"
HEAP_ENV = "EDUPULSE_JVM_HEAP"

_HEAP_RE = re.compile(r"-Xmx\d+[kKmMgG]")
_REQUIRED_FILES = (
    "VnCoreNLP-1.2.jar",
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)
_LINUX_JAVA_HOME = Path("/usr/lib/jvm/default-java")


class SegmenterError(RuntimeError):
    """Không khởi động hoặc không dùng được bộ tách từ; thông báo nêu cách khắc phục."""


def _find_java_home() -> str | None:
    """JAVA_HOME từ môi trường, nếu không có thì thử cách chuẩn của từng hệ điều hành."""
    current = os.environ.get("JAVA_HOME", "").strip()
    if current and Path(current).is_dir():
        return current
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["/usr/libexec/java_home"], capture_output=True, text=True, timeout=10, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        home = result.stdout.strip()
        return home if result.returncode == 0 and home else None
    return str(_LINUX_JAVA_HOME) if _LINUX_JAVA_HOME.is_dir() else None


def _check_files(root: Path) -> None:
    missing = [name for name in _REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise SegmenterError(
            f"thiếu {', '.join(missing)} trong {root} (đặt EDUPULSE_VNCORENLP_DIR nếu để nơi khác)"
        )


class Segmenter:
    """RDRSegmenter dùng chung cho train và serve: nhận chuỗi, trả các từ cách nhau bằng dấu cách.

    Từ ghép nối bằng `_` ("giảng_viên"). Kết quả của các câu trong cùng đầu vào được nối bằng
    dấu cách, vì py_vncorenlp trả danh sách câu (":)" làm tách đôi câu nếu chưa được đổi token).
    """

    def __init__(self, vncorenlp_dir: Path, max_heap: str = DEFAULT_MAX_HEAP) -> None:
        if not _HEAP_RE.fullmatch(max_heap):
            raise SegmenterError(f"{HEAP_ENV} phải dạng -Xmx512m, nhận {max_heap!r}")
        root = Path(vncorenlp_dir).resolve()
        _check_files(root)
        java_home = _find_java_home()
        if java_home:
            os.environ["JAVA_HOME"] = java_home
        self._lock = threading.Lock()
        previous_cwd = os.getcwd()
        try:
            import py_vncorenlp

            self._model = py_vncorenlp.VnCoreNLP(
                annotators=["wseg"], save_dir=str(root), max_heap_size=max_heap
            )
        except SegmenterError:
            raise
        except Exception as exc:  # JVM/pyjnius báo nhiều kiểu lỗi khác nhau
            raise SegmenterError(
                "không khởi động được JVM cho VnCoreNLP "
                f"({type(exc).__name__}: {exc}); cài JRE (apt: default-jre-headless) "
                "hoặc đặt JAVA_HOME"
            ) from exc
        finally:
            os.chdir(previous_cwd)

    def segment(self, text: str) -> str:
        if not text.strip():
            return ""
        with self._lock:
            sentences = self._model.word_segment(text)
        return " ".join(sentences)


_instance: Segmenter | None = None
_instance_lock = threading.Lock()


def get_segmenter() -> Segmenter:
    """Bộ tách từ duy nhất của process; tạo ở lần gọi đầu (JVM nạp mất khoảng 1 giây)."""
    global _instance
    with _instance_lock:
        if _instance is None:
            heap = os.environ.get(HEAP_ENV, "").strip() or DEFAULT_MAX_HEAP
            _instance = Segmenter(load_settings().vncorenlp_dir, heap)
        return _instance
