"""Cấu hình dùng chung: giới hạn cố định, đường dẫn và tham số đọc từ biến môi trường.

Thư mục nặng hoặc tái tạo được (data, models, outputs, artifacts, demo-data, khoá MPS) nằm dưới
EDUPULSE_HOME. Không đặt, hoặc để trống như trong `.env.example`, thì dùng thư mục repo, nên repo
vẫn chạy được trên máy khác, CI và Docker. Biến riêng (EDUPULSE_ARTIFACTS_DIR,
EDUPULSE_DEMO_DIR, ...) ghi đè được từng đường dẫn.
"""

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Giới hạn cố định, không đổi theo môi trường.
MAX_CHARS = 500  # ký tự tối đa của một câu
MAX_UPLOAD_MB = 10
MAX_ROWS = 2000  # dòng tối đa của một file
# Số từ tối đa của câu được giải thích bằng occlusion (tốn số từ + 1 lượt chạy mô hình). Đây là
# giá trị mặc định: EDUPULSE_XAI_MAX_WORDS ghi đè, và mã chạy thật đọc `Settings.xai_max_words`.
XAI_MAX_WORDS = 25
AUTO_XAI_TOP = 20  # số dòng ưu tiên cao nhất được giải thích sẵn khi chạy batch
BUDGETS = (5, 10, 20, 30)  # ngân sách duyệt, % số câu
DEFAULT_BUDGET = 10
AUDIT_RATE = 0.05  # tỉ lệ câu không gắn cờ được lấy mẫu để kiểm
HEARTBEAT_S = 60
MAX_JOBS = 8
MAX_QUEUE = 3

# Trùng giá trị mặc định của `BUNDLE_VERSION` trong Makefile.
DEFAULT_BUNDLE_VERSION = "v1.1"

_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ORIGIN_RE = re.compile(r"https?://[^/\s,]+")
_URL_RE = re.compile(r"https?://\S+")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _find_repo_root() -> Path:
    # Cài editable: src/edupulse/config.py nằm trong repo nên parents[2] là gốc repo.
    # Cài thường (Docker --no-editable) thì không có pyproject.toml ở đó: dùng thư mục chạy.
    candidate = Path(__file__).resolve().parents[2]
    return candidate if (candidate / "pyproject.toml").is_file() else Path.cwd().resolve()


REPO_ROOT = _find_repo_root()


def _text(env: Mapping[str, str], key: str) -> str:
    """Giá trị đã bỏ khoảng trắng thừa; chuỗi rỗng nghĩa là chưa đặt."""
    return env.get(key, "").strip()


def _path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = _text(env, key)
    return Path(raw).expanduser().resolve() if raw else default


def _int(env: Mapping[str, str], key: str, default: int, minimum: int = 1) -> int:
    raw = _text(env, key)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{key} phải là số nguyên, nhận {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{key} phải >= {minimum}, nhận {value}")
    return value


def _positive_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _text(env, key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key} phải là số, nhận {raw!r}") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} phải là số hữu hạn > 0, nhận {raw!r}")
    return value


def _flag(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = _text(env, key).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{key} chỉ nhận 0 hoặc 1 (hoặc true/false), nhận {raw!r}")


def _origins(env: Mapping[str, str], key: str) -> tuple[str, ...]:
    origins = []
    for item in _text(env, key).split(","):
        # Dấu "/" cuối khiến origin không bao giờ khớp header Origin của trình duyệt.
        origin = item.strip().rstrip("/")
        if not origin:
            continue
        if not _ORIGIN_RE.fullmatch(origin):
            raise ValueError(f"{key}: origin phải dạng http(s)://host[:cổng], nhận {item!r}")
        origins.append(origin)
    return tuple(origins)


def _url(env: Mapping[str, str], key: str) -> str:
    value = _text(env, key)
    if value and not _URL_RE.fullmatch(value):
        raise ValueError(f"{key} phải bắt đầu bằng http:// hoặc https://, nhận {value!r}")
    return value


def _version(name: str, value: str) -> str:
    # Phiên bản đi vào đường dẫn nên chặn dấu "/" và "..".
    if not _VERSION_RE.fullmatch(value):
        raise ValueError(f"{name} chỉ gồm chữ, số, '.', '_', '-'; nhận {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    home: Path  # EDUPULSE_HOME
    bundle_version: str  # BUNDLE_VERSION
    bundle_dir: Path  # bundle mà runtime nạp (EDUPULSE_ARTIFACTS_DIR)
    demo_dir: Path  # EDUPULSE_DEMO_DIR
    vncorenlp_dir: Path  # EDUPULSE_VNCORENLP_DIR
    web_dir: Path  # EDUPULSE_WEB_DIR, bản Next.js export tĩnh
    ort_threads: int  # EDUPULSE_ORT_THREADS
    ner: bool  # EDUPULSE_NER
    cors_origins: tuple[str, ...]  # EDUPULSE_CORS_ORIGINS, phân tách bằng dấu phẩy
    source_url: str  # EDUPULSE_SOURCE_URL
    job_ttl_min: float  # EDUPULSE_JOB_TTL_MIN, phút
    xai_max_words: int = XAI_MAX_WORDS  # EDUPULSE_XAI_MAX_WORDS
    model_url: str = ""  # EDUPULSE_MODEL_URL, liên kết tới model trên Hugging Face

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.home / "outputs"

    @property
    def mps_lock_path(self) -> Path:
        # Khoá toàn máy cho GPU MPS: cùng EDUPULSE_HOME thì mọi bản làm việc dùng chung một khoá.
        return self.home / ".mps.lock"

    @property
    def job_ttl_s(self) -> float:
        return self.job_ttl_min * 60

    def artifacts_dir(self, version: str | None = None) -> Path:
        """Nơi build và tải một phiên bản bundle; mặc định là phiên bản hiện hành."""
        return self.home / "artifacts" / _version("version", version or self.bundle_version)


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Đọc cấu hình từ `env` (mặc định `os.environ`); giá trị sai thì ném ValueError nêu rõ biến."""
    env = os.environ if env is None else env
    home = _path(env, "EDUPULSE_HOME", REPO_ROOT)
    version = _version("BUNDLE_VERSION", _text(env, "BUNDLE_VERSION") or DEFAULT_BUNDLE_VERSION)
    return Settings(
        home=home,
        bundle_version=version,
        bundle_dir=_path(env, "EDUPULSE_ARTIFACTS_DIR", home / "artifacts" / version),
        demo_dir=_path(env, "EDUPULSE_DEMO_DIR", home / "demo-data"),
        vncorenlp_dir=_path(env, "EDUPULSE_VNCORENLP_DIR", REPO_ROOT / "vendor" / "vncorenlp"),
        web_dir=_path(env, "EDUPULSE_WEB_DIR", REPO_ROOT / "apps" / "web" / "out"),
        ort_threads=_int(env, "EDUPULSE_ORT_THREADS", 2),
        ner=_flag(env, "EDUPULSE_NER"),
        cors_origins=_origins(env, "EDUPULSE_CORS_ORIGINS"),
        source_url=_url(env, "EDUPULSE_SOURCE_URL"),
        job_ttl_min=_positive_float(env, "EDUPULSE_JOB_TTL_MIN", 30.0),
        xai_max_words=_int(env, "EDUPULSE_XAI_MAX_WORDS", XAI_MAX_WORDS),
        model_url=_url(env, "EDUPULSE_MODEL_URL"),
    )
