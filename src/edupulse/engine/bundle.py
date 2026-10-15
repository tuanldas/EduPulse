"""Nạp và kiểm tra bundle mô hình (`artifacts/<phiên bản>/`) trước khi phục vụ.

Cấu trúc và schema do `edupulse.bundle_schema` định nghĩa. Ở đây thêm các kiểm tra chỉ làm được
lúc chạy: tệp có đủ và đúng sha256, phiên bản chuẩn hoá văn bản khớp mã nguồn (lệch thì mô hình
nhận đầu vào khác lúc huấn luyện và dự đoán sai âm thầm), nhãn khớp thứ tự nhãn trong mã.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from edupulse.bundle_schema import Calibration, Manifest
from edupulse.text.pipeline import NORMALIZER_VERSION

MANIFEST = "manifest.json"
CALIBRATION = "calibration.json"
MODEL = "model.int8.onnx"
TOKENIZER = "tokenizer.json"
REQUIRED_FILES = (MODEL, TOKENIZER, CALIBRATION)

_HOW_TO_GET = (
    "Tạo bằng `make artifacts` hoặc tải bằng `make fetch-artifacts`, rồi đặt BUNDLE_VERSION hoặc "
    "EDUPULSE_ARTIFACTS_DIR trỏ tới bundle. Với Docker, build lại image kèm bundle "
    "(`make docker-build`)."
)


class BundleError(RuntimeError):
    """Bundle thiếu, hỏng hoặc không khớp mã nguồn; thông báo nêu cách khắc phục."""


@dataclass(frozen=True)
class Bundle:
    root: Path
    manifest: Manifest
    calibration: Calibration

    @property
    def model_path(self) -> Path:
        return self.root / MODEL

    @property
    def tokenizer_path(self) -> Path:
        return self.root / TOKENIZER


def _describe(error: ValidationError) -> str:
    """Vị trí và loại lỗi, không chép giá trị (tệp có thể chứa dữ liệu tuỳ ý)."""
    items = [f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}" for e in error.errors()[:5]]
    return "; ".join(items)


def _read_json_model[M: BaseModel](path: Path, model: type[M]) -> M:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise BundleError(f"Không đọc được {path}: {exc.strerror or type(exc).__name__}") from exc
    try:
        return model.model_validate_json(raw)
    except ValidationError as exc:
        raise BundleError(f"{path} không hợp lệ ({_describe(exc)}). {_HOW_TO_GET}") from exc


def _sha256(path: Path) -> str:
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_files(root: Path, files: dict[str, str]) -> None:
    for name, expected in files.items():
        path = root / name
        if not path.is_file():
            raise BundleError(
                f"Bundle thiếu tệp {name} (theo manifest) trong {root}. {_HOW_TO_GET}"
            )
        actual = _sha256(path)
        if actual != expected:
            raise BundleError(
                f"Tệp {name} trong {root} không khớp sha256 của manifest "
                f"(manifest {expected[:12]}…, thực tế {actual[:12]}…). Tải hoặc dựng lại bundle."
            )


def load_bundle(root: Path, *, verify_files: bool = True) -> Bundle:
    """Đọc manifest và hiệu chỉnh, kiểm tra tính toàn vẹn; lỗi thì ném `BundleError`."""
    root = Path(root)
    manifest_path = root / MANIFEST
    if not manifest_path.is_file():
        raise BundleError(f"Không tìm thấy bundle mô hình: thiếu {manifest_path}. {_HOW_TO_GET}")

    manifest = _read_json_model(manifest_path, Manifest)
    if manifest.normalizer_version != NORMALIZER_VERSION:
        raise BundleError(
            f"Bundle {manifest.bundle_version} dựng với chuẩn hoá văn bản phiên bản "
            f"{manifest.normalizer_version}, mã nguồn là {NORMALIZER_VERSION}: mô hình sẽ nhận "
            "đầu vào khác lúc huấn luyện. Dựng lại bundle bằng mã nguồn hiện tại."
        )
    missing = [name for name in REQUIRED_FILES if name not in manifest.files]
    if missing:
        raise BundleError(
            f"Manifest {manifest_path} không liệt kê tệp bắt buộc: {', '.join(missing)}"
        )
    if verify_files:
        _verify_files(root, manifest.files)
    calibration = _read_json_model(root / CALIBRATION, Calibration)
    return Bundle(root=root, manifest=manifest, calibration=calibration)
