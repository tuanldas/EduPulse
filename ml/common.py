"""Tiện ích dùng chung cho các script `ml/`: đường dẫn, băm tệp, đọc ghi JSON, kiểm bản chuẩn hoá.

Mọi đường dẫn đi qua `edupulse.config` (theo EDUPULSE_HOME) nên script không ghi cứng `data/`,
`models/`, `outputs/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from edupulse.config import Settings, load_settings

SPLITS = ("train", "validation", "test")


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
