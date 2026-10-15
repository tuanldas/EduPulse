"""Cấu hình pytest dùng chung.

Test gắn marker `data`, `model`, `java` hoặc `train` tự được bỏ qua kèm lý do rõ khi máy chưa có
tiền điều kiện tương ứng, thay vì lỗi khó hiểu.
"""

import importlib.util
import subprocess
from collections.abc import Callable

import pytest

from edupulse.config import load_settings

pytest_plugins = ["pytester"]  # để test_package.py kiểm tra chính hook bỏ qua bên dưới


def _has_processed_data() -> bool:
    return (load_settings().data_dir / "processed" / "MANIFEST.json").is_file()


def _has_bundle() -> bool:
    return (load_settings().bundle_dir / "manifest.json").is_file()


def _has_java() -> bool:
    # Trên macOS, /usr/bin/java luôn có nhưng báo lỗi khi chưa cài JRE nên phải chạy thử.
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _has_train_group() -> bool:
    return importlib.util.find_spec("torch") is not None


# marker -> (hàm kiểm tra tiền điều kiện, lý do bỏ qua)
REQUIREMENTS: dict[str, tuple[Callable[[], bool], str]] = {
    "data": (_has_processed_data, "chưa có data/processed — chạy `make data`"),
    "model": (_has_bundle, "chưa có artifacts — chạy `make artifacts`"),
    "java": (_has_java, "thiếu JRE — cài Java (apt: default-jre-headless)"),
    "train": (_has_train_group, "thiếu group train — chạy `uv sync --group train`"),
}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    satisfied: dict[str, bool] = {}
    for item in items:
        for marker, (check, reason) in REQUIREMENTS.items():
            if item.get_closest_marker(marker) is None:
                continue
            if marker not in satisfied:
                satisfied[marker] = check()
            if not satisfied[marker]:
                item.add_marker(pytest.mark.skip(reason=reason))
