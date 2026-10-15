"""Fixture của test engine. Test cần bundle thật gắn marker `model` và tự bỏ qua khi chưa có."""

import pytest

from edupulse.config import Settings, load_settings
from edupulse.engine.bundle import Bundle, load_bundle
from edupulse.engine.service import Analyzer
from edupulse.text.segment import SegmenterError


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="session")
def bundle(settings: Settings) -> Bundle:
    return load_bundle(settings.bundle_dir)


@pytest.fixture(scope="session")
def analyzer(settings: Settings) -> Analyzer:
    try:
        return Analyzer.load(settings)
    except SegmenterError as exc:  # máy không có JRE: mọi đường tiền xử lý đều không chạy được
        pytest.skip(f"không khởi động được bộ tách từ: {exc}")
