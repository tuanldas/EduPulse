"""Fixture dùng chung cho test API: cấu hình đọc từ env và thư mục web export thật."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edupulse.api.main import create_app
from edupulse.config import Settings, load_settings


@pytest.fixture(scope="session")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="session")
def web_dir(settings: Settings) -> Path:
    """Thư mục `out/` thật do `pnpm --dir apps/web build` sinh ra."""
    if not (settings.web_dir / "index.html").is_file():
        pytest.skip("chưa build web — chạy `pnpm --dir apps/web build`")
    return settings.web_dir


@pytest.fixture
def client(settings: Settings) -> TestClient:
    # Không vào `with`: bỏ qua lifespan (cần bundle) vì các test dùng fixture này chỉ kiểm tra
    # định tuyến, header và nhật ký.
    return TestClient(create_app(settings))
