"""Fixture dùng chung cho test API: cấu hình đọc từ env, thư mục web export và app có mô hình."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edupulse.api.main import create_app
from edupulse.config import Settings, load_settings
from edupulse.engine.service import Analyzer
from edupulse.text.segment import SegmenterError


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


@pytest.fixture(scope="session")
def live_client(settings: Settings) -> Iterator[TestClient]:
    """App đã qua lifespan, tức đã nạp bundle thật; chỉ dùng cùng marker `model`."""
    try:
        with TestClient(create_app(settings)) as live:
            yield live
    except SegmenterError as exc:  # máy không có JRE
        pytest.skip(f"không khởi động được bộ tách từ: {exc}")


@pytest.fixture(scope="session")
def analyzer(live_client: TestClient) -> Analyzer:
    """Bộ phân tích mà app thật đang dùng (không nạp thêm bản thứ hai)."""
    return live_client.app.state.analyzer


@pytest.fixture(autouse=True)
def empty_batch_store(request: pytest.FixtureRequest) -> Iterator[None]:
    """Mỗi test bắt đầu và kết thúc với kho batch của app thật trống (nếu test có dùng app đó)."""
    live = "live_client" in request.fixturenames or "analyzer" in request.fixturenames
    if live:
        request.getfixturevalue("live_client").app.state.jobs.clear()
    yield
    if live:
        request.getfixturevalue("live_client").app.state.jobs.clear()
