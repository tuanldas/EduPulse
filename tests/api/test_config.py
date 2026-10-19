"""`/api/config`, `/api/model-card` và `/api/sample.csv`."""

import csv
import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edupulse import demo_data
from edupulse.api import schemas
from edupulse.api.limits import MAX_GROUPS
from edupulse.api.main import create_app
from edupulse.bundle_schema import ModelCard
from edupulse.config import (
    AUTO_XAI_TOP,
    BUDGETS,
    DEFAULT_BUDGET,
    HEARTBEAT_S,
    MAX_CHARS,
    MAX_ROWS,
    MAX_UPLOAD_MB,
    Settings,
)
from edupulse.engine.bundle import load_model_card


def write_small_demo(directory: Path, bundle: str) -> None:
    rows = [
        demo_data.DemoRowData(
            display_text="Phòng máy nóng.",
            model_text="phòng máy nóng .",
            sentiment="negative",
            topic="facility",
            sentiment_confidence=0.9,
            topic_confidence=0.9,
            n_words=4,
        )
    ]
    sample = [demo_data.SourceRow("thầy dạy hay .", "positive", "lecturer")]
    ui = [
        demo_data.UiSampleItem(text="Thầy dạy hay.", no_diacritics=False),
        demo_data.UiSampleItem(text="Thay day hay.", no_diacritics=True),
    ]
    demo_data.write_demo_files(
        directory, sample=sample, ui_samples=ui, rows=rows, bundle_version=bundle, revision="rev"
    )


@pytest.fixture
def stateful_client(settings: Settings, tmp_path: Path) -> TestClient:
    """App không nạp mô hình nhưng có sẵn dữ liệu demo trong `app.state`."""
    app = create_app(settings)
    write_small_demo(tmp_path, "b1")
    app.state.demo = demo_data.load_demo(tmp_path, "b1")
    return TestClient(app)


def test_sample_csv_downloads_the_file_built_with_the_image(stateful_client: TestClient):
    response = stateful_client.get("/api/sample.csv")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert 'filename="edupulse-file-mau.csv"' in response.headers["content-disposition"]
    assert response.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"))))
    assert rows[0] == ["Góp ý", "Nhóm (minh hoạ)"]
    assert rows[1][0] == "Thầy dạy hay."


def test_sample_csv_is_a_json_404_when_the_demo_is_unavailable(settings: Settings, tmp_path: Path):
    app = create_app(settings)
    app.state.demo = demo_data.load_demo(tmp_path, "b1")  # thư mục trống
    response = TestClient(app).get("/api/sample.csv")
    assert response.status_code == 404
    detail = schemas.ErrorResponse.model_validate(response.json()).detail
    assert detail.code == "demo_unavailable" and "make demo-data" in detail.message


def test_model_card_is_404_when_the_bundle_has_none(settings: Settings):
    response = TestClient(create_app(settings)).get("/api/model-card")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "model_card_unavailable"


def test_config_needs_the_model_to_be_loaded(stateful_client: TestClient):
    response = stateful_client.get("/api/config")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "engine_unavailable"


@pytest.mark.model
def test_config_exposes_limits_budgets_and_the_model(live_client: TestClient, settings: Settings):
    config = schemas.AppConfig.model_validate(live_client.get("/api/config").json())
    assert config.limits == schemas.Limits(
        max_chars=MAX_CHARS,
        max_upload_mb=MAX_UPLOAD_MB,
        max_rows=MAX_ROWS,
        max_groups=MAX_GROUPS,
        xai_max_words=settings.xai_max_words,
        auto_xai_top=AUTO_XAI_TOP,
    )
    assert config.heartbeat_s == HEARTBEAT_S
    assert config.default_budget == DEFAULT_BUDGET
    assert config.source_url == settings.source_url
    assert config.ner_enabled == settings.ner
    health = live_client.get("/api/health").json()
    assert config.model.model_version == health["model_version"]
    assert config.model.stage == health["stage"]


@pytest.mark.model
def test_config_budgets_come_from_the_calibration(live_client: TestClient, analyzer):
    config = schemas.AppConfig.model_validate(live_client.get("/api/config").json())
    assert [b.budget for b in config.budgets] == list(BUDGETS)
    assert [b.budget for b in config.budgets if b.default] == [DEFAULT_BUDGET]
    for option in config.budgets:
        assert option.efficiency == analyzer.calibration.efficiency[option.budget]
    # Ngân sách càng lớn càng bắt được nhiều lỗi: hiệu quả ước tính phải đơn điệu.
    caught = [b.efficiency.caught_sentiment_errors for b in config.budgets]
    assert caught == sorted(caught)


@pytest.mark.model
def test_config_demo_fields_agree_with_each_other(live_client: TestClient):
    config = schemas.AppConfig.model_validate(live_client.get("/api/config").json())
    if config.demo_available:
        assert config.demo_reason is None
        assert len(config.ui_samples) == 7
        assert [s.no_diacritics for s in config.ui_samples] == [False] * 6 + [True]
    else:
        assert config.demo_reason and config.ui_samples == []


@pytest.mark.model
def test_model_card_is_the_bundle_card_unchanged(live_client: TestClient, analyzer):
    response = live_client.get("/api/model-card")
    assert response.status_code == 200
    assert ModelCard.model_validate(response.json()) == load_model_card(analyzer.bundle)
