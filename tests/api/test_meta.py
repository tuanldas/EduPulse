"""Health, header an toàn, CORS theo env và việc khởi động thiếu bundle."""

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edupulse.api.main import create_app
from edupulse.api.schemas import Health
from edupulse.config import Settings

DEV_ORIGIN = "http://localhost:3000"


def _client(settings: Settings, **changes) -> TestClient:
    return TestClient(create_app(dataclasses.replace(settings, **changes)))


def test_health_reports_status_bundle_version_and_memory(client: TestClient, settings: Settings):
    response = client.get("/api/health")
    assert response.status_code == 200
    health = Health.model_validate(response.json())
    assert health.status == "ok"
    assert health.bundle_version == settings.bundle_version
    assert health.ner_enabled is settings.ner
    assert health.rss_mb > 0


@pytest.mark.parametrize("ner", [True, False])
def test_health_reflects_the_ner_setting(settings: Settings, ner: bool):
    health = _client(settings, ner=ner).get("/api/health").json()
    assert health["ner_enabled"] is ner


def test_every_response_has_the_security_headers(client: TestClient, web_dir: Path):
    for path in ("/api/health", "/api/khong-co", "/", "/khong-co/"):
        headers = client.get(path).headers
        assert headers["x-content-type-options"] == "nosniff", path
        assert headers["x-frame-options"] == "DENY", path
        assert headers["referrer-policy"] == "no-referrer", path


def test_api_responses_are_not_cacheable_but_static_files_are_left_alone(
    client: TestClient, web_dir: Path
):
    assert client.get("/api/health").headers["cache-control"] == "no-store"
    assert client.get("/api/khong-co").headers["cache-control"] == "no-store"
    assert "cache-control" not in client.get("/").headers


def test_openapi_lives_under_api_and_uses_function_names_as_operation_ids(client: TestClient):
    schema = client.get("/api/openapi.json").json()
    assert schema["paths"]["/api/health"]["get"]["operationId"] == "health"
    assert all(path.startswith("/api/") for path in schema["paths"])


def test_cors_is_off_without_configured_origins(client: TestClient):
    response = client.get("/api/health", headers={"Origin": DEV_ORIGIN})
    assert "access-control-allow-origin" not in response.headers


def test_cors_allows_only_the_configured_origins(settings: Settings):
    cors = _client(settings, cors_origins=(DEV_ORIGIN,))

    allowed = cors.get("/api/health", headers={"Origin": DEV_ORIGIN})
    assert allowed.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert "access-control-allow-credentials" not in allowed.headers

    other = cors.get("/api/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in other.headers


def test_cors_preflight_for_a_json_post(settings: Settings):
    preflight = _client(settings, cors_origins=(DEV_ORIGIN,)).options(
        "/api/analyze",
        headers={
            "Origin": DEV_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == DEV_ORIGIN
    assert "POST" in preflight.headers["access-control-allow-methods"]
    assert "x-content-type-options" in preflight.headers


def test_startup_without_a_bundle_stops_with_a_clear_message(settings: Settings, tmp_path: Path):
    app = create_app(dataclasses.replace(settings, bundle_dir=tmp_path / "chua-co-bundle"))
    with pytest.raises(RuntimeError) as failure:
        with TestClient(app):
            pass
    message = str(failure.value)
    assert "chua-co-bundle" in message
    assert "manifest.json" in message
    assert "make artifacts" in message


def test_startup_with_a_bundle_directory_missing_its_manifest_also_stops(
    settings: Settings, tmp_path: Path
):
    app = create_app(dataclasses.replace(settings, bundle_dir=tmp_path))  # tồn tại nhưng rỗng
    with pytest.raises(RuntimeError, match="manifest.json"):
        with TestClient(app):
            pass
