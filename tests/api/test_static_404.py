"""Web export tĩnh cùng origin với API, và JSON 404 cho mọi đường dẫn `/api` không có thật.

Các test dùng thư mục `out/` thật do Next.js sinh ra; thiếu thì bỏ qua kèm lý do.
"""

import dataclasses
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from edupulse.api.main import create_app
from edupulse.config import Settings


def _assert_json_error(response, status: int, code: str) -> None:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/json"
    detail = response.json()["detail"]
    assert detail["code"] == code
    assert detail["message"]


def test_root_serves_exported_index_html(client: TestClient, web_dir: Path):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == (web_dir / "index.html").read_text(encoding="utf-8")
    assert '<html lang="vi"' in response.text


@pytest.mark.parametrize("path", ["/api/khong-co", "/api/khong-co/sau/hon", "/api", "/api/"])
def test_unknown_api_path_is_json_404_not_the_web_404_page(
    client: TestClient, web_dir: Path, path: str
):
    _assert_json_error(client.get(path), 404, "not_found")


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_unknown_api_path_is_json_404_for_every_method(
    client: TestClient, web_dir: Path, method: str
):
    _assert_json_error(client.request(method, "/api/khong-co"), 404, "not_found")


def test_wrong_method_on_a_real_endpoint_is_405_with_allow(client: TestClient):
    response = client.post("/api/health")
    _assert_json_error(response, 405, "method_not_allowed")
    assert "GET" in response.headers["allow"]


def test_unknown_page_returns_the_exported_404_html(client: TestClient, web_dir: Path):
    response = client.get("/khong-co/")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == (web_dir / "404.html").read_text(encoding="utf-8")


def test_page_directory_without_trailing_slash_redirects(client: TestClient, web_dir: Path):
    # `trailingSlash` của Next xuất trang thành <tên>/index.html, nên /<tên> chuyển sang /<tên>/.
    page = next(
        (d.name for d in sorted(web_dir.iterdir()) if d.is_dir() and (d / "index.html").is_file()),
        None,
    )
    if page is None:
        pytest.skip("bản export chưa có trang con nào")
    redirect = client.get(f"/{page}", headers={"host": "evil.example"}, follow_redirects=False)
    assert redirect.status_code == 307
    # Đường dẫn tương đối: không phản chiếu header Host do client gửi.
    assert redirect.headers["location"] == f"/{page}/"
    assert client.get(f"/{page}/").status_code == 200


def test_static_asset_is_served_with_its_content_type(client: TestClient, web_dir: Path):
    script = next((web_dir / "_next" / "static").rglob("*.js"), None)
    if script is None:
        pytest.skip("bản export chưa có file JS")
    response = client.get("/" + script.relative_to(web_dir).as_posix())
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]


@pytest.mark.parametrize(
    "path", ["/%2e%2e/%2e%2e/pyproject.toml", "/..%2f..%2fpyproject.toml", "/%2e%2e%2fMakefile"]
)
def test_path_traversal_cannot_read_files_outside_the_web_dir(
    client: TestClient, web_dir: Path, path: str
):
    response = client.get(path)
    assert response.status_code == 404
    assert "[project]" not in response.text
    assert "THIS_MAKEFILE" not in response.text


def test_write_methods_on_static_paths_are_rejected_as_json(client: TestClient, web_dir: Path):
    # `/` có thật nên là 405; đường dẫn không có thật là 404.
    _assert_json_error(client.post("/"), 405, "method_not_allowed")
    _assert_json_error(client.post("/khong-co/"), 404, "not_found")


def test_without_a_web_export_the_api_still_runs_and_pages_are_json_404(
    settings: Settings, tmp_path: Path
):
    api_only = TestClient(create_app(dataclasses.replace(settings, web_dir=tmp_path / "chua-co")))
    assert api_only.get("/api/health").status_code == 200
    _assert_json_error(api_only.get("/"), 404, "not_found")
    _assert_json_error(api_only.get("/khong-co/"), 404, "not_found")
