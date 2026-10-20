"""Header an toàn: chuyển hướng không phản chiếu header Host do client gửi."""

import pytest
from starlette.applications import Starlette
from starlette.responses import RedirectResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from edupulse.api.security_headers import SecurityHeadersMiddleware, relative_location


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("http://evil.example/batch/", "/batch/"),
        ("https://edupulse.example/model/?x=1#giay-phep", "/model/?x=1#giay-phep"),
        ("http://evil.example", "/"),
        ("/dashboard/", "/dashboard/"),
    ],
)
def test_absolute_redirects_become_relative_paths(location: str, expected: str):
    assert relative_location(location) == expected


def test_middleware_rewrites_a_redirect_built_from_the_host_header():
    def to_slash(request):
        return RedirectResponse(str(request.url.replace(path="/batch/")))

    app = SecurityHeadersMiddleware(Starlette(routes=[Route("/batch", to_slash)]))
    response = TestClient(app).get(
        "/batch?nhom=A", headers={"host": "evil.example"}, follow_redirects=False
    )
    assert response.status_code == 307
    assert response.headers["location"] == "/batch/?nhom=A"
    assert response.headers["x-frame-options"] == "DENY"
