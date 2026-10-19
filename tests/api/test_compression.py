"""Phản hồi lớn được nén gzip nếu trình duyệt cho phép."""

import pytest
from batch_helpers import analyze_and_wait, csv_bytes, sentences
from fastapi.testclient import TestClient

pytestmark = pytest.mark.model

GZIP = {"Accept-Encoding": "gzip"}
IDENTITY = {"Accept-Encoding": "identity"}


def test_small_responses_are_left_alone(live_client: TestClient):
    response = live_client.get("/api/health", headers=GZIP)
    assert response.status_code == 200 and "content-encoding" not in response.headers


def test_large_json_is_compressed_for_clients_that_accept_gzip(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(120)))
    path = f"/api/batches/{batch_id}/results"
    compressed = live_client.get(path, headers=GZIP)
    assert compressed.headers["content-encoding"] == "gzip"
    assert "accept-encoding" in compressed.headers["vary"].lower()
    plain = live_client.get(path, headers=IDENTITY)
    assert "content-encoding" not in plain.headers
    assert compressed.json() == plain.json()  # httpx tự giải nén; nội dung phải y hệt


def test_compressed_responses_keep_their_security_headers(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(120)))
    response = live_client.get(f"/api/batches/{batch_id}/results", headers=GZIP)
    assert response.headers["content-encoding"] == "gzip"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_csv_export_is_compressed_and_still_a_valid_utf8_bom_file(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(120)))
    response = live_client.post(f"/api/batches/{batch_id}/export", json={}, headers=GZIP)
    assert response.headers["content-encoding"] == "gzip"
    assert response.content.startswith(b"\xef\xbb\xbf")
    assert "attachment" in response.headers["content-disposition"]


def test_config_is_compressed_too(live_client: TestClient):
    response = live_client.get("/api/config", headers=GZIP)
    assert response.headers.get("content-encoding") == "gzip" or len(response.content) < 1024
