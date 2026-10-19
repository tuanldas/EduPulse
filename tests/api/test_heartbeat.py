"""Vòng đời batch qua HTTP: heartbeat giữ sống quá TTL, bỏ heartbeat thì hết hạn, close xoá ngay."""

import time
from collections.abc import Iterator

import pytest
from batch_helpers import analyze_and_wait, csv_bytes, sentences, upload
from fastapi.testclient import TestClient

from edupulse.api.jobs import JobStore

pytestmark = pytest.mark.model

TTL_S = 1.0


@pytest.fixture
def short_ttl(live_client: TestClient) -> Iterator[TestClient]:
    """App thật nhưng kho batch có TTL ngắn (cùng bộ phân tích, không nạp mô hình lần nữa)."""
    app = live_client.app
    original = app.state.jobs
    store = JobStore(app.state.analyzer, ttl_s=TTL_S)
    if app.state.demo.available:
        store.set_demo(list(app.state.demo.rows))
    store.start()
    app.state.jobs = store
    try:
        yield live_client
    finally:
        store.shutdown()
        app.state.jobs = original


def test_without_heartbeat_the_batch_expires_after_the_ttl(short_ttl: TestClient):
    batch_id = upload(short_ttl, csv_bytes(sentences(3))).json()["batch_id"]
    assert short_ttl.get(f"/api/batches/{batch_id}").status_code == 200
    time.sleep(TTL_S + 0.5)
    response = short_ttl.get(f"/api/batches/{batch_id}")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "batch_not_found"


def test_heartbeat_keeps_a_batch_alive_far_beyond_the_ttl(short_ttl: TestClient):
    batch_id = upload(short_ttl, csv_bytes(sentences(3))).json()["batch_id"]
    deadline = time.monotonic() + 3 * TTL_S
    while time.monotonic() < deadline:
        assert short_ttl.post(f"/api/batches/{batch_id}/heartbeat").status_code == 204
        time.sleep(TTL_S / 4)
    assert short_ttl.get(f"/api/batches/{batch_id}").status_code == 200  # sống gấp 3 lần TTL
    time.sleep(TTL_S + 0.5)  # ngừng heartbeat
    assert short_ttl.get(f"/api/batches/{batch_id}").status_code == 404


def test_reading_results_also_counts_as_activity(short_ttl: TestClient):
    batch_id, _ = analyze_and_wait(short_ttl, csv_bytes(sentences(6)))
    deadline = time.monotonic() + 2.5 * TTL_S
    while time.monotonic() < deadline:
        assert short_ttl.get(f"/api/batches/{batch_id}/results").status_code == 200
        time.sleep(TTL_S / 4)


def test_heartbeat_on_an_expired_batch_is_404(short_ttl: TestClient):
    batch_id = upload(short_ttl, csv_bytes(sentences(3))).json()["batch_id"]
    time.sleep(TTL_S + 0.5)
    assert short_ttl.post(f"/api/batches/{batch_id}/heartbeat").status_code == 404


def test_expired_batches_make_room_for_new_ones(short_ttl: TestClient):
    for _ in range(8):
        assert upload(short_ttl, csv_bytes(sentences(2))).status_code == 201
    assert upload(short_ttl, csv_bytes(sentences(2))).status_code == 429
    time.sleep(TTL_S + 0.5)
    assert upload(short_ttl, csv_bytes(sentences(2))).status_code == 201


def test_close_works_as_a_beacon_post_without_a_body(short_ttl: TestClient):
    batch_id = upload(short_ttl, csv_bytes(sentences(3))).json()["batch_id"]
    # navigator.sendBeacon chỉ gửi được POST, thân rỗng hoặc text/plain, không header tuỳ biến.
    response = short_ttl.post(
        f"/api/batches/{batch_id}/close", content=b"", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 204 and response.content == b""
    assert short_ttl.get(f"/api/batches/{batch_id}").status_code == 404


def test_the_demo_batch_ignores_ttl_heartbeat_and_close(short_ttl: TestClient):
    if not short_ttl.app.state.demo.available:
        pytest.skip("không có dữ liệu demo — chạy `make demo-data`")
    time.sleep(TTL_S + 0.5)
    for call in (
        short_ttl.post("/api/batches/demo/heartbeat"),
        short_ttl.post("/api/batches/demo/close"),
        short_ttl.delete("/api/batches/demo"),
    ):
        assert call.status_code == 204
    status = short_ttl.get("/api/batches/demo").json()
    assert status["state"] == "done" and status["n_rows"] > 0
