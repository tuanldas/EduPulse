"""Nhật ký chỉ có khuôn route và số lượng: không nội dung, tên file, giá trị nhóm hay id."""

import logging

import pytest
from batch_helpers import SENTENCES, csv_bytes, run, upload, wait_for
from fastapi.testclient import TestClient

pytestmark = pytest.mark.model

CONTENT = "NOIDUNG7f3a"
FILENAME = "TENFILE91bc"
GROUP = "NHOMGIATRI55d1"


def server_logs(caplog: pytest.LogCaptureFixture) -> str:
    """Nhật ký của máy chủ (logger `edupulse*`); log của chính httpx trong test ghi cả URL."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name.startswith("edupulse"))


def test_a_whole_session_leaves_no_user_data_in_the_logs(
    live_client: TestClient, caplog: pytest.LogCaptureFixture
):
    caplog.set_level(logging.DEBUG)
    rows = [[f"{SENTENCES[i % len(SENTENCES)]} {CONTENT}", f"{GROUP}{i % 2}"] for i in range(40)]
    data = csv_bytes(rows, header=("Góp ý", "Nhóm"))

    uploaded = upload(live_client, data, f"{FILENAME}.csv")
    batch_id = uploaded.json()["batch_id"]
    assert run(live_client, batch_id, group_col="Nhóm", review_budget=20).status_code == 202
    assert wait_for(live_client, batch_id)["state"] == "done"
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    explainable = next(r for r in results["rows"] if r["explainable"] and not r["has_explanation"])
    explained = live_client.get(f"/api/batches/{batch_id}/rows/{explainable['index']}/explain")
    assert explained.status_code == 200
    live_client.post(f"/api/batches/{batch_id}/heartbeat")
    live_client.post(f"/api/batches/{batch_id}/close")

    # Cả những đường lỗi cũng không được lặp lại dữ liệu người dùng.
    upload(live_client, b"x", f"{FILENAME}.txt")
    upload(live_client, b"", f"{FILENAME}.csv")
    live_client.get(f"/api/batches/{batch_id}")  # đã đóng: 404
    live_client.post("/api/analyze", json={"text": f"{CONTENT} " + "a" * 600})

    logs = server_logs(caplog)
    for secret in (CONTENT, FILENAME, GROUP, batch_id):
        assert secret not in logs, secret
    # Vẫn có nhật ký truy cập, ghi theo khuôn route.
    assert "POST /api/batches 201" in logs
    assert "GET /api/batches/{batch_id}/results 200" in logs
    assert "POST /api/batches/{batch_id}/close 204" in logs


def test_server_errors_log_only_the_exception_type(
    live_client: TestClient, caplog: pytest.LogCaptureFixture, monkeypatch
):
    caplog.set_level(logging.DEBUG)

    def explode(*args, **kwargs):
        raise RuntimeError(f"hỏng khi xử lý {CONTENT}")

    batch_id = upload(live_client, csv_bytes([[SENTENCES[0] + " " + CONTENT]])).json()["batch_id"]
    monkeypatch.setattr(live_client.app.state.analyzer, "run_batch", explode)
    assert run(live_client, batch_id).status_code == 202
    status = wait_for(live_client, batch_id)
    assert (status["state"], status["error_code"]) == ("failed", "internal_error")
    logs = server_logs(caplog)
    assert CONTENT not in logs and batch_id not in logs
    assert "RuntimeError" in logs
