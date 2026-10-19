"""Luồng vàng của người duyệt qua HTTP với mô hình thật, từ tải file tới đóng tab."""

import csv
import io

import pytest
from batch_helpers import csv_bytes, sentences, upload, wait_for
from fastapi.testclient import TestClient

from edupulse.api import schemas

pytestmark = pytest.mark.model


def test_reviewer_session_from_upload_to_close(live_client: TestClient):
    # 1. Tải file có cột nhóm, một câu dài, một câu chỉ có số điện thoại.
    long_text = " ".join(["Giảng viên dạy rất hay và nhiệt tình"] * 6)
    rows = [[text, f"Nhóm {i % 3}"] for i, (text,) in enumerate(sentences(90))]
    rows[3] = [long_text, "Nhóm 0"]
    rows[7] = ["0912345678", "Nhóm 1"]
    data = csv_bytes(rows, header=("Góp ý", "Môn học"))
    uploaded = schemas.BatchUpload.model_validate(upload(live_client, data).json())
    batch_id = uploaded.batch_id
    assert uploaded.suggested_text_col == "Góp ý" and uploaded.suggested_group_col == "Môn học"

    # 2. Chạy với ngân sách 20% và theo dõi tiến độ.
    run = live_client.post(
        f"/api/batches/{batch_id}/run",
        json={"text_col": "Góp ý", "group_col": "Môn học", "review_budget": 20},
    )
    assert run.status_code == 202
    assert wait_for(live_client, batch_id)["state"] == "done"

    # 3. Giữ phiên bằng heartbeat rồi đọc kết quả.
    assert live_client.post(f"/api/batches/{batch_id}/heartbeat").status_code == 204
    results = schemas.ResultsResponse.model_validate(
        live_client.get(f"/api/batches/{batch_id}/results").json()
    )
    assert results.summary.n_rows == 90 and results.summary.budget == 20
    assert results.summary.n_groups == 3 and results.summary.pii_rows == 1
    assert [s.index for s in results.skipped] == [7]
    queue = [results.rows[i] for i in range(len(results.rows)) if results.rows[i].in_review]
    assert queue and {r.index for r in queue} == set(results.review_order)

    # 4. Mở giải thích: một câu có sẵn, một câu tính khi mở, câu dài bị từ chối.
    ready = next(r for r in results.rows if r.has_explanation)
    lazy = next(r for r in results.rows if r.explainable and not r.has_explanation)
    for row in (ready, lazy):
        explained = live_client.get(f"/api/batches/{batch_id}/rows/{row.index}/explain")
        assert explained.status_code == 200
    refused = live_client.get(f"/api/batches/{batch_id}/rows/3/explain")
    assert refused.status_code == 422 and refused.json()["detail"]["code"] == "xai_too_long"

    # 5. Dashboard của batch, rồi lọc theo nhóm (nhóm đi trong thân yêu cầu).
    everything = schemas.InsightsResponse.model_validate(
        live_client.post(f"/api/batches/{batch_id}/insights", json={}).json()
    )
    one_group = schemas.InsightsResponse.model_validate(
        live_client.post(f"/api/batches/{batch_id}/insights", json={"groups": ["Nhóm 2"]}).json()
    )
    assert everything.kpis.n_rows == 89 and one_group.kpis.n_rows == 30
    assert one_group.selected_groups == ["Nhóm 2"]

    # 6. Sửa hai nhãn, xuất cả hai kiểu file.
    target, other = results.review_order[0], results.review_order[1]
    by_index = {r.index: r for r in results.rows}
    flip = {"negative": "positive", "neutral": "negative", "positive": "negative"}
    overrides = [
        {"index": target, "sentiment": flip[by_index[target].sentiment]},
        {"index": other, "sentiment": flip[by_index[other].sentiment]},
    ]
    everything_csv = live_client.post(
        f"/api/batches/{batch_id}/export", json={"kind": "all", "overrides": overrides}
    )
    corrected_csv = live_client.post(
        f"/api/batches/{batch_id}/export", json={"kind": "corrected", "overrides": overrides}
    )
    assert everything_csv.status_code == corrected_csv.status_code == 200
    all_rows = list(csv.reader(io.StringIO(everything_csv.content.decode("utf-8-sig"))))
    corrected_rows = list(csv.reader(io.StringIO(corrected_csv.content.decode("utf-8-sig"))))
    assert len(all_rows) == 91 and len(corrected_rows) == 3
    assert "0912345678" not in everything_csv.text + corrected_csv.text

    # 7. Đóng tab (sendBeacon): batch biến mất ngay.
    assert live_client.post(f"/api/batches/{batch_id}/close").status_code == 204
    for path in ("", "/results"):
        assert live_client.get(f"/api/batches/{batch_id}{path}").status_code == 404, path
    assert live_client.post(f"/api/batches/{batch_id}/insights", json={}).status_code == 404


def test_demo_dashboard_loads_without_uploading_anything(live_client: TestClient):
    if not live_client.app.state.demo.available:
        pytest.skip("không có dữ liệu demo — chạy `make demo-data`")
    config = live_client.get("/api/config").json()
    assert config["demo_available"] is True
    status = live_client.get("/api/batches/demo").json()
    assert status["state"] == "done"
    insights = live_client.post("/api/batches/demo/insights", json={})
    assert insights.status_code == 200 and insights.json()["source"] == "demo"
