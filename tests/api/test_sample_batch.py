"""Batch tạo từ file mẫu dựng lúc build, và batch `demo` chỉ đọc."""

from pathlib import Path

import pytest
from batch_helpers import csv_bytes, sentences, upload, wait_for
from fastapi.testclient import TestClient

from edupulse import demo_data
from edupulse.api import schemas

pytestmark = pytest.mark.model


@pytest.fixture
def need_demo(live_client: TestClient) -> None:
    if not live_client.app.state.demo.available:
        pytest.skip("không có dữ liệu demo — chạy `make demo-data`")


def create_sample(client: TestClient) -> schemas.BatchUpload:
    response = client.post("/api/batches/sample")
    assert response.status_code == 201, response.text
    return schemas.BatchUpload.model_validate(response.json())


def test_sample_batch_is_marked_illustrative_and_preselects_the_columns(
    live_client: TestClient, need_demo
):
    sample = create_sample(live_client)
    assert sample.columns == ["Góp ý", "Nhóm (minh hoạ)"]
    assert sample.n_rows == 200 and len(sample.preview) == 5
    assert sample.suggested_text_col == "Góp ý"
    assert sample.suggested_group_col == "Nhóm (minh hoạ)"
    assert sample.illustrative_groups and sample.encoding == "utf-8-sig"


def test_illustrative_flag_survives_to_status_results_and_reopening(
    live_client: TestClient, need_demo
):
    batch_id = create_sample(live_client).batch_id
    started = live_client.post(
        f"/api/batches/{batch_id}/run", json={"text_col": "Góp ý", "group_col": "Nhóm (minh hoạ)"}
    )
    assert started.status_code == 202 and started.json()["illustrative_groups"] is True
    assert wait_for(live_client, batch_id)["illustrative_groups"] is True
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert results["summary"]["illustrative_groups"] is True
    assert results["summary"]["n_groups"] == 5
    assert {r["group"] for r in results["rows"]} <= set(demo_data.GROUP_NAMES)
    # "Mở lại ?batch=…": đọc trạng thái lần nữa vẫn thấy cờ.
    assert live_client.get(f"/api/batches/{batch_id}").json()["illustrative_groups"] is True


def test_a_user_file_with_an_illustrative_group_column_is_marked(live_client: TestClient):
    rows = [[text, f"Nhóm {i % 2}"] for i, (text,) in enumerate(sentences(12))]
    marked = upload(live_client, csv_bytes(rows, header=("Góp ý", "Nhóm (minh hoạ)"))).json()
    assert marked["illustrative_groups"] is True  # tên cột ghi "minh hoạ"
    plain = upload(live_client, csv_bytes(rows, header=("Góp ý", "Nhóm"))).json()
    assert plain["illustrative_groups"] is False
    started = live_client.post(
        f"/api/batches/{plain['batch_id']}/run", json={"text_col": "Góp ý", "group_col": "Nhóm"}
    )
    assert started.json()["illustrative_groups"] is False
    started = live_client.post(
        f"/api/batches/{marked['batch_id']}/run",
        json={"text_col": "Góp ý", "group_col": "Nhóm (minh hoạ)"},
    )
    assert started.json()["illustrative_groups"] is True


def test_the_sample_file_can_be_downloaded_and_uploaded_back(live_client: TestClient, need_demo):
    downloaded = live_client.get("/api/sample.csv")
    assert downloaded.status_code == 200
    again = upload(live_client, downloaded.content, "edupulse-file-mau.csv").json()
    assert again["n_rows"] == 200 and again["illustrative_groups"] is True
    assert again["suggested_group_col"] == "Nhóm (minh hoạ)"


def test_demo_batch_is_done_but_only_serves_insights(live_client: TestClient, need_demo):
    status = schemas.BatchStatus.model_validate(live_client.get("/api/batches/demo").json())
    assert status.state == "done" and status.progress == 1.0 and status.n_rows == 3166
    assert not status.illustrative_groups
    for path in ("/api/batches/demo/results", "/api/batches/demo/rows/0/explain"):
        response = live_client.get(path)
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == "batch_readonly"
    run = live_client.post("/api/batches/demo/run", json={"text_col": "x"})
    assert run.status_code == 409 and run.json()["detail"]["code"] == "batch_readonly"


def test_without_demo_data_the_sample_batch_is_a_json_404(
    live_client: TestClient, tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(live_client.app.state, "demo", demo_data.load_demo(tmp_path, "b1"))
    response = live_client.post("/api/batches/sample")
    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "demo_unavailable"
    assert live_client.get("/api/sample.csv").status_code == 404
    config = live_client.get("/api/config").json()
    assert config["demo_available"] is False and config["ui_samples"] == []
