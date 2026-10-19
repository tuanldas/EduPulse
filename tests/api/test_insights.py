"""`POST /api/batches/{id}/insights`: batch demo, batch người dùng, lọc nhóm qua thân yêu cầu."""

import pytest
from batch_helpers import analyze_and_wait, csv_bytes, sentences, upload
from fastapi.testclient import TestClient

from edupulse.api import schemas

pytestmark = pytest.mark.model


def insights(client: TestClient, batch_id: str, **body):
    return client.post(f"/api/batches/{batch_id}/insights", json=body)


def parsed(response) -> schemas.InsightsResponse:
    assert response.status_code == 200, response.text
    return schemas.InsightsResponse.model_validate(response.json())


@pytest.fixture
def need_demo(live_client: TestClient) -> None:
    if not live_client.app.state.demo.available:
        pytest.skip("không có dữ liệu demo — chạy `make demo-data`")


def grouped_batch(client: TestClient, n: int = 100) -> tuple[str, dict]:
    rows = [[text, f"Nhóm {i % 3}"] for i, (text,) in enumerate(sentences(n))]
    data = csv_bytes(rows, header=("Góp ý", "Môn học"))
    return analyze_and_wait(client, data, group_col="Môn học")


# --- Batch demo ---------------------------------------------------------------------------------


def test_demo_insights_cover_the_whole_test_set_without_groups_or_flags(
    live_client: TestClient, need_demo
):
    out = parsed(insights(live_client, "demo"))
    assert out.source == "demo" and not out.illustrative_groups
    assert out.kpis.n_rows == 3166 and out.kpis.n_groups is None
    assert out.kpis.n_flagged is None and out.kpis.flagged_share is None
    assert out.groups is None and out.selected_groups is None and out.heatmap is None
    assert [t.topic for t in out.topics] == ["lecturer", "training_program", "facility", "others"]
    assert sum(t.n for t in out.topics) == 3166
    assert [b.topic for b in out.keywords] == ["lecturer", "training_program", "facility"]
    assert all(b.keywords for b in out.keywords)
    assert [b.topic for b in out.quotes] == ["lecturer", "training_program", "facility", "others"]
    assert out.recommendations


def test_demo_insights_ignore_a_group_filter_because_there_are_no_groups(
    live_client: TestClient, need_demo
):
    out = parsed(insights(live_client, "demo", groups=["Nhóm A"]))
    assert out.kpis.n_rows == 3166 and out.selected_groups is None


def test_demo_kpis_are_close_to_the_training_baseline(live_client: TestClient, need_demo):
    out = parsed(insights(live_client, "demo"))
    assert abs(out.kpis.delta_points.negative) < 5  # tập test cùng phân phối với train
    assert out.alerts == []  # "không chủ đề nào vượt mức nền đáng kể"


# --- Batch của người dùng -------------------------------------------------------------------------


def test_insights_wait_for_the_analysis_to_finish(live_client: TestClient):
    batch_id = upload(live_client, csv_bytes(sentences(5))).json()["batch_id"]
    response = insights(live_client, batch_id)
    assert response.status_code == 409 and response.json()["detail"]["code"] == "not_done"


def test_insights_of_a_user_batch_agree_with_its_results(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(80)), review_budget=20)
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    out = parsed(insights(live_client, batch_id))
    assert out.source == "batch" and out.kpis.n_rows == results["summary"]["n_analyzed"]
    assert out.kpis.n_flagged == results["summary"]["n_flagged"]
    negatives = sum(r["sentiment"] == "negative" for r in results["rows"])
    assert out.kpis.shares.negative == pytest.approx(negatives / 80)
    per_topic = {t.topic: t.n for t in out.topics}
    for topic in per_topic:
        assert per_topic[topic] == sum(r["topic"] == topic for r in results["rows"])


def test_insights_skip_rows_that_have_no_label(live_client: TestClient):
    rows = [*sentences(20), ["0912345678"]]
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(rows))
    assert parsed(insights(live_client, batch_id)).kpis.n_rows == 20


def test_group_filter_travels_in_the_body_and_narrows_the_numbers(live_client: TestClient):
    batch_id, _ = grouped_batch(live_client)
    everything = parsed(insights(live_client, batch_id))
    assert [g.name for g in everything.groups] == ["Nhóm 0", "Nhóm 1", "Nhóm 2"]
    assert sum(g.n for g in everything.groups) == 100 and everything.heatmap is not None
    one = parsed(insights(live_client, batch_id, groups=["Nhóm 1"]))
    assert one.selected_groups == ["Nhóm 1"] and one.kpis.n_rows == 33
    assert one.kpis.n_groups == 1 and [g.n for g in one.groups] == [g.n for g in everything.groups]
    assert one.heatmap.groups == ["Nhóm 1"]
    two = parsed(insights(live_client, batch_id, groups=["Nhóm 2", "Nhóm 1"]))
    assert two.selected_groups == ["Nhóm 1", "Nhóm 2"] or two.selected_groups == [
        "Nhóm 2",
        "Nhóm 1",
    ]
    assert two.kpis.n_rows == 66


def test_filter_order_does_not_matter_and_unknown_groups_are_ignored(live_client: TestClient):
    batch_id, _ = grouped_batch(live_client)
    a = insights(live_client, batch_id, groups=["Nhóm 0", "Nhóm 2"]).json()
    b = insights(live_client, batch_id, groups=["Nhóm 2", "Nhóm 0"]).json()
    assert a == b
    ignored = parsed(insights(live_client, batch_id, groups=["Không có"]))
    assert ignored.selected_groups is None and ignored.kpis.n_rows == 100


def test_heatmap_cells_add_up_to_the_batch(live_client: TestClient):
    batch_id, _ = grouped_batch(live_client)
    heatmap = parsed(insights(live_client, batch_id)).heatmap
    assert len(heatmap.cells) == 3 and all(len(line) == 4 for line in heatmap.cells)
    assert sum(cell.n for line in heatmap.cells for cell in line) == 100
    assert all(not cell.shaded for line in heatmap.cells for cell in line if cell.n < 20)


def test_sample_batch_insights_say_the_groups_are_illustrative(live_client: TestClient, need_demo):
    created = live_client.post("/api/batches/sample").json()["batch_id"]
    live_client.post(
        f"/api/batches/{created}/run", json={"text_col": "Góp ý", "group_col": "Nhóm (minh hoạ)"}
    )
    from batch_helpers import wait_for

    assert wait_for(live_client, created)["state"] == "done"
    out = parsed(insights(live_client, created))
    assert out.illustrative_groups is True
    assert len(out.groups) == 5 and out.kpis.n_rows == 200


def test_the_same_request_gets_the_same_answer_every_time(live_client: TestClient):
    batch_id, _ = grouped_batch(live_client)
    first = insights(live_client, batch_id, groups=["Nhóm 0"]).json()
    assert insights(live_client, batch_id, groups=["Nhóm 0"]).json() == first
    assert insights(live_client, batch_id).json() != first


def test_cache_is_bounded_but_answers_stay_correct(live_client: TestClient):
    batch_id, _ = grouped_batch(live_client, 30)
    batch = live_client.app.state.jobs.get(batch_id)
    for combo in (["Nhóm 0"], ["Nhóm 1"], ["Nhóm 2"], ["Nhóm 0", "Nhóm 1"], ["Nhóm 0", "Nhóm 2"]):
        insights(live_client, batch_id, groups=combo)
    assert 1 <= len(batch.insights_cache) <= 16


# --- Lỗi ---------------------------------------------------------------------------------------


def test_insights_of_an_unknown_batch_is_a_json_404(live_client: TestClient):
    response = insights(live_client, "khong-co")
    assert response.status_code == 404 and response.json()["detail"]["code"] == "batch_not_found"


@pytest.mark.parametrize(
    "body",
    [
        {"groups": [""]},
        {"groups": ["x" * 101]},
        {"groups": [str(i) for i in range(31)]},
        {"groups": "Nhóm A"},
        {"nhom": ["A"]},
    ],
)
def test_bad_filters_are_validation_errors(live_client: TestClient, body: dict):
    batch_id = upload(live_client, csv_bytes(sentences(3))).json()["batch_id"]
    response = live_client.post(f"/api/batches/{batch_id}/insights", json=body)
    assert response.status_code == 422 and response.json()["detail"]["code"] == "validation_error"


def test_a_bundle_without_background_numbers_cannot_serve_insights(
    live_client: TestClient, monkeypatch
):
    monkeypatch.setattr(live_client.app.state, "background", None)
    response = insights(live_client, "demo")
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "insights_unavailable"


def test_group_names_are_never_part_of_the_url(live_client: TestClient):
    batch_id, _ = grouped_batch(live_client, 30)
    response = live_client.post(f"/api/batches/{batch_id}/insights", json={"groups": ["Nhóm 1"]})
    assert response.status_code == 200
    assert "Nhóm" not in str(response.request.url) and response.request.url.query == b""
