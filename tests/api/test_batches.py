"""Phân tích hàng loạt qua HTTP với mô hình thật: tải file, chạy, tiến độ, kết quả, giải thích."""

import io
import math
import time

import openpyxl
import pytest
from batch_helpers import (
    SENTENCES,
    analyze_and_wait,
    csv_bytes,
    run,
    sentences,
    upload,
)
from fastapi.testclient import TestClient

from edupulse.api import schemas
from edupulse.config import AUDIT_RATE, MAX_ROWS, MAX_UPLOAD_MB

pytestmark = pytest.mark.model

MB = 1 << 20


def error(response) -> schemas.ErrorBody:
    return schemas.ErrorResponse.model_validate(response.json()).detail


# --- Tải file -----------------------------------------------------------------------------------


def test_upload_describes_the_file_and_guesses_the_columns(live_client: TestClient):
    rows = [[str(i), s, f"Nhóm {i % 3}"] for i, s in enumerate(SENTENCES)]
    data = csv_bytes(rows, header=("STT", "Góp ý", "Môn học"))
    response = upload(live_client, data, "gopy_hk1.csv")
    assert response.status_code == 201
    body = schemas.BatchUpload.model_validate(response.json())
    assert body.filename == "gopy_hk1.csv"
    assert body.n_rows == len(SENTENCES) and body.columns == ["STT", "Góp ý", "Môn học"]
    assert len(body.preview) == 5 and body.preview[0] == ["0", SENTENCES[0], "Nhóm 0"]
    assert body.suggested_text_col == "Góp ý" and body.suggested_group_col == "Môn học"
    assert body.encoding == "utf-8-sig" and body.warnings == []
    assert not body.illustrative_groups
    assert len(body.batch_id) >= 22


def test_very_long_file_names_are_cut_for_display_but_still_judged_by_extension(
    live_client: TestClient,
):
    long = upload(live_client, csv_bytes(sentences(3)), "a" * 1000 + ".csv")
    assert long.status_code == 201 and long.json()["filename"] == "a" * 255
    assert upload(live_client, csv_bytes(sentences(3)), "a" * 1000 + ".txt").status_code == 415


def test_each_upload_gets_its_own_unguessable_id(live_client: TestClient):
    ids = {upload(live_client, csv_bytes(sentences(3))).json()["batch_id"] for _ in range(3)}
    assert len(ids) == 3


def test_preview_cells_are_cut_to_a_readable_length(live_client: TestClient):
    data = csv_bytes([["x" * 1000]] * 3)
    cell = upload(live_client, data).json()["preview"][0][0]
    assert len(cell) == 200


def test_upload_responses_are_never_cached(live_client: TestClient):
    response = upload(live_client, csv_bytes(sentences(2)))
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    ("name", "data", "status", "code"),
    [
        ("a.txt", b"Gop y\nhay\n", 415, "unsupported_file_type"),
        ("a.xls", b"\xd0\xcf\x11\xe0" + b"\x00" * 100, 415, "unsupported_file_type"),
        ("a.csv", b"", 422, "no_rows"),
        ("a.csv", b"Gop y\n", 422, "no_rows"),
        ("a.csv", b"\xff\xfeG\x00", 422, "decode_failed"),
        ("a.xlsx", b"PK\x03\x04" + b"junk" * 40, 422, "decode_failed"),
    ],
)
def test_upload_errors_have_stable_codes(live_client, name, data, status, code):
    response = upload(live_client, data, name)
    assert response.status_code == status
    assert error(response).code == code


def test_an_xlsx_with_one_far_corner_cell_is_refused_before_it_can_exhaust_memory(
    live_client: TestClient,
):
    # ~5KB, nhưng calamine sẽ dựng ma trận đặc ~52 triệu ô (~1,6GB) nếu được mở.
    book = openpyxl.Workbook()
    book.active.append(["Góp ý"])
    book.active.cell(row=1_048_576, column=50, value="x")
    out = io.BytesIO()
    book.save(out)
    assert len(out.getvalue()) < 10_000
    started = time.monotonic()
    response = upload(live_client, out.getvalue(), "xa.xlsx")
    assert response.status_code == 422 and error(response).code == "decode_failed"
    assert "vùng ô" in error(response).message
    assert time.monotonic() - started < 5


def test_too_many_rows_reports_the_real_count(live_client: TestClient):
    response = upload(live_client, csv_bytes(sentences(MAX_ROWS + 1)))
    assert response.status_code == 422
    detail = error(response)
    assert detail.code == "too_many_rows"
    assert detail.params == {"n_rows": MAX_ROWS + 1, "max_rows": MAX_ROWS}
    assert upload(live_client, csv_bytes(sentences(MAX_ROWS))).status_code == 201


def test_a_file_one_byte_over_the_limit_is_413(live_client: TestClient):
    over = b"x" * (MAX_UPLOAD_MB * MB + 1)
    response = upload(live_client, over)
    assert response.status_code == 413
    assert error(response).code == "file_too_large"
    assert error(response).params == {"max_mb": MAX_UPLOAD_MB}
    assert response.headers["x-content-type-options"] == "nosniff"  # qua middleware ngoài


def test_a_file_exactly_at_the_limit_is_accepted(live_client: TestClient):
    limit = MAX_UPLOAD_MB * MB
    header = csv_bytes([])  # BOM + tiêu đề + CRLF
    rows, extra = MAX_ROWS, 0
    per_row = (limit - len(header)) // rows - 2  # trừ CRLF cuối dòng
    extra = limit - len(header) - rows * (per_row + 2)
    lines = [b"a" * per_row] * (rows - 1) + [b"a" * (per_row + extra)]
    data = header + b"".join(line + b"\r\n" for line in lines)
    assert len(data) == limit
    response = upload(live_client, data)
    assert response.status_code == 201 and response.json()["n_rows"] == rows


def test_a_50mb_upload_is_rejected_by_the_middleware_before_parsing(live_client: TestClient):
    response = upload(live_client, b"x" * (50 * MB))
    assert response.status_code == 413
    assert error(response).code == "file_too_large"
    assert response.headers["content-type"] == "application/json"


def test_the_request_must_carry_a_file(live_client: TestClient):
    response = live_client.post("/api/batches", data={"x": "y"})
    assert response.status_code == 422 and error(response).code == "validation_error"


# --- Chạy --------------------------------------------------------------------------------------


def test_run_validates_columns_and_budget(live_client: TestClient):
    batch_id = upload(live_client, csv_bytes(sentences(5))).json()["batch_id"]
    assert run(live_client, batch_id, text_col="Không có").status_code == 422
    assert error(run(live_client, batch_id, text_col="Không có")).code == "invalid_column"
    assert error(run(live_client, batch_id, group_col="Góp ý")).code == "invalid_column"
    assert error(run(live_client, batch_id, review_budget=7)).code == "validation_error"
    assert error(run(live_client, batch_id, nhom="x")).code == "validation_error"


def test_run_on_an_unknown_batch_is_404(live_client: TestClient):
    response = run(live_client, "khong-co")
    assert response.status_code == 404 and error(response).code == "batch_not_found"


def test_run_twice_is_a_conflict(live_client: TestClient):
    batch_id = upload(live_client, csv_bytes(sentences(30))).json()["batch_id"]
    first = run(live_client, batch_id)
    assert first.status_code == 202
    assert schemas.BatchStatus.model_validate(first.json()).state in ("queued", "running")
    second = run(live_client, batch_id)
    assert second.status_code == 409 and error(second).code == "already_running"


def test_run_queue_is_limited_and_asks_to_retry(live_client: TestClient):
    big = csv_bytes(sentences(MAX_ROWS))
    ids = [upload(live_client, big).json()["batch_id"] for _ in range(4)]
    statuses = [run(live_client, batch_id) for batch_id in ids]
    assert [r.status_code for r in statuses] == [202, 202, 202, 429]
    assert error(statuses[3]).code == "busy"
    assert statuses[3].headers["retry-after"] == "5"
    for batch_id in ids:
        assert live_client.delete(f"/api/batches/{batch_id}").status_code == 204


def test_at_most_eight_batches_are_kept(live_client: TestClient):
    ids = [upload(live_client, csv_bytes(sentences(2))).json()["batch_id"] for _ in range(8)]
    ninth = upload(live_client, csv_bytes(sentences(2)))
    assert ninth.status_code == 429 and error(ninth).code == "busy"
    assert live_client.post(f"/api/batches/{ids[0]}/close").status_code == 204
    assert upload(live_client, csv_bytes(sentences(2))).status_code == 201


# --- Tiến độ và kết quả -------------------------------------------------------------------------


def test_full_flow_reports_monotonic_progress_and_consistent_results(live_client: TestClient):
    batch_id = upload(live_client, csv_bytes(sentences(120))).json()["batch_id"]
    assert run(live_client, batch_id, review_budget=20).status_code == 202
    progress, steps = [], []
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        status = schemas.BatchStatus.model_validate(
            live_client.get(f"/api/batches/{batch_id}").json()
        )
        progress.append(status.progress)
        if status.step and (not steps or steps[-1] != status.step):
            steps.append(status.step)
        assert status.n_rows == 120
        if status.state == "done":
            break
        assert status.state in ("queued", "running")
        time.sleep(0.02)
    assert progress == sorted(progress) and progress[-1] == 1.0
    assert steps == [s for s in ("prepare", "predict", "flag", "explain") if s in steps]

    results = schemas.ResultsResponse.model_validate(
        live_client.get(f"/api/batches/{batch_id}/results").json()
    )
    summary = results.summary
    assert summary.n_rows == summary.n_analyzed == len(results.rows) == 120
    assert summary.budget == 20 and summary.budget_target == math.ceil(0.2 * 120)
    assert summary.n_flagged == summary.budget_target  # câu có dấu: không câu nào bị cờ ép
    assert summary.n_audit == math.ceil(AUDIT_RATE * (120 - summary.n_flagged))
    assert summary.n_review == len(results.review_order) == summary.n_flagged + summary.n_audit
    assert summary.n_explained == min(20, summary.n_flagged)
    assert summary.n_groups is None and not summary.illustrative_groups
    assert [r.index for r in results.rows] == list(range(120))
    assert sum(r.source == "flag" for r in results.rows) == summary.n_flagged
    assert sum(r.source == "audit" for r in results.rows) == summary.n_audit
    assert all(r.in_review == (r.source != "none") for r in results.rows)
    assert all(r.needs_review == (r.source == "flag") for r in results.rows)
    by_index = {r.index: r for r in results.rows}
    assert set(results.review_order) == {i for i, r in by_index.items() if r.in_review}


def test_results_match_what_analyze_says_about_each_sentence(live_client: TestClient):
    batch_id, status = analyze_and_wait(live_client, csv_bytes(sentences(30)))
    assert status["state"] == "done"
    rows = live_client.get(f"/api/batches/{batch_id}/results").json()["rows"]
    for row in rows[: len(SENTENCES)]:
        single = live_client.post("/api/analyze", json={"text": row["text"]}).json()
        assert row["sentiment"] == single["sentiment"]["label"]
        assert row["topic"] == single["topic"]["label"]
        assert row["sentiment_confidence"] == single["sentiment"]["confidence"]
        assert row["topic_confidence"] == single["topic"]["confidence"]
        assert row["n_words"] == single["n_words"]


def test_results_before_the_run_finishes_are_a_conflict(live_client: TestClient):
    batch_id = upload(live_client, csv_bytes(sentences(5))).json()["batch_id"]
    for path in ("results", "rows/0/explain"):
        response = live_client.get(f"/api/batches/{batch_id}/{path}")
        assert response.status_code == 409 and error(response).code == "not_done"


def test_the_same_batch_always_gives_the_same_results(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(60)))
    first = live_client.get(f"/api/batches/{batch_id}/results").json()
    second = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert first == second


def test_same_file_twice_flags_the_same_rows(live_client: TestClient):
    data = csv_bytes(sentences(60))
    flags = []
    for _ in range(2):
        batch_id, _ = analyze_and_wait(live_client, data)
        rows = live_client.get(f"/api/batches/{batch_id}/results").json()["rows"]
        flags.append([r["needs_review"] for r in rows])
    assert flags[0] == flags[1]


def test_groups_travel_with_the_rows(live_client: TestClient):
    rows = [[s, f"Nhóm {i % 3}"] for i, (s,) in enumerate(sentences(30))]
    data = csv_bytes(rows, header=("Góp ý", "Môn học"))
    batch_id, _ = analyze_and_wait(live_client, data, group_col="Môn học")
    body = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert body["summary"]["n_groups"] == 3
    assert [r["group"] for r in body["rows"][:4]] == ["Nhóm 0", "Nhóm 1", "Nhóm 2", "Nhóm 0"]


def test_blank_group_cells_get_a_visible_label(live_client: TestClient):
    data = csv_bytes(
        [[SENTENCES[0], ""], [SENTENCES[1], "A"], [SENTENCES[2], "A"]], ("Góp ý", "Nhóm")
    )
    batch_id, _ = analyze_and_wait(live_client, data, group_col="Nhóm")
    groups = [
        r["group"] for r in live_client.get(f"/api/batches/{batch_id}/results").json()["rows"]
    ]
    assert groups == ["(trống)", "A", "A"]


def test_personal_information_is_masked_counted_and_never_returned(live_client: TestClient):
    texts = [
        "Thầy Nam dạy rất nhiệt tình.",
        "Em xin số 0912345678 để hỏi bài thêm.",
        "Liên hệ sv.test@example.com nếu cần tài liệu.",
        "Phòng máy nóng và máy chiếu mờ.",
    ]
    batch_id, status = analyze_and_wait(live_client, csv_bytes([[t] for t in texts]))
    assert status["pii_rows"] == 3
    response = live_client.get(f"/api/batches/{batch_id}/results")
    body = response.json()
    assert body["summary"]["pii_rows"] == 3
    counts = {c["type"]: c["rows"] for c in body["summary"]["pii_counts"]}
    assert counts == {"name": 1, "phone": 1, "email": 1}
    for secret in ("0912345678", "sv.test@example.com", "Nam"):
        assert secret not in response.text
    assert "[SĐT]" in body["rows"][1]["text"] and body["rows"][1]["pii_types"] == ["phone"]


def test_rows_with_nothing_to_analyze_are_listed_as_skipped(live_client: TestClient):
    rows = [[SENTENCES[0]], ["0912345678"], [SENTENCES[1]], ["🙂"], [SENTENCES[2]]]
    batch_id, status = analyze_and_wait(live_client, csv_bytes(rows))
    assert status["state"] == "done"
    body = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert body["summary"]["n_rows"] == 5 and body["summary"]["n_analyzed"] == 3
    assert [s["index"] for s in body["skipped"]] == [1, 3]
    assert [s["source_row"] for s in body["skipped"]] == [2, 4]
    assert body["skipped"][0]["text"] == "[SĐT]"
    assert [r["index"] for r in body["rows"]] == [0, 2, 4]  # `index` không bị dồn lại


def test_a_batch_with_nothing_to_analyze_fails_with_a_stable_code(live_client: TestClient):
    batch_id, status = analyze_and_wait(live_client, csv_bytes([["0912345678"], ["a@b.vn"]]))
    assert (status["state"], status["error_code"]) == ("failed", "no_content")
    assert live_client.get(f"/api/batches/{batch_id}/results").status_code == 409


def test_blank_lines_are_dropped_but_source_rows_stay_true_to_the_file(live_client: TestClient):
    data = (
        b"G\xc3\xb3p \xc3\xbd\n" + SENTENCES[0].encode() + b"\n\n\n" + SENTENCES[1].encode() + b"\n"
    )
    batch_id, _ = analyze_and_wait(live_client, data)
    rows = live_client.get(f"/api/batches/{batch_id}/results").json()["rows"]
    assert [(r["index"], r["source_row"]) for r in rows] == [(0, 1), (1, 4)]


def test_very_long_cells_are_cut_to_the_sentence_limit_and_counted(live_client: TestClient):
    long_text = ("Thầy dạy hay nhưng bài giảng hơi nhanh và nhiều. " * 30).strip()
    batch_id, _ = analyze_and_wait(live_client, csv_bytes([[long_text], [SENTENCES[0]]]))
    body = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert body["summary"]["n_truncated"] == 1
    assert len(body["rows"][0]["text"]) <= 500


def test_sentences_typed_without_diacritics_are_always_flagged(live_client: TestClient):
    rows = [["thay giang day nhiet tinh de hieu"], *sentences(30)]
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(rows), review_budget=5)
    body = live_client.get(f"/api/batches/{batch_id}/results").json()
    first = body["rows"][0]
    assert first["no_diacritics"] and first["needs_review"] and first["source"] == "flag"
    assert first["reasons"][0]["code"] == "no_diacritics"
    assert body["summary"]["n_no_diacritics"] == 1
    assert body["summary"]["n_flagged"] >= body["summary"]["budget_target"]


# --- Giải thích từng dòng -----------------------------------------------------------------------


def test_row_explanations_are_precomputed_for_the_top_flags_and_cached_for_the_rest(
    live_client: TestClient,
):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(100)), review_budget=30)
    rows = live_client.get(f"/api/batches/{batch_id}/results").json()["rows"]
    cached = next(r for r in rows if r["has_explanation"])
    lazy = next(r for r in rows if r["explainable"] and not r["has_explanation"])

    first = live_client.get(f"/api/batches/{batch_id}/rows/{cached['index']}/explain")
    assert first.status_code == 200
    explanation = schemas.ExplainResponse.model_validate(first.json())
    assert len(explanation.sentiment.tokens) == cached["n_words"]
    assert explanation.sentiment.label == cached["sentiment"]
    assert explanation.topic.label == cached["topic"]

    computed = live_client.get(f"/api/batches/{batch_id}/rows/{lazy['index']}/explain")
    assert computed.status_code == 200
    assert live_client.get(f"/api/batches/{batch_id}/rows/{lazy['index']}/explain").json() == (
        computed.json()
    )
    after = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert after["summary"]["n_explained"] == before_count(rows) + 1
    assert next(r for r in after["rows"] if r["index"] == lazy["index"])["has_explanation"]


def before_count(rows: list[dict]) -> int:
    return sum(r["has_explanation"] for r in rows)


def test_explanation_agrees_with_the_single_sentence_endpoint(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(20)), review_budget=30)
    rows = live_client.get(f"/api/batches/{batch_id}/results").json()["rows"]
    row = next(r for r in rows if r["explainable"])
    from_batch = live_client.get(f"/api/batches/{batch_id}/rows/{row['index']}/explain").json()
    alone = live_client.post("/api/explain", json={"text": row["text"]}).json()
    assert from_batch == alone


def test_explaining_a_long_row_or_a_missing_row_has_its_own_error(live_client: TestClient):
    long_text = " ".join(["Giảng viên dạy rất hay và nhiệt tình"] * 6)
    rows = [[long_text], ["0912345678"], [SENTENCES[0]]]
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(rows))
    path = f"/api/batches/{batch_id}/rows"
    too_long = live_client.get(f"{path}/0/explain")
    assert too_long.status_code == 422 and error(too_long).code == "xai_too_long"
    assert error(too_long).params["n_words"] > error(too_long).params["max_words"]
    no_content = live_client.get(f"{path}/1/explain")
    assert no_content.status_code == 422 and error(no_content).code == "no_content"
    missing = live_client.get(f"{path}/99/explain")
    assert missing.status_code == 404 and error(missing).code == "row_not_found"
    assert live_client.get(f"{path}/-1/explain").status_code == 422
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    assert [r["explainable"] for r in results["rows"]] == [False, True]


# --- Hết hạn và đóng ----------------------------------------------------------------------------


def test_unknown_batches_are_a_json_404_on_every_route(live_client: TestClient):
    for method, path in [
        ("GET", "/api/batches/khong-co"),
        ("POST", "/api/batches/khong-co/heartbeat"),
        ("POST", "/api/batches/khong-co/close"),
        ("DELETE", "/api/batches/khong-co"),
        ("GET", "/api/batches/khong-co/results"),
        ("GET", "/api/batches/khong-co/rows/0/explain"),
    ]:
        response = live_client.request(method, path)
        assert response.status_code == 404, path
        assert error(response).code == "batch_not_found", path


def test_close_and_delete_remove_the_batch_immediately(live_client: TestClient):
    first = upload(live_client, csv_bytes(sentences(3))).json()["batch_id"]
    assert live_client.post(f"/api/batches/{first}/close").status_code == 204
    assert live_client.get(f"/api/batches/{first}").status_code == 404
    assert live_client.post(f"/api/batches/{first}/close").status_code == 404  # lần hai: vô hại
    second = upload(live_client, csv_bytes(sentences(3))).json()["batch_id"]
    assert live_client.delete(f"/api/batches/{second}").status_code == 204
    assert live_client.get(f"/api/batches/{second}").status_code == 404


def test_closing_a_finished_batch_frees_its_results(live_client: TestClient):
    batch_id, _ = analyze_and_wait(live_client, csv_bytes(sentences(10)))
    assert live_client.get(f"/api/batches/{batch_id}/results").status_code == 200
    live_client.post(f"/api/batches/{batch_id}/close")
    assert live_client.get(f"/api/batches/{batch_id}/results").status_code == 404


def test_batch_ids_are_not_guessable_by_prefix(live_client: TestClient):
    batch_id = upload(live_client, csv_bytes(sentences(3))).json()["batch_id"]
    assert live_client.get(f"/api/batches/{batch_id[:10]}").status_code == 404


def test_finished_reads_the_result_only_once():
    """Tab đóng ngay sau phép kiểm (kết quả bị giải phóng) không được làm route đọc lại ra None."""
    from edupulse.api.routes.batches import _finished

    class ClosingBatch:
        readonly = False
        state = "done"
        reads = 0

        @property
        def result(self):
            self.reads += 1
            return "kết quả" if self.reads == 1 else None

    batch = ClosingBatch()
    assert _finished(batch) == "kết quả"
    assert batch.reads == 1
