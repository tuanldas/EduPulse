"""Xuất CSV: BOM, chống formula injection, nhãn sửa đúng dòng, cột cho huấn luyện lại."""

import csv
import hashlib
import io

import pytest
from batch_helpers import SENTENCES, analyze_and_wait, csv_bytes, sentences, upload
from fastapi.testclient import TestClient

from edupulse.api import export
from edupulse.api.export import (
    ALL_HEADER,
    CORRECTED_HEADER,
    DANGEROUS_LEADS,
    guard_cell,
    unguard_cell,
)
from edupulse.labels import LABELS

# --- Chống formula injection (không cần mô hình) --------------------------------------------------


@pytest.mark.parametrize("lead", DANGEROUS_LEADS)
def test_every_dangerous_lead_gets_a_quote_in_front(lead: str):
    assert guard_cell(f"{lead}cmd|' /C calc'!A0") == f"'{lead}cmd|' /C calc'!A0"


def test_the_full_width_variants_are_covered_too():
    assert {"＝", "＋", "－", "＠"} <= set(DANGEROUS_LEADS)
    assert guard_cell("＝1+1") == "'＝1+1"


@pytest.mark.parametrize(
    "text", ["", "Thầy dạy hay", "5 điểm - rất tốt", "email a@b.vn", " =1+1", "'=1"]
)
def test_other_cells_are_left_alone(text: str):
    assert guard_cell(text) == text


@pytest.mark.parametrize("lead", DANGEROUS_LEADS)
def test_unguard_undoes_guard(lead: str):
    cell = f"{lead}abc"
    assert unguard_cell(guard_cell(cell)) == cell
    assert unguard_cell("Thầy dạy hay") == "Thầy dạy hay"
    assert unguard_cell("'Thầy") == "'Thầy"  # dấu nháy không đứng trước ký tự nguy hiểm: giữ


def test_writer_guards_every_cell_and_emits_bom_and_crlf():
    data = export._write([["=SUM(A1)", "bình thường"], ["@x", 'có "nháy", phẩy\nvà xuống dòng']])
    assert data.startswith(b"\xef\xbb\xbf") and b"\r\n" in data
    parsed = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert parsed[0] == ["'=SUM(A1)", "bình thường"]
    assert parsed[1] == ["'@x", 'có "nháy", phẩy\nvà xuống dòng']


# --- Qua HTTP với mô hình thật --------------------------------------------------------------------


def export_csv(client: TestClient, batch_id: str, **body):
    return client.post(f"/api/batches/{batch_id}/export", json=body)


def table_of(response) -> list[list[str]]:
    assert response.status_code == 200, response.text
    return list(csv.reader(io.StringIO(response.content.decode("utf-8-sig"), newline="")))


@pytest.fixture
def finished(live_client: TestClient):
    """Batch 60 dòng có cột nhóm, một dòng chỉ có số điện thoại, đã phân tích xong."""
    rows = [[text, f"Nhóm {i % 3}"] for i, (text,) in enumerate(sentences(60))]
    rows[10] = ["0912345678", "Nhóm 1"]
    data = csv_bytes(rows, header=("Góp ý", "Môn học"))
    batch_id, _ = analyze_and_wait(live_client, data, group_col="Môn học", review_budget=20)
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    return batch_id, results


@pytest.mark.model
def test_export_all_is_an_excel_friendly_csv_with_a_row_per_input_row(
    live_client: TestClient, finished
):
    batch_id, results = finished
    response = export_csv(live_client, batch_id)
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert 'filename="edupulse-ket-qua.csv"' in response.headers["content-disposition"]
    assert response.content.startswith(b"\xef\xbb\xbf")
    table = table_of(response)
    assert table[0] == [*ALL_HEADER, "Môn học"]
    assert len(table) - 1 == results["summary"]["n_rows"] == 60
    assert [int(r[0]) for r in table[1:]] == list(range(1, 61))  # STT = số dòng trong file gốc


@pytest.mark.model
def test_export_all_repeats_what_the_results_endpoint_says(live_client: TestClient, finished):
    batch_id, results = finished
    body = table_of(export_csv(live_client, batch_id))[1:]
    by_stt = {int(r[0]): r for r in body}
    vi_sentiment = {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
    vi_topic = {
        "lecturer": "Giảng viên",
        "training_program": "Chương trình",
        "facility": "Cơ sở vật chất",
        "others": "Khác",
    }
    for row in results["rows"]:
        line = by_stt[row["source_row"]]
        assert line[1] == row["text"]
        assert line[2] == vi_sentiment[row["sentiment"]]
        assert line[3] == str(round(row["sentiment_confidence"] * 100))
        assert line[4] == vi_topic[row["topic"]]
        assert line[5] == str(round(row["topic_confidence"] * 100))
        assert line[6] == ("Có" if row["needs_review"] else "Không")
        assert line[7] == "; ".join(r["message"] for r in row["reasons"])
        assert (
            line[8] == {"flag": "Gắn cờ", "audit": "Kiểm tra ngẫu nhiên", "none": ""}[row["source"]]
        )
        assert line[9] == line[10] == ""  # chưa sửa gì
        assert line[11] == row["group"]
    flagged = sum(r[6] == "Có" for r in body)
    assert flagged == results["summary"]["n_flagged"]


@pytest.mark.model
def test_rows_without_content_are_kept_with_a_note(live_client: TestClient, finished):
    batch_id, _ = finished
    skipped = table_of(export_csv(live_client, batch_id))[11]  # STT 11 = dòng chỉ có số điện thoại
    assert skipped[0] == "11" and skipped[1] == "[SĐT]"
    assert skipped[2:7] == ["", "", "", "", ""]
    assert skipped[7] == "Không có nội dung để phân tích" and skipped[11] == "Nhóm 1"


@pytest.mark.model
def test_overrides_land_on_the_right_rows_and_only_when_they_change_something(
    live_client: TestClient, finished
):
    batch_id, results = finished
    rows = {r["index"]: r for r in results["rows"]}
    a, b, c = (rows[0], rows[1], rows[2])
    flip = {"negative": "positive", "neutral": "negative", "positive": "negative"}
    other_topic = {
        "lecturer": "facility",
        "training_program": "lecturer",
        "facility": "others",
        "others": "lecturer",
    }
    overrides = [
        {"index": a["index"], "sentiment": flip[a["sentiment"]]},
        {"index": b["index"], "topic": other_topic[b["topic"]], "sentiment": flip[b["sentiment"]]},
        {"index": c["index"], "sentiment": c["sentiment"], "topic": c["topic"]},  # chỉ xác nhận
    ]
    table = table_of(export_csv(live_client, batch_id, overrides=overrides))
    by_stt = {int(r[0]): r for r in table[1:]}
    vi_sentiment = {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
    vi_topic = {
        "lecturer": "Giảng viên",
        "training_program": "Chương trình",
        "facility": "Cơ sở vật chất",
        "others": "Khác",
    }
    line_a = by_stt[a["source_row"]]
    assert line_a[9] == vi_sentiment[flip[a["sentiment"]]] and line_a[10] == ""
    line_b = by_stt[b["source_row"]]
    assert line_b[9] == vi_sentiment[flip[b["sentiment"]]]
    assert line_b[10] == vi_topic[other_topic[b["topic"]]]
    line_c = by_stt[c["source_row"]]
    assert line_c[9] == line_c[10] == ""  # trùng nhãn dự đoán: không tính là sửa
    corrected_cells = sum(bool(r[9] or r[10]) for r in table[1:])
    assert corrected_cells == 2
    assert line_a[2] == vi_sentiment[a["sentiment"]]  # cột nhãn dự đoán giữ nguyên


@pytest.mark.model
def test_corrected_file_has_machine_columns_and_only_the_changed_rows(
    live_client: TestClient, finished
):
    batch_id, results = finished
    rows = {r["index"]: r for r in results["rows"]}
    a, b = rows[0], rows[5]
    overrides = [
        {"index": 0, "sentiment": "positive" if a["sentiment"] != "positive" else "negative"},
        {"index": 5, "topic": "others" if b["topic"] != "others" else "lecturer"},
    ]
    response = export_csv(live_client, batch_id, kind="corrected", overrides=overrides)
    assert 'filename="edupulse-nhan-da-sua.csv"' in response.headers["content-disposition"]
    table = table_of(response)
    assert table[0] == list(CORRECTED_HEADER)
    assert len(table) == 3
    health = live_client.get("/api/health").json()
    for line, original, override in zip(table[1:], (a, b), overrides, strict=True):
        record = dict(zip(CORRECTED_HEADER, line, strict=True))
        assert record["text_masked"] == original["text"]
        assert record["model_version"] == health["model_version"]
        assert record["pred_sentiment"] == original["sentiment"]
        assert record["pred_topic"] == original["topic"]
        assert float(record["conf_sentiment"]) == pytest.approx(
            original["sentiment_confidence"], abs=1e-4
        )
        assert record["label_sentiment"] == override.get("sentiment", original["sentiment"])
        assert record["label_topic"] == override.get("topic", original["topic"])
        assert record["source"] == original["source"] and record["reviewed"] == "1"
        assert record["id_hash"] == hashlib.sha256(record["text_model"].encode()).hexdigest()[:16]
        assert record["text_model"] and record["text_model"] == record["text_model"].lower()


@pytest.mark.model
def test_the_retrain_script_reads_the_corrected_file_back(
    live_client: TestClient, finished, tmp_path
):
    pytest.importorskip("pandas", reason="script huấn luyện lại cần group train")
    from ml import retrain

    batch_id, results = finished
    original = next(r for r in results["rows"] if r["index"] == 0)
    fixed = "positive" if original["sentiment"] != "positive" else "negative"
    path = tmp_path / "edupulse-nhan-da-sua.csv"
    response = export_csv(
        live_client, batch_id, kind="corrected", overrides=[{"index": 0, "sentiment": fixed}]
    )
    path.write_bytes(response.content)
    feedback = retrain.read_feedback(path)
    assert len(feedback) == 1
    assert LABELS["sentiment"][feedback["label_s"].iloc[0]] == fixed
    assert LABELS["topic"][feedback["label_t"].iloc[0]] == original["topic"]


@pytest.mark.model
def test_corrected_file_without_real_changes_is_just_a_header(live_client: TestClient, finished):
    batch_id, results = finished
    row = results["rows"][0]
    same = [{"index": row["index"], "sentiment": row["sentiment"]}]
    table = table_of(export_csv(live_client, batch_id, kind="corrected", overrides=same))
    assert table == [list(CORRECTED_HEADER)]
    assert table_of(export_csv(live_client, batch_id, kind="corrected")) == [list(CORRECTED_HEADER)]


@pytest.mark.model
def test_corrected_file_never_carries_personal_data(live_client: TestClient):
    texts = [
        "Thầy Nam dạy rất nhiệt tình, số 0912345678 để hỏi bài.",
        "Liên hệ sv.test@example.com nếu cần tài liệu nhé.",
        *[text for (text,) in sentences(10)],
    ]
    batch_id, _ = analyze_and_wait(live_client, csv_bytes([[t] for t in texts]))
    overrides = [{"index": i, "sentiment": "neutral"} for i in range(len(texts))]
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    overrides = [
        {
            "index": r["index"],
            "sentiment": "positive" if r["sentiment"] != "positive" else "negative",
        }
        for r in results["rows"]
    ]
    for kind in ("all", "corrected"):
        response = export_csv(live_client, batch_id, kind=kind, overrides=overrides)
        for secret in ("0912345678", "sv.test@example.com", "Nam dạy"):
            assert secret not in response.text, (kind, secret)
    first = table_of(export_csv(live_client, batch_id, kind="corrected", overrides=overrides))[1]
    record = dict(zip(CORRECTED_HEADER, first, strict=True))
    assert "[TÊN]" in record["text_masked"] and "wzjwz" in record["text_model"]


@pytest.mark.model
def test_nothing_in_the_file_can_start_a_formula(live_client: TestClient):
    evil = [
        '=HYPERLINK("http://x","bấm") thầy dạy hay',
        "+1+1 phòng nóng",
        "-2 điểm vì máy chiếu mờ",
        "@SUM(1) giáo trình cũ",
    ]
    rows = [[text, f"={i}+cmd"] for i, text in enumerate(evil)]
    rows += [[text, "Nhóm A"] for (text,) in sentences(20)]
    data = csv_bytes(rows, header=("Góp ý", "=Nhóm"))
    batch_id, _ = analyze_and_wait(live_client, data, group_col="=Nhóm", review_budget=30)
    results = live_client.get(f"/api/batches/{batch_id}/results").json()
    overrides = [
        {"index": r["index"], "sentiment": "neutral" if r["sentiment"] != "neutral" else "positive"}
        for r in results["rows"]
    ]
    for kind in ("all", "corrected"):
        table = table_of(export_csv(live_client, batch_id, kind=kind, overrides=overrides))
        assert len(table) > 1
        for line in table:
            for cell in line:
                assert not cell.startswith(DANGEROUS_LEADS), (kind, cell)
    first = table_of(export_csv(live_client, batch_id))[1]
    assert first[1].startswith("'=")  # được giữ nguyên nội dung, chỉ thêm dấu nháy
    assert unguard_cell(first[1]).startswith("=HYPERLINK")


@pytest.mark.model
def test_vietnamese_text_survives_the_round_trip(live_client: TestClient):
    texts = [t for t in SENTENCES]
    batch_id, _ = analyze_and_wait(live_client, csv_bytes([[t] for t in texts]))
    table = table_of(export_csv(live_client, batch_id))
    assert [line[1] for line in table[1:]] == texts


# --- Lỗi ------------------------------------------------------------------------------------------


@pytest.mark.model
@pytest.mark.parametrize(
    "overrides",
    [
        [{"index": 999, "sentiment": "negative"}],
        [{"index": 0, "sentiment": "negative"}, {"index": 0, "topic": "others"}],
    ],
    ids=["out-of-range", "duplicate"],
)
def test_invalid_overrides_are_rejected(live_client: TestClient, finished, overrides):
    batch_id, _ = finished
    response = export_csv(live_client, batch_id, overrides=overrides)
    assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_override"


@pytest.mark.model
def test_overriding_a_row_that_has_no_label_is_rejected(live_client: TestClient, finished):
    batch_id, results = finished
    skipped = results["skipped"][0]["index"]
    response = export_csv(
        live_client, batch_id, overrides=[{"index": skipped, "sentiment": "positive"}]
    )
    assert response.status_code == 422 and response.json()["detail"]["code"] == "invalid_override"


@pytest.mark.model
@pytest.mark.parametrize(
    "body",
    [
        {"kind": "everything"},
        {"overrides": [{"index": -1}]},
        {"overrides": [{"index": 0, "sentiment": "angry"}]},
        {"x": 1},
    ],
)
def test_malformed_requests_are_validation_errors(live_client: TestClient, finished, body):
    batch_id, _ = finished
    response = export_csv(live_client, batch_id, **body)
    assert response.status_code == 422 and response.json()["detail"]["code"] == "validation_error"


@pytest.mark.model
def test_export_needs_a_finished_batch_that_exists(live_client: TestClient):
    unknown = export_csv(live_client, "khong-co")
    assert unknown.status_code == 404 and unknown.json()["detail"]["code"] == "batch_not_found"
    batch_id = upload(live_client, csv_bytes(sentences(3))).json()["batch_id"]
    early = export_csv(live_client, batch_id)
    assert early.status_code == 409 and early.json()["detail"]["code"] == "not_done"


@pytest.mark.model
def test_the_demo_batch_cannot_be_exported(live_client: TestClient):
    if not live_client.app.state.demo.available:
        pytest.skip("không có dữ liệu demo — chạy `make demo-data`")
    response = export_csv(live_client, "demo")
    assert response.status_code == 409 and response.json()["detail"]["code"] == "batch_readonly"
