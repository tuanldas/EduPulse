"""Hàm dùng chung của các test batch qua HTTP: dựng file, tải lên, chờ chạy xong."""

import csv
import io
import time

from fastapi.testclient import TestClient

# Câu viết tay đủ ba cảm xúc và bốn chủ đề; không lấy từ UIT-VSFC.
SENTENCES = [
    "Thầy giải thích rõ ràng, em nghe một lần là hiểu.",
    "Phòng máy nóng, máy chiếu mờ, quạt kêu to.",
    "Giáo trình cần được cập nhật thêm ví dụ thực tế.",
    "Em không có ý kiến gì thêm về môn học này.",
    "Cô đến lớp đúng giờ và hỗ trợ sinh viên tận tình.",
    "Lịch thi dày đặc, mong nhà trường dàn đều hơn.",
    "Wifi trong giảng đường yếu nên không tra cứu tài liệu được.",
    "Bài giảng quá nhanh, em không theo kịp phần cuối.",
    "Môn học hữu ích và gắn với thực tế công việc.",
    "Thầy hay nghỉ đột xuất mà không báo trước cho lớp.",
]


def csv_bytes(
    rows: list[list[str]], header: tuple[str, ...] = ("Góp ý",), bom: bool = True
) -> bytes:
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(header)
    writer.writerows(rows)
    data = out.getvalue().encode("utf-8")
    return b"\xef\xbb\xbf" + data if bom else data


def sentences(n: int, offset: int = 0) -> list[list[str]]:
    return [[SENTENCES[(offset + i) % len(SENTENCES)]] for i in range(n)]


def upload(client: TestClient, data: bytes, name: str = "gopy.csv"):
    return client.post("/api/batches", files={"file": (name, data, "text/csv")})


def run(client: TestClient, batch_id: str, **body):
    payload = {"text_col": "Góp ý"} | body
    return client.post(f"/api/batches/{batch_id}/run", json=payload)


def wait_for(client: TestClient, batch_id: str, timeout: float = 90.0) -> dict:
    """Hỏi trạng thái tới khi `done` hoặc `failed`."""
    deadline = time.monotonic() + timeout
    status: dict = {}
    while time.monotonic() < deadline:
        response = client.get(f"/api/batches/{batch_id}")
        assert response.status_code == 200, response.text
        status = response.json()
        if status["state"] in ("done", "failed"):
            return status
        time.sleep(0.05)
    raise AssertionError(f"batch chưa xong sau {timeout}s: {status}")


def analyze_and_wait(client: TestClient, data: bytes, **run_body) -> tuple[str, dict]:
    """Tải lên, chạy, chờ xong; trả về (id, trạng thái cuối)."""
    uploaded = upload(client, data)
    assert uploaded.status_code == 201, uploaded.text
    batch_id = uploaded.json()["batch_id"]
    started = run(client, batch_id, **run_body)
    assert started.status_code == 202, started.text
    return batch_id, wait_for(client, batch_id)
