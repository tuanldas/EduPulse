"""Upload song song và file toàn dòng trống: trần số file đang parse, kiểm sức chứa trước khi đọc.

Không cần mô hình: kho batch không dùng bộ phân tích khi chỉ tạo batch.
"""

import threading
import time
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from edupulse.api import parsing
from edupulse.api.errors import ApiError
from edupulse.api.jobs import JobStore
from edupulse.api.limits import MAX_PARALLEL_PARSES
from edupulse.config import MAX_ROWS

CSV = "Góp ý\nThầy giải thích rõ ràng.\nPhòng máy nóng, quạt kêu to.\n".encode()


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_store(**kwargs) -> JobStore:
    kwargs.setdefault("ttl_s", 100)
    return JobStore(None, **kwargs)  # type: ignore[arg-type]


def code_of(call) -> tuple[int, str]:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value.status_code, caught.value.detail["code"]


def table():
    return parsing.parse_upload("a.csv", CSV)


# --- Kho: sức chứa và chỗ parse ------------------------------------------------------------------


def test_a_full_store_says_busy_before_any_file_is_read():
    store = make_store(max_batches=2)
    store.ensure_capacity()
    store.create("a.csv", table())
    store.ensure_capacity()
    store.create("b.csv", table())
    assert code_of(store.ensure_capacity) == (429, "busy")
    assert code_of(lambda: store.create("c.csv", table())) == (429, "busy")  # lưới cuối vẫn còn


def test_expired_batches_free_the_capacity_the_check_looks_at():
    clock = Clock()
    store = make_store(max_batches=1, clock=clock)
    store.create("a.csv", table())
    assert code_of(store.ensure_capacity) == (429, "busy")
    clock.now += 101
    store.ensure_capacity()  # batch cũ đã hết hạn nên có chỗ


def test_only_a_few_files_are_parsed_at_once():
    store = make_store()
    running = peak = 0
    guard = threading.Lock()

    def upload() -> None:
        nonlocal running, peak
        with store.parsing():
            with guard:
                running += 1
                peak = max(peak, running)
            time.sleep(0.05)
            with guard:
                running -= 1

    threads = [threading.Thread(target=upload) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert peak == MAX_PARALLEL_PARSES == 2
    assert running == 0


def test_waiting_too_long_for_a_parse_slot_is_a_429_not_a_hung_request():
    store = make_store(parse_slots=1, parse_wait_s=0.05)

    def take_a_slot() -> None:
        with store.parsing():
            pass

    with store.parsing():
        assert code_of(take_a_slot) == (429, "busy")
    take_a_slot()  # chỗ đã trả lại sau khi người giữ xong


def test_a_failing_parse_gives_its_slot_back():
    store = make_store(parse_slots=1, parse_wait_s=0.05)
    with pytest.raises(RuntimeError), store.parsing():
        raise RuntimeError("file hỏng")
    with store.parsing():
        pass


# --- Route: kiểm sức chứa rồi mới đọc ------------------------------------------------------------


@pytest.fixture
def web(client: TestClient) -> TestClient:
    client.app.state.jobs = make_store(max_batches=1)
    return client


def upload(client: TestClient, data: bytes = CSV):
    return client.post("/api/batches", files={"file": ("gopy.csv", data, "text/csv")})


def test_a_full_store_refuses_the_upload_without_parsing_the_file(web: TestClient, monkeypatch):
    assert upload(web).status_code == 201
    parsed = []
    original = parsing.parse_upload
    monkeypatch.setattr(
        "edupulse.api.routes.batches.parse_upload",
        lambda *args: parsed.append(args) or original(*args),
    )
    response = upload(web)
    assert response.status_code == 429 and response.json()["detail"]["code"] == "busy"
    assert response.headers["retry-after"] == "5"
    assert parsed == []  # không đọc, không parse


def test_the_upload_is_checked_then_parsed_inside_a_parse_slot_then_stored(
    web: TestClient, monkeypatch
):
    store = web.app.state.jobs
    steps: list[str] = []
    original = {
        "ensure_capacity": store.ensure_capacity,
        "parsing": store.parsing,
        "create": store.create,
    }
    monkeypatch.setattr(store, "ensure_capacity", lambda: steps.append("capacity"))

    @contextmanager
    def spy_parsing():
        steps.append("slot taken")
        with original["parsing"]():
            yield
        steps.append("slot released")

    def spy_parse(*args):
        steps.append("parse")
        return parsing.parse_upload(*args)

    def spy_create(*args, **kwargs):
        steps.append("create")
        return original["create"](*args, **kwargs)

    monkeypatch.setattr(store, "parsing", spy_parsing)
    monkeypatch.setattr("edupulse.api.routes.batches.parse_upload", spy_parse)
    monkeypatch.setattr(store, "create", spy_create)
    assert upload(web).status_code == 201
    assert steps == ["capacity", "slot taken", "parse", "slot released", "create"]


# --- CSV toàn dòng trống hoặc toàn dấu phân cách ---------------------------------------------


def test_sniffing_the_delimiter_looks_at_a_bounded_number_of_raw_rows(monkeypatch):
    consumed = 0
    original = parsing._iter_csv

    def counting(text: str, delimiter: str):
        nonlocal consumed
        for row in original(text, delimiter):
            consumed += 1
            yield row

    monkeypatch.setattr(parsing, "_iter_csv", counting)
    parsing.choose_delimiter("\n" * 200_000 + "Góp ý,Nhóm\n")
    assert consumed <= 3 * parsing.SNIFF_RAW_ROWS


def test_a_file_of_blank_lines_is_refused_without_scanning_it_all():
    blank = "\r\n" * (parsing.MAX_BLANK_ROWS + 1)
    started = time.monotonic()
    for text in (f"Góp ý\r\n{blank}hay\r\n", f"{blank}Góp ý\r\nhay\r\n"):  # sau và trước tiêu đề
        with pytest.raises(ApiError) as caught:
            parsing.parse_upload("a.csv", text.encode())
        assert caught.value.detail["code"] == "decode_failed"
        assert "dòng trống" in caught.value.detail["message"]
    assert time.monotonic() - started < 5


def test_as_many_blank_lines_as_the_limit_are_still_fine():
    blank = "\r\n" * parsing.MAX_BLANK_ROWS
    table = parsing.parse_upload("a.csv", f"Góp ý\r\n{blank}hay\r\n".encode())
    assert [r[0] for r in table.rows] == ["hay"]
    assert [(w.code, w.count) for w in table.warnings] == [
        ("blank_rows_skipped", parsing.MAX_BLANK_ROWS)
    ]
    assert parsing.MAX_BLANK_ROWS >= MAX_ROWS  # sheet XLSX tô định dạng dưới dữ liệu vẫn đọc được


def test_a_single_line_of_millions_of_separators_is_refused_before_it_is_built():
    started = time.monotonic()
    with pytest.raises(ApiError) as caught:
        parsing.parse_upload("a.csv", b"Cot A" + b"," * (parsing.MAX_DELIMITERS + 1) + b"\nhay\n")
    assert caught.value.detail["code"] == "decode_failed"
    assert "quá nhiều dấu phân cách" in caught.value.detail["message"]
    assert time.monotonic() - started < 5


def test_a_normal_wide_file_is_not_mistaken_for_a_separator_bomb():
    header = ",".join(f"c{i}" for i in range(200))
    rows = "\n".join(",".join(["x"] * 200) for _ in range(MAX_ROWS))
    table = parsing.parse_upload("a.csv", f"{header}\n{rows}\n".encode())
    assert len(table.columns) == 200 and len(table.rows) == MAX_ROWS
