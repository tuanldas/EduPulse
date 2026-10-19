"""Kho batch: TTL, giới hạn, hàng đợi, khoá và worker nền (dùng bộ phân tích thật)."""

import random
import threading
import time

import pytest

from edupulse.api.errors import ApiError
from edupulse.api.jobs import JobStore
from edupulse.api.limits import DEMO_BATCH_ID, MAX_GROUPS
from edupulse.api.parsing import parse_upload
from edupulse.demo_data import DemoRowData
from edupulse.engine.service import Analyzer

pytestmark = pytest.mark.model

SENTENCES = [
    "[đã gỡ câu UIT-VSFC].",
    "Phòng máy nóng, máy chiếu mờ.",
    "Giáo trình cần cập nhật thêm.",
    "Em không có ý kiến gì thêm.",
    "Cô đến lớp đúng giờ và hỗ trợ sinh viên tận tình.",
    "Lịch thi dày đặc, mong dàn đều hơn.",
]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def table(n: int = 6, group: bool = False):
    lines = ["Góp ý,Nhóm" if group else "Góp ý"]
    for i in range(n):
        text = f'"{SENTENCES[i % len(SENTENCES)]}"'
        lines.append(f"{text},Nhóm {i % 3}" if group else text)
    return parse_upload("a.csv", "\n".join(lines).encode("utf-8"))


def code_of(call) -> tuple[int, str]:
    with pytest.raises(ApiError) as caught:
        call()
    return caught.value.status_code, caught.value.detail["code"]


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(analyzer: Analyzer, clock: Clock):
    """Kho chưa chạy worker nên batch được `run` nằm yên ở trạng thái `queued`."""
    created = JobStore(analyzer, ttl_s=100, clock=clock)
    yield created
    created.shutdown()


# --- Tạo, tra cứu, xoá ---------------------------------------------------------------------------


def test_batches_get_unguessable_urlsafe_ids(store: JobStore):
    ids = {store.create("a.csv", table()).id for _ in range(5)}
    assert len(ids) == 5
    assert all(len(i) >= 22 and i.replace("-", "").replace("_", "").isalnum() for i in ids)
    assert DEMO_BATCH_ID not in ids


def test_unknown_ids_are_404(store: JobStore):
    assert code_of(lambda: store.get("khong-co")) == (404, "batch_not_found")
    assert code_of(lambda: store.remove("khong-co")) == (404, "batch_not_found")


def test_removing_a_batch_drops_its_data_and_cancels_it(store: JobStore):
    batch = store.create("a.csv", table())
    store.remove(batch.id)
    assert batch.table is None and batch.cancel.is_set()
    assert code_of(lambda: store.get(batch.id)) == (404, "batch_not_found")
    assert code_of(lambda: store.remove(batch.id)) == (404, "batch_not_found")


# --- TTL và heartbeat ----------------------------------------------------------------------------


def test_a_batch_expires_after_the_ttl_without_access(store: JobStore, clock: Clock):
    batch = store.create("a.csv", table())
    clock.now += 99
    assert store.get(batch.id) is batch
    clock.now += 101
    assert code_of(lambda: store.get(batch.id)) == (404, "batch_not_found")
    assert store.count() == 0


def test_every_access_extends_the_life_of_the_batch(store: JobStore, clock: Clock):
    batch = store.create("a.csv", table())
    for _ in range(10):  # tổng 800 s, gấp tám lần TTL
        clock.now += 80
        assert store.get(batch.id) is batch  # heartbeat chính là một lần tra cứu
    clock.now += 101
    assert code_of(lambda: store.get(batch.id)) == (404, "batch_not_found")


def test_expired_batch_data_is_released(store: JobStore, clock: Clock):
    batch = store.create("a.csv", table())
    clock.now += 500
    store.count()
    assert batch.table is None and batch.cancel.is_set()


# --- Giới hạn -------------------------------------------------------------------------------------


def test_at_most_eight_batches_then_busy_until_one_goes(store: JobStore):
    batches = [store.create("a.csv", table()) for _ in range(8)]
    status, code = code_of(lambda: store.create("a.csv", table()))
    assert (status, code) == (429, "busy")
    store.remove(batches[0].id)
    assert store.create("a.csv", table())


def test_expired_batches_free_their_slot(store: JobStore, clock: Clock):
    for _ in range(8):
        store.create("a.csv", table())
    clock.now += 500
    assert store.create("a.csv", table())


def test_busy_error_asks_the_client_to_retry_later(store: JobStore):
    for _ in range(8):
        store.create("a.csv", table())
    with pytest.raises(ApiError) as caught:
        store.create("a.csv", table())
    assert caught.value.headers == {"Retry-After": "5"}


def test_run_queue_holds_at_most_three_active_batches(analyzer: Analyzer, clock: Clock):
    store = JobStore(analyzer, ttl_s=100, clock=clock)  # không start: batch ở lại `queued`
    batches = [store.create("a.csv", table()) for _ in range(4)]
    for batch in batches[:3]:
        assert store.run(batch, text_col="Góp ý", group_col=None, budget=10).state == "queued"
    status, code = code_of(
        lambda: store.run(batches[3], text_col="Góp ý", group_col=None, budget=10)
    )
    assert (status, code) == (429, "busy")
    store.remove(batches[0].id)  # một chỗ trống
    assert store.run(batches[3], text_col="Góp ý", group_col=None, budget=10).state == "queued"


# --- Kiểm tra tham số chạy ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"text_col": "Không có"}, (422, "invalid_column")),
        ({"text_col": "Góp ý", "group_col": "Không có"}, (422, "invalid_column")),
        ({"text_col": "Góp ý", "group_col": "Góp ý"}, (422, "invalid_column")),
        ({"text_col": "Nhóm", "group_col": "Nhóm"}, (422, "invalid_column")),
        ({"text_col": "Góp ý", "budget": 7}, (422, "invalid_budget")),
    ],
)
def test_run_validates_the_chosen_columns_and_budget(store: JobStore, kwargs, expected):
    batch = store.create("a.csv", table(group=True))
    params = {"text_col": "Góp ý", "group_col": None, "budget": 10} | kwargs
    assert code_of(lambda: store.run(batch, **params)) == expected
    assert batch.state == "uploaded"  # lỗi tham số không làm đổi trạng thái


def test_a_text_column_without_any_text_is_refused(store: JobStore):
    data = "Góp ý,Ghi chú\n,một\n,hai\n".encode()
    batch = store.create("a.csv", parse_upload("a.csv", data))
    assert code_of(lambda: store.run(batch, text_col="Góp ý", group_col=None, budget=10)) == (
        422,
        "no_text_column",
    )


def test_too_many_group_values_are_refused_with_the_count(store: JobStore):
    lines = ["Góp ý,Nhóm", *(f"câu {i},nhóm {i}" for i in range(MAX_GROUPS + 1))]
    batch = store.create("a.csv", parse_upload("a.csv", "\n".join(lines).encode()))
    with pytest.raises(ApiError) as caught:
        store.run(batch, text_col="Góp ý", group_col="Nhóm", budget=10)
    assert caught.value.detail["code"] == "too_many_groups"
    assert caught.value.detail["params"] == {"n_groups": MAX_GROUPS + 1, "max_groups": MAX_GROUPS}


def test_exactly_the_group_limit_is_fine(store: JobStore):
    lines = ["Góp ý,Nhóm", *(f"câu {i},nhóm {i}" for i in range(MAX_GROUPS))]
    batch = store.create("a.csv", parse_upload("a.csv", "\n".join(lines).encode()))
    assert store.run(batch, text_col="Góp ý", group_col="Nhóm", budget=10).state == "queued"


def test_a_batch_cannot_be_run_twice_but_a_failed_one_can_be_retried(store: JobStore):
    batch = store.create("a.csv", table())
    store.run(batch, text_col="Góp ý", group_col=None, budget=10)
    assert code_of(lambda: store.run(batch, text_col="Góp ý", group_col=None, budget=10)) == (
        409,
        "already_running",
    )
    batch.state = "failed"
    batch.error_code = "internal_error"
    snapshot = store.run(batch, text_col="Góp ý", group_col=None, budget=20)
    assert (snapshot.state, snapshot.error_code, snapshot.progress) == ("queued", None, 0.0)


def test_illustrative_groups_follow_the_sample_flag_or_the_group_column_name(store: JobStore):
    sample = store.create("mau.csv", table(group=True), kind="sample", illustrative=True)
    store.run(sample, text_col="Góp ý", group_col="Nhóm", budget=10)
    assert sample.illustrative_groups
    data = "Góp ý,Nhóm (minh hoạ)\nhay,A\ntệ,B\n".encode()
    marked = store.create("a.csv", parse_upload("a.csv", data))
    assert not marked.illustrative_groups
    store.run(marked, text_col="Góp ý", group_col="Nhóm (minh hoạ)", budget=10)
    assert marked.illustrative_groups
    plain = store.create("a.csv", table(group=True))
    store.run(plain, text_col="Góp ý", group_col="Nhóm", budget=10)
    assert not plain.illustrative_groups


# --- Batch demo -----------------------------------------------------------------------------------


def demo_rows() -> list[DemoRowData]:
    return [
        DemoRowData(
            display_text=f"Câu {i}.",
            model_text=f"câu {i} .",
            sentiment="negative",
            topic="facility",
            sentiment_confidence=0.9,
            topic_confidence=0.9,
            n_words=3,
        )
        for i in range(4)
    ]


def test_demo_batch_is_done_readonly_and_never_expires(store: JobStore, clock: Clock):
    assert code_of(lambda: store.get(DEMO_BATCH_ID)) == (404, "batch_not_found")  # chưa nạp
    store.set_demo(demo_rows())
    clock.now += 10_000
    demo = store.get(DEMO_BATCH_ID)
    assert demo.readonly and demo.state == "done" and len(demo.insight_rows) == 4
    assert all(r.group is None and r.flagged is None for r in demo.insight_rows)
    store.remove(DEMO_BATCH_ID)  # không xoá được
    assert store.get(DEMO_BATCH_ID) is demo
    assert store.count() == 0  # không tính vào giới hạn
    assert code_of(lambda: store.run(demo, text_col="x", group_col=None, budget=10)) == (
        409,
        "batch_readonly",
    )


# --- Worker nền -----------------------------------------------------------------------------------


def wait_for(store: JobStore, batch, states: tuple[str, ...], timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = store.snapshot(batch)
        if snapshot.state in states:
            return snapshot
        time.sleep(0.05)
    raise AssertionError(f"batch không tới {states} sau {timeout}s: {store.snapshot(batch)}")


@pytest.fixture
def live_store(analyzer: Analyzer):
    created = JobStore(analyzer, ttl_s=600)
    created.start()
    yield created
    created.shutdown()


def test_worker_finishes_a_batch_and_drops_the_raw_data(live_store: JobStore):
    batch = live_store.create("a.csv", table(30, group=True))
    live_store.run(batch, text_col="Góp ý", group_col="Nhóm", budget=10)
    snapshot = wait_for(live_store, batch, ("done", "failed"))
    assert snapshot.state == "done" and snapshot.progress == 1.0 and snapshot.step is None
    assert batch.table is None  # dữ liệu thô đã được giải phóng
    assert batch.result is not None and len(batch.result.rows) == 30
    assert {r.group for r in batch.result.rows} == {"Nhóm 0", "Nhóm 1", "Nhóm 2"}


def test_progress_never_goes_backwards_and_steps_come_in_order(live_store: JobStore):
    batch = live_store.create("a.csv", table(120))
    live_store.run(batch, text_col="Góp ý", group_col=None, budget=20)
    seen, steps = [], []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        snapshot = live_store.snapshot(batch)
        seen.append(snapshot.progress)
        if snapshot.step and (not steps or steps[-1] != snapshot.step):
            steps.append(snapshot.step)
        if snapshot.state == "done":
            break
        time.sleep(0.01)
    assert seen == sorted(seen) and seen[-1] == 1.0
    order = ["prepare", "predict", "flag", "explain"]
    assert steps == sorted(steps, key=order.index)


def test_batches_run_one_after_another(live_store: JobStore):
    batches = [live_store.create("a.csv", table(40)) for _ in range(3)]
    for batch in batches:
        live_store.run(batch, text_col="Góp ý", group_col=None, budget=10)
    running_together = 0
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline and any(b.state != "done" for b in batches):
        running_together = max(running_together, sum(b.state == "running" for b in batches))
        time.sleep(0.01)
    assert all(b.state == "done" for b in batches)
    assert running_together == 1


def test_closing_a_running_batch_stops_it_and_the_worker_keeps_serving(live_store: JobStore):
    big = live_store.create("a.csv", table(2000))
    live_store.run(big, text_col="Góp ý", group_col=None, budget=10)
    wait_for(live_store, big, ("running",))
    started = time.monotonic()
    live_store.remove(big.id)
    follower = live_store.create("a.csv", table(10))
    live_store.run(follower, text_col="Góp ý", group_col=None, budget=10)
    assert wait_for(live_store, follower, ("done",)).state == "done"
    assert time.monotonic() - started < 20  # không phải chờ hết 2.000 câu của batch đã đóng
    assert big.result is None


def test_a_batch_whose_rows_are_all_empty_fails_with_a_stable_code(live_store: JobStore):
    data = "Góp ý\n0912345678\nhttps://example.com/x\n".encode()
    batch = live_store.create("a.csv", parse_upload("a.csv", data))
    live_store.run(batch, text_col="Góp ý", group_col=None, budget=10)
    snapshot = wait_for(live_store, batch, ("failed", "done"))
    assert (snapshot.state, snapshot.error_code) == ("failed", "no_content")
    assert batch.table is not None  # còn dữ liệu để chạy lại với cột khác


def test_shutdown_stops_the_worker_thread(analyzer: Analyzer):
    store = JobStore(analyzer, ttl_s=100)
    store.start()
    worker = store._thread
    assert worker is not None and worker.is_alive()
    store.shutdown()
    assert not worker.is_alive()


# --- Đồng thời ------------------------------------------------------------------------------------


def test_concurrent_requests_never_deadlock_or_corrupt_the_store(analyzer: Analyzer):
    store = JobStore(analyzer, ttl_s=0.05, max_batches=8, max_queue=3)
    failures: list[BaseException] = []

    def hammer(seed: int) -> None:
        rng = random.Random(seed)
        mine: list = []
        try:
            for _ in range(300):
                action = rng.choice(["create", "get", "run", "remove", "count"])
                try:
                    if action == "create":
                        mine.append(store.create("a.csv", table(3)))
                    elif action == "get" and mine:
                        store.get(rng.choice(mine).id)
                    elif action == "run" and mine:
                        store.run(rng.choice(mine), text_col="Góp ý", group_col=None, budget=10)
                    elif action == "remove" and mine:
                        store.remove(mine.pop().id)
                    elif action == "count":
                        assert 0 <= store.count() <= 8
                except ApiError:
                    pass  # 404/409/429 là kết quả hợp lệ khi chạy đua
        except BaseException as exc:  # noqa: BLE001 - gom lại để báo ở luồng chính
            failures.append(exc)

    threads = [threading.Thread(target=hammer, args=(seed,)) for seed in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not any(thread.is_alive() for thread in threads), "bế tắc"
    assert failures == []
