"""Worker rảnh vẫn xoá batch hết hạn, dù không có yêu cầu nào chạm vào kho."""

import time

from edupulse.api import parsing
from edupulse.api.jobs import JobStore
from edupulse.api.limits import PURGE_EVERY_S


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def a_table():
    return parsing.parse_upload("a.csv", "Góp ý\nThầy giải thích rõ ràng.\n".encode())


def wait_until(condition, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return condition()


def test_the_default_idle_purge_is_once_a_minute():
    assert PURGE_EVERY_S == 60.0
    assert JobStore(None, ttl_s=100).purge_every_s == 60.0  # type: ignore[arg-type]


def test_an_expired_upload_is_dropped_by_the_idle_worker_and_its_raw_table_released():
    clock = Clock()
    store = JobStore(None, ttl_s=100, purge_every_s=0.02, clock=clock)  # type: ignore[arg-type]
    store.start()
    try:
        batch = store.create("a.csv", a_table())
        assert batch.table is not None
        clock.now += 101  # quá TTL, và không ai gọi get/count/create từ giờ
        assert wait_until(lambda: not store._batches)
        assert batch.table is None and batch.cancel.is_set()
    finally:
        store.shutdown()


def test_the_idle_purge_keeps_batches_that_are_still_alive():
    clock = Clock()
    store = JobStore(None, ttl_s=100, purge_every_s=0.02, clock=clock)  # type: ignore[arg-type]
    store.start()
    try:
        batch = store.create("a.csv", a_table())
        clock.now += 99
        time.sleep(0.3)  # nhiều chu kỳ dọn
        assert list(store._batches) == [batch.id] and batch.table is not None
    finally:
        store.shutdown()
