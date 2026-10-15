"""Đọc bộ nhớ tiến trình và cgroup: số hợp lý, và không lỗi khi không chạy trong container."""

from pathlib import Path

import pytest

from edupulse.engine.memory import cgroup_peak_mb, current_rss_mb


def test_rss_is_a_positive_number_of_megabytes():
    rss = current_rss_mb()
    assert 10 < rss < 100_000  # tiến trình Python đang chạy pytest chiếm vài chục MB trở lên


@pytest.mark.skipif(
    not Path("/proc/self/statm").exists(), reason="macOS chỉ có mức đỉnh, không có RSS hiện tại"
)
def test_rss_grows_when_memory_is_allocated_and_touched():
    before = current_rss_mb()
    block = bytearray(64 * 1024 * 1024)
    block[::4096] = b"x" * len(block[::4096])  # chạm từng trang để hệ điều hành cấp thật
    assert current_rss_mb() >= before + 32
    del block


def test_cgroup_peak_is_a_positive_number_or_none_outside_a_container():
    peak = cgroup_peak_mb()
    assert peak is None or peak > 0
