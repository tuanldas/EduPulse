"""Đọc bộ nhớ của tiến trình và của cgroup (container)."""

import resource
import sys
from pathlib import Path

_MB = 1024 * 1024
_CGROUP_PEAK = Path("/sys/fs/cgroup/memory.peak")


def current_rss_mb() -> float:
    """Bộ nhớ thường trú hiện tại (MB). macOS không có /proc nên trả mức đỉnh của tiến trình."""
    try:
        with open("/proc/self/statm", "rb") as statm:
            resident_pages = int(statm.read().split()[1])
        return round(resident_pages * resource.getpagesize() / _MB, 1)
    except (OSError, ValueError, IndexError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss tính bằng byte trên macOS, KB trên Linux.
        return round(peak / (_MB if sys.platform == "darwin" else 1024), 1)


def cgroup_peak_mb() -> float | None:
    """Đỉnh bộ nhớ của cgroup v2 mà tiến trình đang chạy trong (container); None nếu không có."""
    try:
        return round(int(_CGROUP_PEAK.read_text()) / _MB, 1)
    except (OSError, ValueError):
        return None
