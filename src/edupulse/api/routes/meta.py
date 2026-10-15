"""Endpoint mô tả trạng thái dịch vụ."""

import resource
import sys

from fastapi import APIRouter, Request

from edupulse.api.schemas import Health

router = APIRouter(prefix="/api", tags=["meta"])

_MB = 1024 * 1024


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


# `async def` để health check không phải chờ luồng trong threadpool đang bận suy luận.
@router.get("/health")
async def health(request: Request) -> Health:
    settings = request.app.state.settings
    return Health(
        status="ok",
        bundle_version=settings.bundle_version,
        ner_enabled=settings.ner,
        rss_mb=current_rss_mb(),
    )
