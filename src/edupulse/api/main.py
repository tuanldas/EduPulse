"""Ứng dụng FastAPI: API `/api/*` và bản web export tĩnh phục vụ chung một origin.

Bản web được gắn bằng `app.frontend()`: route tĩnh có ưu tiên thấp, chỉ được xét khi không route
API nào khớp. Nhờ vậy `/api/sai` không rơi vào trang 404 HTML của web mà trả JSON 404, còn gọi
sai method vào một endpoint có thật vẫn ra 405 kèm `Allow`.

Chạy với đúng 1 worker (trạng thái nằm trong RAM) và `--no-access-log` (log của uvicorn ghi cả
query string); nhật ký truy cập do `AccessLogMiddleware` đảm nhiệm.
"""

import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from edupulse.api.access_log import AccessLogMiddleware
from edupulse.api.errors import http_error_handler, install_error_handlers
from edupulse.api.paths import is_api_path
from edupulse.api.routes import analyze, meta
from edupulse.api.security_headers import SecurityHeadersMiddleware
from edupulse.config import Settings, load_settings
from edupulse.engine.service import Analyzer

logger = logging.getLogger("edupulse.api")


def _configure_logging() -> None:
    """Uvicorn chỉ cấu hình logger của nó; gắn handler cho `edupulse.*` để log hiện ra stderr."""
    root = logging.getLogger("edupulse")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    root.setLevel(logging.INFO)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Nạp mô hình trước khi nhận kết nối; thiếu hoặc lệch bundle thì dừng với thông báo rõ."""
    settings: Settings = app.state.settings
    # Nạp mất vài giây (băm tệp, ONNX, JVM): chạy ngoài vòng lặp sự kiện.
    app.state.analyzer = await run_in_threadpool(Analyzer.load, settings)
    manifest = app.state.analyzer.bundle.manifest
    logger.info(
        "Sẵn sàng: bundle %s (%s, mô hình %s), web %s",
        manifest.bundle_version,
        manifest.stage,
        manifest.model_version,
        settings.web_dir,
    )
    yield


def _web_not_found_handler(
    page: Path,
) -> Callable[[Request, StarletteHTTPException], Awaitable[Response]]:
    """Trình duyệt mở trang không có thì nhận `404.html` do Next sinh; còn lại là JSON."""

    async def handler(request: Request, exc: StarletteHTTPException) -> Response:
        wants_page = request.method in ("GET", "HEAD") and not is_api_path(request.scope["path"])
        if wants_page and page.is_file():
            return FileResponse(page, status_code=404)
        return await http_error_handler(request, exc)

    return handler


def _serve_web(app: FastAPI, web_dir: Path) -> None:
    """Phục vụ bản Next.js export (`trailingSlash`): `/batch` chuyển hướng sang `/batch/`."""
    if not web_dir.is_dir():
        logger.warning(
            "Chưa có bản web export tại %s, chỉ phục vụ API (`pnpm --dir apps/web build` để dựng).",
            web_dir,
        )
        return
    # fallback=None: file thiếu thì ném 404 để handler bên dưới chọn JSON (API) hay trang HTML.
    app.frontend("/", directory=web_dir, fallback=None, check_dir=True)
    app.add_exception_handler(404, _web_not_found_handler(web_dir / "404.html"))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    _configure_logging()

    app = FastAPI(
        title="EduPulse API",
        lifespan=_lifespan,
        # Tài liệu nằm dưới /api để route web ở "/" không đụng tên trang.
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
        swagger_ui_oauth2_redirect_url=None,
        # Tên hàm làm operationId, để kiểu TypeScript sinh ra có tên ổn định.
        generate_unique_id_function=lambda route: route.name,
    )
    app.state.settings = settings

    install_error_handlers(app)
    app.include_router(meta.router)
    app.include_router(analyze.router)
    _serve_web(app, settings.web_dir)

    # Middleware thêm sau nằm ngoài: nhật ký bọc tất cả, header an toàn phủ cả phản hồi CORS.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
            allow_credentials=False,
            max_age=600,
        )
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(AccessLogMiddleware)
    return app


app = create_app()
