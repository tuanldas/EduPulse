# syntax=docker/dockerfile:1
#
# Image EduPulse: FastAPI phục vụ API và bản web Next.js export tĩnh trên cổng 8000, 1 worker.
#
# Bundle mô hình không nằm trong repo nên phải đưa vào bằng build context tên `artifacts`:
#   make docker-build BUNDLE_VERSION=v1.0 IMAGE=edupulse:v1.0
#   docker compose build            (compose khai sẵn additional_contexts)
# Build cần mạng (npm, PyPI, apt).

ARG NODE_IMAGE=node:24.15.0-trixie-slim
ARG PYTHON_IMAGE=python:3.12-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.22

# ---- web: dựng Next.js thành HTML/JS tĩnh trong /web/out ----
FROM ${NODE_IMAGE} AS web
ENV COREPACK_ENABLE_DOWNLOAD_PROMPT=0 \
    NEXT_TELEMETRY_DISABLED=1 \
    CI=true
WORKDIR /web
RUN corepack enable
COPY apps/web/package.json apps/web/pnpm-lock.yaml apps/web/pnpm-workspace.yaml ./
RUN --mount=type=cache,target=/pnpm-store \
    pnpm install --frozen-lockfile --store-dir /pnpm-store
COPY apps/web/ ./
RUN pnpm build

# ---- deps: môi trường Python chỉ gồm group `app` (không torch, pandas, pyarrow) ----
FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PYTHON_PREFERENCE=only-system
WORKDIR /app
# Lớp phụ thuộc đứng riêng để chỉ đổi mã nguồn thì không cài lại.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group app --no-install-project --no-editable
COPY README.md LICENSE ./
COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group app --no-editable

# ---- runtime ----
FROM ${PYTHON_IMAGE} AS runtime
ARG BUNDLE_VERSION=v1.0
LABEL org.opencontainers.image.title="EduPulse" \
      org.opencontainers.image.licenses="AGPL-3.0-only"

# JRE cho py_vncorenlp (tách từ RDRSegmenter). uid 1000 để khớp quyền ghi khi gắn volume.
RUN apt-get update \
    && apt-get install -y --no-install-recommends default-jre-headless \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 1000 --create-home --shell /usr/sbin/nologin app

ENV JAVA_HOME=/usr/lib/jvm/default-java \
    PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BUNDLE_VERSION=${BUNDLE_VERSION} \
    EDUPULSE_ARTIFACTS_DIR=/app/artifacts \
    EDUPULSE_WEB_DIR=/app/web \
    EDUPULSE_VNCORENLP_DIR=/app/vendor/vncorenlp
WORKDIR /app

# Mã chạy do root sở hữu (chỉ đọc); dữ liệu app đọc thuộc uid 1000 để không phụ thuộc quyền
# file trên máy build.
COPY --from=deps /app/.venv /app/.venv
COPY --chown=1000:1000 vendor/vncorenlp /app/vendor/vncorenlp/
COPY --from=artifacts --chown=1000:1000 . /app/artifacts/
COPY --from=web --chown=1000:1000 /web/out /app/web/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"]
# Đúng 1 worker (trạng thái nằm trong RAM); --no-access-log vì log của uvicorn ghi cả query string.
CMD ["uvicorn", "edupulse.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--proxy-headers", "--no-access-log"]
