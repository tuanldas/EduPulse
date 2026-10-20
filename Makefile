# EduPulse — các lệnh tái lập. Xem `make help`.
# Chạy được với GNU Make 3.81, bản mặc định của macOS.
#
# Mọi lệnh Python đi qua `uv run`. Đường dẫn dữ liệu, model, bundle lấy từ edupulse.config
# (theo EDUPULSE_HOME), nên lệnh không ghi cứng data/, models/, outputs/.
# Lệnh nối HF/GitHub không nằm ở đây: chúng ra ngoài máy nên chạy tay khi đến mốc đẩy.

THIS_MAKEFILE := $(lastword $(MAKEFILE_LIST))
.DEFAULT_GOAL := help

# Cấu hình cục bộ trong `.env`: cú pháp KEY=value, không dấu nháy, không `$`, không chú thích
# cuối dòng. Giá trị trong `.env` thắng biến môi trường cùng tên; muốn đổi tạm thì truyền
# `make VAR=giá_trị`. `export` đưa mọi biến sang lệnh con (uvicorn, pnpm, script ml/).
-include .env
export

# Biến để trống trong `.env` (vd. EDUPULSE_HOME=) coi như chưa đặt.
BUNDLE_VERSION := $(or $(strip $(BUNDLE_VERSION)),v1.1)
EDUPULSE_HOME  := $(or $(strip $(EDUPULSE_HOME)),$(CURDIR))
IMAGE          ?= edupulse:$(BUNDLE_VERSION)
# Bundle đưa vào image qua build context; khớp Settings.bundle_dir trong src/edupulse/config.py.
BUNDLE_DIR     := $(or $(strip $(EDUPULSE_ARTIFACTS_DIR)),$(EDUPULSE_HOME)/artifacts/$(BUNDLE_VERSION))
# Chỉ truyền cờ build dữ liệu demo khi có đặt EDUPULSE_BUILD_DEMO (0 để bỏ stage demo).
BUILD_DEMO_ARG := $(if $(strip $(EDUPULSE_BUILD_DEMO)),--build-arg EDUPULSE_BUILD_DEMO=$(EDUPULSE_BUILD_DEMO))

PY := uv run python

# Chạy pnpm TỪ TRONG apps/web để corepack đọc đúng `packageManager` (pnpm ghim) và không làm đổi lockfile.
define web
@if [ -f apps/web/package.json ]; then cd apps/web && pnpm $(1); \
else echo "(bỏ qua) chưa có apps/web/package.json: pnpm $(1)"; fi
endef

# Biến nội bộ của Makefile không cần đưa sang lệnh con.
unexport THIS_MAKEFILE PY IMAGE BUNDLE_DIR BUILD_DEMO_ARG web

.PHONY: help env setup data eda baseline train export calibrate experiments freeze evaluate hitl report \
	artifacts demo-data api web gen-api test e2e check-raw lint docker-build smoke up fetch-artifacts

help: ## Liệt kê các lệnh
	@awk -F':.*## ' '/^[a-z][a-z0-9-]*:.*## /{printf "  %-16s %s\n", $$1, $$2}' $(THIS_MAKEFILE)

env: ## In cấu hình đang áp dụng (từ .env hoặc mặc định)
	@env | grep -E '^(EDUPULSE_|BUNDLE_|NEXT_PUBLIC_)' | sort
	@echo "bundle đưa vào image: $(BUNDLE_DIR)"

setup: ## Cài môi trường Python (mọi group) và phụ thuộc web
	uv sync --all-groups
	$(call web,install)

data: ## Tải UIT-VSFC, chuẩn hoá, tách từ vào $EDUPULSE_HOME/data
	$(PY) -m ml.download_data
	$(PY) -m ml.preprocess

eda: ## Phân tích khám phá dữ liệu, ghi hình và bảng vào results/
	$(PY) -m ml.eda

baseline: ## Huấn luyện 3 baseline TF-IDF
	$(PY) -m ml.train_baseline

train: ## Huấn luyện PhoBERT đa nhiệm, một run (seed 42)
	$(PY) -m ml.train_phobert --heads mtl --seed 42

experiments: ## Chạy đủ 9 run PhoBERT (đa nhiệm, đơn nhiệm, 3 seed); ngoài phạm vi từ 2026-10-17
	bash ml/run-experiments.sh

export: ## Xuất run tốt nhất sang ONNX int8
	$(PY) -m ml.export_onnx --run best

calibrate: ## Hiệu chỉnh xác suất (temperature scaling) và ngưỡng duyệt trên val
	$(PY) -m ml.calibrate

freeze: ## Đóng băng cấu hình đánh giá cuối vào results/freeze.json (commit file này trước evaluate)
	$(PY) -m ml.evaluate --freeze

evaluate: ## Chấm test đúng một lần (cần results/freeze.json đã commit)
	$(PY) -m ml.evaluate --split test --final

hitl: ## Mô phỏng vòng người duyệt bằng LinearSVC, chỉ dùng train và validation
	$(PY) -m ml.hitl_simulation

report: ## Kiểm định, phân tích lỗi, CheckList, ECE trên test
	$(PY) -m ml.significance
	$(PY) -m ml.error_analysis
	$(PY) -m ml.checklist
	$(PY) -m ml.calibrate --report test

artifacts: ## Gom bundle phiên bản BUNDLE_VERSION vào $EDUPULSE_HOME/artifacts/
	$(PY) -m ml.build_artifacts --version $(BUNDLE_VERSION)

demo-data: ## Tính dữ liệu demo từ UIT-VSFC vào thư mục demo-data (cần mạng)
	$(PY) -m edupulse.demo_data

api: ## Chạy API FastAPI để dev (cổng 8000, tự nạp lại khi sửa src/)
	uv run uvicorn edupulse.api.main:app --reload --reload-dir src --no-access-log --port 8000

web: ## Chạy Next.js dev (cổng 3000)
	cd apps/web && pnpm dev

gen-api: ## Xuất OpenAPI rồi sinh lại kiểu TypeScript cho web
	$(PY) -m edupulse.api.export_openapi
	cd apps/web && pnpm gen:api

test: ## Chạy pytest (và Vitest khi đã có apps/web)
	uv run pytest
	$(call web,run --if-present test)

e2e: ## Chạy Playwright trên container đang chạy (cần `make up`; E2E_BASE_URL để trỏ nơi khác)
	cd apps/web && E2E_BASE_URL=$${E2E_BASE_URL:-http://127.0.0.1:$${EDUPULSE_PORT:-8000}} pnpm exec playwright test

check-raw: ## Kiểm câu gốc UIT-VSFC và bí mật trong mọi commit trước khi đẩy; ARGS="--head" hoặc dải rev để thu hẹp
	$(PY) -m ml.check_raw_sentences $(ARGS)

lint: ## ruff, eslint và tsc --noEmit
	uv run ruff check .
	$(call web,run --if-present lint)
	$(call web,exec tsc --noEmit)

docker-build: ## Build image (tag IMAGE); bundle lấy qua build context `artifacts`
	docker build --build-arg BUNDLE_VERSION=$(BUNDLE_VERSION) $(BUILD_DEMO_ARG) \
		--build-context artifacts=$(BUNDLE_DIR) -t $(IMAGE) .

smoke: ## Smoke test giới hạn tài nguyên; thêm tham số bằng ARGS="--url ..."
	$(PY) scripts/smoke_test.py $(ARGS)

up: ## Dựng và chạy bằng docker compose
	docker compose up -d --build

fetch-artifacts: ## Tải bundle từ Hugging Face Hub về $EDUPULSE_HOME/artifacts/
	bash scripts/fetch-artifacts.sh
