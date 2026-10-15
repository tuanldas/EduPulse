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
BUNDLE_VERSION := $(or $(strip $(BUNDLE_VERSION)),v1.0)
EDUPULSE_HOME  := $(or $(strip $(EDUPULSE_HOME)),$(CURDIR))
IMAGE          ?= edupulse:$(BUNDLE_VERSION)
# Bundle đưa vào image qua build context; khớp Settings.bundle_dir trong src/edupulse/config.py.
BUNDLE_DIR     := $(or $(strip $(EDUPULSE_ARTIFACTS_DIR)),$(EDUPULSE_HOME)/artifacts/$(BUNDLE_VERSION))
# Chỉ truyền cờ build dữ liệu demo khi có đặt EDUPULSE_BUILD_DEMO (0 để bỏ stage demo).
BUILD_DEMO_ARG := $(if $(strip $(EDUPULSE_BUILD_DEMO)),--build-arg EDUPULSE_BUILD_DEMO=$(EDUPULSE_BUILD_DEMO))

PY := uv run python

# Chạy pnpm trong apps/web nếu đã có package.json (phần web được dựng ở phase sau).
define web
@if [ -f apps/web/package.json ]; then pnpm --dir apps/web $(1); \
else echo "(bỏ qua) chưa có apps/web/package.json: pnpm $(1)"; fi
endef

# Biến nội bộ của Makefile không cần đưa sang lệnh con.
unexport THIS_MAKEFILE PY IMAGE BUNDLE_DIR BUILD_DEMO_ARG web

.PHONY: help env setup data eda baseline train export calibrate experiments evaluate report \
	artifacts demo-data api web gen-api test e2e lint docker-build smoke up fetch-artifacts

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

experiments: ## Chạy toàn bộ các run PhoBERT (đa nhiệm, đơn nhiệm, 3 seed)
	bash ml/run-experiments.sh

export: ## Xuất run tốt nhất sang ONNX int8
	$(PY) -m ml.export_onnx --run best

calibrate: ## Hiệu chỉnh xác suất (temperature scaling) và ngưỡng duyệt trên val
	$(PY) -m ml.calibrate

evaluate: ## Đóng băng cấu hình rồi chấm test đúng một lần
	$(PY) -m ml.evaluate --freeze
	$(PY) -m ml.evaluate --split test --final

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
	pnpm --dir apps/web dev

gen-api: ## Xuất OpenAPI rồi sinh lại kiểu TypeScript cho web
	$(PY) -m edupulse.api.export_openapi
	pnpm --dir apps/web gen:api

test: ## Chạy pytest (và Vitest khi đã có apps/web)
	uv run pytest
	$(call web,run --if-present test)

e2e: ## Chạy Playwright trên container đang chạy (cần `make up`)
	pnpm --dir apps/web exec playwright test

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
