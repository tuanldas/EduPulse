---
title: "Phase 1: Thiết lập repo & tooling"
status: todo
priority: P1
effort: 4h
dates: "2026-10-15"
depends_on: []
---

# Phase 1: Thiết lập repo & tooling
<!-- Updated: Validation Session 1–2 - commit không cần duyệt từng lần (dừng ở mốc); thêm group `demo`; Makefile thêm `report`, `demo-data`, `-include .env`, `BUNDLE_VERSION`; `artifacts/<version>/` + `demo-data/` vào gitignore; `EDUPULSE_SOURCE_URL` (runtime) thay `NEXT_PUBLIC_SOURCE_URL`; module dùng chung trong bố cục; sửa trích dẫn BibTeX; đánh dấu bước [smoke] -->

## Context links
- [PDR](../../docs/project-overview-pdr.md): ràng buộc + license (L33–40), tiêu chí nghiệm thu 1 "tái lập bằng vài lệnh" (L51).
- [tech-stack.md](../../docs/tech-stack.md): banner đổi kiến trúc (L5); dòng lỗi thời "Giao diện" (L17), "Host" (L19); lưu ý code (L35–38).
- [r1 modeling §9](../reports/researcher-261014-1913-phobert-vsfc-modeling.md): pin đã chạy thật (L123–124).
- [r2 deploy §8](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): uv groups, pin runtime (L59–65).
- [r6 kiến trúc §2, §6](./research/researcher-02-architecture.md): phiên bản đã kiểm (L16–21), bố cục repo (L47–63).
- BibTeX: nội dung ở `docs/wireframes/04-mo-hinh-dao-duc.html:563–580`; design-guidelines L141 chỉ nhắc khối `bibtex`.
- `.gitignore` hiện có: `/data/`, `/models/`, `/outputs/` (L10–13). Quy trình duyệt: [plan.md § Dependencies](./plan.md) (dừng ở mốc).

## Overview
- Ngày: 2026-10-15 (thứ Năm) · Effort: 4h · Priority: P1 (chặn mọi phase) · Trạng thái: chưa bắt đầu.
- Dựng project uv Python 3.12 (groups `app`/`train`/`dev`/`demo`, pin chính xác), khung package `src/edupulse`, ruff + pytest, `Makefile` làm "vài lệnh" tái lập (đọc `.env`), LICENSE AGPL-3.0 + ghi nguồn, cập nhật `docs/tech-stack.md` theo kiến trúc đã chốt, README khung.
- Bước [smoke] (1–2, 4, 6) làm trước để phase 2–4 bắt đầu sớm; phần còn lại làm xen kẽ.

## Key Insights
- Máy có Python 3.14.6 → uv 0.11.22 tự tải CPython 3.12 nhờ `.python-version`; numpy 2.5.4/scipy 1.18.1 cần ≥3.12 (r1 L87).
- Runtime không torch/pandas ⇒ phụ thuộc runtime ở group `app`; Docker cài `uv sync --frozen --no-default-groups --group app --no-editable` (cờ có trong `uv sync --help` 0.11.22). Group `demo` (pyarrow + huggingface-hub) CHỈ dùng ở stage Docker tính dữ liệu demo lúc build (phase 6), không vào image cuối.
- `pyproject.toml`, `uv.lock`, `Makefile` là file dùng chung ⇒ khai báo ĐỦ pin và target cho cả dự án ngay phase này. Phase sau cần thêm dep/target → báo lead, một người sửa.
- Repo không chứa câu gốc UIT-VSFC (Validation S1 #3): `data/`, `demo-data/`, `artifacts/` đều gitignore; test dùng câu tự viết hoặc dữ liệu tải về lúc chạy; chỉ ≤30 câu ví dụ trong `results/` cho báo cáo.
- Duyệt "dừng ở mốc" (Validation S1 #1): agent commit theo conventional commits, không chờ duyệt từng commit. Repo đã `git init` (nhánh `main`), chưa có commit. Không đưa mã plan/phase vào tên file code, comment, commit message.
<!-- Updated: Session 4 (lead, môi trường) - EDUPULSE_HOME ngoài thư mục Syncthing, dùng chung giữa các worktree; khoá MPS toàn máy -->
- **Ràng buộc môi trường (Session 4, lead phát hiện lúc bắt đầu code):**
  - Repo nằm trong `~/Desktop/Sync` do **Syncthing** đồng bộ. `.stignore` của Sync đã loại `.venv`, `node_modules`, `out`, `build`, `dist`, `vendor`, nhưng KHÔNG loại `data/`, `models/`, `outputs/`, `artifacts/`, `demo-data/`, `.next`.
  - Team code chạy song song trên nhiều **git worktree**.
  - ⇒ Mọi thư mục nặng/tái tạo được đi qua `EDUPULSE_HOME`:
    - `$EDUPULSE_HOME/{data,models,outputs,artifacts/<version>,demo-data}`;
    - khoá MPS ở `$EDUPULSE_HOME/.mps.lock` (khoá toàn máy, không theo worktree);
    - cache HF mặc định của thư viện, giữ nguyên.
  - Không đặt `EDUPULSE_HOME` thì mặc định là thư mục repo (`./data`, `./artifacts/<version>`…), để repo vẫn chạy được ở máy khác, CI và Docker.
  - Máy này đặt `EDUPULSE_HOME=$HOME/.cache/edupulse` trong `.env` (đường dẫn tuyệt đối, mọi worktree dùng chung).
  - Docker không COPY file nặng từ repo. Bundle đi vào image qua BuildKit `--build-context artifacts=$EDUPULSE_HOME/artifacts/$(BUNDLE_VERSION)` (compose: `additional_contexts`) hoặc qua `scripts/fetch-artifacts.sh` tải từ HF.

## Requirements
- `uv sync --all-groups` chạy được trên macOS arm64; `uv.lock` universal (macOS arm64, Linux x86_64/aarch64).
- Cây phụ thuộc group `app` không chứa `torch`, `pandas`, `pyarrow`.
- `uv run pytest` và `uv run ruff check .` exit 0; `make help` liệt kê mọi target trong bảng dưới; `make api`/`make web` nhận biến từ `.env`.
- Hằng dùng chung: thứ tự nhãn UIT-VSFC (`labels.py`); giới hạn + đường dẫn đọc từ env (`config.py`).
- `config.py` có `EDUPULSE_HOME` + các đường dẫn suy ra (`data_dir`, `models_dir`, `outputs_dir`, `artifacts_dir(version)`, `demo_dir`, `mps_lock_path`). Biến riêng (`EDUPULSE_ARTIFACTS_DIR`, `EDUPULSE_DEMO_DIR`…) vẫn ghi đè được. `Makefile` và mọi script `ml/` dùng các đường dẫn này, không ghi cứng `data/`, `models/`, `outputs/`. Có test cho cả 2 trường hợp: không đặt và có đặt `EDUPULSE_HOME`.
- `LICENSE` AGPL-3.0 + `THIRD_PARTY_NOTICES.md`: PhoBERT-base-v2 AGPL-3.0, VnCoreNLP GPL-3.0+, UIT-VSFC học thuật phi thương mại, Be Vietnam Pro/JetBrains Mono OFL-1.1, Material Symbols Apache-2.0; BibTeX UIT-VSFC + PhoBERT.
- `docs/tech-stack.md` phản ánh: Next.js export tĩnh do FastAPI phục vụ (1 container), Docker + compose trên VPS (build ngay trên VPS), JRE `default-jre-headless` trong image, runtime `csv` + `python-calamine` (pandas chỉ ở train), dữ liệu demo tính lúc build image, test pytest/Vitest/Playwright.

## Architecture
Bố cục repo (tham chiếu cho mọi phase):
```
EduPulse/
├─ pyproject.toml · uv.lock · .python-version · Makefile · .env.example · LICENSE · THIRD_PARTY_NOTICES.md · README.md
├─ src/edupulse/      config.py · labels.py · scoring.py (3) · bundle_schema.py (3) · demo_data.py (6)
│  ├─ text/           normalize · pii · segment · ngrams · pipeline  — MỘT nguồn tiền xử lý cho train & serve (2)
│  ├─ engine/         bundle · onnx_model · flags · explain · service · bench (4) · baseline (6)
│  ├─ insights/       stats · rules · build (6)
│  └─ api/            main · schemas · errors · access_log · routes/ (4) · jobs · upload_limit · parsing · export · export_openapi (6)
├─ ml/                script train/eval/export (group train; import edupulse.{text,labels,config,scoring,bundle_schema}; KHÔNG import api/engine/insights)
├─ apps/web/          Next.js export tĩnh (scaffold 4, giao diện 7)
├─ vendor/vncorenlp/  jar + wordsegmenter, commit (GPL-3.0+) (2)
├─ tests/ · scripts/  pytest (+ tests/fixtures/golden_sentences.json câu tự viết) · smoke_test.py, fetch-artifacts.sh, record_fixtures.py
├─ results/           hình/bảng/metric cho báo cáo — COMMIT
├─ data/ models/ outputs/ artifacts/<version>/ demo-data/   gitignore; tái tạo bằng make hoặc tải từ HF Hub / lúc build
└─ Dockerfile · .dockerignore · docker-compose.yml · deploy/Caddyfile
```

| Group | Gói (pin chính xác) | Cài ở |
|---|---|---|
| `app` | fastapi 0.143.1 · uvicorn 0.54.0 · python-multipart 0.0.32 · onnxruntime 1.31.0 · tokenizers 0.23.3 · numpy 2.5.4 · py-vncorenlp 0.1.4 · pyjnius 1.9.0 · python-calamine 0.8.3 · scikit-learn 1.9.1 · skops 0.16.0 · underthesea 9.5.0 | image cuối, máy dev |
| `demo` | pyarrow 26.0.0 · huggingface-hub 2.2.0 | stage Docker `demo` (build), máy dev |
| `train` | torch 2.14.1 · transformers 5.19.0 · accelerate 1.15.0 · safetensors 0.8.0 · huggingface-hub 2.2.0 · pandas 3.0.6 · pyarrow 26.0.0 · scipy 1.18.1 · onnx 1.23.2 · onnxscript 0.7.2 · matplotlib (bản ổn định mới nhất có wheel cp312, ghi pin lúc cài) | Mac, Colab |
| `dev` | pytest 9.1.1 · httpx 0.28.1 · ruff 0.17.0 | máy dev |

Target `Makefile` (đầu file `-include .env` + `export`; mọi lệnh Python qua `uv run`; `BUNDLE_VERSION ?= v1.0`; target của phase sau khai báo trước làm hợp đồng; KHÔNG có target đẩy HF/GitHub — hành động ra ngoài chạy tay ở mốc M3):

| Target | Lệnh | Hiện thực ở |
|---|---|---|
| `setup` | `uv sync --all-groups` + `pnpm --dir apps/web install` | 1, 4 |
| `data` · `eda` | `python -m ml.download_data && python -m ml.preprocess` · `python -m ml.eda` | 2 |
| `baseline` · `train` · `export` · `calibrate` | `ml.train_baseline` · `ml.train_phobert --heads mtl --seed 42` · `ml.export_onnx --run best` · `ml.calibrate` | 3, 5 |
| `experiments` · `evaluate` · `report` | `bash ml/run-experiments.sh` · `ml.evaluate --freeze && ml.evaluate --split test --final` · `ml.significance && ml.error_analysis && ml.checklist && ml.calibrate --report test` | 5 |
| `artifacts` · `demo-data` | `ml.build_artifacts --version $(BUNDLE_VERSION)` → `artifacts/$(BUNDLE_VERSION)/` · `python -m edupulse.demo_data --out demo-data` | 3, 5 · 6 |
| `api` · `web` · `gen-api` | `uvicorn edupulse.api.main:app --reload --no-access-log --port 8000` · `pnpm --dir apps/web dev` · `python -m edupulse.api.export_openapi && pnpm --dir apps/web gen:api` | 4, 6, 7 |
| `test` · `e2e` · `lint` | pytest + vitest · playwright (cần `make up`) · ruff + eslint + `tsc --noEmit` | 1 → 8 |
| `docker-build` · `smoke` · `up` · `fetch-artifacts` | `docker build --build-arg BUNDLE_VERSION=$(BUNDLE_VERSION)` · `python scripts/smoke_test.py` · `docker compose up -d --build` · `bash scripts/fetch-artifacts.sh` | 4, 6, 9 |

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/pyproject.toml`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/uv.lock`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.python-version`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/Makefile`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.env.example`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/LICENSE`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/THIRD_PARTY_NOTICES.md`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/README.md`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/config.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/labels.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/conftest.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/test_package.py`

Modify:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.gitignore`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docs/tech-stack.md`

## Implementation Steps
1. [smoke] `.python-version` = `3.12`. `pyproject.toml`: `[project] name="edupulse"`, `requires-python="==3.12.*"`, `dependencies=[]`, build backend `uv_build` (layout `src/`); `[dependency-groups]` theo bảng; `[tool.uv] default-groups` gồm cả 4 group (máy dev cài đủ, Docker tự chọn).
2. [smoke] `uv lock` → `uv sync`; kiểm `import torch, transformers, onnxruntime, tokenizers, pyarrow`.
3. `uv tree --no-default-groups --group app` → xác nhận không có `torch`/`pandas`/`pyarrow`; gói nào kéo theo → ghi lại, đổi gói hoặc import lười.
4. [smoke] `labels.py`: `SENTIMENT = ("negative", "neutral", "positive")`, `TOPIC = ("lecturer", "training_program", "facility", "others")` đúng id gốc 0..n của UIT-VSFC + tên tiếng Việt. `config.py`: dataclass đọc env `EDUPULSE_ARTIFACTS_DIR` (thư mục 1 phiên bản, vd `artifacts/v1.0`), `EDUPULSE_DEMO_DIR` (`demo-data`), `EDUPULSE_VNCORENLP_DIR`, `EDUPULSE_WEB_DIR`, `EDUPULSE_ORT_THREADS` (2), `EDUPULSE_NER` (0/1), `EDUPULSE_CORS_ORIGINS` (rỗng), `EDUPULSE_SOURCE_URL`, `EDUPULSE_JOB_TTL_MIN` (30, test đặt nhỏ); hằng `MAX_CHARS=500`, `MAX_UPLOAD_MB=10`, `MAX_ROWS=2000`, `XAI_MAX_WORDS=40`, `AUTO_XAI_TOP=20`, `BUDGETS=(5,10,20,30)`, `DEFAULT_BUDGET=10`, `AUDIT_RATE=0.05`, `HEARTBEAT_S=60`, `MAX_JOBS=8`, `MAX_QUEUE=3`.
5. ruff (line 100; E, F, I, B, UP) + pytest (`testpaths=["tests"]`, `addopts="-ra"` để in lý do skip; markers `data`, `java`, `model`, `train`, `slow`) trong pyproject. `tests/conftest.py`: helper skip kèm lý do rõ ("chưa có data/processed — chạy `make data`", "chưa có artifacts — chạy `make artifacts`", "thiếu JRE", "thiếu group train").
6. [smoke] `Makefile` đủ target theo bảng + `help`; `-include .env` + `export`.
7. `.gitignore` thêm `/artifacts/`, `/demo-data/`, `node_modules/`, `apps/web/.next/`, `apps/web/out/`, `apps/web/playwright-report/`, `apps/web/test-results/`.
8. `LICENSE` (văn bản AGPL-3.0 gốc) + `THIRD_PARTY_NOTICES.md` + BibTeX (chép từ `docs/wireframes/04-mo-hinh-dao-duc.html:563–580`).
9. `.env.example`: biến ở bước 4 + `BUNDLE_VERSION`, `NEXT_PUBLIC_API_BASE=http://localhost:8000` (chỉ dev), `EDUPULSE_SOURCE_URL=https://github.com/<user>/edupulse`, `EDUPULSE_HF_REPO=<user>/edupulse-phobert-mtl`, `EDUPULSE_HF_REVISION`; không secret.
10. `docs/tech-stack.md`: sửa dòng Giao diện, Host (Docker/compose build trên VPS), Dữ liệu (pandas chỉ train; runtime csv + calamine; dữ liệu demo tính lúc build), Tách từ (JRE apt trong Docker, bỏ jdk4py), Kiểm thử; đổi banner thành "chốt 2026-10-15"; "Không dùng" thêm Streamlit + Community Cloud, pandas ở runtime, TanStack Table/Query, SSE, rewrites của Next ở dev (r6 L63).
11. README khung: 1 đoạn mô tả, trạng thái "đang phát triển", `make setup`, link docs + plan (bản cuối ở phase 9).
12. Kiểm `uv run pytest -q`, `uv run ruff check .`, `make help`; commit `chore: scaffold python project and tooling` + `docs: update tech stack for nextjs, fastapi and docker`.

## Todo list
- [ ] `.python-version` + `pyproject.toml` (4 group, pin chính xác) + `uv.lock`
- [ ] `uv tree` group app: không torch/pandas/pyarrow
- [ ] `src/edupulse/{__init__,config,labels}.py` + `ml/__init__.py`
- [ ] ruff + pytest config + `tests/conftest.py` (skip có lý do) + `tests/test_package.py`
- [ ] `Makefile` đủ target (`report`, `demo-data`, `-include .env`) + `help`
- [ ] `.gitignore`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `.env.example`
- [ ] Cập nhật `docs/tech-stack.md`
- [ ] README khung
- [ ] Kiểm pytest/ruff/make help + commit

## Success Criteria
- `uv sync --all-groups` OK; `uv run python -c "import edupulse, torch, transformers, onnxruntime, tokenizers"` OK.
- `uv tree --no-default-groups --group app` có 0 dòng `torch`, `pandas`, `pyarrow`.
- `uv run pytest` exit 0 với ≥1 test thật (thứ tự nhãn, đọc env); `uv run ruff check .` exit 0.
- `make help` in đủ target; `make -n api` cho thấy biến từ `.env` được export; `docs/tech-stack.md` không còn Streamlit/Community Cloud là lựa chọn hiện hành.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| `uv lock` không giải được (torch Linux kéo gói CUDA; pyjnius/underthesea thiếu wheel 1 nền tảng) | Thấp × TB | Chạy `uv lock` ngay bước 2; giới hạn nền tảng bằng `tool.uv.environments` nếu cần; NER là tuỳ chọn → bỏ được |
| Gói group app kéo theo pandas/pyarrow | Thấp × Thấp | `uv tree` bước 3; đổi gói hoặc import lười |
| Pin 14/10 có bản vá mới | TB × Thấp | Giữ pin để tái lập; chỉ nâng khi gặp lỗi |

Rollback: chỉ là file khung, `git revert` hoặc xoá file; không đụng dữ liệu.

## Security Considerations
- `.env.example` không chứa secret; `.gitignore` đã chặn `.env*`; Makefile chỉ đọc `.env` cục bộ.
- Tuân thủ license: repo AGPL-3.0, ghi nguồn bên thứ ba; không commit `data/`, `models/`, `artifacts/`, `demo-data/`.

## Next steps
- Phase 2 bắt đầu ngay sau bước 2. Phần Next scaffold/Docker của phase 4 bắt đầu song song khi `pyproject.toml` xong.
