---
title: "Phase 4: API skeleton, Docker, smoke test giới hạn tài nguyên"
status: todo
priority: P1
effort: 9.5h
dates: "2026-10-16 → 2026-10-17 (muộn nhất sáng 18/10)"
depends_on: [1, 3]  # phần A chỉ cần 1; phần B cần artifacts/v0-smoke của 3 (gián tiếp gồm 2)
---

# Phase 4: API skeleton, Docker, smoke test giới hạn tài nguyên
<!-- Updated: Validation Session 1–2 - depends_on [1, 3] (F17); engine dùng `scoring.py`, bỏ `engine/calibration.py` (F11); test golden 20 câu tự viết thay test theo demo (F8); `artifacts/<version>/` qua build-arg (F8); tắt access log, log theo route template (F4); ngưỡng 1.000 câu gồm auto-XAI ≤30 s (F15); smoke từ chối đo khi khoá MPS đang giữ, không build Docker khi train (F14); mốc dừng M1; lịch smoke mục tiêu 17/10 -->

<!-- Updated: Session 4 (lead, môi trường) - bundle vào image qua --build-context artifacts, không COPY từ repo -->

## Context links
- [r6 kiến trúc](./research/researcher-02-architecture.md): §1 topology 1 origin (L3–15); §5 bẫy Next export (L46); §6 dev không rewrites (L63); §7 endpoint + quy tắc vận hành (L64–79, nhất là L78).
- [r2 deploy](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): JRE trong `python:3.12-slim` đã kiểm (L27); số đo + ước lượng x86 (L38–47); Dockerfile mẫu (L66–79).
- [r3 intelligence](../reports/researcher-261014-1913-intelligent-layer-xai-insights.md): ngân sách môi trường yếu (L13); occlusion (L23–28).
- [PDR](../../docs/project-overview-pdr.md): mục tiêu hiệu năng (L55).
- [design-guidelines](../../docs/design-guidelines.md): cờ 1 câu theo ngân sách mặc định 10% + lý do bằng lời (L116); không dấu / PII / >40 từ (L119).
- Báo cáo kiểm chứng: [F4, F8, F11, F14, F15, F17](./reports/validation-verification.md).
- Prototype (scratchpad phiên): `scratchpad/r2/dk3/Dockerfile`, `r2/bench.py`, `r3/scripts/bench_xai.py`, `r3/scripts/mem_xai.py`.

## Overview
- Ngày: 2026-10-16 → 2026-10-17 (muộn nhất sáng 18/10) · Effort: 9.5h (A 3,25h ‖ phase 3; B 6,25h) · Priority: P1 (giảm rủi ro lớn nhất trước khi làm UI) · Trạng thái: chưa bắt đầu.
- Phần A (song song phase 3, chưa cần model): scaffold Next.js tối thiểu, khung FastAPI (log không nội dung), Dockerfile multi-stage, compose cơ bản.
- Phần B (cần `artifacts/v0-smoke/`): engine (ONNX + `scoring.py` + cờ + occlusion), `/api/analyze`, `/api/explain`, CLI bench, smoke test trong container giới hạn như VPS nhỏ. **Mốc dừng M1 = smoke go/no-go** (người dùng xem số đo). Mục tiêu 17/10 (ngày 4); 16/10 tối chỉ khi các bước [smoke] ở phase 1–3 trơn tru. Luôn TRƯỚC 9 run MPS của phase 5 và trước mọi việc UI.

## Key Insights
- Một origin, một tiến trình: Next `output:"export"` + `trailingSlash:true` → FastAPI `StaticFiles(html=True)` mount `/` SAU router `/api`; mount `/` biến `/api/*` sai thành 404 HTML → catch-all JSON 404 (r6 L4, L78).
- `--workers 1` bắt buộc (trạng thái trong RAM); endpoint CPU là `def`; 1 khoá inference theo mini-batch 16–32 để `/analyze` chen vào ≤ ~1 giây khi batch chạy (r6 L78).
- uvicorn 0.54.0 ghi access log KÈM query string (F4) ⇒ chạy `--no-access-log`; middleware `api/access_log.py` tự ghi `method · route template (vd /api/batches/{batch_id}/results) · status · ms` — không id, không query, không body.
- `python:3.12-slim` + `default-jre-headless` (OpenJDK 21, +194MB) chạy uid 1000 đã kiểm: nạp 0,45s, 1.000 câu tách từ 0,45s (r2 L27).
- Occlusion: N+1 forward (N ≤ 40 từ RDR), batch 16 (<40MB RAM thêm), MTL giải thích 2 head trong 1 lượt (r3 L23–28); tạo biến thể bằng bỏ từ rồi tokenize lại.
- M1 nhanh hơn VPS x86 chia sẻ nhiều (r2 ước x86 2 vCPU int8: 30–50ms/câu, 1.000 câu 15–35s — L47) ⇒ đọc số M1 với hệ số ×3. Ngưỡng 1.000 câu tính CẢ auto-XAI: 30 s × 3 = 90 s < 120 s, còn biên 30 s (F15).
- Tài nguyên độc quyền (F14): không build/chạy Docker trong lúc run MPS của phase 3 (12 phút); `smoke_test.py` từ chối đo khi `outputs/.mps.lock` đang bị giữ.
- VPS 2GB còn ~1,5GB cho container sau OS + dockerd ⇒ smoke chạy `--cpus=2 --memory=1536m --memory-swap=1536m`. Docker Desktop đang tắt: bật khi không train, cấp VM ≥4GB RAM / ≥4 CPU.
- Kịp mốc: nếu run 4 epoch chưa xong, export checkpoint epoch 1 (model đã fine-tune thật, không phải model giả) để đo RAM/latency; đo lại với bundle đầy đủ khi có.
- Bundle theo phiên bản (Session 4: bundle nằm ở `$EDUPULSE_HOME`, ngoài repo): `make docker-build BUNDLE_VERSION=v0-smoke IMAGE=edupulse:smoke` truyền `--build-context artifacts=<bundle_dir>` → Dockerfile `COPY --from=artifacts . /app/artifacts/`, `EDUPULSE_ARTIFACTS_DIR=/app/artifacts` (F8). Không COPY file nặng từ repo.
- Dev: `next dev` :3000 + uvicorn :8000, CORS bật bằng env, `NEXT_PUBLIC_API_BASE`; KHÔNG rewrites (r6 L63). `next dev` tự sinh `AGENTS.md` (tắt `agentRules:false`); font Google tải lúc build ⇒ build cần mạng (r6 L46).

## Requirements
- Engine `src/edupulse/engine/`: `bundle.py` (đọc qua `bundle_schema`, kiểm sha256, `normalizer_version` = hằng code, thứ tự nhãn), `onnx_model.py` (tokenizers + ORT, `intra_op_num_threads` theo env, sắp theo độ dài, batch 32, truncation 128), `flags.py` (gọi `scoring.py`; cờ 1 câu ngân sách 10%; không dấu luôn cờ; >40 từ không tự cờ; lý do "phân vân giữa A x% và B y%"), `explain.py` (top-3 ủng hộ + 1 phản đối mỗi head, Δp "93% → 46%"; hiển thị `_`→dấu cách, `wzjwz<k>`→`[TÊN]`, token emoticon → ký hiệu gốc), `service.py` (`Analyzer.analyze`, `analyze_many(progress_cb)`, `explain`), `bench.py` (CLI đo).
- API: `create_app()` + lifespan nạp `Analyzer` (thiếu/lệch bundle → dừng với thông báo rõ); `GET /api/health` (`status`, `model_version`, `bundle_version`, `stage`, `ner_enabled`, `rss_mb`); `POST /api/analyze` `{text ≤500}`; `POST /api/explain` `{text}`; catch-all `/api/{path}` → JSON 404; StaticFiles từ `EDUPULSE_WEB_DIR` nếu có; CORS chỉ khi `EDUPULSE_CORS_ORIGINS`; header `X-Content-Type-Options`, `Referrer-Policy`, `X-Frame-Options: DENY`; lỗi `{detail:{code, message}}`; middleware `access_log`.
- Web scaffold: Next 16.4.1 + React 19.3.1 + TS 5.9.3 (ghim vì openapi-typescript) + Tailwind 4.3.3, pnpm (ghim `packageManager`, corepack), `next.config.ts` (`output:"export"`, `trailingSlash:true`, `images.unoptimized:true`), 1 trang placeholder tiếng Việt, `.nvmrc` 24.15.0.
- Docker: stage `web` (node build `out/`) → stage `runtime` (python + JRE + `uv sync --frozen --no-default-groups --group app --no-editable` + `src/` + `vendor/vncorenlp/` + `artifacts/${BUNDLE_VERSION}/` + `out/`), user uid 1000, `COPY --chown`, HEALTHCHECK, `CMD uvicorn … --workers 1 --proxy-headers --no-access-log`; `.dockerignore`; compose cơ bản (port 8000, `cpus: 2`, `mem_limit: 1536m`, `restart: unless-stopped`). Phase 6 thêm stage `demo` + `final`.
- `scripts/smoke_test.py` + báo cáo `results/perf/`.

## Architecture
```
POST /api/analyze {text}
  → validate (rỗng/ >500 → 422) → prepare(): PII 2 view · lowercase · dấu thanh · token đặc biệt · RDR
  → tokenizers (truncation 128, pad id 1) → [khoá inference] ORT int8 1 forward → logits CX(3), CĐ(4)
  → scoring: softmax(logits/T) → q = phân vị CDF val → cờ (q < τ_10 ở CX hoặc CĐ; không dấu luôn cờ)
  → {sentiment{label, probs}, topic{label, probs}, flag{needs_review, reasons[]}, display_text, pii_types, no_diacritics, xai_available}
POST /api/explain {text}  (gọi song song để kết quả hiện trước)
  → prepare → ≤40 từ? → N+1 biến thể bỏ-từng-từ → batch 16 → Δp lớp dự đoán mỗi head → top-3 ủng hộ + 1 phản đối
log: access_log middleware → "POST /api/analyze 200 412ms" (không nội dung, không query)
```
Smoke test (ghi `results/perf/smoke-<ngày>-arm64.{json,md}`; từ chối chạy nếu khoá MPS đang giữ):
1. `make docker-build BUNDLE_VERSION=v0-smoke IMAGE=edupulse:smoke` (BuildKit `--build-context artifacts=…`); ghi kích thước image.
2. `docker run -d --cpus=2 --memory=1536m --memory-swap=1536m -p 8000:8000 edupulse:smoke`; đo thời gian tới `/api/health` 200.
3. `docker cp` file 1.000 câu val thật (tải local, không commit) → `docker exec … python -m edupulse.engine.bench --n 1000 --xai 20`: thời gian từng bước + tổng end-to-end; đọc `memory.peak` của cgroup v2.
4. Từ host: 30 lần `/api/analyze` + `/api/explain` (câu ≤40 từ) → p50/p95; lặp lại khi bench đang chạy (độ trễ chen ngang).
5. `GET /api/khong-co` → JSON 404; `GET /` → HTML; `GET /khong-co/` → 404.html; `docker logs` không có body/query.
6. Tham khảo: lặp bước 2–4 với `--cpus=1`.

| Chỉ số (M1, 2 CPU / 1,5GB) | Ngưỡng go | Lý do |
|---|---|---|
| OOM-kill | 0 | — |
| `memory.peak` (NER bật) | ≤ 1,2GB | chừa ≥300MB; >1,2GB → tắt NER rồi đo lại |
| 1 câu + XAI, p95 | ≤ 0,6 s | ×3 = 1,8 s < 2 s (PDR L55) |
| 1.000 câu end-to-end (prepare + predict + cờ + auto-XAI 20 câu) | ≤ 30 s | ×3 = 90 s < 2 phút, còn biên 30 s (F15) |
| Khởi động tới health 200 | ≤ 20 s | JVM + ORT + tokenizer |

Không đạt → áp dụng lần lượt, đo lại sau mỗi bước: tắt NER → `enable_cpu_mem_arena=False` → batch 16 → JVM `-Xmx256m` → auto-XAI top-10 → int8 per-tensor (136MB, phải qua lại cổng ≤1 điểm). Vẫn trượt → dừng (M1/M6) báo người dùng (cân nhắc VPS 4GB).

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/bundle.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/onnx_model.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/flags.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/explain.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/service.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/bench.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/main.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/schemas.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/errors.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/access_log.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/routes/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/routes/analyze.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/routes/meta.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/package.json`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/pnpm-lock.yaml`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/next.config.ts`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/tsconfig.json`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/postcss.config.mjs`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/eslint.config.mjs`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/layout.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/page.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/globals.css`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.nvmrc`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/Dockerfile`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.dockerignore`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docker-compose.yml`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/scripts/smoke_test.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/engine/test_bundle.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/engine/test_flags.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/engine/test_explain.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/engine/test_onnx_model.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/engine/test_golden.py` (đường serve trên 20 câu tự viết = `golden_predictions.json`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_meta.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_analyze.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_static_404.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_access_log.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/test_runtime_guard.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/perf/smoke-<ngày>-arm64.json` + `.md` (sinh, commit)

Modify: không (`scoring.py`, `bundle_schema.py` do phase 3 sở hữu; target `Makefile` đã khai báo ở phase 1 — lệnh khác thì báo lead).

## Implementation Steps
Phần A (song song phase 3; không build Docker trong 12 phút run MPS):
1. Scaffold `apps/web` bằng create-next-app (App Router, TS, Tailwind, ESLint, `src/`, alias `@/*`), ghim phiên bản; `pnpm build` → `out/index.html`. pnpm ≥10 chặn script build của dependency theo mặc định → allowlist nếu build lỗi.
2. `api/main.py`, `errors.py`, `access_log.py`, `routes/meta.py` khung; static mount + catch-all JSON 404 + `tests/api/test_static_404.py` (dùng `out/` thật) + `tests/api/test_access_log.py` (log không chứa query/body/id).
3. `Dockerfile` (stage `web` + `runtime`, `ARG BUNDLE_VERSION`), `.dockerignore` (loại `.git`, `.env*`, `data/`, `models/`, `outputs/`, `demo-data/`, `node_modules/`, `apps/web/.next/`), `docker-compose.yml`; build thử (chưa có artifacts → container dừng với thông báo rõ).

Phần B (cần `artifacts/v0-smoke/`, toàn bộ là [smoke]):
4. Engine + test (marker `model`, skip kèm lý do nếu thiếu artifacts): golden 20 câu tự viết khớp `golden_predictions.json` (nhãn bằng nhau, |Δp| ≤ 1e-4); τ_10 cho ~10% câu val bị cờ (marker `data`); occlusion trả đúng N+1 forward; câu >40 từ trả `xai_too_long`.
5. `routes/analyze.py` (`/api/analyze`, `/api/explain`) + `tests/api/test_analyze.py` qua `TestClient` với model thật (rỗng → 422, 501 ký tự → 422, câu không dấu → cờ, có SĐT → `[SĐT]`, "Thầy Nam" → `[TÊN]`).
6. `engine/bench.py`.
7. `tests/test_runtime_guard.py`: chạy trong env chỉ group app (`uv run --no-default-groups --group app --group dev pytest tests/test_runtime_guard.py`) — import `edupulse.api.main` rồi assert `torch`, `pandas`, `pyarrow` không có trong `sys.modules`.
8. Bật Docker Desktop (khi không train); build image với `v0-smoke`; chạy smoke theo protocol; ghi kết quả; **dừng mốc M1**: báo số đo + quyết định go/no-go.
9. Commit (`feat: add inference engine and analyze api`, `build: add dockerfile and compose`, `test: add resource-limited smoke test`).

## Todo list
- [ ] Scaffold Next.js export tĩnh (ghim TS 5.9.3, pnpm, `.nvmrc`)
- [ ] Khung FastAPI: lifespan, lỗi JSON, static mount, catch-all 404 JSON, CORS env, header an toàn, `access_log` (không access log uvicorn)
- [ ] Dockerfile `web` + `runtime` (`ARG BUNDLE_VERSION`) + `.dockerignore` + compose cơ bản
- [ ] Engine: bundle, onnx_model, flags (qua `scoring.py`), explain, service + test golden (model thật)
- [ ] `/api/health`, `/api/analyze`, `/api/explain` + test TestClient
- [ ] `engine/bench.py` + guard "không torch/pandas/pyarrow"
- [ ] Smoke test trong container (2 CPU/1,5GB) + báo cáo `results/perf/` + mốc M1
- [ ] Commit

## Success Criteria
- Mọi chỉ số trong bảng go đạt (hoặc đạt sau fallback, ghi rõ fallback nào); người dùng đã xem số ở M1.
- `pytest tests/engine tests/api tests/test_runtime_guard.py` pass (test cần model skip kèm lý do khi thiếu artifacts, không mock); golden 20/20 khớp.
- `GET /api/khong-co` → 404 JSON `{detail:{code:"not_found"}}`; `GET /` → HTML; `/batch` → 307 `/batch/` (khi trang tồn tại); log container chỉ có method + route template + status + ms.
- Image ghi nhận kích thước (kỳ vọng ≈0,9–1,2GB); container chạy user không phải root.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Smoke trễ sang 18/10 | TB × TB | Thứ tự [smoke] ở phase 1–3; checkpoint epoch 1; 2 luồng ở phase 3 |
| RAM vượt ngân sách | TB × Cao | Đo sớm; chuỗi fallback ở trên; NER tuỳ chọn; baseline nạp lười (phase 6) |
| Số M1 lạc quan so với VPS x86 | Cao × TB | Hệ số ×3; kiểm nhãn amd64 qua QEMU (phase 8); đo thật bằng `smoke_test.py --url` trên VPS người dùng |
| pyjnius/JVM lỗi trong container | Thấp × Cao | r2 đã kiểm; `JAVA_HOME=/usr/lib/jvm/default-java`; test segment trong container ngay bước 3 |
| Next 16 / create-next-app khác trí nhớ | TB × Thấp | Đọc file sinh ra + docs; chỉ cần export tĩnh tối thiểu |
| Docker chạy trùng lúc train (swap) | TB × TB | Quy tắc độc quyền + `smoke_test.py` kiểm khoá MPS |

Rollback: image tag riêng `edupulse:smoke`; mỗi bước commit riêng, `git revert` được; fallback hiệu năng là cấu hình env, đảo lại được.

## Security Considerations
- Container chạy uid 1000; không secret trong image; `.dockerignore` chặn `.env*`, dữ liệu, model thô, `demo-data/`.
- Input ≤500 ký tự; uvicorn chạy `--no-access-log`; middleware chỉ ghi route template (không id/query/body); handler lỗi chỉ log loại exception + route.
- CORS tắt ở prod (cùng origin); chỉ bật bằng env khi dev.

## Next steps
- Sau M1: phase 5 chạy 9 run MPS (Docker tắt). Phase 6 mở rộng API + thêm stage `demo`. Phase 7 dựng giao diện trên scaffold. Phase 8/9 chạy lại smoke với bundle v1, compose cuối và trên VPS thật.
