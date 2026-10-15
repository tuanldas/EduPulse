---
title: "Phase 8: Kiểm thử, review, hiệu năng dưới giới hạn tài nguyên"
status: todo
priority: P1
effort: 12h
dates: "2026-10-23 → 2026-10-26"
depends_on: [5, 6, 7]
---

# Phase 8: Kiểm thử, review, hiệu năng dưới giới hạn tài nguyên
<!-- Updated: Validation Session 1–2 - E2E dùng chuỗi đánh dấu ở cả văn bản lẫn cột nhóm để chứng minh log sạch (F4); E2E heartbeat/TTL + close (F5, S2 #2); E2E image không có demo (`EDUPULSE_BUILD_DEMO=0`) (S1 #3); badge minh hoạ trên file mẫu (S1 #4); kiểm nhãn amd64 qua QEMU (lead chốt); ngưỡng 1.000 câu gồm auto-XAI ≤30 s (F15); không đo khi đang train (F14); chạy lại trên compose cuối ở phase 9 (F16); ước lượng lại 12h -->

## Context links
- [PDR](../../docs/project-overview-pdr.md): hiệu năng + upload 5MB (L55), test tự động (L56), model card/CheckList (L57).
- [r6 kiến trúc §8](./research/researcher-02-architecture.md): chiến lược test thật, 1 kịch bản E2E vàng trên site do uvicorn phục vụ (L80–83).
- [r2 deploy](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): x86 không VNNI dùng U8S8, giữ per-channel + reduce_range, đo lại trên host (L50).
- [design-guidelines](../../docs/design-guidelines.md): a11y (L143–148), mục cần kiểm trên trình duyệt (§10, đã cập nhật ở phase 7).
- [Phase 4](./phase-04-api-skeleton-docker-smoke-test.md): protocol + ngưỡng smoke (bảng go/no-go). Báo cáo kiểm chứng: [F4, F5, F14–F16, Unverified 1–2](./reports/validation-verification.md).

## Overview
- Ngày: 2026-10-23 → 2026-10-26 · Effort: 12h · Priority: P1 · Trạng thái: chưa bắt đầu.
- E2E vàng + edge trên container thật; đo hiệu năng cuối với `artifacts/v1.0/` + UI thật dưới `--cpus=2 --memory=1536m`; kiểm nhãn trên amd64 bằng QEMU; QA a11y/tương phản; review code các vùng rủi ro cao; kiểm bảo mật; sửa lỗi kèm test hồi quy. Đóng băng tính năng sau M5 (24/10) — sau đó chỉ sửa lỗi. Bản compose cuối (phase 9) chạy lại smoke + E2E vàng.

## Key Insights
- Playwright chạy trên image đã build (kiểm luôn `out/`, static mount, JSON 404, dữ liệu demo build-time) thay vì `next dev` (r6 L83).
- Đo hiệu năng khi không có job MPS nào (script từ chối nếu khoá đang giữ); đọc số M1 với hệ số ×3; ngưỡng 1.000 câu gồm auto-XAI ≤30 s (F15).
- File 5MB mà ≤2.000 dòng ⇒ mỗi dòng ~2,5KB (đoạn văn dài): chi phí tăng gần tuyến tính theo token, truncation 128 giới hạn trần mỗi dòng; PDR chỉ yêu cầu "chạy được" ⇒ đo + ghi thời gian, kiểm tiến trình/ETA, không đặt ngưỡng cứng.
- "Không log nội dung" phải được chứng minh với cả giá trị cột nhóm (F4): file E2E sinh lúc test từ câu tự viết, chèn chuỗi đánh dấu duy nhất vào văn bản + cột nhóm + tên file → grep `docker logs` = 0.
- Vòng đời batch (S2 #2): container E2E đặt `EDUPULSE_JOB_TTL_MIN=1`; tab mở > 90 s vẫn xuất được (heartbeat); đóng trang → trạng thái 404.
- Kiểm x86 (lead chốt): QEMU chỉ để so NHÃN (golden 20 câu tự viết + 200 câu val tải local) giữa image arm64 và amd64 build với `EDUPULSE_BUILD_DEMO=0`; KHÔNG đọc tốc độ. Đo thật bằng `smoke_test.py --url` trên VPS của người dùng; model card giữ "chưa đo x86" tới khi đo.
- Review tập trung: parity chuẩn hoá train↔serve, `scoring.py`, occlusion, thống kê insight, upload middleware/parse/export, JobStore + heartbeat + JVM, Dockerfile (stage demo).

## Requirements
- `apps/web/e2e/golden-path.spec.ts` (desktop 1440): trang 1 chọn câu mẫu → thấy nhãn + XAI → trang 2 "Dùng file mẫu" → thấy badge minh hoạ → chạy → sửa 1 nhãn → mở Sheet giải thích → "Câu sau" → tải CSV (kiểm BOM + nhãn đã sửa) → "Xem dashboard" → trang 3 có KPI + cột chồng + heatmap (minh hoạ) → trang 4 có bảng chỉ số.
- `apps/web/e2e/edge-cases.spec.ts`: file >2.000 dòng → nút khoá + cảnh báo; file sai định dạng → lỗi tiếng Việt; câu không dấu → cờ; câu có SĐT → `[SĐT]`; "Thầy Nam" → `[TÊN]`; `<img src=x onerror=alert(1)>` hiển thị như chữ; `GET /api/khong-co` → JSON 404; công tắc tối giữ qua điều hướng; heartbeat/TTL + close; image `EDUPULSE_BUILD_DEMO=0` → câu mẫu, file mẫu, nguồn demo bị ẩn kèm thông báo.
- `scripts/smoke_test.py` mở rộng: luồng batch qua HTTP (upload 1.000 câu → run → poll → done), upload ~5MB (2.000 dòng đoạn văn ghép từ câu val tải local), `/api/analyze` chen ngang khi batch chạy, chế độ `--url <host>` cho VPS.
- QEMU: `docker buildx build --platform linux/amd64 --build-arg EDUPULSE_BUILD_DEMO=0 --load` → so nhãn với image arm64 → `results/perf/qemu-amd64-labels.json`.
- QA tay theo design §10: tương phản nút/tab dark, bàn phím qua nav/Select/Slider/ToggleGroup/Sheet (Esc đóng), focus ring, font tiếng Việt, 390px không cuộn ngang trang. (Could) axe qua `@axe-core/playwright`.
- Review bằng agent `code-reviewer` + sửa; mỗi lỗi sửa có test hồi quy.

## Architecture
```
docker build (final) → docker run (--cpus=2 --memory=1536m, EDUPULSE_JOB_TTL_MIN=1 cho E2E)
   ├─► Playwright (BASE_URL=http://localhost:8000): golden-path + edge-cases + ảnh sáng/tối
   ├─► smoke_test.py: bench trong container + luồng HTTP batch + upload 5MB + chen ngang
   └─► docker logs → grep chuỗi đánh dấu (phải rỗng)
docker buildx --platform linux/amd64 (EDUPULSE_BUILD_DEMO=0) → so nhãn golden 20 + 200 val với arm64
make lint · make test (pytest đủ marker + vitest)
```
Ma trận test tổng (kiểm đủ trước khi đóng phase):

| Tầng | Nội dung | Công cụ | Dữ liệu |
|---|---|---|---|
| Đơn vị Python | normalize, PII (luật tên), segment, ngrams, pipeline parity, `scoring.py`, schema bundle, cờ/ngân sách, occlusion, Wilson/log-odds/luật, upload middleware, parsing, export, JobStore/heartbeat, demo_data, guard retrain | pytest | câu val tải lúc chạy + chuỗi PII tổng hợp + câu tự viết |
| Mô hình | int8 vs FP32 (cổng), tokenizer parity, golden 20 câu tự viết | pytest `-m model` | artifacts thật |
| API | TestClient: mọi endpoint, mã lỗi, snapshot OpenAPI, static + 404 JSON, log route template | pytest | model + file sinh từ câu thật |
| Guard runtime | không torch/pandas/pyarrow trong env app | pytest (env app) | — |
| FE đơn vị | format, nhãn, XAI, ReviewTable, polling, session/heartbeat, badge minh hoạ | Vitest + RTL | fixture ghi từ API thật trên câu tự viết |
| E2E | golden path + edge cases + ảnh sáng/tối | Playwright | container thật |
| Hiệu năng | RAM đỉnh, p95 1 câu+XAI, 1.000 câu gồm auto-XAI, 5MB, chen ngang | smoke_test.py | câu val tải local |
| Kiến trúc | nhãn amd64 (QEMU) = arm64 | buildx + bench | golden 20 + 200 val |
| ML | test 1 lần, kiểm định, CheckList | script `ml/` (phase 5) | split test/val |

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/e2e/golden-path.spec.ts`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/e2e/edge-cases.spec.ts`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/perf/final-<ngày>-arm64.json` + `.md` (sinh, commit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/perf/qemu-amd64-labels.json` (sinh, commit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/screenshots/*.png` (4 trang × sáng/tối, sinh từ Playwright, commit)

Modify:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/scripts/smoke_test.py` (luồng HTTP batch, 5MB, chen ngang, `--url`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/bench.py` (chế độ chỉ xuất nhãn cho so sánh QEMU)
- File nguồn có lỗi do review/test phát hiện (thuộc phase gốc; lúc này không phase nào chạy song song trên chúng) + test hồi quy tương ứng trong `tests/` hoặc `apps/web/tests/`

## Implementation Steps
1. Build image `final` với `artifacts/v1.0/` (có mạng → demo ok); `make up` (hoặc `docker run` giới hạn); viết + chạy `golden-path.spec.ts`, `edge-cases.spec.ts` (kèm image `EDUPULSE_BUILD_DEMO=0` cho ca thiếu demo); lưu ảnh vào `results/screenshots/`.
2. Mở rộng `smoke_test.py`; chạy đo cuối (2 CPU/1,5GB, thêm 1 lượt `--cpus=1` tham khảo) → `results/perf/final-*.{json,md}`; so với smoke phase 4 và ngưỡng go.
3. Grep `docker logs` sau E2E: không xuất hiện chuỗi đánh dấu (văn bản, giá trị nhóm, tên file), batch id hay query string.
4. QEMU amd64: build `EDUPULSE_BUILD_DEMO=0`, so nhãn golden 20 + 200 val với arm64 → `results/perf/qemu-amd64-labels.json`; lệch → ghi vào model card + deployment.md.
5. QA tay a11y/tương phản theo design §10; (Could) axe.
6. Review `code-reviewer`: `src/edupulse/text/`, `scoring.py`, `engine/{flags,explain}.py`, `insights/`, `api/{jobs,upload_limit,parsing,export}.py`, `demo_data.py`, `Dockerfile`; kiểm bảo mật: zip bomb, CSV injection qua UI, CORS tắt ở prod (không header `Access-Control-Allow-Origin`), batch id không đoán được, container non-root, image không chứa câu gốc ngoài `demo-data/` sinh lúc build.
7. Sửa lỗi (High trước), thêm test hồi quy; chạy lại `make lint test` + E2E.
8. Commit (`test: add end-to-end golden path`, `fix: …`), ghi kết quả vào `results/`.

## Todo list
- [ ] E2E golden path trên container thật (file mẫu + badge minh hoạ)
- [ ] E2E edge cases (giới hạn, lỗi, không dấu, PII/tên, XSS, 404 JSON, dark, heartbeat/TTL/close, thiếu demo)
- [ ] `smoke_test.py`: luồng HTTP batch + 5MB + chen ngang + `--url`
- [ ] Đo hiệu năng cuối (2 CPU/1,5GB) → `results/perf/final-*`
- [ ] Chứng minh log không có nội dung (chuỗi đánh dấu ở văn bản + nhóm)
- [ ] So nhãn amd64 qua QEMU → `results/perf/qemu-amd64-labels.json`
- [ ] QA a11y/tương phản (design §10)
- [ ] Review code + kiểm bảo mật
- [ ] Sửa lỗi + test hồi quy; `make lint test` + E2E xanh; commit

## Success Criteria
- `make lint`, `make test` (pytest mọi marker có artifacts + Vitest) và Playwright (golden + edge) đều pass trên image `final`.
- Hiệu năng cuối đạt bảng go của phase 4 (M1, ×3 an toàn): 1 câu+XAI p95 ≤0,6 s; 1.000 câu gồm auto-XAI ≤30 s; RAM đỉnh ≤1,2GB; 0 OOM; upload ~5MB chạy xong, có thời gian ghi lại và tiến trình đúng.
- QEMU: golden 20/20 cùng nhãn; 200 câu val ≥98% cùng nhãn (liệt kê câu lệch theo id, không in văn bản).
- 0 phát hiện review mức High còn mở; grep log = 0 kết quả.
- Ảnh 4 trang sáng/tối trong `results/screenshots/`.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Lỗi muộn nhiều hơn thời gian sửa | Cao × TB | Đóng băng tính năng sau M5; ưu tiên High; cắt Could/Should còn lại |
| Hiệu năng cuối kém smoke (UI + bundle v1) | TB × Cao | Chuỗi fallback phase 4; giảm auto-XAI top-10; dừng M6 báo người dùng |
| Nhãn amd64 lệch nhiều (U8S8 bão hoà) | Thấp × Cao | r2 L50: per-channel + reduce_range đã chọn; lệch >2% → báo người dùng, cân nhắc ONNX FP32 trên x86 |
| QEMU build/chạy quá chậm | TB × Thấp | Bỏ demo stage; chỉ 220 câu; không đo tốc độ |
| E2E chập chờn (polling, tải file, heartbeat) | TB × Thấp | Chờ theo trạng thái UI, không `sleep` cố định; retries=1 |
| File 5MB đoạn dài chạy rất lâu trên VPS | TB × TB | Truncation 128; tiến trình + ETA; ghi số đo vào deployment.md |

Rollback: chỉ thêm test/sửa lỗi nhỏ, `git revert` từng commit sửa; không đổi kiến trúc trong phase này.

## Security Considerations
- Kiểm thực tế: upload vượt giới hạn (Content-Length + chunked), zip bomb (header giả), CSV injection, XSS qua văn bản, CORS ở prod, non-root, không secret trong image (`docker history`), log không nội dung/không giá trị nhóm/không id.
- Ghi các rủi ro chấp nhận (không rate-limit theo IP; VPS tự quản; tên viết thường có thể sót) vào `docs/deployment.md` ở phase 9.

## Next steps
- Phase 9: đóng gói deploy VPS, chạy lại smoke + E2E vàng trên compose cuối, cập nhật docs, gom chỉ mục artifact báo cáo; `smoke_test.py --url` làm checklist nghiệm thu trên VPS thật.
