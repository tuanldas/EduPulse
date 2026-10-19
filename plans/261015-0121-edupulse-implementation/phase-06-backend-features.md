---
title: "Phase 6: Backend đủ tính năng"
status: todo
priority: P1
effort: 24h
dates: "2026-10-18 → 2026-10-22"
depends_on: [4]
---

# Phase 6: Backend đủ tính năng
<!-- Updated: Validation Session 1–2 - so sánh mô hình, retrain, XLSX, heatmap thành Must (bảo vệ) (S1 #2); dữ liệu demo/file mẫu/câu mẫu tính lúc build image: `demo_data.py` + stage Docker `demo` + `make demo-data`, HF lỗi → ẩn demo (S1 #3); file mẫu có cột "Nhóm (minh hoạ)" A–E + cờ `illustrative_groups` (S1 #4, S2 #3); heartbeat 60 s + `close` qua sendBeacon + TTL 30' làm lưới an toàn (F5, S2 #2); middleware giới hạn upload theo Content-Length + đếm byte (F3); insights nhận nhóm qua POST body, log theo route template (F4); `source_url` qua `/api/config` (F9); model card theo schema (F7); n-gram dùng chung (F11); lọc câu nghi còn tên (F12); XLSX giải nén có trần thật; ngưỡng 1.000 câu ≤30 s (F15); mốc dừng M4 -->

## Context links
- [r6 kiến trúc](./research/researcher-02-architecture.md): luồng OpenAPI → client (L41); endpoint + quy tắc vận hành (L64–78); parse/an toàn (L79); test thật (L80–83).
- [r3 intelligence](../reports/researcher-261014-1913-intelligent-layer-xai-insights.md): ngân sách + mẫu kiểm tra (L41–53); export + retrain (L54–56); insight log-odds, câu tiêu biểu, Wilson, ~12 luật (L58–65); baseline giải thích bằng coef (L29).
- [r2 deploy](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): CSV `utf-8-sig` + chống formula injection (L55).
- [design-guidelines](../../docs/design-guidelines.md): hành vi trang 2 (L122–128), trang 3 (L130–136), giọng văn trung lập (L150–154).
- Wireframe: [trang 1](../../docs/wireframes/01-phan-tich-cau.html) (câu mẫu L431), [trang 2](../../docs/wireframes/02-phan-tich-hang-loat.html), [trang 3](../../docs/wireframes/03-dashboard-insight.html) — KPI demo khớp mức nền toàn train (tiêu cực 46,6%, tích cực 49,4%).
- Báo cáo kiểm chứng: [F2–F5, F7, F9, F11, F12, F15, Unverified 7](./reports/validation-verification.md).

## Overview
- Ngày: 2026-10-18 → 2026-10-22 · Effort: 22.5h · Priority: P1 · Trạng thái: chưa bắt đầu.
- Sáng 18/10 chốt hợp đồng API (phase 7 làm song song), rồi dựng `demo_data.py` sớm (UI cần câu mẫu/file mẫu/demo), sau đó: upload CSV/XLSX có middleware giới hạn, job nền + polling + heartbeat, cờ theo ngân sách + mẫu kiểm tra, XAI top-20 + theo yêu cầu, insight + khuyến nghị + heatmap, export CSV an toàn, model card / file mẫu / cấu hình, so sánh LinearSVC, script retrain, stage Docker tính demo lúc build. **Mốc dừng M4** = xong khối API.

## Key Insights
- Contract-first: toàn bộ schema + mã lỗi chốt ở bước 1; route chưa xong trả 501 `not_implemented` (không dữ liệu giả); `python -m edupulse.api.export_openapi` → `apps/web/openapi.json`; snapshot test bắt lệch. `generate_unique_id_function=lambda r: r.name`; enum `Literal` khoá tiếng Anh, UI ánh xạ tiếng Việt; lỗi `{detail:{code,…}}` (r6 L41). `/api/model-card` trả kiểu `ModelCard` của `bundle_schema` ⇒ FE có type (F7).
- Upload (F3): Starlette 1.8.0 parse HẾT multipart (file >1MB spool ra file tạm) trước khi handler chạy, `max_part_size` không áp cho file ⇒ middleware ASGI `upload_limit.py` cho `POST /api/batches`: `Content-Length` > 10MB + 64KB → 413 ngay, không đọc body; không có/khai sai → bọc `receive` đếm byte, vượt → 413. File tạm nằm ở `/tmp` (tmpfs ở prod), xoá ngay sau parse; dữ liệu đã parse giữ trong RAM của JobStore.
- JobStore: dict + Lock, tối đa 8 batch, hàng đợi (đang chạy + chờ) > 3 ⇒ 429 `busy`; 1 worker thread; id `secrets.token_urlsafe(16)`. Vòng đời (S2 #2): FE gửi heartbeat 60 s khi tab mở; `pagehide` → `navigator.sendBeacon` tới `POST …/close` (sendBeacon chỉ gửi POST, xử lý giống DELETE); TTL 30' từ lần truy cập cuối chỉ là lưới an toàn ⇒ duyệt lâu không mất batch, đóng tab thì xoá ngay.
- Cờ batch qua `scoring.rank_batch`: cờ ⌈B·n⌉ câu điểm min(q_s, q_t) thấp nhất; không dấu luôn cờ, tính vào ngân sách trước (vượt thì vẫn cờ, báo số thực); >40 từ không tự cờ. Mẫu kiểm tra = 5% phần không cờ, seed theo batch id. Hàng đợi duyệt = cờ + kiểm tra, sắp độ tin cậy tăng dần, thứ tự cố định (design L124–126). Auto-XAI 20 câu cờ thấp nhất (≤40 từ); câu khác giải thích khi mở (~1 s), cache.
- Insight (r3 L59–63): n-gram qua `text/ngrams.py` (cùng hàm đếm nền, F11); log-odds Dirichlet α0 = 1000 cho ô chủ đề × tiêu cực; cảnh báo khi n ≥ 20 VÀ cận dưới Wilson 95% > nền chủ đề + 10 điểm; CSVC theo tỉ trọng (cận dưới > 4,4%); ≤3 thẻ; KPI so với nền toàn train; heatmap nhóm × chủ đề (Must); câu tiêu biểu: Σz/√số từ, conf ≥ 0,7, 6–30 từ, bỏ gần trùng (Jaccard tập từ > 0,8), bỏ câu còn từ Viết hoa ngay sau từ xưng hô (nghi tên chưa che, F12). Tính trên NHÃN DỰ ĐOÁN (wireframe trang 3), không phản ánh nhãn sửa. Nhóm lọc gửi qua POST body, không qua query string (F4).
- Dữ liệu demo tính lúc BUILD IMAGE (S1 #3): `edupulse/demo_data.py` tải `default/{validation,test}/0000.parquet` @ `2d769061…` (HF, không token) → `sample.csv` (200 câu val phân tầng, làm đẹp nhẹ, cột `Góp ý` + `Nhóm (minh hoạ)` = Nhóm A–E gán ngẫu nhiên seed 42), `ui_samples.json` (6 câu val ngắn phủ chủ đề × cảm xúc + 1 bản bỏ dấu), `demo_results.json` (3.166 câu test chạy qua `Analyzer` trên kiến trúc máy build), `demo_manifest.json` (`ok`, revision, `bundle_version`, `arch`). HF lỗi → `ok:false` + lý do, build KHÔNG fail ⇒ API `demo_available=false`, UI ẩn demo/file mẫu/câu mẫu kèm thông báo. Stage `demo` dùng group `demo` (pyarrow) — không vào image cuối. QEMU/offline: `--build-arg EDUPULSE_BUILD_DEMO=0`.
- `illustrative_groups` (S1 #4): true khi batch tạo từ `POST /api/batches/sample` hoặc tên cột nhóm chứa "minh hoạ"; có trong trạng thái, results, insights ⇒ nhãn "dữ liệu minh hoạ" không mất khi mở lại `?batch=`.
- Nhãn sửa giữ ở client, gửi `overrides` khi export (r6 L78). Export: `utf-8-sig`; ô bắt đầu `= + - @` (cả full-width `＝ ＋ － ＠`), TAB, CR → thêm `'` (OWASP); file "nhãn đã sửa" theo r3 L54 + cột `text_model` (đã chuẩn hoá, tên là `wzjwz<k>`).
- Retrain (Must): cổng trên val + tập kiểm tra (audit), KHÔNG dùng test (giữ số test sạch, khác r3 L55); dedup theo hash `text_model`; loại câu trùng val/test. So sánh LinearSVC (Must): skops nạp lười lần đầu `compare=true`; giải thích = coef × tf-idf (r3 L29). XLSX (Must): giải nén từng member bằng `zipfile` theo luồng với trần 100MB tổng (không tin `file_size` trong header) rồi mới đưa calamine.

## Requirements
| Method · Path | Request → Response | Lỗi (`detail.code`) |
|---|---|---|
| GET `/api/health` | → `status, model_version, bundle_version, stage, ner_enabled, rss_mb` | — |
| GET `/api/config` | → giới hạn (500 ký tự, 10MB, 2.000 dòng, 40 từ), ngân sách + hiệu quả val từng mức, mặc định 10, `heartbeat_s` 60, `ui_samples[]`, `demo_available` + lý do, `source_url` (env runtime) | — |
| POST `/api/analyze` | `{text, compare?}` → kết quả 1 câu (+ `baseline` khi `compare`) | 422 `text_empty`/`text_too_long` |
| POST `/api/explain` | `{text}` → giải thích 2 head | 422 `xai_too_long` |
| POST `/api/batches` | multipart → `batch_id, filename, n_rows, columns, preview[5], suggested_text_col, encoding, warnings[], illustrative_groups` | 413 `file_too_large` · 415 `unsupported_file_type` · 422 `too_many_rows`/`no_rows`/`decode_failed` · 429 `busy` |
| POST `/api/batches/sample` | → như trên, từ file mẫu build-time, `illustrative_groups: true` | 404 `demo_unavailable` · 429 |
| POST `/api/batches/{id}/run` | `{text_col, group_col?, review_budget ∈ {5,10,20,30}}` → 202 trạng thái | 404 `batch_not_found` · 409 `already_running` · 422 `invalid_column`/`no_text_column` |
| GET `/api/batches/{id}` | → `state, step, done, total, eta_s, error_code, illustrative_groups` | 404 |
| POST `/api/batches/{id}/heartbeat` | → 204 (gia hạn) | 404 |
| POST `/api/batches/{id}/close` · DELETE `/api/batches/{id}` | → 204 (sendBeacon dùng POST) | 404 |
| GET `/api/batches/{id}/results` | → `rows[]` (text ẩn PII, nhãn, xác suất, cờ, lý do, `in_review`, `source`, nhóm, `has_explanation`) + `summary` | 404 · 409 `not_done` |
| GET `/api/batches/{id}/rows/{i}/explain` | → giải thích | 404 · 422 `xai_too_long` |
| POST `/api/batches/{id}/insights` | `{groups?: string[]}` → KPI, cảnh báo, cột chồng, heatmap (nếu có nhóm), từ khoá, câu tiêu biểu (pool ~50/chủ đề kèm từ khớp để UI lọc), khuyến nghị, `illustrative_groups` | 404 |
| POST `/api/batches/{id}/export` | `{kind: all\|corrected, overrides[]}` → `text/csv` | 404 · 422 |
| GET `/api/model-card` · `/api/sample.csv` | → `ModelCard` (schema) · CSV 200 câu (build-time) | 404 `demo_unavailable` (sample) |
- Batch `demo` (3.166 câu test, dự đoán tính lúc build) nạp lúc khởi động từ `demo_results.json`, không suy luận lại, không hết hạn, chỉ đọc, không có cột nhóm.
- Upload: middleware giới hạn 10MB; CSV qua `csv.Sniffer` (`,` `;` TAB), `utf-8-sig` → rơi về `cp1258` + cảnh báo; XLSX sheet đầu (giải nén có trần thật); >2.000 dòng → 422 (UI khoá nút); đoán cột văn bản theo tên (`góp ý|ý kiến|nội dung|feedback|text`) rồi độ dài trung bình.
- Tiến trình theo bước khớp wireframe: ẩn thông tin cá nhân → tách từ → dự đoán → gắn cờ → tự giải thích 20 câu; ETA từ thông lượng đo được.

## Architecture
```
POST /api/batches ─► upload_limit (Content-Length + đếm byte) ─► parsing (mã hoá, cột, XLSX trần giải nén) ─► JobStore[id] (RAM)
POST /api/batches/sample ─► demo-data/sample.csv (build-time) ─► JobStore[id]{illustrative_groups:true}
run ─► worker 1 luồng: prepare_many (từng bước) ─► Analyzer.predict mini-batch 32 (khoá inference chung)
     ─► scoring.rank_batch(B) + audit 5% ─► explain top-20 ─► state=done
client: poll 1 Hz · heartbeat 60 s · pagehide → sendBeacon POST /close · TTL 30' lưới an toàn
results ─► (UI sửa nhãn, giữ client) ─► export {overrides} ─► CSV
POST insights {groups} (batch|demo) ─► insights.build: KPI · Wilson · log-odds (ngrams + prior nền) · luật · câu tiêu biểu · heatmap
Docker: web → runtime → demo (FROM runtime + group demo; python -m edupulse.demo_data) → final (FROM runtime + COPY /demo → /app/demo-data)
```
- `insights/stats.py` (Wilson LB, log-odds Dirichlet), `insights/rules.py` (~12 luật chủ đề × từ điển → hành động; GV kèm "Tổng hợp theo môn, không gắn với cá nhân"), `insights/build.py` (payload dashboard).

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/routes/batches.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/jobs.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/upload_limit.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/parsing.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/export.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/export_openapi.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/insights/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/insights/stats.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/insights/rules.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/insights/build.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/baseline.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/demo_data.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/retrain.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/openapi.json` (sinh)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_batches.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_parsing.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_upload_limit.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_heartbeat.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_sample_batch.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_export.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_limits.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/api/test_openapi_snapshot.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/insights/test_stats.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/insights/test_rules.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/insights/test_build.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/engine/test_baseline.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_retrain.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/test_demo_data.py`

Modify:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/main.py` (router mới, middleware upload, batch demo, dọn job)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/schemas.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/errors.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/routes/analyze.py` (`compare`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/api/routes/meta.py` (`/config`, `/model-card`, `/sample.csv`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/flags.py` (xếp hạng batch, audit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/engine/service.py` (progress theo bước)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/Dockerfile` (stage `demo` + `final`, `ARG EDUPULSE_BUILD_DEMO`)

## Implementation Steps
1. Hợp đồng (sáng 18/10): schema mọi endpoint ở bảng (gồm heartbeat/close/sample, insights POST, `illustrative_groups`, trường `/api/config`) + mã lỗi; route chưa xong trả 501; `export_openapi` → `apps/web/openapi.json`; snapshot test. Báo phase 7 chạy `make gen-api`.
2. `demo_data.py` + `make demo-data` (nguồn HF @ SHA hoặc `--source-dir data/raw` offline) + `tests/test_demo_data.py` (offline: 200 dòng, Nhóm A–E cố định theo seed, 7 câu mẫu, `demo_results` qua schema, nhánh `ok:false`).
3. `upload_limit.py` + `parsing.py` + test (file sinh lúc test từ câu val tải local: UTF-8 BOM, `;`, cp1258, XLSX, 2.001 dòng, rỗng, 10MB + 1 byte; `Content-Length` 50MB → 413 ngay; chunked 50MB → 413 khi vượt ~10MB; XLSX khai `file_size` giả → bị chặn).
4. `jobs.py` + `routes/batches.py` (upload, sample, run, status, heartbeat, close/DELETE) + worker + tiến trình/ETA + TTL; `flags.rank_batch` + audit; auto-XAI top-20.
5. `results` + `rows/{i}/explain` (cache).
6. `insights/` + `POST …/insights` + batch `demo`; test số đã biết: Wilson 4/5 → 0,38 · 28/40 → 0,55 · 70/100 → 0,60 (r3 L62); log-odds đúng dấu trên dữ liệu train (CSVC × tiêu cực có "máy chiếu", "phòng"); luật khớp regex trên câu thật; heatmap + cờ minh hoạ; lọc câu nghi còn tên.
7. `export.py` + test (BOM, injection mọi ký tự đầu nguy hiểm, overrides áp đúng dòng, cột retrain).
8. `/api/model-card` (kiểu `ModelCard`), `/api/sample.csv`, `/api/config`.
9. (Must) `engine/baseline.py` + `compare` trong `/api/analyze` + test.
10. (Must) `ml/retrain.py --feedback … --model {baseline,phobert}`; test đường baseline (5 câu tự viết + 5 câu val đọc lúc chạy phải bị loại vì trùng val); đường PhoBERT chạy thử `--max-steps 20` sau phase 5 (khoá MPS, Docker tắt).
10b. (Must, Validation S3 #2; **đổi ở Session 6**: chạy vòng mô phỏng người duyệt bằng LinearSVC trên CPU — cùng thiết kế nửa duyệt/nửa đánh giá, ngân sách 10%, so trước/sau, KHÔNG dùng test — và chỉ smoke retrain PhoBERT 20 bước trên CPU; không train MPS) Thiết kế gốc: 1 lần retrain PhoBERT ĐẦY ĐỦ mô phỏng vòng người duyệt cho báo cáo: chia val cố định (seed) thành nửa "duyệt" + nửa "đánh giá"; trên nửa duyệt, gắn cờ theo ngân sách 10% bằng mô hình v1, lấy nhãn gốc của câu bị cờ làm "nhãn đã sửa" (người duyệt mô phỏng, ghi rõ trong báo cáo); retrain MTL cùng cấu hình (~12 phút MPS, khoá MPS, Docker tắt); so trước/sau trên nửa đánh giá (acc, macro-F1 từng task, F1 lớp trung tính/"khác", McNemar). KHÔNG dùng test. Ghi `results/tables/hitl_retrain.md` + `results/metrics/hitl_retrain.json`; không thay bundle v1 đang phục vụ.
11. `Dockerfile`: stage `demo` (FROM runtime, `uv sync … --group app --group demo`, chạy `python -m edupulse.demo_data --out /demo` khi `EDUPULSE_BUILD_DEMO=1`) + stage `final` (FROM runtime, `COPY --from=demo /demo /app/demo-data`, `EDUPULSE_DEMO_DIR`); build 2 lần: có mạng → demo ok; `EDUPULSE_BUILD_DEMO=0` → `demo_available=false`, image vẫn chạy.
12. Test luồng đầy đủ với model thật (marker `model`): upload → run → heartbeat → poll → results → explain → insights (POST nhóm) → export → close; 404 JSON, 413, 415, 422, 429; hết hạn khi không heartbeat (`EDUPULSE_JOB_TTL_MIN` nhỏ); log không chứa nội dung/giá trị nhóm/id (`caplog`).
13. Sinh lại `openapi.json` nếu đổi; **dừng M4** (xong khối API: danh sách endpoint + kết quả test); commit (`feat: add batch analysis jobs`, `feat: add insights and recommendations`, `feat: add build-time demo data`, `feat: add csv export and retrain script`).

## Todo list
- [x] Hợp đồng API (heartbeat/close/sample, insights POST, `illustrative_groups`, config) + `openapi.json` + snapshot (sáng 18/10)
- [x] `demo_data.py` + `make demo-data` + test offline
- [x] Middleware giới hạn upload + parsing CSV/XLSX (trần giải nén thật) + test
- [x] JobStore + worker + tiến trình/ETA + heartbeat/close + TTL + 429
- [x] Cờ theo ngân sách + audit 5% + auto-XAI top-20; results + explain theo dòng
- [x] Insights (ngrams, Wilson, log-odds, luật, câu tiêu biểu, heatmap, cờ minh hoạ) + batch demo
- [x] Export CSV (BOM, chống injection, overrides, cột retrain; `ml/retrain.py` đọc đúng header `CORRECTED_HEADER`, có test đọc ngược)
- [x] `/config`, `/model-card`, `/sample.csv`, `POST /batches/sample`
- [x] So sánh LinearSVC nạp lười (Must)
- [x] `ml/retrain.py` + test baseline + chạy thử PhoBERT (Must)
- [x] ~~1 lần retrain PhoBERT đầy đủ mô phỏng người duyệt~~ → thay theo Session 6: mô phỏng bằng LinearSVC (nửa val duyệt/đánh giá, ngân sách 10%) + smoke PhoBERT 20 bước CPU → `results/tables/hitl_retrain.md` (Must)
<!-- Updated: Validation Session 3 - retrain PhoBERT đầy đủ 1 lần mô phỏng vòng người duyệt cho báo cáo -->


- [ ] Dockerfile stage `demo` + `final` (+ `EDUPULSE_BUILD_DEMO=0`)
- [ ] Test luồng đầy đủ + giới hạn + TTL + log sạch; mốc M4; commit

## Success Criteria
- Không còn route 501; snapshot OpenAPI khớp `apps/web/openapi.json`.
- `pytest tests/api tests/insights tests/engine tests/test_demo_data.py` pass với model thật (skip kèm lý do khi thiếu artifacts/dữ liệu).
- Batch 1.000 câu end-to-end gồm auto-XAI ≤30 s trên M1 (F15); tiến trình tăng đơn điệu; `/api/analyze` trong lúc batch chạy ≤1 s thêm.
- Số câu cờ = ⌈B·n⌉ (cộng phần không dấu vượt ngân sách nếu có); audit = 5% phần không cờ; cùng batch cho cùng kết quả.
- Upload: `Content-Length` >10MB → 413 trước khi đọc body; chunked 50MB → 413 khi vượt ~10MB; XLSX header giả → chặn. Có heartbeat → batch sống quá TTL; không heartbeat → hết hạn; sau `close` → 404.
- Insight khớp số kiểm tay trên 1 file nhỏ; mọi khuyến nghị có "Căn cứ: n câu…"; tên người che theo luật (best effort, có disclaimer); batch từ file mẫu mang `illustrative_groups: true`.
- Build có mạng → demo đủ (3.166 câu, file mẫu 200 dòng Nhóm A–E, 7 câu mẫu); `EDUPULSE_BUILD_DEMO=0`/HF lỗi → build vẫn thành công, `demo_available=false`.
- Export mở bằng Excel tiếng Việt đúng dấu (kiểm tay 1 lần); 0 ô bắt đầu bằng ký tự nguy hiểm mà thiếu `'`; log chỉ có route template.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Khối lượng 22,5h vượt lịch 18–22/10 | TB × Cao | Contract-first để FE song song; cắt Could rồi Should trước (NER underthesea; lọc câu tiêu biểu theo từ khoá chủ yếu ở UI, backend chỉ trả pool); Must (bảo vệ) chỉ cắt khi người dùng quyết ở M6 |
| HF không truy cập được lúc build | TB × TB | Fallback `ok:false` + thông báo trên UI; build lại khi có mạng |
| Build image lâu/ngốn RAM do stage demo (VPS 2GB) | TB × TB | Swap ≥2GB (phase 9); ghi thời gian build đo trên Mac; `EDUPULSE_BUILD_DEMO=0` khi cần build nhanh |
| Worker thread gọi JVM (pyjnius) lỗi | TB × TB | Mọi lời gọi segmenter qua Lock; test đa luồng; giữ JVM trong 1 process |
| Hợp đồng đổi giữa chừng làm vỡ FE | TB × TB | Snapshot test + `make gen-api` bắt buộc khi đổi; báo phase 7 |
| Luật khuyến nghị sai ngữ cảnh | TB × TB | Luôn kèm căn cứ + cảnh báo "chỉ để gợi ý"; (Could) precision@50 |
| Tốn RAM khi 8 batch × 2.000 dòng | Thấp × TB | Dòng nhỏ (~1MB/batch); TTL + heartbeat + close; đo ở phase 8 |

Rollback: mỗi endpoint commit riêng, `git revert` từng phần; hợp đồng có snapshot nên lùi đồng bộ được với FE; stage `demo` tắt bằng `EDUPULSE_BUILD_DEMO=0`.

## Security Considerations
- Giới hạn upload ở tầng ASGI trước khi parse; file tạm chỉ ở `/tmp` và xoá sau parse; XLSX giải nén có trần thật; dữ liệu batch chỉ trong RAM, xoá khi close/DELETE/TTL.
- Batch id 128-bit ngẫu nhiên; batch `demo` chỉ đọc; `close` idempotent (id lạ → 404 vô hại).
- Chống CSV/formula injection; PII che trong mọi output (tên: best effort + disclaimer); `text_model` không chứa tên thật.
- Không log nội dung câu, tên file, giá trị cột/nhóm, batch id; nhóm lọc đi qua body; handler lỗi chỉ log mã lỗi.
- Hàng đợi giới hạn → 429 thay vì treo; endpoint CPU là `def` (threadpool), 1 worker.
- Repo/bundle không chứa câu gốc: dữ liệu demo chỉ sinh trong image lúc build (`demo-data/` gitignore + dockerignore).

## Next steps
- Phase 7 tích hợp trang 2–3 khi bước 3–6 xong (dùng `make demo-data` ở máy dev). Phase 8 test E2E + đo hiệu năng batch qua HTTP. Phase 9 hoàn thiện compose trên stage `final`, ghi hướng dẫn retrain + build trên VPS vào docs.
