# EduPulse — Kiến trúc Next.js + FastAPI (r6-architecture)
> 2026-10-15 · host đích giả định yếu (1–2 vCPU, 1–3 GB RAM; chưa chốt, r5). Đã chạy thử trong thư mục tạm, không đụng dự án: export Next, Starlette TestClient, esbuild, sinh client OpenAPI, `shadcn add`, RSS import.
## 1. Kết luận (đã xếp hạng)
1. **Một origin, một tiến trình:** Next `output:"export"` + `trailingSlash:true` → FastAPI `StaticFiles(html=True)` phục vụ `out/`, mount sau router `/api`. Đã đo: `/dashboard`→307→`/dashboard/`→200, query string chạy; thiếu `trailingSlash` thì `/dashboard` 404. Mất SSR/Route Handler/rewrites/Server Actions — app này không cần.
2. **Batch = job nền + polling 1 Hz.** Đồng bộ bị loại: PDR cho 1.000 câu ≤ 2 phút ⇒ 2.000 câu có thể > 4 phút, vượt timeout proxy; UI cần “640/1.000 · còn ~25 giây”. SSE bị loại: thêm `sse-starlette`, reconnect, proxy buffer mà không thêm giá trị.
3. **Hợp đồng:** `openapi-typescript` + `openapi-fetch`, **ghim TypeScript 5.9.3** (TS 7.0.2 làm openapi-typescript sập, đã chạy thử).
4. **UI:** shadcn/ui + Tailwind v4; Recharts qua shadcn `chart`; heatmap/confusion matrix = CSS grid; bảng duyệt = `Table` + `Select` từng dòng, **không TanStack Table** (thiết kế cố định thứ tự, không sort/lọc cột).
5. **Python:** 1 project uv + dependency groups (khớp tech-stack.md); runtime không pandas (`csv` + `python-calamine`).
6. **Công sức:** web + API ≈ 8 / 13 / 22 ngày công (tốt/thường/xấu) so với ~10 ngày công còn lại (15→28/10) mà ML + báo cáo cũng cần ⇒ phải cắt phạm vi (mục 9).

| Topology | Tiến trình | CORS | Node lúc chạy | Hợp host yếu | Hạng |
|---|---|---|---|---|---|
| Export tĩnh, FastAPI phục vụ cả web | 1 | không | không | tốt nhất (web stack Python ~73 MB) | **1** |
| Tách host (web tĩnh host A, API host B) | 2 dịch vụ | cần + `NEXT_PUBLIC_API_BASE` | không | RAM ổn; thêm 1 dịch vụ, cần kiểm giới hạn body 10 MB qua proxy | 2 |
| `next start` + FastAPI | 2 | qua proxy | có (RSS chưa đo) | kém | 3 |
## 2. Phiên bản đã kiểm (npm/PyPI/GitHub, 2026-10-15)
- **Web:** next 16.4.1 (Node ≥20.9) · react/react-dom 19.3.1 · typescript 7.0.2 (**dùng 5.9.3**; 6.x mới nhất 6.0.3) · tailwindcss + @tailwindcss/postcss 4.3.3 · shadcn 4.21.5 (kéo `@base-ui/react` 1.9.0, recharts 3.10.1, lucide-react 1.57.0, sonner 2.0.8, next-themes 0.4.6, tw-animate-css 1.4.0) · @tanstack/react-table 9.3.1 · react-dropzone 20.1.3 · echarts 6.1.0 · vega-embed 7.3.0 / vega-lite 6.5.0.
- **Client API:** openapi-typescript 7.13.0 · openapi-fetch 0.17.0 · @hey-api/openapi-ts 0.99.0 · orval 8.43.0 (hai gói sau cần Node ≥22.18).
- **Test/tool:** vitest 5.0.4 · jsdom 30.1.2 (Node ^22.22.2 hoặc ^24.15; máy có 24.15.0, sát ngưỡng) · @testing-library/react 16.3.3 · user-event 14.6.8 · jest-dom 7.0.1 · @playwright/test 1.64.1 · eslint 10.12.0 · eslint-config-next 16.4.1 · pnpm 12.11.2.
- **Python:** fastapi 0.143.1 · starlette 1.8.0 · pydantic 2.14.1 · pydantic-settings 2.16.0 · uvicorn 0.54.0 · python-multipart 0.0.32 · python-calamine 0.8.3 · openpyxl 3.1.5 (**bản cuối 2024-06**) · pandas 3.0.6 · onnxruntime 1.31.0 · tokenizers 0.23.3 · numpy 2.5.4 · pytest 9.1.1 · httpx 0.28.1 (2024-12) · ruff 0.17.0.
- **Đã chạy thật:** Next 16.4.1 export với TS 7.0.2 và 5.9.3 · `next/font/google` Be Vietnam Pro + JetBrains Mono (`subsets:["vietnamese","latin"]`) → ≈12 woff2 tự host, `out/` 992 KB · `shadcn init` + `add` 21 file component + trang BarChart 100% có `ReferenceLine` → build OK (JS gzip ≈346 KB cho site mẫu 2 trang) · `pnpm install` OK (14,5 s).
## 3. Chart — đo bundle (tree-shaken, minify / gzip, React external)
| | Recharts 3.10.1 | ECharts 6.1.0 | Vega-Lite (vega-embed) |
|---|---|---|---|
| Kích thước | 403 / **116 KB** | 604 / 204 KB | 820 / 282 KB |
| Cột chồng 100% + vạch nền | `stackOffset="expand"` + `ReferenceLine` (build OK) | tự tính % + `markLine` | `stack:"normalize"` + lớp rule |
| Heatmap (thang 6 bậc, n<20 viền đứt) | không có chart sẵn → CSS grid | `heatmap` + `visualMap` | `rect` |
| Reliability diagram | `ComposedChart` + đường chéo | OK | OK |
| Dark mode / token | biến CSS, `ChartContainer` của shadcn | khởi tạo lại theo theme | config theo theme |
| Rủi ro | 27,6k★, phát hành đều (3.10.1 ngày 2026-07-25) | 67,5k★ (Apache); wrapper React 3.0.6 | nặng; chưa có mã Altair nào để tái dùng |
| **Hạng** | **1** | 2 (nếu cần heatmap lớn) | 3 |
Ô ECharts/Vega theo tài liệu, chưa dựng. Heatmap/confusion matrix nhỏ, có số trong ô + viền đứt: `<table>`/CSS grid dễ và đúng a11y hơn thư viện. Màu để một module `lib/colors.ts` (design §4).
## 4. Client OpenAPI — chạy trên schema FastAPI thử (6 endpoint, có upload)
| | openapi-typescript + openapi-fetch | @hey-api/openapi-ts | orval |
|---|---|---|---|
| Đầu ra | 448 dòng `.d.ts`, runtime nhỏ | ~2.400 dòng (client/core kèm) | 477 dòng, hàm theo endpoint |
| `UploadFile` | `file: string` → cần `bodySerializer` (có trong docs) | `Blob \| File` | `Blob \| File` |
| TS 7 | **sập** (issue mở) | issue “TypeScript 7 support” mở | chưa thử |
| Bảo trì | 9,5M tải/tuần; bản cuối 2026-02-11; dep Redocly đã archive (chỉ lúc sinh) | 0.99.0, chưa 1.0 ⇒ rủi ro đổi API giữa minor | rất tích cực; hướng react-query/mock, cấu hình nhiều |
| **Hạng** | **1** — `.d.ts` commit sẵn nên không khoá runtime | 2 | 3 |
Luồng: `python -m edupulse.api.export_openapi` → `apps/web/openapi.json` → `pnpm gen:api` → `src/lib/api/schema.d.ts` (commit cả hai); pytest so `app.openapi()` với file để bắt lệch. FastAPI: `generate_unique_id_function=lambda r: r.name`; enum `Literal` khoá tiếng Anh, UI ánh xạ tiếng Việt; lỗi `{detail:{code,…}}`, UI dịch theo `code`.
## 5. Ánh xạ UI Streamlit → Next
`st.navigation`→`Sidebar` (có Sheet mobile) · `metric(border)`→`Card`+`Metric` tự viết · `badge`→`Badge` cặp màu token + icon (lucide gần đúng; đúng Material Symbols: `@material-symbols/svg-400` 0.48) · `segmented_control`→`ToggleGroup` (chặn giá trị rỗng) · `select_slider` 5/10/20/30→`Slider` step=1 (chỉ số→giá trị) · `pills`→chip · `tabs`→`Tabs` · `expander`→`Collapsible` · `status+progress`→`Progress` + bước từ polling · `data_editor`→`Table` + `Select` + nút “Giải thích” (thay CheckboxColumn), thứ tự cố định, 50 dòng/trang · `dialog(position=right)`→`Sheet side="right"` + Câu trước/sau · toast→`Sonner` · `file_uploader`→`<input type=file>` + vùng thả (react-dropzone hạng 2) · `download_button`→link/Blob · `warning/info`→`Alert`.
**State:** không TanStack Query (YAGNI): hook `useBatchPolling` + `?batch=` trên URL (route động `[id]` cần `generateStaticParams`, tránh); `useSearchParams` bọc `<Suspense>` (đã build OK); `demo` là batch mặc định của dashboard.
**Cạm bẫy shadcn/TanStack:** CLI shadcn 4.21.5 mặc định **Base UI**, không phải Radix ⇒ ví dụ/trí nhớ cũ (`asChild`, `type="single"`) sai, đọc mã sinh ra; `init` cần alias `@/*` trong tsconfig; `add` kéo thêm dep lạ `cn@^0.4.0` (không có clsx/tailwind-merge) — chưa đọc `lib/utils.ts`, kiểm trước khi dùng. TanStack Table `latest` = v9 (`useTable` + `tableFeatures`), mã v8 `useReactTable` theo trí nhớ sẽ lệch. Quy tắc escape markdown (design §5) bỏ: React tự escape; cấm `dangerouslySetInnerHTML`, tô XAI bằng `<mark>`.
**Cạm bẫy Next export:** `next dev` tự sinh `AGENTS.md` (tắt: `agentRules:false`); font Google tải lúc build ⇒ Docker build cần mạng (hoặc `next/font/local`); `images.unoptimized:true`. Dark mode: `next-themes` 0.4.6 (bản cuối 2025-03, repo còn commit 2026-02; dự phòng provider 30 dòng) + biến CSS theo bảng token §2, `<html lang="vi" suppressHydrationWarning>`. Font: `Be_Vietnam_Pro({subsets:["vietnamese","latin"],weight:["400","500","600","700"],variable:"--font-sans"})` + JetBrains Mono 400/500.
## 6. Sơ đồ thư mục (1 project uv + groups; workspace 3 member là hạng 2, chỉ hơn ở chỗ ép ranh giới import)
```
EduPulse/
├─ pyproject.toml · uv.lock        # groups: app (runtime) · train · dev
├─ package.json                    # script gốc: dev, gen:api, test, e2e (pnpm --dir apps/web)
├─ src/edupulse/
│  ├─ text/     # normalize · pii · segment — MỘT nguồn tiền xử lý cho train & serve
│  ├─ engine/   # ONNX session · calibration · cờ · occlusion
│  ├─ insights/ # luật r3: Wilson, từ khoá, khuyến nghị
│  └─ api/      # main.py (lifespan nạp model, mount out/) · schemas · routes/ · jobs.py · parsing.py · export.py
├─ ml/          # train/eval/export (group train; chỉ import edupulse.text)
├─ apps/web/    # Next export: src/app/{page,batch,dashboard,model}/ · components/ui · lib/{api,colors}.ts
├─ artifacts/   # onnx int8 · tokenizer.json · calibration.json · model_card.json · demo_results.json (tải từ HF Hub)
├─ tests/ · apps/web/{tests,e2e}
└─ Dockerfile   # stage node: pnpm build → out/ ; stage python: uv sync --only-group app, COPY out/
```
Đã đo: `uv sync --only-group app` chỉ cài group `app`; guard bằng test `import edupulse.api` trong env đó và `assert "torch" not in sys.modules`. Chỉ 1 gói JS ⇒ không cần `pnpm-workspace.yaml`; ghim `.nvmrc` 24.15. Dev: `next dev` :3000 + `uvicorn --reload` :8000, trình duyệt gọi thẳng `NEXT_PUBLIC_API_BASE` (CORS bật bằng env). **Không dùng rewrites ở dev**: với `trailingSlash:true`, Next redirect `/api/upload`→`/api/upload/` (thân phản hồi = đường dẫn mới), POST 9,5 MB không tới FastAPI (gọi thẳng thì tới đủ 9.961.472 byte); rewrites + export chỉ cảnh báo lúc build (exit 0).
## 7. Endpoint đề xuất (tiền tố `/api`, JSON)
| Method · Path | Việc | Ghi chú |
|---|---|---|
| GET `/health` | `ready`, `model_version`, RSS | dùng làm health check của host |
| POST `/analyze` | `{text≤500}` → 2 nhãn + xác suất đã hiệu chỉnh, cờ + lý do, text đã ẩn PII, cảnh báo không dấu | sync, `def` |
| POST `/explain` | `{text}` → occlusion 2 head (từ, Δ xác suất) | gọi song song `/analyze` để kết quả hiện trước |
| POST `/batches` | multipart CSV/XLSX → `{batch_id, columns, n_rows, preview, encoding, warnings}` | parse sync, ≤10 MB, ≤2.000 dòng |
| POST `/batches/{id}/run` | `{text_col, group_col?, review_budget∈{5,10,20,30}}` → 202 | job nền |
| GET `/batches/{id}` · DELETE | `{status, done, total, eta_s, error_code}` · xoá ngay | poll 1 Hz; 404 = hết phiên |
| GET `/batches/{id}/results` | mọi dòng: text ẩn PII, nhãn, xác suất, cờ, lý do, `in_review` | gzip, ~1 MB/2.000 dòng (ước) |
| GET `/batches/{id}/rows/{i}/explain` | occlusion theo yêu cầu (~1 s) | cache trong batch; top-20 tính sẵn |
| GET `/batches/{id}/insights?group=…` | KPI, cảnh báo Wilson, dữ liệu cột chồng + heatmap, từ khoá, câu tiêu biểu, khuyến nghị | `id=demo` nạp sẵn lúc khởi động |
| POST `/batches/{id}/export` | `{kind: all\|corrected, overrides}` → CSV `utf-8-sig` | chống CSV injection |
| GET `/model-card` · `/sample.csv` | JSON tĩnh trong `artifacts/` · file mẫu | cùng thư mục với model ⇒ số liệu khớp bản triển khai |
**Quy tắc:** `--workers 1` bắt buộc (trạng thái trong RAM); endpoint CPU dùng `def`, không `async def`; 1 thread inference + khoá toàn cục theo mini-batch 16–32 câu để `/analyze` chen vào ≤ ~1 giây; `JobStore` dict+Lock, TTL 30 phút từ lần truy cập cuối, tối đa 8 batch, dọn lười, hàng đợi >3 ⇒ 429 `busy`; sửa nhãn giữ ở client, gửi `overrides` khi export (khớp “tải file trước khi đóng tab”); log không ghi nội dung câu; `uvicorn --proxy-headers` (Starlette redirect bằng URL tuyệt đối); mount static ở `/` biến `/api/*` sai thành 404 HTML (đã đo) ⇒ thêm catch-all JSON 404.
**Parse/an toàn:** stdlib `csv` + `Sniffer` (Excel locale VN hay dùng `;`); giải mã `utf-8-sig`, rơi về `cp1258` + cảnh báo; XLSX bằng `python-calamine`, kiểm tổng `file_size` trong zip ≤ ~100 MB trước khi mở; đếm byte khi đọc upload (413); xuất CSV thêm `'` đầu ô bắt đầu bằng `= + - @`. RSS import đo trên M1: numpy+onnxruntime+tokenizers 52 MB, + FastAPI/uvicorn/pydantic 73 MB, pandas thêm ~40–70 MB (nhiễu ±30), openpyxl/calamine ≈ 0. Hạng: csv+calamine > pandas(engine=calamine) > openpyxl (bản cuối 2024-06, cần thêm defusedxml 0.7.1 từ 2021).
## 8. Test (dữ liệu thật, không mock)
- **pytest 9.1.1 + `TestClient`** (cần httpx 0.28.1): nhóm nhanh trên fixture thật (tập con UIT-VSFC; CSV/XLSX mẫu có BOM, `;`, cp1258) + nhóm `@pytest.mark.model` chạy ONNX thật; snapshot OpenAPI; golden tiền xử lý train↔serve trên câu thật; ngân sách thời gian (1 câu < 2 s, 1.000 câu < 2 phút) đo trên host thật.
- **Vitest 5 + Testing Library:** `ReviewTable` (sửa `Select` không đổi thứ tự), hook polling, định dạng số kiểu Việt; fixture là JSON ghi lại từ API thật, không viết tay.
- **Playwright 1.64:** chạy trên site đã export do `uvicorn` phục vụ (kiểm luôn cấu trúc `out/`); 1 kịch bản vàng: phân tích 1 câu → file mẫu → chạy → sửa nhãn → mở ngăn kéo → xuất CSV → dashboard demo; chụp ảnh sáng/tối.
## 9. Công sức (1 dev + AI; giả định r1/r3 giao engine + luật insight dạng thư viện; ngày công, ước lượng của tôi, không đo)
| Hạng mục | Tốt | Thường | Xấu |
|---|---|---|---|
| Scaffold: monorepo, Next+shadcn+token+dark+font+sidebar, pipeline OpenAPI, Dockerfile | 0,5 | 1 | 2 |
| API lõi: engine glue, analyze/explain, store/giới hạn, model-card, health | 1 | 1,5 | 2,5 |
| API batch: parse CSV/XLSX, job + tiến trình, results, insights, export | 1,5 | 2,5 | 4 |
| Trang 1 Phân tích câu (+ tô XAI) | 0,75 | 1 | 2 |
| Trang 2 Hàng loạt (upload→ngân sách→tiến trình→bảng duyệt→ngăn kéo→xuất) | 1,5 | 2,5 | 4 |
| Trang 3 Dashboard (KPI, thẻ cảnh báo, cột chồng, heatmap, từ khoá, câu tiêu biểu) | 1,25 | 2 | 3,5 |
| Trang 4 Mô hình & đạo đức | 0,5 | 1 | 1,5 |
| Test + QA sáng/tối + smoke triển khai | 1 | 1,5 | 2,5 |
| **Tổng** | **8** | **13** | **22** |
Còn ~10 ngày công (15→28/10) dùng chung với train/đánh giá/báo cáo ⇒ **rủi ro cao** nếu cùng một người. Làm được khi: contract-first (chốt OpenAPI ngày 1–2 để API và 4 trang chạy song song bằng nhiều agent), engine/insights xong trước ngày 4, và cắt: bỏ toggle LinearSVC, bỏ sparkline trong metric, bảng duyệt chỉ phân trang (không sort/lọc/ảo hoá), polling thay SSE, CSV trước rồi XLSX, 1 kịch bản E2E, dark mode chỉ kiểm tay trang 1–3.
**Rủi ro theo thứ tự:** (1) băng thông 1 người chia cho ML; (2) RAM/CPU host: ONNX + JVM ~600 MB RSS (đo trước đó) + web 73 MB, mốc 1.000 câu < 2 phút chưa đo trên x86; (3) stack toàn bản mới (Next 16, React 19.3, TS 7, Tailwind 4.3, shadcn 4/Base UI, TanStack 9, pnpm 12, Vitest 5, ESLint 10, jsdom 30) ⇒ mã AI viết theo trí nhớ dễ lệch, ghim phiên bản; (4) rời Streamlit nên design §10 mục 1–5 thành QA tương phản/dark mode thủ công trên trình duyệt; (5) host chưa chốt (r5).
## 10. Rủi ro chấp nhận (độ chín muồi)
`next-themes` (bản cuối 2025-03) · `openapi-typescript` (bản cuối 2026-02, TS 6/7 chưa hỗ trợ) · `@hey-api/openapi-ts` 0.x · `openpyxl` (2024-06) · `httpx` 0.28.1 (2024-12, bắt buộc cho TestClient) · TanStack Table v9 mới · Base UI mặc định của shadcn. Giảm thiểu: ghim exact; đầu ra generator là `.d.ts` commit sẵn; shadcn copy mã vào repo nên thay được từng component.
## 11. Hạn chế nghiên cứu
Nguồn: npm registry + deps.dev + GitHub API, docs Next (static-exports, export-no-custom-routes), shadcn (llms.txt, trang docs), docs openapi-fetch, Google Fonts CSS, cộng thử nghiệm thật trong thư mục tạm. Chưa làm: chạy app/model thật (độ trễ lấy từ PDR và số đo trước); `pnpm run build` (install OK, build lỗi do tôi quên copy `hooks/`, không phải lỗi pnpm); Next dev + `skipTrailingSlashRedirect`; dựng ECharts/Vega thật (chỉ đo bundle); axe/tương phản; máy x86; RSS Node SSR; nguồn dep `cn@^0.4.0`. Chưa đọc HTML wireframe (index 535 dòng + 4 trang) — có thể port sang JSX nhanh hơn viết lại.
## Câu hỏi chưa giải quyết
1. Host cuối (r5) có chạy được 1 container Python phục vụ cả static không? Nếu buộc tách (Vercel + API) → cần CORS và kiểm giới hạn body 10 MB.
2. Nhãn sửa chỉ ở client (khuyến nghị) hay lưu server? Dashboard có phản ánh nhãn đã sửa không?
3. shadcn: Base UI (mặc định) hay Radix (khớp tài liệu cũ)? Flag chọn base khi `init` chưa kiểm.
4. Icon: lucide (gần đúng) hay Material Symbols Rounded đúng thiết kế?
5. 1 project uv + groups (khuyến nghị) hay workspace 3 member như gợi ý ban đầu?
6. Có được commit tập con UIT-VSFC làm fixture/demo không (giấy phép học thuật, phi thương mại)?
7. Có giữ toggle PhoBERT↔LinearSVC không (thêm RAM và ~1 ngày)?
