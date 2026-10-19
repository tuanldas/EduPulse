---
title: "Phase 7: Frontend 4 trang"
status: todo
priority: P1
effort: 32.5h
dates: "2026-10-18 → 2026-10-24"
depends_on: [4, 6]
---

# Phase 7: Frontend 4 trang
<!-- Updated: Validation Session 1–2 - so sánh mô hình + heatmap thành Must (bảo vệ), lọc câu tiêu biểu theo từ khoá vẫn Should (S1 #2); câu mẫu/file mẫu/demo lấy từ dữ liệu build-time, thiếu thì ẩn + thông báo (S1 #3); nhãn "dữ liệu minh hoạ" theo `illustrative_groups` (S1 #4); heartbeat 60 s + sendBeacon `close` + cảnh báo beforeunload (F5, S2 #2); link mã nguồn lấy `source_url` từ `/api/config` (F9); insights gọi POST (F4); fixture Vitest từ 20 câu tự viết (S1 #3, F8); danh sách trạng thái theo đúng mục wireframe (F17); trang 4 hiện stage v0/v1 + ghi chú x86; ước lượng lại 32,5h (F13); mốc dừng M5 -->

## Context links
- [design-guidelines](../../docs/design-guidelines.md): nguyên tắc (L5–12), token màu (L14–45), chữ/khoảng cách (L47–53), quy tắc từng trang (L113–141), a11y (L143–148), giọng văn (L150–154), icon/logo (L156–158), mục cần kiểm (L160–166).
- Wireframe đã duyệt: [index](../../docs/wireframes/index.html), [trang 1](../../docs/wireframes/01-phan-tich-cau.html) (trạng thái: "Các trạng thái khác của khu vực kết quả" L550), [trang 2](../../docs/wireframes/02-phan-tich-hang-loat.html) ("Các trạng thái khác" L549), [trang 3](../../docs/wireframes/03-dashboard-insight.html) ("Trạng thái mặc định" L543), [trang 4](../../docs/wireframes/04-mo-hinh-dao-duc.html) (không có mục trạng thái; BibTeX L563–580); ảnh 1440/390/dark ở `docs/wireframes/screenshots/`.
- [r6 kiến trúc](./research/researcher-02-architecture.md): chart (L22–32), ánh xạ component + state (L42–44), bẫy shadcn/Next (L45–46), test Vitest (L82); công sức trang theo kịch bản (L84–96).
- Hợp đồng API: `apps/web/openapi.json` (phase 6 bước 1). Báo cáo kiểm chứng: [F4, F5, F9, F13, F17](./reports/validation-verification.md).

## Overview
- Ngày: 2026-10-18 → 2026-10-24 · Effort: 32.5h (nền tảng 7h · trang 1 4h · trang 4 4h · trang 2 7,5h · trang 3 6h · Vitest 2h · xem trước 2h) · Priority: P1 (khối lượng lớn nhất, rủi ro tiến độ cao) · Trạng thái: chưa bắt đầu.
- Cập nhật design-guidelines cho Next.js, dựng nền tảng (token, font, theme, khung, client API, vòng đời batch), rồi 4 trang đúng wireframe, 100% tiếng Việt, sáng/tối. Sau nền tảng chia 2 agent: (A) trang 1 + 4 (~8h), (B) trang 2 + 3 (~13,5h). Dùng bundle v0 + `make demo-data` trước; số liệu tự đúng khi phase 5 ra v1. **Mốc dừng M5** = xong khối UI (người dùng xem ảnh 4 trang).

## Key Insights
- shadcn CLI 4.21.5 mặc định Base UI, KHÔNG Radix: ví dụ cũ (`asChild`, `type="single"`) sai → đọc mã sinh ra + docs hiện hành; `init` cần alias `@/*`; `add` kéo dep lạ `cn@^0.4.0` → kiểm `lib/utils.ts`, gói không rõ nguồn thì thay bằng `clsx` + `tailwind-merge` (r6 L45).
- Export tĩnh: không SSR/Route Handler/Server Actions/rewrites; route động tránh → dùng `?batch=`; `useSearchParams` bọc `<Suspense>`; `images.unoptimized` (r6 L44–46). Chuyển trang bằng `<Link>` (client) để batch không bị `close` khi sang dashboard.
- Vòng đời batch (S2 #2): hook `use-batch-session` gửi heartbeat mỗi `heartbeat_s` (60 s, lấy từ `/api/config`) khi trang batch/dashboard đang mở; `pagehide` → `navigator.sendBeacon('/api/batches/{id}/close')` (sendBeacon chỉ gửi POST); có nhãn sửa chưa xuất → `beforeunload` hỏi lại. 404 = phiên đã kết thúc.
- Không `dangerouslySetInnerHTML`; React tự escape nên bỏ quy tắc escape markdown của Streamlit (design L111); tô XAI bằng `<mark>`: ủng hộ = đậm + số hạng ¹²³ + nền `rgba(91,104,238,.45/.30/.18)`; phản đối = nghiêng + "−" + nền `rgba(139,144,154,.30)` (design L44).
- Token §2 → biến CSS `:root`/`.dark` (next-themes class) + 1 module `lib/colors.ts` cho chart/heatmap (không rải hex). Chữ màu primary trên nền tối dùng `#8B96FF`; nút primary dark chữ trắng trên `#5B68EE` (4,5:1) — design L161.
- Cột chồng 100% có vạch mức nền KHÁC NHAU mỗi hàng: `ReferenceLine` của Recharts không đủ → tự vẽ shape hoặc dựng bằng CSS flex (cùng kiểu heatmap/confusion matrix bằng CSS grid, r6 L32); Recharts cho reliability diagram (Bar + đường chéo) và sparkline (Could).
- Icon đúng thiết kế: Material Symbols Rounded — chép path SVG ~30 icon (Apache-2.0) vào 1 component, tô `currentColor`; lucide chỉ còn trong component shadcn.
- Bảng duyệt: shadcn `Table` + `Select` từng dòng, thứ tự cố định, phân trang 50, "Giải thích" mở `Sheet side="right"` có Câu trước/Câu sau (r6 L43); nhãn sửa lưu `sessionStorage` theo batch id.
- Dữ liệu build-time (S1 #3): câu mẫu trang 1, "Dùng file mẫu", nguồn "Dữ liệu demo" chỉ hiện khi `/api/config.demo_available`; thiếu → ẩn + thông báo "Dữ liệu demo chưa sẵn (bản build không tải được dataset)". Nhóm minh hoạ (S1 #4): `illustrative_groups` → badge "Dữ liệu minh hoạ — nhóm gán ngẫu nhiên" ở chọn cột nhóm, heatmap, thẻ cảnh báo theo nhóm, bộ lọc nhóm.
- Số liệu luôn lấy từ API (`/api/config`, `/api/model-card`, insights); số wireframe chỉ minh hoạ (design L166). Số kiểu Việt `54,1%` · `1.000` · `0,018`, chênh lệch ghi "điểm" (design L50). Chép chuỗi tiếng Việt từ HTML wireframe (bỏ khung chú thích hồng); giọng văn trung lập, không tên người (design §8).
- Ước lượng theo kịch bản "Thường" của r6 (trang + khung + QA ≈ 9 ngày công người) × ~50% nhờ agent ⇒ 32,5h; bản cũ 24h sát kịch bản "Tốt" (F13).

## Requirements
- Khung: sidebar (logo, 4 mục: Phân tích câu `rate_review` · Phân tích hàng loạt `upload_file` · Dashboard & insight `insights` · Mô hình & đạo đức `balance`; "Mô hình đang dùng" + phiên bản từ `/api/health`; disclaimer; "Mã nguồn · AGPL-3.0" → `source_url` từ `/api/config`; "Không lưu dữ liệu bạn nhập"; công tắc Sáng/Tối/Hệ thống); ≤640px → Sheet điều hướng; đầu trang: tiêu đề + icon + 1 câu mục đích.
- Trang 1 (design L114–120): textarea ≤500 + đếm ký tự; chip câu mẫu từ `ui_samples` (chọn = chạy luôn); Phân tích/Xoá; badge "Cần xem xét" + lý do bằng lời; 2 thẻ Cảm xúc | Chủ đề (nhãn lớn + icon + %, thanh "Độ tin cậy (đã hiệu chỉnh)", xác suất từng lớp); XAI (ToggleGroup CX/CĐ, câu tô, chú giải, bảng "Bỏ từ này thì xác suất …: 93% → 46%"); trạng thái A–E ở wireframe 01 L550 (không dấu, đã ẩn thông tin cá nhân, >40 từ, đủ tin cậy, chưa nhập) + thiếu câu mẫu; Switch "So sánh với mô hình cơ sở" mặc định TẮT (Must).
- Trang 2 (L122–128): thông báo riêng tư (tên che best effort, khuyên ẩn danh trước khi tải); B1 tải file (kéo thả + chọn, CSV/XLSX ≤10MB) + "Dùng file mẫu (200 câu UIT-VSFC)" (`POST /api/batches/sample`) + xem trước 5 dòng; B2 chọn cột văn bản/nhóm (+ badge minh hoạ), Slider ngân sách 5/10/20/30 (mặc định 10) + thẻ "Hiệu quả ước tính (validation)" 3 chỉ số (sparkline: Could), nút "Phân tích n câu" + ước tính thời gian; khi chạy: danh sách bước + progress "640/1.000 câu · còn khoảng 25 giây"; B3 4 metric + Tabs "Cần duyệt (n)" / "Tất cả kết quả (n)" + bảng duyệt (# · Câu · Lý do · Cảm xúc · Sửa cảm xúc · Chủ đề · Sửa chủ đề · Giải thích · Tin cậy CX/CĐ · Nhóm) + Sheet giải thích; B4 "Tải toàn bộ kết quả (CSV)", "Tải nhãn đã sửa", link dashboard, cảnh báo tải trước khi đóng tab; trạng thái A–D ở wireframe 02 L549 (chưa có file, file lỗi, >2.000 câu khoá nút, toast tạo file) + phiên đã kết thúc + thiếu file mẫu.
- Trang 3 (L130–136): nguồn (Dữ liệu demo · UIT-VSFC | File vừa tải) + lọc nhóm (chỉ khi có cột nhóm; gửi qua body `POST …/insights`); 4 KPI (tiêu cực/tích cực kèm "điểm so với nền"); ≤3 thẻ cảnh báo (nhận định + câu có số + căn cứ KTC 95%, n, badge + "Xem câu liên quan"); cột chồng 100% chủ đề × cảm xúc + vạch nền + n; heatmap nhóm × chủ đề theo chênh so với nền (thang 6 bậc, n<20 không tô + viền đứt) (Must) + badge minh hoạ; từ khoá 3 khối (CSVC, CT, GV), chọn để lọc câu tiêu biểu (Should); câu tiêu biểu theo Tabs chủ đề (từ khoá đậm, badge cảm xúc + nhóm); khuyến nghị + cảnh báo "chỉ để gợi ý"; "Trạng thái mặc định" ở wireframe 03 L543 (demo không nhóm: ẩn lọc + heatmap; thông báo khi không chủ đề nào vượt nền) + demo chưa sẵn.
- Trang 4 (L138–141): cảnh báo disclaimer + mục lục neo; Hiệu năng (nhãn theo `stage`: "bản smoke · số trên val" hoặc "tập test"): ToggleGroup CX/CĐ → 4 metric (Accuracy, Macro-F1, ECE, Lỗi được bắt ở ngân sách 10%), bảng từng lớp (F1 thanh), confusion matrix (% theo hàng), bảng so sánh 3 baseline + đơn nhiệm + đa nhiệm int8 (dòng đang dùng đậm), reliability diagram + "Đọc các chỉ số thế nào?"; Model card 2 cột (gồm ghi chú "chưa đo trên x86" và demo tính lúc build); Quyền riêng tư & vai trò con người (3 khối); Giấy phép, mã nguồn & trích dẫn (bảng license, link GitHub/HF, BibTeX + nút sao chép).
- Mỗi biểu đồ có tiêu đề + caption "cách đọc" + số trong ô/bảng tương đương (design L146); chữ ≥12px; điều hướng bàn phím + focus ring.

## Architecture
```
apps/web/src/
├─ app/ layout.tsx (lang="vi", font, ThemeProvider, AppShell) · globals.css (token) · page.tsx · batch/ · dashboard/ · model/ · not-found.tsx
├─ components/ ui/ (shadcn) · shell/ · common/ (icon, badge nhãn, metric, thanh xác suất, câu XAI, badge minh hoạ, thông báo thiếu demo) · analyze/ · batch/ · dashboard/ · model/
└─ lib/ api/{client.ts, schema.d.ts (sinh), errors.ts} · colors.ts · format.ts · labels.ts · use-batch-polling.ts · use-batch-session.ts · overrides-store.ts · app-config.ts
```
- Dữ liệu: component client gọi `openapi-fetch` (không TanStack Query — YAGNI); `/api/config` nạp 1 lần (context) cho giới hạn, ngân sách, câu mẫu, `demo_available`, `source_url`, `heartbeat_s`; upload multipart dùng `bodySerializer` hoặc `fetch` thường; tải CSV = Blob + `<a download>`.
- Trạng thái: `?batch=` cho dashboard; batch gần nhất + nhãn sửa trong `sessionStorage`; polling 1 Hz dừng khi `done`/`error`/404; heartbeat 60 s + `close` khi `pagehide`.
- Lỗi: `detail.code` → câu tiếng Việt trong `lib/api/errors.ts` (vd. `busy` → "Máy chủ đang bận, thử lại sau ít phút"; `demo_unavailable` → "Dữ liệu demo chưa sẵn").
- Sở hữu file khi chia 2 agent: nền tảng sở hữu `app/layout.tsx`, `globals.css`, `components/{ui,shell,common}`, `lib/`; agent A sở hữu `app/page.tsx`, `app/model/`, `components/{analyze,model}/`; agent B sở hữu `app/batch/`, `app/dashboard/`, `components/{batch,dashboard}/`. Cần component dùng chung mới → báo người giữ nền tảng.

## Related code files
Modify:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docs/design-guidelines.md` (§4 token → biến CSS + `lib/colors.ts`; §5 khung Next; §6.1 "So sánh mô hình: đã duyệt, mặc định tắt"; §10 danh sách kiểm trên trình duyệt)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/package.json` (shadcn, next-themes, recharts, openapi-typescript 7.13.0, openapi-fetch 0.17.0, vitest 5.0.4, jsdom 30.1.2, @testing-library/react 16.3.3, user-event 14.6.8, jest-dom 7.0.1, @playwright/test 1.64.1; script `gen:api`, `test`, `e2e`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/layout.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/globals.css`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/page.tsx`

Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/components.json`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/public/logo.svg`, `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/public/logo-mark.svg` (chép từ `docs/wireframes/assets/`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/batch/page.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/dashboard/page.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/model/page.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/app/not-found.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/ui/*.tsx` (shadcn sinh: button, card, badge, select, slider, tabs, toggle-group, sheet, table, progress, alert, tooltip, collapsible, sonner, chart, skeleton, separator, textarea, switch, dropdown-menu)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/shell/{app-shell,app-sidebar,mobile-nav,page-header,theme-toggle}.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/common/{icon,label-badge,flag-badge,metric-card,prob-bars,confidence-bar,xai-sentence,xai-table,illustrative-badge,demo-unavailable-notice}.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/analyze/{analyze-form,sample-chips,result-cards,explain-panel,compare-panel,state-notices}.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/batch/{upload-step,settings-step,budget-efficiency,run-progress,review-table,explain-sheet,export-step}.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/dashboard/{source-switch,kpi-row,alert-cards,sentiment-by-topic,delta-heatmap,keyword-chips,representative-sentences,recommendations}.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/components/model/{perf-section,per-class-table,confusion-matrix,comparison-table,reliability-chart,metrics-guide,model-card,privacy-blocks,license-citation}.tsx`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/lib/api/{client.ts,schema.d.ts,errors.ts}`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/src/lib/{colors,format,labels,use-batch-polling,use-batch-session,overrides-store,app-config}.ts`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/vitest.config.ts`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/playwright.config.ts`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/e2e/visual.spec.ts` (chụp 4 trang 1440/390 × sáng/tối)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/tests/{format,labels,xai-sentence,review-table,use-batch-polling,use-batch-session,illustrative-badge}.test.ts(x)`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/apps/web/tests/fixtures/*.json` (ghi từ API thật trên 20 câu tự viết `tests/fixtures/golden_sentences.json` + 1 batch dựng từ chúng có nhóm minh hoạ; không chứa câu UIT-VSFC)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/scripts/record_fixtures.py`

## Implementation Steps
1. Cập nhật `docs/design-guidelines.md` (§4, §5, §6.1, §10) cho Next.js TRƯỚC khi code UI; giữ nguyên token và quy tắc trang.
2. Nền tảng (1 agent, ~6h sau bước 1): `shadcn init` (Base UI) + kiểm `cn`; `add` component; `globals.css` token sáng/tối; font Be Vietnam Pro (`subsets: ["vietnamese","latin"]`, 400–700) + JetBrains Mono; next-themes (`<html lang="vi" suppressHydrationWarning>`); khung sidebar + mobile Sheet + page header; `icon.tsx`; `lib/{colors,format,labels,app-config}`; `pnpm gen:api` → `schema.d.ts`; `lib/api/{client,errors}`; `use-batch-polling`, `use-batch-session`, `overrides-store`; badge minh hoạ + thông báo thiếu demo.
3. Agent A — trang 1 (API phân tích/giải thích có từ phase 4): form, chip mẫu, thẻ kết quả, XAI, trạng thái; compare (Must, khi phase 6 bước 9 xong).
4. Agent A — trang 4 (`/api/model-card` có từ bundle v0): hiệu năng (nhãn stage), CM, so sánh, reliability, model card (ghi chú x86/demo), quyền riêng tư, giấy phép + BibTeX.
5. Agent B — trang 2 (tích hợp khi phase 6 bước 2–5 xong): upload + file mẫu, thiết lập + hiệu quả ước tính, tiến trình, bảng duyệt + Sheet, xuất CSV, heartbeat/close/beforeunload, trạng thái.
6. Agent B — trang 3 (khi insights xong): nguồn/lọc (POST body), KPI, cảnh báo, cột chồng, heatmap + badge minh hoạ, từ khoá (lọc: Should), câu tiêu biểu, khuyến nghị, trạng thái demo/không có demo.
7. Vitest (fixture ghi từ API thật bằng `scripts/record_fixtures.py` trên 20 câu tự viết): định dạng số Việt; ánh xạ nhãn; `XaiSentence`; `ReviewTable` (đổi Select không đổi thứ tự, phân trang 50, overrides đúng dòng); `useBatchPolling` (dừng khi done/404); `useBatchSession` (heartbeat theo chu kỳ, beacon khi `pagehide`); badge minh hoạ.
8. Xem trước bằng Playwright (`e2e/visual.spec.ts`): 4 trang × 1440/390 × sáng/tối, đặt cạnh `docs/wireframes/screenshots/`; sửa lệch; kiểm tay tương phản dark (design L161).
9. `pnpm lint && pnpm exec tsc --noEmit && pnpm test && pnpm build` sạch; **dừng M5** (người dùng xem ảnh 4 trang); commit theo trang (`feat(web): …`).

## Todo list
- [x] Cập nhật design-guidelines §4, §5, §6.1, §10 cho Next.js
- [x] Nền tảng: shadcn (Base UI, kiểm `cn`), token, font, theme, khung, icon, lib, client API, `use-batch-session`, badge minh hoạ
- [ ] Trang 1 Phân tích câu (+ XAI, trạng thái, compare Must)
- [ ] Trang 4 Mô hình & đạo đức (nhãn stage, ghi chú x86/demo)
- [ ] Trang 2 Phân tích hàng loạt (upload/file mẫu → ngân sách → tiến trình → bảng duyệt → Sheet → xuất; heartbeat/close)
- [ ] Trang 3 Dashboard & insight (heatmap Must; lọc từ khoá Should)
- [ ] Vitest + fixture ghi từ API thật trên câu tự viết
- [ ] Ảnh Playwright 1440/390 sáng/tối, so wireframe, sửa lệch
- [ ] Lint + typecheck + test + build sạch; mốc M5; commit

## Success Criteria
- `pnpm build` sinh `out/index.html`, `out/batch/index.html`, `out/dashboard/index.html`, `out/model/index.html`; 0 lỗi TS/ESLint.
- Trạng thái ở wireframe 01 L550 (A–E), 02 L549 (A–D), 03 L543 (mặc định) hiển thị được với dữ liệu thật; thêm trạng thái "thiếu dữ liệu demo" ở trang 1–3 và "bản smoke" ở trang 4.
- Vitest pass; 0 mã hex rải trong component (chỉ ở `globals.css`/`lib/colors.ts`); không fixture nào chứa câu UIT-VSFC.
- 1 câu: nhãn hiển thị ngay khi `/api/analyze` trả về, XAI hiện sau không chặn kết quả; duyệt >30' không mất batch khi tab mở (heartbeat).
- Ảnh chụp 4 trang khớp bố cục wireframe; người dùng đã xem ở M5.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Khối lượng vượt 32,5h | Cao × Cao | 2 agent song song; cắt theo thứ tự: sparkline → polish mobile 390 → axe (Could) → Vitest đầy đủ → lọc câu tiêu biểu theo từ khoá (Should); compare + heatmap là Must (bảo vệ), chỉ cắt khi người dùng quyết ở M6 |
| API Base UI khác trí nhớ / ví dụ cũ | Cao × TB | Đọc mã shadcn sinh ra, docs hiện hành; không chép ví dụ Radix |
| Hợp đồng API đổi | TB × TB | `make gen-api` + snapshot (phase 6); TS báo lỗi ngay |
| `pagehide` không bắn trên vài trình duyệt/di động | TB × Thấp | TTL 30' là lưới an toàn; heartbeat dừng thì batch tự hết hạn |
| Recharts không vẽ được vạch nền từng hàng | TB × Thấp | Dựng bằng CSS flex |
| Font Google lúc build trong Docker cần mạng | Thấp × Thấp | Build có mạng; dự phòng `next/font/local` |

Rollback: commit theo trang; `git revert` 1 trang không ảnh hưởng trang khác (sở hữu theo thư mục).

## Security Considerations
- Không `dangerouslySetInnerHTML`; văn bản người dùng chỉ render qua JSX; link ngoài `rel="noopener noreferrer"`.
- Không secret trong `NEXT_PUBLIC_*` (chỉ `NEXT_PUBLIC_API_BASE` ở dev); `source_url` lấy lúc chạy từ `/api/config`.
- Kiểm nguồn gốc gói `cn` mà shadcn kéo theo trước khi giữ; ghim chính xác phiên bản, commit `pnpm-lock.yaml`.
- Nhãn sửa chỉ ở `sessionStorage` của tab; không gửi đi đâu ngoài export; đóng tab → `close` xoá batch phía server.

## Next steps
- Phase 8 viết E2E vàng + edge (heartbeat/TTL, thiếu demo) trên container + QA a11y/tương phản. Phase 9 chụp ảnh cuối vào `results/screenshots/` và README.
