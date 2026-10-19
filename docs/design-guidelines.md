# EduPulse — Design guidelines (Next.js + Tailwind CSS v4 + shadcn/ui)

> Nguồn chuẩn cho giao diện app. Wireframe duyệt: [`docs/wireframes/index.html`](wireframes/index.html) (+ ảnh trong `wireframes/screenshots/`). Cập nhật 2026-10-19: giao diện chuyển từ Streamlit sang Next.js export tĩnh (xem [`tech-stack.md`](tech-stack.md)); bố cục, nội dung và token màu của wireframe giữ nguyên, chỉ đổi cách hiện thực. Mã token: `apps/web/src/app/globals.css` và `apps/web/src/lib/colors.ts`. Danh sách đã kiểm và còn lại ở mục 10.

## 1. Nguyên tắc
1. **AI gợi ý, người quyết định.** Mọi nhãn kèm độ tin cậy đã hiệu chỉnh; câu dưới ngưỡng có cờ “Cần xem xét” + lý do. Không có thao tác tự động.
2. **Trung lập, không quy lỗi cá nhân.** Tổng hợp theo chủ đề/môn; không hiện tên giảng viên; câu chữ mô tả dữ liệu, không phán xét người.
3. **Màu không bao giờ là tín hiệu duy nhất.** Luôn kèm icon + chữ (+ số khi có).
4. **shadcn/ui (Base UI) trước, tự viết khi cần.** Tô từ XAI bằng thẻ `<mark>`; React tự escape văn bản người dùng và cấm `dangerouslySetInnerHTML`. Màu chỉ khai ở `globals.css` / `lib/colors.ts`, component dùng tên token, không mã hex.
5. **Lời trước, số sau.** Thẻ cảnh báo là câu hoàn chỉnh; chi tiết thống kê để ở caption/tooltip.
6. **Dày thông tin nhưng quét được:** KPI → cảnh báo → biểu đồ → chi tiết. Mỗi biểu đồ có tiêu đề + caption “cách đọc”.
7. **Trình chiếu được:** desktop 1440 là chính; chữ ≥ 14px; viền widget ≥ 3:1.
8. **Số liệu luôn từ API** (`/api/config`, `/api/model-card`, insight). Số trên wireframe chỉ minh hoạ; UI không ghi cứng số đo.

## 2. Màu
### 2.1 Nền, chữ, primary
| Token | Light | Dark | Biến CSS · ghi chú (tương phản) |
|---|---|---|---|
| bg | `#FAFAF8` | `#111419` | `--background` |
| bg2 | `#F3F3EE` | `#1A1E25` | `--muted`, `--secondary` — sidebar, ô nhập |
| raised | `#FFFFFF` | `#1E232B` | `--popover` — Sheet, menu, hộp thoại |
| hover | `#E7E6E0` | `#262B34` | `--accent` — mục điều hướng đang mở, hover |
| text | `#14171E` | `#E7E9EE` | `--foreground` — 17,2 / 15,2:1 |
| muted | `#5B6170` | `#A2A8B4` | `--muted-foreground` — 5,9 / 7,7:1 (trên bg2: 5,6 / 7,0:1) |
| border | `#918F87` | `#5F6573` | `--border`, `--input` — 3,1 / 3,2:1 (WCAG 1.4.11 cho ô nhập, khung, nút viền) |
| divider | `#DCDAD2` | `#2C313A` | `--divider` — đường kẻ trang trí, tab, sidebar |
| grid | `#E4E2DB` | `#272C35` | `--grid` — lưới trong bảng, nền thanh tiến độ |
| primary | `#3A4BD0` | `#5B68EE` | `--primary` — nền nút/công tắc/thanh; chữ trắng trên nút 6,8 / 4,5:1. Hover `--primary-hover`: `#2F3EBB` / `#4F5CE0` (tối chọn bậc **tối hơn** để chữ trắng vẫn ≥ 5,3:1; `#6D79F2` của wireframe sáng hơn nên chỉ 3,7:1, axe bắt được khi con trỏ nằm trên nút) |
| primary-text | `#3A4BD0` | `#8B96FF` | `--primary-text` — **chữ** màu primary (chip chọn, tab, số hạng): 6,5 / 7,0:1. `#5B68EE` làm chữ trên nền tối chỉ 4,1:1 nên không dùng |
| primary-soft | `#E8EAFB` | `#22284F` | `--primary-soft` — nền mục đang chọn (chip, phân đoạn); chữ `primary-text` trên đó 5,7 / 5,3:1 |
| link | `#3A4BD0` | `#8B96FF` | `--link` — 6,5 / 7,0:1 |

### 2.2 Màu ngữ nghĩa (thứ tự cố định)
| Ý nghĩa | Icon (Material Symbols) | Fill biểu đồ `--{khoá}`, chung 2 theme | Chữ / nền badge — light `--{khoá}-fg` / `-bg` | Chữ / nền badge — dark | Khoá Tailwind |
|---|---|---|---|---|---|
| Tiêu cực | `sentiment_dissatisfied` | `#C94827` | `#C0401E` / `#FDEBE5` | `#EF6C4B` / `#41201A` | `neg` |
| Trung tính | `sentiment_neutral` | `#8B909A` | `#62666F` / `#EDEEF1` | `#A3A8B2` / `#2A2D33` | `neu` |
| Tích cực | `sentiment_satisfied` | `#1B9A6E` | `#007752` / `#DDF4E9` | `#3DB083` / `#0E3326` | `pos` |
| Giảng viên | `co_present` | `#0167BB` | `#0F64B4` / `#E5EFFC` | `#5AA2F2` / `#132C47` | `gv` |
| Chương trình | `menu_book` | `#944A76` | `#93496F` / `#F9E8F1` | `#CF84AF` / `#3B2130` | `ct` |
| Cơ sở vật chất | `domain` | `#B88818` | `#835F00` / `#FBEFD3` | `#CFA03A` / `#382809` | `csvc` |
| Khác | `more_horiz` | `#66837A` (chỉ biểu đồ) | dùng chung cặp gray (`khac-fg` / `khac-bg` trùng `neu`) | dùng chung cặp gray | `khac` |
| Cần xem xét (cờ) | `flag` | `#F2C230` — không làm fill biểu đồ | `#7A5E00` / `#FCEFC4` | `#E3B52B` / `#352905` | `flag` |

Dùng qua lớp Tailwind: `bg-neg` (fill), `text-neg-fg`, `bg-neg-bg` (badge); `lib/colors.ts` có `TONE_CLASS[tone]` và `toneVar(tone)` cho SVG.

- Đã đo bằng test tự động `apps/web/tests/contrast.test.ts` (đọc thẳng `globals.css`, chạy ở cả hai theme): mỗi fill ≥ 3:1 trên nền trang; mỗi cặp badge ≥ 4,5:1, kể cả chữ/icon `-fg` đặt trực tiếp trên nền trang; mỗi cặp thông báo và mỗi nền tô XAI ≥ 4,5:1. Mô phỏng protan/deutan/tritan (Machado 2009): ΔE00 nhỏ nhất 15,8 giữa 3 cảm xúc, 11,0 giữa 4 chủ đề, ≥ 8,8 giữa hai nhóm.
- Cảm xúc và chủ đề **không bao giờ** chung một kênh màu trong một biểu đồ. “Khác” và “Trung tính” chung badge xám → phân biệt bằng icon + chữ.
- Nhãn trong cột chồng: chữ trắng trên Tiêu cực (4,7:1), chữ `#14171E` trên Trung tính/Tích cực (≥ 5,0:1); chọn màu chữ theo ô bằng `TEXT_ON_SOLID[tone]` (`text-white` hoặc `text-ink`, cố định ở hai theme vì nền đặc không đổi theo theme). Riêng “Khác” `#66837A` không màu chữ nào đạt 4,5:1 (chữ tối 4,35:1): không đặt chữ nhỏ lên nền đó.
- Heatmap “chênh so với mức nền” (điểm %), thang tường minh `DELTA_SCALE` trong `lib/colors.ts`: ≤ 0 `#EEF0F2` · 0–5 `#FBEAE4` · 5–10 `#F5C9B8` · 10–15 `#EDA286` · 15–20 `#E07656` · ≥ 20 `#B5401F` (chữ trắng; còn lại chữ `#14171E`). Ô n < 20: không tô (trong suốt) + viền nét đứt (`border-dashed`) + “n”.
- Thang tuần tự primary (10 bậc, `SEQUENTIAL_SCALE`; confusion matrix dùng làm thang tường minh, bậc = `ceil(p·10)`): `#F0F3FF #DCE3FF #C7D2FF #AEBDFC #97A8F7 #8193F1 #6A7BE5 #5664D3 #444EBB #343A9F`. Chữ `#14171E` ở bậc 1–7, chữ trắng ở bậc 8–10. Hai thang này giống nhau ở hai theme.
- XAI: từ ủng hộ = chữ đậm + số hạng ¹²³ + nền `rgba(91,104,238, .45/.30/.18)` theo hạng (`bg-xai-3/2/1`); từ phản đối = chữ nghiêng + “−” + nền `rgba(139,144,154,.30)` (`bg-xai-opp`). Một bộ alpha cho cả hai theme (chữ trên nền tô ≥ 8,5:1).
- Thông báo: component `Callout` (info / warn / ok / err), cặp chữ/nền theo theme — info `#1D4E89`/`#E6EFFA` (dark `#8DB9F4`/`#15263C`) · warn `#6F5300`/`#FCF0C8` (`#E8C25C`/`#33290C`) · ok `#0B6B4B`/`#E0F3EA` (`#5CC79B`/`#0F2C21`) · err `#A3341A`/`#FCE7E1` (`#F28C70`/`#3B1C14`).

## 3. Chữ, khoảng cách, bo góc
- Font: **Be Vietnam Pro** (có subset `vietnamese`, thiết kế riêng cho dấu chồng tầng) 400/500/600/700 cho toàn app; **JetBrains Mono** 400/500 cho code và thẻ `[TÊN]`. Nạp bằng `next/font/google` (tự host lúc build). Không dùng Inter/Poppins.
- Thang (px, cỡ/dòng · đậm): title 32/40 · 700 · header 24/32 · 600 · subheader 20/28 · 600 · body 16/24 · nhãn widget, caption, bảng 14/22 · badge, trục 12–14 · số metric 32. Không dưới 12px.
- Số kiểu Việt: `54,1%` · `1.000` · `0,018` (hàm trong `lib/format.ts`); chênh lệch tỉ lệ ghi “điểm” (“cao hơn khoảng 15 điểm”), dấu trừ đúng “−”.
- Khoảng cách: nhịp 4/8/12/16/24/32/48; khối cách nhau 16px (`gap-4`). Không lồng viền container quanh metric có viền.
- Bo góc: `--radius: 0.5rem` (8px) cho nút, ô nhập, khung; badge 6px.
- Bố cục: thiết kế cho 1440×900: sidebar 288px, nội dung đệm 80px hai bên (tối đa 1.200px). ≤ 640px: sidebar thành Sheet mở từ thanh đầu trang, cột xếp dọc, bảng rộng cuộn ngang trong khung của nó (không cuộn ngang cả trang).

## 4. Nền tảng giao diện: token, theme, component
| Nơi | Vai trò |
|---|---|
| `src/app/globals.css` | Mọi token màu (`:root` sáng, `.dark` tối) và ánh xạ sang Tailwind bằng `@theme inline`; nơi **duy nhất** khai màu giao diện. |
| `src/lib/colors.ts` | Lớp Tailwind theo khoá ngữ nghĩa (`TONE_CLASS`), thang tường minh cho confusion matrix và heatmap (chứa hex, không đổi theo theme). Mã hex chỉ được nằm ở hai file này. |
| `src/app/layout.tsx` | Font (`next/font`), `ThemeProvider` của `next-themes` (`attribute="class"`, mặc định theo hệ thống; công tắc Sáng / Tối / Theo hệ thống), `TooltipProvider`, `Toaster`, `AppShell`. `<html lang="vi" suppressHydrationWarning>`. |
| `components.json` | shadcn CLI 4.x, preset `base-nova`: dựa trên **Base UI**, không phải Radix. Ví dụ cũ dùng `asChild` hoặc `type="single"` là của Radix, đọc mã sinh ra thay vì chép ví dụ. |
| `src/components/ui/` | Component shadcn đã chỉnh theo wireframe (nút cao 40px, badge 6px, bảng có lưới `--grid`, tab gạch chân, nhóm nút dạng phân đoạn…). |
| `src/components/shared/` | Component dùng chung: `Icon`, `LabelBadge`, `FlagBadge`, `Callout`, `MetricCard`, `ProbBars`, `ConfidenceBar`, `XaiSentence` / `XaiLegend` / `XaiTable`, `PiiChip` / `PiiText`, `IllustrativeBadge`, `DemoUnavailableNotice`, `FailureNotice`, `SegmentedControl`, `SectionHeading`, `Caption`. |
| `src/components/shell/` | `AppShell`, `AppSidebar`, `MobileNav`, `ThemeToggle`, `PageHeader`. |
| `src/lib/` | `api/` (client `openapi-fetch`, lỗi tiếng Việt theo `detail.code`, kiểu sinh từ OpenAPI), `app-config.ts` (nạp `/api/config` một lần), `format.ts`, `labels.ts`, `colors.ts`, `xai.ts`, `reliability.ts`. |

Ánh xạ tên token trong tài liệu này sang mã:

| Tên thiết kế | Biến CSS | Lớp Tailwind |
|---|---|---|
| bg / bg2 / raised / hover | `--background` / `--muted` / `--popover` / `--accent` | `bg-background` / `bg-muted` / `bg-popover` / `bg-accent` |
| text / muted | `--foreground` / `--muted-foreground` | `text-foreground` / `text-muted-foreground` |
| border / divider / grid | `--border` (= `--input`) / `--divider` / `--grid` | `border-border` / `border-divider` / `bg-grid` |
| primary / primary-text / primary-soft | `--primary` / `--primary-text` / `--primary-soft` | `bg-primary` / `text-primary-text` / `bg-primary-soft` |
| nhãn, cờ, thông báo, XAI | `--neg`, `--neg-fg`, `--neg-bg`, …, `--info-fg`, …, `--xai-1` … | `bg-neg`, `text-neg-fg`, `bg-neg-bg`, `text-info-fg`, `bg-xai-1` … |

Quy tắc:
- Thêm màu mới: khai biến ở cả `:root` và `.dark`, ánh xạ trong `@theme inline`, rồi thêm cặp vào `tests/contrast.test.ts`.
- **Chữ** màu primary dùng `text-primary-text`, không dùng `text-primary` (ở theme tối `--primary` chỉ đủ cho nền nút và đồ hoạ ≥ 3:1).
- Font: `Be_Vietnam_Pro({ subsets: ["vietnamese", "latin"], weight: ["400","500","600","700"], variable: "--font-be-vietnam" })` và `JetBrains_Mono` 400/500 (`--font-jetbrains`).
- Hàm gộp lớp `cn` lấy từ gói `cn` 0.4.0 (do tác giả shadcn phát hành, repo `shadcn-ui/cn`, MIT, không phụ thuộc) mà shadcn CLI cài sẵn; ghim đúng phiên bản.
- Trong code biểu đồ giữ **một** nơi hằng số màu (`lib/colors.ts`); không rải mã hex.

## 5. Khung chung
- Logo: `public/logo.svg` (wordmark 32px, bọc `<Link>` về trang 1) trong sidebar; `public/logo-mark.svg` trên thanh đầu trang di động; favicon `src/app/icon.svg`.
- Điều hướng ở sidebar (`NAV_ITEMS`, mục đang mở có `aria-current="page"`): Phân tích câu `rate_review` · Phân tích hàng loạt `upload_file` · Dashboard & insight `monitoring` · Mô hình & đạo đức `balance`.
- Dưới nav: nhãn “Mô hình đang dùng” + `model_version · bundle_version` từ `/api/config` (bản smoke ghi thêm “số đo chưa phải kết quả cuối”); khối disclaimer ngắn có viền; nút “Mã nguồn · AGPL-3.0” mở `source_url` của `/api/config` (rỗng thì thay bằng liên kết tới mục giấy phép ở trang 4); dòng “Không lưu dữ liệu bạn nhập”.
- Công tắc Sáng / Tối / Theo hệ thống ở góc phải trên (desktop) hoặc trên thanh đầu trang (di động).
- Đầu mỗi trang: `PageHeader` = tiêu đề h1 có icon trang + một câu nói mục đích. Có liên kết “Bỏ qua điều hướng” cho bàn phím.
- Văn bản người dùng/sinh viên (câu tiêu biểu, từ XAI) chỉ render qua JSX; React tự escape nên không cần escape markdown, và **cấm** `dangerouslySetInnerHTML`. Liên kết ra ngoài có `rel="noopener noreferrer"`.

## 6. Quy tắc component theo trang
**6.1 Phân tích câu**
- Ô nhập `Textarea` (tối đa `limits.max_chars` = 500 ký tự, có đếm) + chip câu mẫu từ `ui_samples` của `/api/config` (chọn = điền câu rồi chạy luôn; thiếu dữ liệu demo thì ẩn chip và hiện `DemoUnavailableNotice`); nút “Phân tích” (primary), “Xoá”; Ctrl/⌘ + Enter cũng chạy.
- **Nhãn hiện trước, giải thích tải sau:** gọi `/api/analyze` rồi hiện ngay nhãn, cờ, xác suất; sau đó mới gọi `/api/explain` (skeleton trong lúc chờ) và không chặn kết quả. Yêu cầu cũ bị huỷ khi đổi câu.
- Kết quả: “Kết quả” + cờ “Cần xem xét” (`FlagBadge`) + lý do bằng lời (“Chủ đề không chắc — mô hình phân vân giữa Giảng viên (58%) và Cơ sở vật chất (31%)”) hoặc “Đủ tin cậy, không cần xem xét”. Ngưỡng cờ = ngưỡng của ngân sách duyệt mặc định (`default_budget`, 10%), không có thanh trượt; dòng chú thích + liên kết “đổi ngân sách ở trang Phân tích hàng loạt”.
- Hai khung viền: Cảm xúc | Chủ đề. Mỗi khung: nhãn lớn (icon + màu + %) → thanh “Độ tin cậy (đã hiệu chỉnh)” (ghi “đạt ngưỡng” hoặc “dưới ngưỡng cờ”) → thanh ngang xác suất từng lớp, lớp cao nhất in đậm.
- XAI: nút phân đoạn Cảm xúc/Chủ đề → câu tô (đơn vị từ ghép, chữ hoa/thường khôi phục từ câu gốc) + chú giải + bảng “Bỏ từ này thì xác suất …: 93% → 46%” + câu “Cách đọc”. Số từ được tô là số thật từ API (0–3 từ ủng hộ, 0–1 từ phản đối); không có từ nào đủ lớn thì nói rõ thay vì để trống. **Giải thích chỉ cho câu ≤ `limits.xai_max_words` (mặc định 25 từ, đặt bằng `EDUPULSE_XAI_MAX_WORDS`).**
- Không dấu: `Callout` cảnh báo (cờ luôn bật). Đã ẩn thông tin cá nhân: `Callout` info đếm “[TÊN] ×1 · [SĐT] ×1” + câu đã ẩn + `[SĐT] [EMAIL] [MSSV] [CCCD] [URL]` (quy tắc) và `[TÊN]` (**best effort**; luôn kèm khuyến cáo tự ẩn danh). Câu dài hơn giới hạn từ: thông báo “Câu dài N từ (> 25) nên không tô từ để giữ tốc độ”, vẫn giữ kết quả phân loại và **không** tự gắn cờ. Câu không còn nội dung (mã `no_content`): thông báo riêng. Lỗi mạng hoặc máy chủ bận: `FailureNotice` có nút “Thử lại”.
- **So sánh mô hình: đã duyệt, mặc định tắt.** Công tắc “So sánh với mô hình cơ sở” (PhoBERT ↔ TF-IDF + LinearSVC) lấy kết quả mô hình cơ sở bằng một yêu cầu riêng (`/api/analyze` với `compare: true`), nên không làm chậm hay làm hỏng kết quả chính; hiện hai khung kèm từ quyết định và câu kết luận cùng/khác nhãn.

**6.2 Phân tích hàng loạt** (4 bước đánh số bằng tiêu đề mục)
- `Callout` riêng tư → vùng tải file (`<input type="file">` + kéo thả, CSV/XLSX tối đa `limits.max_upload_mb`) + nút “Dùng file mẫu” → xem trước 5 dòng. Tối đa **`limits.max_rows` câu mỗi lần** (2.000); vượt thì cảnh báo đề nghị tách file và khoá nút chạy.
- Chọn cột văn bản, cột nhóm (tuỳ chọn; nhóm của file mẫu mang nhãn “Dữ liệu minh hoạ — nhóm gán ngẫu nhiên”); `Slider` “**Ngân sách duyệt**” 5/10/20/30% (mặc định `default_budget` ≈ 100/1.000 câu) = tỉ lệ câu độ tin cậy thấp nhất được gắn cờ. Cơ chế: cờ các câu độ tin cậy thấp nhất ở cảm xúc **hoặc** chủ đề; lý do ghi theo tác vụ không chắc; câu gõ không dấu luôn bị cờ (ưu tiên trong ngân sách). Cạnh đó khung “Hiệu quả ước tính (validation)” gồm 3 chỉ số theo mức ngân sách — lỗi cảm xúc được bắt · lỗi chủ đề được bắt · phần không cờ đúng cả 2 nhãn — lấy từ `budgets[].efficiency` của `/api/config`. Ghi rõ là ước tính.
- Khi chạy: danh sách bước + thanh tiến độ “640/1.000 câu · còn khoảng 25 giây” (hỏi tiến độ mỗi giây); giải thích tự chạy chỉ cho **top-`limits.auto_xai_top`** (20) câu cờ có độ tin cậy thấp nhất. Xong: 4 chỉ số có viền (Cần duyệt mặc định “100 · 10,0% số câu”); tab “Cần duyệt (n)” / “Tất cả kết quả (n)”.
- Bảng duyệt theo thứ tự thao tác: # · Câu (đã ẩn thông tin cá nhân, 1 dòng “…”) · Lý do · Cảm xúc (chữ màu) · Sửa cảm xúc (chọn) · Chủ đề · Sửa chủ đề · Giải thích (nút) · Tin cậy CX/CĐ (thanh) · Nhóm. Gồm câu bị cờ + 5–10% mẫu ngẫu nhiên (không bị cờ, để đo độ chính xác không thiên lệch), sắp độ tin cậy tăng dần; thứ tự cố định, không sắp lại sau khi sửa; 50 dòng mỗi trang.
- Bấm “Giải thích” → ngăn kéo bên phải (`Sheet`) có câu đầy đủ, tô từ, nút Câu trước/Câu sau; câu ngoài top-20 tính khi mở (~1 giây); câu dài hơn giới hạn từ (`explainable = false`) ghi “không tô từ”.
- Xuất: “Tải toàn bộ kết quả (CSV)” + “Tải nhãn đã sửa”; cảnh báo “… tải file trước khi đóng tab”; liên kết sang dashboard. Tab mở thì gửi heartbeat; đóng tab thì báo `close` để máy chủ xoá batch.

**6.3 Dashboard & insight**
- Điều khiển: nút phân đoạn nguồn (Dữ liệu demo · File vừa tải) + bộ lọc nhóm (chỉ hiện khi có cột nhóm; gửi qua thân yêu cầu, không qua URL).
- 4 chỉ số có viền; tỉ lệ tiêu cực tô màu cảnh báo khi cao hơn nền, ghi “điểm so với nền”.
- Thẻ cảnh báo (≤ 3, khung viền): tiêu đề nhận định + 1 câu có số + dòng căn cứ (khoảng tin cậy 95%, n, badge chủ đề) + “Xem câu liên quan”. Luật (r3): n ≥ 20 và cận dưới Wilson 95% > nền + 10 điểm; CSVC xét tỉ trọng. Mức nền của từng chủ đề lấy từ bundle (`InsightBackground`), không ghi cứng.
- Biểu đồ: cột chồng 100% chủ đề × cảm xúc + vạch mức nền (dựng bằng CSS vì mức nền mỗi hàng khác nhau); heatmap nhóm × chủ đề tô theo **chênh so với mức nền** (không tô % thô vì nền từng chủ đề khác nhau), kèm nhãn minh hoạ khi `illustrative_groups`. Nhãn tiếng Việt không bị cắt: cột nhãn đủ rộng, xuống dòng thay vì cắt.
- Từ khoá: 3 khối (CSVC, CT, GV) dạng chip kèm số câu, chọn để lọc câu tiêu biểu; “Khác” không hiện từ khoá.
- Câu tiêu biểu: tab theo chủ đề, từ khoá in đậm, badge cảm xúc + nhóm. Khuyến nghị: hành động + “Căn cứ: n câu nhắc …” + badge chủ đề; cuối mục `Callout` “chỉ để gợi ý”.

**6.4 Mô hình & đạo đức**
- `Callout` disclaimer đầu trang; mục lục link neo (Hiệu năng · Model card · Quyền riêng tư · Giấy phép & trích dẫn).
- Hiệu năng (tên tập và nhãn bản lấy từ `stage` và `metrics.split` của `/api/model-card`: “bản smoke · số trên validation” hoặc “tập test”): nút phân đoạn Cảm xúc/Chủ đề → 4 chỉ số (Accuracy, Macro-F1, ECE, Lỗi được bắt ở ngân sách duyệt mặc định) · bảng từng lớp (F1 có thanh) · confusion matrix (% theo hàng, dựng bằng `<table>`) · bảng so sánh mô hình cơ sở với PhoBERT (dòng đang dùng in đậm; mô hình nhiều seed ghi trung bình ± độ lệch chuẩn) · biểu đồ độ tin cậy (SVG theo bề rộng khung, kèm bảng số trong mục “Xem bảng số”) + hộp “Đọc các chỉ số thế nào?” với ví dụ bằng số thật.
- Model card 2 cột (mục đích · không dùng cho · dữ liệu · mô hình · giới hạn · kiểm tra hành vi · phiên bản · ghi chú triển khai gồm “chưa đo trên x86” và dữ liệu demo tính lúc build); 3 khối quyền riêng tư; giấy phép + nút mở mã nguồn và mô hình Hugging Face (từ `source_url` / `model_url`, ẩn khi chưa cấu hình) + BibTeX có nút sao chép.

## 7. Accessibility (WCAG 2.1 AA)
- Mọi cặp ở mục 2 đã đo bằng test tự động (`tests/contrast.test.ts`): chữ ≥ 4,5:1, viền và fill ≥ 3:1, ở cả hai theme.
- Không chỉ dùng màu: badge = icon + chữ; XAI = số hạng, chữ đậm / chữ nghiêng + “−” (kèm chữ ẩn “ủng hộ, hạng 1” cho trình đọc màn hình); heatmap có số trong ô và viền nét đứt cho n < 20; confusion matrix có số trong ô.
- Mỗi biểu đồ có caption nêu ý chính (thay alt text) và bảng số tương đương (reliability diagram có bảng số; confusion matrix và bảng từng lớp vốn là bảng).
- Điều hướng bàn phím do Base UI; vòng tiêu điểm là `focus-visible:ring-ring/70` (độ đậm 70% mới đạt ≥ 3:1 trên nền trang cả hai theme, 50% chỉ 2,3–2,5:1) và phần tử không tự vẽ vòng dùng viền mặc định màu `--ring` đặc; điều khiển phân đoạn đổi mục bằng phím mũi tên. Có liên kết “Bỏ qua điều hướng” đưa tiêu điểm vào nội dung chính. Kết quả phân tích báo cho trình đọc màn hình bằng một câu ngắn ở vùng `role="status"`, không đọc lại cả trang.
- Không tự thêm animation; `prefers-reduced-motion: reduce` tắt mọi chuyển động (`globals.css`).
- Mobile 390px: cột xếp dọc, không cuộn ngang trang (bảng rộng chỉ cuộn trong khung của nó); nút cao 40px (desktop-first, chấp nhận dưới 44px). Có kiểm bằng Playwright: trang 1 và 4 ở 390 (`e2e/visual.spec.ts`) và 320 / 720 / 1024 (`e2e/golden-path.spec.ts`); trang 2–3 ở 320 / 390 / 900 / 1100 (`e2e/reflow-batch-dashboard.spec.ts`).
- Nhãn dài (vd. “Dữ liệu minh hoạ — nhóm gán ngẫu nhiên”, `IllustrativeBadge`) co theo khung chứa và xuống dòng, không dùng `whitespace-nowrap` / `shrink-0`; tên mô hình ở bảng so sánh của trang 4 cũng xuống dòng để bảng vừa khung ở 1440.

## 8. Giọng văn
- Tiếng Việt có dấu đầy đủ, câu ngắn, chủ ngữ là dữ liệu: “Góp ý về giảng dạy ở môn X có tỉ lệ tiêu cực cao hơn mức nền”, không “Giảng viên môn X kém”.
- Không dùng: “đánh giá giảng viên”, “vi phạm”, “kém”, “tệ”, tên người. Luôn kèm căn cứ (n câu, từ khoá); khi nói về giảng dạy thêm “Tổng hợp theo môn, không gắn với cá nhân”.
- Thuật ngữ cố định: Cảm xúc / Chủ đề · Tiêu cực / Trung tính / Tích cực · Giảng viên / Chương trình / Cơ sở vật chất / Khác · **Độ tin cậy** (xác suất đã hiệu chỉnh của *một câu*) ≠ **khoảng tin cậy 95%** (thống kê của *nhóm câu*) · Cần xem xét · Ngân sách duyệt · Lỗi được bắt · Mức nền · Thông tin cá nhân (không viết “PII” trên UI) · Mô hình cơ sở.
- Câu mẫu hiển thị viết hoa chữ đầu, giữ nguyên lỗi chính tả gốc của sinh viên.

## 9. Icon & logo
- Material Symbols Rounded (không emoji), nhúng dạng path SVG qua `<Icon name="…" />` (`components/shared/icon-paths.ts`, sinh từ `@material-symbols/svg-400` 0.48.0, Apache-2.0; tô `currentColor`, cỡ xs 16 / sm 18 / md 20 / lg 24). Thêm icon: lấy thuộc tính `d` của `https://cdn.jsdelivr.net/npm/@material-symbols/svg-400@0.48.0/rounded/<tên>.svg` (bản đặc: `<tên>-fill.svg`) rồi thêm vào file. `lucide-react` chỉ nằm trong các component shadcn sinh ra.
- Gói 0.48.0 không có `insights`, `expand_more`, `expand_less`: dùng `monitoring` (Dashboard & insight), `keyboard_arrow_down`, `keyboard_arrow_up`.
- Ngoài icon ở mục 2 và 5: `play_arrow` chạy · `download` xuất · `manage_search` giải thích · `compare_arrows` so sánh · `shield_person` / `visibility_off` ẩn thông tin cá nhân · `spellcheck` không dấu · `trending_up` cảnh báo tăng · `campaign` nhắc nhiều · `lightbulb` khuyến nghị · `format_quote` câu tiêu biểu · `sell` từ khoá · `lock` riêng tư · `gavel` disclaimer/giấy phép · `code` mã nguồn.
- Logo: `docs/wireframes/assets/logo.svg` (wordmark, chữ đã chuyển path từ Be Vietnam Pro) và `logo-mark.svg` (bong bóng + nhịp tim, rõ ở 16–32px), đã chép vào `apps/web/public/`. Một file dùng cho cả hai theme (màu `#5B68EE`/`#727886` ≥ 3,7:1 trên sidebar sáng và tối).

## 10. Đã kiểm và còn lại (trên trình duyệt)
**Đã chốt khi dựng giao diện (2026-10-19):**
1. **Chữ trên nút primary (dark):** nút giữ `#5B68EE` với chữ trắng (4,5:1). Chữ màu primary trên nền tối dùng `--primary-text` `#8B96FF` (chữ chọn của chip/tab/phân đoạn đạt 5,3:1; dùng `#5B68EE` chỉ 3,1–4,1:1 và test tương phản đã bắt lỗi này).
2. **Tương phản:** đã tự động hoá trong `tests/contrast.test.ts` (mọi cặp mục 2, hai theme). Không còn việc đo tay.
3. **Font:** Be Vietnam Pro hiển thị đúng dấu chồng tầng ở cả sáng/tối, 1440 và 390 (ảnh `apps/web/test-results/visual/`); font tự host nhờ `next/font`.
4. **Không cuộn ngang trang** cả bốn trang ở các cỡ nêu ở mục 7 (kiểm bằng e2e).
5. **Axe** (`@axe-core/playwright`, WCAG A/AA + best-practice): 0 vi phạm ở bốn trang × 1440 / 390 × sáng / tối, gồm các trạng thái rỗng, lỗi và kết quả (`e2e/a11y.spec.ts`, `e2e/a11y-batch-dashboard.spec.ts`).
6. **Bảng so sánh mô hình (trang 4)** vừa khung ở 1440 với tên của bản phát hành (cột tên xuống dòng; không ghi lặp “1 seed (1 seed)” vì tên do bundle đặt đã nói số seed); ở 390 bảng cuộn trong khung.
7. **BibTeX** đã đối chiếu với trang gốc: UIT-VSFC (KSE 2018, tr. 19–24, DOI 10.1109/KSE.2018.8573337) và PhoBERT (Findings of EMNLP 2020, tr. 1037–1042, DOI 10.18653/v1/2020.findings-emnlp.92).
8. **Câu trong repo là câu tự viết:** wireframe và ảnh của chúng, fixture và test của web không chứa câu UIT-VSFC (đã quét 8-gram từ với tập này, không còn trùng). Câu của tập này chỉ xuất hiện lúc chạy, từ dữ liệu demo dựng lúc build (chip câu mẫu, danh sách “Câu tiêu biểu”). Ghi lại fixture bằng `pnpm record:fixtures` / `pnpm record:batch-fixtures` vẫn giữ được điều này vì câu golden g01 được thay trong `apps/web/scripts/fixture-sentences.mjs`.

**Còn lại:**
1. Số liệu trên wireframe là minh hoạ (số val của r1/r3, file 1.000 câu tự dựng) — trừ khung “Trạng thái mặc định” trang 3 (số thật tập test, nhãn gốc). UI lấy mọi số từ API. Bộ ảnh `apps/web/test-results/visual/` chụp trên bản export do FastAPI phục vụ với bundle `release` v1.0 (trang 4 hiện số thật trên tập test; bundle `smoke` hiện số validation kèm thông báo). Ảnh dashboard của dữ liệu demo phủ khối xám lên danh sách “Câu tiêu biểu” vì đó là câu của UIT-VSFC; chip câu mẫu ở trang 1 là 6 câu validation ngắn của tập đó (kèm một bản bỏ dấu) và không bị phủ.
2. Liên kết mã nguồn lấy từ `EDUPULSE_SOURCE_URL` và liên kết mô hình Hugging Face từ `EDUPULSE_MODEL_URL` lúc chạy (`source_url`, `model_url` của `/api/config`); trường nào rỗng thì ẩn nút tương ứng (không có liên kết giả).
3. Viền trái thẻ cảnh báo của dashboard (`acard`) và các trạng thái trang 2–3 đã có trong bộ ảnh (`02-*`, `03-*`).
4. Chưa kiểm Safari/Firefox và trình đọc màn hình thật (axe đã chạy, xem mục trên); nút chỉ báo của `next dev` (chữ N, góc trái dưới) chỉ có khi dev và được ẩn trong ảnh chụp.
