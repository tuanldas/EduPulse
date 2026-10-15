# EduPulse — Design guidelines (Streamlit 1.65)

> Nguồn chuẩn cho giao diện app. Wireframe duyệt: [`docs/wireframes/index.html`](wireframes/index.html) (+ ảnh trong `wireframes/screenshots/`). Cập nhật 2026-10-15, đã đối chiếu `plans/reports/researcher-261014-1913-streamlit-theming.md`. Điểm chưa kiểm trên app thật nằm ở mục 10.

## 1. Nguyên tắc
1. **AI gợi ý, người quyết định.** Mọi nhãn kèm độ tin cậy đã hiệu chỉnh; câu dưới ngưỡng có cờ “Cần xem xét” + lý do. Không có thao tác tự động.
2. **Trung lập, không quy lỗi cá nhân.** Tổng hợp theo chủ đề/môn; không hiện tên giảng viên; câu chữ mô tả dữ liệu, không phán xét người.
3. **Màu không bao giờ là tín hiệu duy nhất.** Luôn kèm icon + chữ (+ số khi có).
4. **Native Streamlit trước, 0 custom HTML.** Tô từ XAI bằng directive `:color[…]{background=…}`. CSS chỉ tuỳ chọn (viền trái thẻ cảnh báo qua `.st-key-*`), bỏ đi không mất nghĩa.
5. **Lời trước, số sau.** Thẻ cảnh báo là câu hoàn chỉnh; chi tiết thống kê để ở caption/help.
6. **Dày thông tin nhưng quét được:** KPI → cảnh báo → biểu đồ → chi tiết. Mỗi biểu đồ có tiêu đề + caption “cách đọc”.
7. **Trình chiếu được:** desktop 1440 là chính; chữ ≥ 14px; viền widget ≥ 3:1.

## 2. Màu
### 2.1 Nền, chữ, primary
| Token | Light | Dark | Khoá config / ghi chú (tương phản) |
|---|---|---|---|
| bg | `#FAFAF8` | `#111419` | `backgroundColor` |
| bg2 | `#F3F3EE` | `#1A1E25` | `secondaryBackgroundColor` — sidebar, ô nhập |
| text | `#14171E` | `#E7E9EE` | `textColor` — 17,2 / 15,2:1; nếu Streamlit pha chữ phụ ~60% thì vẫn ≥ 4,5:1 |
| muted | `#5B6170` | `#A2A8B4` | chữ phụ tự viết (nếu cần) — 5,9 / 7,7:1 |
| border | `#918F87` | `#5F6573` | `borderColor` — 3,1 / 3,2:1 (WCAG 1.4.11 cho ô nhập) |
| divider | `#DCDAD2` | `#2C313A` | `dataframeBorderColor` — lưới bảng, đường kẻ trang trí |
| primary | `#3A4BD0` | `#5B68EE` | `primaryColor` — chữ trắng 6,8 / 4,5:1 |
| link | `#3A4BD0` | `#8B96FF` | `linkColor` — 6,5 / 7,0:1 |

### 2.2 Màu ngữ nghĩa (thứ tự cố định)
| Ý nghĩa | Icon `:material/…:` | Fill biểu đồ (`{c}Color`, chung 2 theme) | Chữ / nền badge — light | Chữ / nền badge — dark | Màu Streamlit `{c}` |
|---|---|---|---|---|---|
| Tiêu cực | `sentiment_dissatisfied` | `#C94827` | `#C0401E` / `#FDEBE5` | `#EF6C4B` / `#41201A` | red |
| Trung tính | `sentiment_neutral` | `#8B909A` | `#62666F` / `#EDEEF1` | `#A3A8B2` / `#2A2D33` | gray |
| Tích cực | `sentiment_satisfied` | `#1B9A6E` | `#007752` / `#DDF4E9` | `#3DB083` / `#0E3326` | green |
| Giảng viên | `co_present` | `#0167BB` | `#0F64B4` / `#E5EFFC` | `#5AA2F2` / `#132C47` | blue |
| Chương trình | `menu_book` | `#944A76` | `#93496F` / `#F9E8F1` | `#CF84AF` / `#3B2130` | violet |
| Cơ sở vật chất | `domain` | `#B88818` | `#835F00` / `#FBEFD3` | `#CFA03A` / `#382809` | orange |
| Khác | `more_horiz` | `#66837A` (chỉ biểu đồ) | dùng chung cặp gray | dùng chung cặp gray | gray |
| Cần xem xét (cờ) | `flag` | `#F2C230` — không làm fill biểu đồ | `#7A5E00` / `#FCEFC4` | `#E3B52B` / `#352905` | yellow |

- Đã đo: mỗi fill ≥ 3:1 trên cả `#FAFAF8` và `#111419` (scale tường minh của Altair không đổi theo theme); mỗi cặp badge ≥ 4,5:1. Mô phỏng protan/deutan/tritan (Machado 2009): ΔE00 nhỏ nhất 15,8 giữa 3 cảm xúc, 11,0 giữa 4 chủ đề, ≥ 8,8 giữa hai nhóm.
- Cảm xúc và chủ đề **không bao giờ** chung một kênh màu trong một biểu đồ. “Khác” và “Trung tính” chung badge xám (theme chỉ có một cặp gray) → phân biệt bằng icon + chữ.
- Nhãn trong cột chồng: chữ trắng trên Tiêu cực (4,7:1), chữ `#14171E` trên Trung tính/Tích cực (≥ 5,0:1). Altair: `alt.condition` chọn màu chữ theo ô.
- Heatmap “chênh so với mức nền” (điểm %), scale tường minh: ≤ 0 `#EEF0F2` · 0–5 `#FBEAE4` · 5–10 `#F5C9B8` · 10–15 `#EDA286` · 15–20 `#E07656` · ≥ 20 `#B5401F` (chữ trắng; còn lại chữ `#14171E`). Ô n < 20: không tô (trong suốt) + viền nét đứt (`strokeDash`) + “n”; chữ để màu mặc định theo theme.
- Thang tuần tự primary (10 bậc, `chartSequentialColors`; confusion matrix dùng làm scale tường minh): `#F0F3FF #DCE3FF #C7D2FF #AEBDFC #97A8F7 #8193F1 #6A7BE5 #5664D3 #444EBB #343A9F`. Chữ `#14171E` ở bậc 1–7, chữ trắng ở bậc 8–10.
- XAI: từ ủng hộ = chữ đậm + số hạng ¹²³ + nền `rgba(91,104,238, .45/.30/.18)` theo hạng; từ phản đối = chữ nghiêng + “−” + nền `rgba(139,144,154,.30)`. Một bộ alpha cho cả hai theme (chữ trên nền tô ≥ 8,5:1).
- Thông báo: giữ `st.info / st.warning / st.success / st.error` mặc định; đo lại tương phản sau khi đặt theme.

## 3. Chữ, khoảng cách, bo góc
- Font: **Be Vietnam Pro** (Google Fonts, có subset `vietnamese`, thiết kế riêng cho dấu chồng tầng) 400/500/600/700 cho toàn app; **JetBrains Mono** 400/500 cho code. Không dùng Inter/Poppins.
- Thang (px, cỡ/dòng · đậm): title 32/40 · 700 · header 24/32 · 600 · subheader 20/28 · 600 · body 16/24 · nhãn widget, caption, bảng 14/22 · badge, trục 12–14 · số metric 32. Không dưới 12px.
- Số kiểu Việt: `54,1%` · `1.000` · `0,018`; chênh lệch tỉ lệ ghi “điểm” (“cao hơn khoảng 15 điểm”).
- Khoảng cách: nhịp 4/8/12/16/24/32/48; giữa khối để mặc định Streamlit (1rem); `st.columns(..., gap="medium")`. Không lồng viền container quanh metric có viền.
- Bo góc: `baseRadius = "0.5rem"` (8px); badge, dialog giữ mặc định.
- Bố cục: `st.set_page_config(layout="wide", page_title="EduPulse")`. Thiết kế cho 1440×900; ≤ 640px Streamlit tự xếp cột dọc, bảng cuộn ngang trong khung.

## 4. `.streamlit/config.toml`
```toml
[client]
toolbarMode = "viewer"            # ẩn Deploy/Rerun, giữ công tắc System/Light/Dark

[server]
maxUploadSize = 10                # MB; thêm st.file_uploader(max_upload_size=10)

[theme]                           # chung; [theme.light]/[theme.dark] kế thừa rồi ghi đè
font = "'Be Vietnam Pro':https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap"
codeFont = "'JetBrains Mono':https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap"
headingFontSizes = ["2rem", "1.5rem", "1.25rem"]
headingFontWeights = [700, 600, 600]
metricValueFontSize = "2rem"
baseRadius = "0.5rem"
showWidgetBorder = true
showSidebarBorder = true
redColor = "#C94827"
orangeColor = "#B88818"
yellowColor = "#F2C230"
greenColor = "#1B9A6E"
blueColor = "#0167BB"
violetColor = "#944A76"
grayColor = "#8B909A"
chartCategoricalColors = ["#C94827", "#8B909A", "#1B9A6E", "#0167BB", "#944A76", "#B88818", "#66837A"]

[theme.light]
primaryColor = "#3A4BD0"
backgroundColor = "#FAFAF8"
secondaryBackgroundColor = "#F3F3EE"
textColor = "#14171E"
linkColor = "#3A4BD0"
borderColor = "#918F87"
dataframeBorderColor = "#DCDAD2"
chartSequentialColors = ["#F0F3FF", "#DCE3FF", "#C7D2FF", "#AEBDFC", "#97A8F7", "#8193F1", "#6A7BE5", "#5664D3", "#444EBB", "#343A9F"]
redTextColor = "#C0401E"          # + redBackgroundColor = "#FDEBE5"; 6 màu còn lại theo bảng 2.2 (light)

[theme.dark]
primaryColor = "#5B68EE"
backgroundColor = "#111419"
secondaryBackgroundColor = "#1A1E25"
textColor = "#E7E9EE"
linkColor = "#8B96FF"
borderColor = "#5F6573"
dataframeBorderColor = "#2C313A"
chartSequentialColors = ["#343A9F", "#444EBB", "#5664D3", "#6A7BE5", "#8193F1", "#97A8F7", "#AEBDFC", "#C7D2FF", "#DCE3FF", "#F0F3FF"]
redTextColor = "#EF6C4B"          # + redBackgroundColor = "#41201A"; 6 màu còn lại theo bảng 2.2 (dark)
```
- Cú pháp font: tên có khoảng trắng đặt trong nháy đơn; **không** thêm `, sans-serif` sau URL (Streamlit tự nối fallback).
- Trong code giữ **một** module hằng số màu cho `alt.Scale(domain=…, range=…)`; không rải mã hex.

## 5. Khung chung
- `st.logo("assets/logo.svg", icon_image="assets/logo-mark.svg", size="large")` (32px).
- `st.navigation` ở sidebar: Phân tích câu `:material/rate_review:` · Phân tích hàng loạt `:material/upload_file:` · Dashboard & insight `:material/insights:` · Mô hình & đạo đức `:material/balance:`.
- Dưới nav: caption “Mô hình đang dùng” + phiên bản; `st.container(border=True)` chứa disclaimer ngắn; `st.link_button("Mã nguồn · AGPL-3.0")`; caption “Không lưu dữ liệu bạn nhập”.
- Đầu mỗi trang: `st.title` có icon trang + `st.caption` một câu nói mục đích.
- Văn bản của người dùng/sinh viên đưa vào `st.markdown` (câu tiêu biểu, từ XAI) **phải escape** ký tự markdown: ``re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", tok)`` (research: thô thì 13/26 mẫu vỡ).

## 6. Quy tắc component theo trang
**6.1 Phân tích câu**
- `st.text_area(max_chars=500, key=…)` + `st.pills` câu mẫu thật (chọn = ghi session_state rồi chạy); `st.button(type="primary")` “Phân tích”.
- Kết quả: “Kết quả” + `st.badge("Cần xem xét", icon=":material/flag:", color="yellow")` + lý do bằng lời (“Chủ đề không chắc — phân vân giữa Giảng viên 58% và Cơ sở vật chất 31%”). Ngưỡng cờ = ngưỡng của ngân sách duyệt mặc định 10%, không có thanh trượt; `st.caption` + `st.page_link` “đổi ngân sách ở trang Phân tích hàng loạt”.
- 2 `st.container(border=True)`: Cảm xúc | Chủ đề. Mỗi khối: nhãn lớn `:red[:material/…: Tiêu cực]` + % → `st.progress` “Độ tin cậy (đã hiệu chỉnh)” → Altair thanh ngang xác suất từng lớp.
- XAI: `st.segmented_control` Cảm xúc/Chủ đề → câu tô (≤ 40 từ, đơn vị từ ghép, `_` → dấu cách) + chú giải + `st.dataframe` “Bỏ từ này thì xác suất …: 93% → 46%”.
- Không dấu: `st.warning`; đã ẩn thông tin cá nhân: `st.info` + `[SĐT] [EMAIL] [MSSV] [CCCD] [URL]` (quy tắc) và `[TÊN]` (**best effort** — NER tuỳ chọn vì tốn RAM; luôn kèm khuyến cáo tự ẩn danh); câu > 40 từ: `st.info` “không tô từ”, giữ kết quả, **không** tự gắn cờ.
- So sánh PhoBERT ↔ LinearSVC qua `st.toggle` (mặc định tắt) — **tuỳ chọn, chờ duyệt**.

**6.2 Phân tích hàng loạt** (4 bước đánh số bằng `st.subheader`)
- `st.info` riêng tư → `st.file_uploader(type=["csv","xlsx"], max_upload_size=10)` + nút “Dùng file mẫu” → `st.expander` xem trước. Tối đa **2.000 câu mỗi lần** (hằng số cấu hình); vượt thì `st.warning` đề nghị tách file và khoá nút chạy.
- `st.selectbox` cột văn bản, cột nhóm (tuỳ chọn); `st.select_slider` “**Ngân sách duyệt**” 5/10/20/30% (mặc định 10% ≈ 100/1.000 câu) = tỉ lệ câu độ tin cậy thấp nhất được gắn cờ. Cơ chế: cờ các câu độ tin cậy thấp nhất ở cảm xúc **hoặc** chủ đề; lý do ghi theo tác vụ không chắc; câu gõ không dấu luôn bị cờ (ưu tiên trong ngân sách). Cạnh đó `st.container(border=True)` “Hiệu quả ước tính (validation)” gồm 3 `st.metric` có sparkline `chart_data` qua 4 mức — lỗi cảm xúc được bắt · lỗi chủ đề được bắt · phần không cờ đúng cả 2 nhãn: 5% → 41% · 21% · 88,5% · 10% → 54% · 41% · 90,7% · 20% → 66% · 65% · 93,7% · 30% → 72% · 80% · 95,0% (không cờ: 86,5%). Ghi rõ là ước tính.
- Khi chạy: `st.status(expanded=True)` các bước + `st.progress` “640/1.000 câu · còn khoảng 25 giây”; XAI tự chạy chỉ cho **top-20** câu cờ có độ tin cậy thấp nhất. Xong: 4 `st.metric(border=True)` (Cần duyệt mặc định “100 · 10,0% số câu”); `st.tabs(["Cần duyệt (n)", "Tất cả kết quả (n)"])`.
- `st.data_editor(num_rows="fixed", column_order=…)` theo thứ tự thao tác: # · Câu (đã ẩn PII, 1 dòng “…”) · Lý do · Cảm xúc (chỉ đọc, chữ màu bằng `Styler.map`) · Sửa cảm xúc (`SelectboxColumn`) · Chủ đề · Sửa chủ đề · Giải thích (`CheckboxColumn`) · Tin cậy CX/CĐ (`ProgressColumn`) · Nhóm. Gồm câu bị cờ + 5–10% mẫu ngẫu nhiên (không bị cờ, để đo độ chính xác không thiên lệch), sắp độ tin cậy tăng dần; giữ df cố định trong session_state, không sắp lại sau khi sửa.
- Tick “Giải thích” → `@st.dialog(…, position="right")` (ngăn kéo phải, 1.65) có câu đầy đủ, tô từ, nút Câu trước/Câu sau; câu ngoài top-20 tính khi mở (~1 giây).
- Xuất: `st.download_button(on_click="ignore", type="primary")` “Tải toàn bộ kết quả (CSV)” + “Tải nhãn đã sửa”; `st.warning` “… tải file trước khi đóng tab”; `st.page_link` sang dashboard.

**6.3 Dashboard & insight**
- Điều khiển: `st.segmented_control` nguồn (Dữ liệu demo · File vừa tải) + `st.multiselect` lọc nhóm (chỉ hiện khi có cột nhóm).
- 4 `st.metric(border=True)`; tỉ lệ tiêu cực dùng `delta_color="inverse"`, delta ghi “điểm so với nền”.
- Thẻ cảnh báo (≤ 3, `st.container(border=True)`): tiêu đề nhận định + 1 câu có số + dòng căn cứ (khoảng tin cậy 95%, n, badge chủ đề) + “Xem câu liên quan”. Luật (r3): n ≥ 20 và cận dưới Wilson 95% > nền + 10 điểm; CSVC xét tỉ trọng (nền 4,4%). Nền tiêu cực: GV 35,6% · CT 78,0% · CSVC 95,4% · Khác 40,2%.
- Biểu đồ: cột chồng 100% chủ đề × cảm xúc + vạch mức nền; heatmap nhóm × chủ đề tô theo **chênh so với mức nền** (không tô % thô vì nền từng chủ đề khác nhau). Altair: `alt.Axis(labelLimit=240, titlePadding=24, labelAngle=0)` để nhãn tiếng Việt không bị cắt; `width="stretch"`.
- Từ khoá: 3 khối (CSVC, CT, GV) `st.pills` kèm số câu, chọn để lọc câu tiêu biểu; “Khác” không hiện từ khoá.
- Câu tiêu biểu: `st.tabs` theo chủ đề, từ khoá in đậm, badge cảm xúc + nhóm. Khuyến nghị: hành động + “Căn cứ: n câu nhắc …” + badge chủ đề; cuối mục `st.warning` “chỉ để gợi ý”.

**6.4 Mô hình & đạo đức**
- `st.warning` disclaimer đầu trang; mục lục link neo.
- Hiệu năng: `st.segmented_control` Cảm xúc/Chủ đề → 4 metric · bảng từng lớp (F1 `ProgressColumn`) · confusion matrix · bảng so sánh 3 baseline với PhoBERT (dòng đang dùng in đậm) · reliability diagram + hộp “Đọc các chỉ số thế nào?”.
- Model card 2 cột (mục đích · không dùng cho · dữ liệu · mô hình · giới hạn · kiểm tra hành vi · phiên bản); 3 khối quyền riêng tư; giấy phép + `st.link_button` GitHub/HF + `st.code(language="bibtex")`.

## 7. Accessibility (WCAG 2.1 AA)
- Mọi cặp ở mục 2 đã đo ≥ 4,5:1 (chữ) / ≥ 3:1 (viền, fill). Ngoại lệ đã biết: mục 10, ý 1.
- Không chỉ dùng màu: badge = icon + chữ; XAI = số hạng, chữ đậm / chữ nghiêng + “−”; heatmap có số trong ô và viền nét đứt cho n < 20.
- Mỗi biểu đồ có caption nêu ý chính (thay alt text) và bảng số tương đương khi cần.
- Không tự thêm animation; dùng widget gốc để giữ điều hướng bàn phím và focus ring.
- Mobile 390px: cột xếp dọc, không cuộn ngang trang; nút cao 40px (desktop-first, chấp nhận dưới 44px).

## 8. Giọng văn
- Tiếng Việt có dấu đầy đủ, câu ngắn, chủ ngữ là dữ liệu: “Góp ý về giảng dạy ở môn X có tỉ lệ tiêu cực cao hơn mức nền”, không “Giảng viên môn X kém”.
- Không dùng: “đánh giá giảng viên”, “vi phạm”, “kém”, “tệ”, tên người. Luôn kèm căn cứ (n câu, từ khoá); khi nói về giảng dạy thêm “Tổng hợp theo môn, không gắn với cá nhân”.
- Thuật ngữ cố định: Cảm xúc / Chủ đề · Tiêu cực / Trung tính / Tích cực · Giảng viên / Chương trình / Cơ sở vật chất / Khác · **Độ tin cậy** (xác suất đã hiệu chỉnh của *một câu*) ≠ **khoảng tin cậy 95%** (thống kê của *nhóm câu*) · Cần xem xét · Ngân sách duyệt · Lỗi được bắt · Mức nền · Thông tin cá nhân (không viết “PII” trên UI) · Mô hình cơ sở.
- Câu mẫu hiển thị viết hoa chữ đầu, giữ nguyên lỗi chính tả gốc của sinh viên.

## 9. Icon & logo
- Material Symbols Rounded qua `:material/name:` (không emoji). Ngoài icon ở mục 2 và 5: `play_arrow` chạy · `download` xuất · `manage_search` giải thích · `compare_arrows` so sánh · `shield_person` / `visibility_off` ẩn thông tin cá nhân · `spellcheck` không dấu · `trending_up` cảnh báo tăng · `campaign` nhắc nhiều · `lightbulb` khuyến nghị · `format_quote` câu tiêu biểu · `sell` từ khoá · `lock` riêng tư · `gavel` disclaimer/giấy phép · `code` mã nguồn.
- Logo: `docs/wireframes/assets/logo.svg` (wordmark, chữ đã chuyển path từ Be Vietnam Pro) và `logo-mark.svg` (bong bóng + nhịp tim, rõ ở 16–32px). Một file dùng cho cả hai theme (màu `#5B68EE`/`#727886` ≥ 3,7:1 trên sidebar sáng và tối). Chép vào `app/assets/` khi code.

## 10. Cần xác minh (trên app thật)
1. **Chữ trên nút primary (dark):** nếu Streamlit luôn dùng chữ trắng → giữ `#5B68EE` (nút 4,5:1; nhưng chữ trạng thái chọn của pills/tabs nếu tô bằng primary chỉ 4,1:1). Nếu Streamlit tự chọn chữ tối khi primary sáng → đổi sang `#8B96FF` (đạt mọi cặp; report theming dùng `#8B93FF` nhưng chưa nói chữ nút).
2. `CheckboxColumn` → `@st.dialog`: kiểm cách mở khi tick, không mở lại sau rerun, giữ dấu tick.
3. Community Cloud: font Google, công tắc `[theme.light]/[theme.dark]`, `maxUploadSize` (smoke test ngày 2, như report theming).
4. `page_icon` nhận SVG không (nếu không: `:material/monitor_heart:`); `st.logo` hiển thị SVG wordmark đúng tỉ lệ ở 32px.
5. Viền trái thẻ cảnh báo dùng `.st-key-*` — docs ghi không ổn định giữa phiên bản → ghim 1.65.0, chụp lại sau mỗi lần nâng; bỏ nếu vỡ.
6. Số liệu trên wireframe là minh hoạ (số val của r1/r3, file 1.000 câu tự dựng) — trừ khung “Trạng thái mặc định” trang 3 (số thật tập test, nhãn gốc). Thay bằng số test sau khi chốt mô hình; đối chiếu BibTeX với bài gốc; thay placeholder `github.com/<user>/edupulse`, `huggingface.co/<user>/edupulse-phobert-mtl`.
