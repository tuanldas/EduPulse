# Streamlit 1.65.0: theming + component cho EduPulse (kiểm 2026-10-15)
**Khuyến nghị:** 1 `config.toml` (font Be Vietnam Pro qua URL Google, bảng màu ngữ nghĩa, `[theme.light]`+`[theme.dark]`) + directive markdown native, gần như 0 CSS. Ghim `streamlit==1.65.0`: PyPI đã có 1.66.0 (14/10, mới 1 ngày; thêm `theme.paddingTop/Bottom`), chờ bản vá. Rủi ro nâng cấp: thấp với config (key mới gần như mỗi bản, key lạ chỉ in warning), cao với CSS tay (mục 7).
**Cách kiểm:** venv tạm 1.65.0 (altair 6.3.0, pandas 3.0.6): `inspect.signature`, `streamlit config show`, đọc bundle JS, AppTest, render Chrome headless (light/dark, desktop + 390px). Ảnh: `scratchpad/r4/shots` (ngoài dự án).
## 1. config.toml mẫu (parse 0 warning + render đúng trên Chrome light/dark; màu là placeholder)
```toml
[client]
toolbarMode = "viewer"   # ẩn Deploy/Rerun, GIỮ công tắc System/Light/Dark ("minimal" ẩn luôn menu theo docs)

[theme]                  # dùng chung; [theme.light]/[theme.dark] kế thừa rồi ghi đè
font = "'Be Vietnam Pro':https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap"
baseRadius = "0.75rem"
showWidgetBorder = true
headingFontSizes = ["2rem", "1.5rem", "1.25rem"]   # h1-h3; h4-h6 giữ mặc định
metricValueFontSize = "2rem"
redColor = "#E5484D"     # tiêu cực: :red[] :red-background[] :red-badge[] st.badge(color="red") đều lấy từ đây
greenColor = "#30A46C"   # tích cực
grayColor = "#8B8D98"    # trung tính
chartCategoricalColors = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]

[theme.light]
primaryColor = "#4F46E5"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F4F6FA"   # nền sidebar + widget
textColor = "#1F2430"
borderColor = "#DCE0EA"
chartSequentialColors = ["#EEF2FF", "#E0E7FF", "#C7D2FE", "#A5B4FC", "#818CF8", "#6366F1", "#4F46E5", "#4338CA", "#3730A3", "#312E81"]  # đúng 10 màu

[theme.dark]
primaryColor = "#8B93FF"
backgroundColor = "#0F1220"
secondaryBackgroundColor = "#1A1F33"
textColor = "#E8EAF2"
borderColor = "#2B3150"
chartSequentialColors = ["#312E81", "#3730A3", "#4338CA", "#4F46E5", "#6366F1", "#818CF8", "#A5B4FC", "#C7D2FE", "#E0E7FF", "#EEF2FF"]  # đảo chiều: thấp=tối, cao=sáng
```
## 2. Font tiếng Việt (xếp hạng: URL Google > `fontFaces` tự host)
- Cú pháp `font = "'Tên có cách':<css_url>"` (tách ở dấu `:` đầu; tên có khoảng trắng cần nháy đơn trong; `headingFont`/`codeFont` cùng cú pháp; 1 `family=` mỗi URL). KHÔNG thêm `, sans-serif` sau URL (ví dụ fallback trong docs không chạy như mô tả: parser 1.65 nuốt nó vào URL, đã test); Streamlit tự nối fallback (computed: `"Be Vietnam Pro", "Source Sans", sans-serif`).
- Không cần subset: Google trả 3 khối `@font-face`/weight (vietnamese, latin-ext, latin) có `unicode-range`; frontend chèn `<link rel=stylesheet>` nên trình duyệt tự tải khối cần (Chrome: `document.fonts.check` true, 3 khối 400 `loaded`). Chạy ở trình duyệt người dùng, không phụ thuộc host (Community Cloud chưa test thật).
- Tự host (offline/GDPR): `[[theme.fontFaces]]` (`family`, `url="app/static/x.woff2"`, `weight`, `style`, `unicodeRange`) + `[server] enableStaticServing=true` + thư mục `static/` cạnh entrypoint (file ≤200MB, thư mục ≤1GB); chỉ ở `[theme]`, đổi phải restart; woff2 phải có glyph vietnamese; Be Vietnam Pro license OFL. Chưa chạy thử.
## 3. Key theme và dark mode
- `[theme]` 1.65 (`streamlit config show`): base, primaryColor, backgroundColor, secondaryBackgroundColor, textColor, linkColor, linkUnderline, codeTextColor, codeBackgroundColor, {red,orange,yellow,blue,green,violet,gray}{Color,BackgroundColor,TextColor}, font, headingFont, codeFont, fontFaces, baseFontSize/Weight, headingFontSizes/Weights, codeFontSize/Weight, metricValueFontSize/Weight, baseRadius, buttonRadius (none/small/medium/large/full/"0.75rem"), borderColor, dataframeBorderColor, dataframeHeaderBackgroundColor, dataframeHeaderTextColor, showWidgetBorder, showSidebarBorder, chartCategoricalColors, chartSequentialColors (đúng 10), chartDivergingColors (đúng 10).
- `[theme.sidebar]`, `[theme.light]`, `[theme.dark]`, `[theme.light.sidebar]`, `[theme.dark.sidebar]` có hầu hết key; chỉ `[theme]` có: base, fontFaces, baseFontSize/Weight, metricValueFont*, showSidebarBorder. Sidebar mặc định nền = `secondaryBackgroundColor` (đảo với main), kế thừa font/màu; heading sizes/weights không kế thừa.
- Dark mode: có `[theme.light]`/`[theme.dark]` → menu ⋮ có công tắc System/Light/Dark, mặc định System theo OS; mỗi mode = `[theme]` + ghi đè (base tự gán). Chỉ `[theme]` → theme cố định, không công tắc (đã kiểm: OS dark vẫn nền sáng). `st.context.theme.type` đọc "light"/"dark" (có thể sai lúc vừa đổi theme).
## 4. Tô màu chữ cho XAI (xếp hạng: `:color[]{background=rgba}` > `:green-background[]` > `st.html`)
- Palette: `red orange yellow green blue violet gray/grey primary` (+`rainbow` chữ gradient, không có nền/badge). `:c[..]` chữ, `:c-background[..]` nền, `:c-badge[..]`; còn `:small[]`, `:shimmer[]`. Chỉnh bằng `{c}Color`/`{c}BackgroundColor`/`{c}TextColor` (7 màu x 3).
- Đo Chrome: nền = màu 10% (light)/20% (dark) alpha, chữ giữ `textColor` nên luôn đọc được; chữ màu tự đậm/nhạt ~15% theo mode (`:red[]` light rgb(196,28,34), dark rgb(238,139,142)); `primary` theo primaryColor từng mode.
- Màu riêng từng span (1.55+): `:color[chữ]{foreground="#hex" background="rgba(48,164,108,0.4)"}`; nhận named/hex/rgb(a)/hsl(a) dấu phẩy; KHÔNG đổi theo theme, nên dùng rgba alpha ≤0.5 (đã thử tới 0.5) (màu đặc như `hsl(60,100%,90%)` chói ở dark).
- An toàn: `st.markdown` mặc định escape HTML; `unsafe_allow_html=True` và `st.html` đều sanitize (docs: DOMPurify): `onerror`, `<script>`, `javascript:` không chạy (đã kiểm `window.__pwn*` rỗng). Màu cứng trong HTML không đổi theo dark mode; docs khuyên tránh HTML/CSS tay.
- Text người dùng/token PHẢI escape: thô thì 13/26 mẫu vỡ (đậm, code, link, `$` LaTeX, `:smile:`, `:red[x]` bị tiêm, ngoặc); `` re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", tok) `` đúng 24/26, 2 mẫu còn lại là URL vẫn auto-link.
## 5. Component (1.65.0: `inspect.signature` + chạy AppTest/Chrome; tất cả đều CÓ)
| Component | Có? | Chữ ký / ghi chú (kwargs chính) |
|---|---|---|
| `st.badge` | Có | `(label, *, icon, color="blue", width="content", help)`; color: red/orange/yellow/blue/green/violet/gray/grey/primary (= `:green-badge[..]`) |
| `st.pills`, `st.segmented_control` | Có | `(label, options, *, selection_mode="single"/"multi", default, required, format_func, key, help, on_change, disabled, label_visibility, width="content", wrap)`; nhãn option nhận directive: `:green[●] Tích cực` |
| `st.select_slider` | Có | `(label, options, value, format_func, key, help, on_change, *, disabled, label_visibility, width="stretch")` |
| `st.dialog` | Có | `(title, *, width="small"/"medium"/"large", position="center"/"left"/"right", dismissible, icon, on_dismiss="ignore")`; left/right = drawer (mới 1.65) |
| `st.metric` | Có | `(label, value, delta, delta_color, *, help, icon, label_visibility, border=False, width, height, chart_data, chart_type, delta_arrow, format, delta_description)` |
| `st.container` | Có | `(*, border, key, width, height, horizontal=False, wrap=True, horizontal_alignment, vertical_alignment, gap="small", autoscroll)`; viền container + viền metric lồng nhau nhìn nặng |
| `st.columns` | Có | `(spec, *, gap="small" (xxsmall..xxlarge/int/None), vertical_alignment="top"/"center"/"bottom", border=False, width, wrap=True)` |
| `st.data_editor` | Có | `(data, *, width, height, hide_index, column_order, column_config, num_rows="fixed", disabled=bool/list, key, on_change, row_height, placeholder)`; `SelectboxColumn(label, *, width, help, disabled, required, pinned, default, options, format_func)` KHÔNG có `color`; `ProgressColumn(..., format, min_value, max_value, step, color)`; `TextColumn(..., disabled, required, max_chars, validate)` |
| `st.dataframe` | Có | như trên + `on_select`, `selection_mode`; Styler: giá trị/màu/font-weight, KHÔNG bar/hover/caption, chỉ áp cột không sửa được ở data_editor; pandas 3: dùng `Styler.map` (`applymap` đã xóa) |
| `st.status`, `st.toast` | Có | `status(label, *, expanded=False, state="running"/"complete"/"error", width)`; `toast(body, *, icon, duration="short"/"long"/"infinite"/int)` |
| `st.download_button` | Có | `(label, data, file_name, mime, key, help, on_click="rerun"/"ignore"/callable, ..., *, type, icon, width)` |
| `st.file_uploader` | Có | `(label, type, accept_multiple_files, key, help, on_change, ..., *, max_upload_size=<MB>, disabled, label_visibility, width)`; ghi đè `server.maxUploadSize` (200MB); dropzone hiện "5MB per file" |
| `st.logo`, `st.navigation`, `st.Page` | Có | `logo(image, *, size="small"/"medium"/"large" (20/24/32px), link="http(s)://..", icon_image)`; `navigation(pages/{section: pages}, *, position="sidebar"/"top"/"hidden", expanded)`; `Page(page, *, title, icon, url_path, default, visibility)` |
- Icon `:material/ten:` (rounded) dùng ở `Page(icon=)`, `logo(image=)`, `badge/button/dialog/expander/metric/toast/download_button(icon=)` và trong nhãn markdown; emoji 1 ký tự cũng được. Top nav: 4 trang phẳng = thanh ngang có icon, trang active tô nền; dict section = dropdown; ở 390px top nav rơi vào drawer sidebar (nút »). `layout="wide"` chỉ đặt qua `st.set_page_config` (centered ~700px ở màn 1280).
## 6. Altair 6.3 trong `st.altair_chart`
- `theme="streamlit"` mặc định: áp font + `chartCategoricalColors` + `chartSequentialColors` theo mode (heatmap đổi ramp light/dark: đã chụp). Scale tường minh thắng theme: `alt.Color("nhan:N", scale=alt.Scale(domain=["Tiêu cực","Trung tính","Tích cực"], range=["#E5484D","#8B8D98","#30A46C"]))` (đã render đỏ/xám/xanh, cố định cả 2 mode; đủ tương phản trên #FFF và #0F1220).
- Heatmap: `mark_rect()` + `alt.Color("n:Q")` dùng ramp theme, không cần `scheme`. Số đè bằng `mark_text` cần `alt.condition` đổi màu chữ theo ô (chưa thử; chữ đen trên ô đậm khó đọc).
- Nhãn tiếng Việt hiển thị đúng, nhưng web font rộng hơn Source Sans: nhãn trục y dài ("Chương trình học") bị cắt khi `title=None`, chạm tiêu đề khi mặc định (bỏ font thì hết lỗi; đã so sánh). Sửa: `alt.Axis(titlePadding=24, labelLimit=240)` hoặc `.properties(padding={"left": 24})`; trục x `labelAngle=0`.
- Streamlit yêu cầu `altair>=5,<7` nên pip lấy 6.3.0; `use_container_width` bị deprecate, dùng `width="stretch"`; 6000 hàng không lỗi.
## 7. Custom CSS
- Còn hỗ trợ (kiểm Chrome): `st.html("<style>..</style>")`, `st.html(Path("style.css"))` (tự bọc `<style>`, không chiếm chỗ; chuỗi path bị coi là HTML từ 1.61 nên truyền `Path`), `st.markdown("<style>..", unsafe_allow_html=True)`. Hook ổn nhất: `st.container(key="kpi")` → class `st-key-kpi`.
- Rủi ro: docs ghi prefix/attribute/vị trí DOM của `st-key-` "aren't guaranteed to be stable between versions"; tránh `st-emotion-*` (hash tự sinh). Khuyến nghị: bắt đầu 0 CSS; nếu cần chỉ ≤20 dòng dùng `.st-key-*`, ghim phiên bản, chụp lại ảnh sau mỗi lần nâng.
## 8. Không làm được native → cách thay thế
1. Độ đậm màu liên tục theo trọng số từ: palette chỉ 1 mức → `:color[tok]{background="rgba(..,α~|w|)"}` (native, đã chạy).
2. Màu từng option trong `SelectboxColumn`/pills: không có → emoji 🟢🟡🔴 trong `format_func`, hoặc `:green[●]` trong nhãn pills/segmented.
3. Tô màu ô trong bảng sửa nhãn: Styler không áp cột sửa được → cột "Dự đoán" read-only tô bằng `Styler.map` + cột "Nhãn đã sửa" `SelectboxColumn`; độ tin cậy = `ProgressColumn(color=..)`.
4. Card/KPI có bóng, nền riêng, hover: theme chỉ chỉnh radius/border/màu chung → `st.container(border=True, key=..)` + vài dòng CSS `.st-key-*`.
5. Đổi theme bằng code / theme riêng từng trang: không (`st.context.theme` chỉ đọc) → người dùng đổi ở menu ⋮.
6. Việt hoá chuỗi built-in (Upload, Print, Record screen, toolbar bảng): không có cấu hình i18n → chấp nhận tiếng Anh; `toolbarMode="minimal"` ẩn menu nhưng mất công tắc dark.
## Câu hỏi chưa giải quyết
- Community Cloud chưa test: font Google, `[theme.light/dark]`, `maxUploadSize` có bị host ghi đè không (docs chỉ nói config ở `.streamlit/config.toml` gốc repo) → smoke test ngày 2.
- Chỉ kiểm Chrome headless macOS; chưa Safari/Firefox/thiết bị thật; chưa đo hiệu năng với >300 span directive/câu hay bảng lớn.
- `fontFaces` tự host chưa chạy thử; chưa tìm thấy cách Việt hoá UI built-in; 1.66.0 chưa đọc breaking change (nâng sau 1.66.1, chạy lại smoke test).
## Nguồn
Docs: https://docs.streamlit.io/develop/concepts/configuration/theming-customize-fonts · https://docs.streamlit.io/develop/concepts/configuration/theming · https://docs.streamlit.io/develop/concepts/configuration/theming-customize-colors-and-borders · https://docs.streamlit.io/develop/quick-reference/release-notes/2026 · https://docs.streamlit.io/develop/concepts/architecture/widget-behavior · https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization
Gốc: `streamlit config show`, docstring, `runtime/theme_util.py`, `runtime/app_session.py` của 1.65.0 đã cài; https://pypi.org/pypi/streamlit/json · https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700&display=swap · https://github.com/google/fonts/blob/main/ofl/bevietnampro/METADATA.pb
