# EduPulse: nghiên cứu deploy MIỄN PHÍ & runtime (Streamlit + PhoBERT)

**Khuyến nghị chính:** Streamlit Community Cloud (miễn phí, không thẻ) + ONNX Runtime int8 + `tokenizers` (không torch) + py_vncorenlp/pyjnius/`jdk4py` (không apt); model để ở repo model public trên HF Hub (miễn phí). **Dự phòng:** chạy local + Cloudflare Quick Tunnel cho ngày bảo vệ. HF Spaces chỉ còn miễn phí cho Static và ZeroGPU/Gradio (phải viết lại UI) → không khuyến nghị.

Ngày 2026-10-15. Số đo = tự chạy trên M1 Pro (arm64, máy đang tải, 2 thread) → chỉ so sánh tương đối, KHÔNG phải x86/Linux. Script/log: scratchpad `r2/` (ngoài repo). Chưa deploy thật lên host nào (không có tài khoản); Docker Desktop bị tắt nên chưa build trọn Dockerfile.

## 1. Streamlit Community Cloud (khuyến nghị)
- Tài nguyên: CPU 0.078-2 lõi, RAM 690MB-2.7GB, đĩa ≤50GB (FAQ staff 02/2024; trang Limits trong app ghi "2.7 GBs/app, 1 app private, public không giới hạn", forum 118484, 08/2025). "Đổi không báo trước"; vượt RAM → app bị hạn chế tạm thời (forum 107013) hoặc tài khoản bị chặn fair-use 403 (forum 120966, 03/2026). Có form xin tăng tài nguyên cho tổ chức giáo dục/phi lợi nhuận (~1 tuần).
- Đủ cho PhoBERT? CÓ, với dư địa: đo trên macOS (đã gồm streamlit+pandas+JVM `-Xmx512m`+1000 câu) peak RSS = 615MB (ORT int8 284MB) và 551MB (torch FP32); Linux ước 0.7-0.9GB (ORT) và 1.0-1.3GB (torch) vì RSS macOS đếm khác → đều < 2.7GB nhưng > 690MB "đảm bảo" nên có rủi ro nghẽn khi node bận. ORT int8 giúp ít hơn về RAM, nhiều hơn về tốc độ, kích thước tải (284MB so với 540MB) và không cần torch (-196MB wheel).
- Ngủ/đánh thức: ngủ sau 12h KHÔNG có traffic (docs; commit không tính, forum 97582); khách bấm "Yes, get this app back up!"; không có số chính thức về thời gian wake → mở app trước buổi bảo vệ 10-15 phút.
- Repo GitHub: public hoặc private đều được (private cần quyền OAuth `repo`/deploy key; app kế thừa quyền repo; chỉ 1 app private cùng lúc, đổi sang public được; developer phải là admin repo). Cập nhật từ GitHub ≤5 lần/phút.
- Phụ thuộc: `requirements.txt` (xử lý bằng uv); ưu tiên file cạnh entrypoint trước gốc repo, thứ tự `uv.lock` > Pipfile > environment.yml > requirements.txt > pyproject → đặt `app/requirements.txt` cạnh entrypoint để `uv.lock` dev ở gốc không ăn trước. `packages.txt` (apt) có hỗ trợ, phải ở GỐC repo, nền Debian 11; 07-09/09/2026 apt hỏng do repo bullseye-security hết hạn, staff vá 09/09 (forum 122388/122392) → dùng `jdk4py` (pip) thay vì `default-jre-headless`. Python mặc định 3.12, đổi ở Advanced settings.
- Cấu hình bị ép: `enableXsrfProtection=true`, `gatherUsageStats=true` (không tắt được telemetry), `showErrorDetails=false`; `maxUploadSize` không bị ép: mặc định 200MB, đặt trần bằng `st.file_uploader(max_upload_size=MB)`. Logs chỉ developer xem ("Manage app").
- Tải model: tài khoản HF free tạo được repo model public (public storage "best-effort", vài trăm MB là nhỏ); rate limit resolver 3000 req/5 phút/IP ẩn danh, 5000 với tài khoản free → nếu gặp 429 thì đặt read-token trong Secrets. File >100MB không lên GitHub được → model ở HF Hub, VnCoreNLP (28MB) commit vào repo.

## 2. Giữ trên HF miễn phí: Gradio + ZeroGPU (không khuyến nghị)
- Điều kiện (docs `spaces-zerogpu`): tối đa 2 Space cho tài khoản free "in good standing" = email xác minh + tài khoản >30 ngày (tài khoản mới tạo thì chưa dùng được). CHỈ SDK Gradio (≥4, hiện 6.30.0), Python 3.12.12/3.10.13, PyTorch 2.8-2.13 (torch 2.14.1 chưa có trong danh sách), không `torch.compile`; model phải `.to("cuda")` ở mức module, cần ít nhất một hàm `@spaces.GPU` (template CPU không có hàm này bị lỗi hàng loạt, forum 177703). Streamlit KHÔNG chạy được.
- Quota GPU tính theo NGƯỜI GỌI/ngày: ẩn danh 2 phút (ưu tiên thấp), tài khoản free 5 phút, PRO 40 phút; GPU = nửa RTX Pro 6000 Blackwell 48GB. PhoBERT 1000 câu chỉ vài giây GPU nên quota không phải nút thắt, nhưng hàng đợi/cold start không có số công bố; đặt `duration` ngắn (≤10s).
- Công sức đổi UI (ước lượng của tôi, 2-4 ngày + chạy lại runtime bằng torch GPU thay ONNX CPU): phân tích 1 câu dễ (`gr.Textbox`, `gr.Label`, `gr.HighlightedText` hợp XAI hơn Streamlit). Batch: `gr.File` + `gr.Dataframe(interactive=True)` nhưng cột chỉ có `str/number/bool/date/markdown`, KHÔNG có ô chọn → sửa nhãn phải gõ tay hoặc tự code `select` + `gr.Dropdown`; xuất file bằng `gr.DownloadButton`. Dashboard: `gr.BarPlot`/`gr.Plot`; nhiều trang: `gr.Navbar`/`Blocks.route`. Yếu hơn spec ở data editor; docs ZeroGPU cũng cảnh báo "unexpected issues may arise".

## 3. Nền tảng miễn phí khác (đã xem trang giá/docs 2026-10)
- Loại vì RAM: Render free 512MB, ngủ sau 15 phút, wake ~1 phút (render.com/pricing, /docs/free) < peak ~615MB.
- Loại vì thẻ/không free: Fly.io (tổ chức mới không có free tier; trial 2 giờ/7 ngày, sau đó cần thẻ), Modal (docs billing: "must have a payment method on file"), Cloud Run (free tier 180k vCPU-s, 360k GiB-s, 2M req/tháng nhưng cần billing account). Koyeb: trang giá chỉ nêu "monthly free credit", không rõ gói free cho web service → chưa xác minh, bỏ. HF Static + chạy trong trình duyệt: không có JVM/onnxruntime Python → loại.
- Đáng giữ làm dự phòng: máy cá nhân + Cloudflare Quick Tunnel (docs: không cần tài khoản, URL trycloudflare.com tạm thời, "for testing", không SLA, ≤200 request đồng thời); hoặc GitHub Codespaces (free 120 core-giờ + 15GB-tháng/tháng, tự tắt khi rảnh).

## 4. Java / tách từ RDRSegmenter (Q2, đã chỉnh theo Community Cloud)
- Đã kiểm: py_vncorenlp 0.1.4 (wrapper bên thứ ba, 2023-02, phụ thuộc pyjnius 1.9.0 có wheel manylinux cp311-314). Trong container `python:3.12-slim` (arm64) + `default-jre-headless` (OpenJDK 21, +194MB) chạy uid 1000: nạp 0.45s, 1000 câu 0.45s. macOS/OpenJDK 26: nạp 1.0s, 1000 câu 0.30s; 4 thread×200 câu không lỗi.
- Không apt: `jdk4py==25.0.2.1` (JRE trong wheel, 35MB x86_64). Đã chạy py_vncorenlp+jdk4py trên macOS arm64 (nạp 2.5s, kết quả giống hệt); wheel Linux x86_64 chỉ kiểm tĩnh (ELF x86-64, cùng 36 module) → thử trên Community Cloud ngày 2. Đặt `os.environ["JAVA_HOME"]=str(jdk4py.JAVA_HOME)` TRƯỚC `import py_vncorenlp`.
- Bẫy py_vncorenlp: constructor `os.chdir(save_dir)` vĩnh viễn (lưu cwd, chdir lại); `download_model()` gọi `wget` và tải cả NER/parse → chỉ commit `VnCoreNLP-1.2.jar` 27.4MB + `models/wordsegmenter/{vi-vocab,wordsegmenter.rdr}` (VnCoreNLP GPL-3.0+); `word_segment` trả list theo câu → `" ".join`; đặt `-Xmx512m`; khởi tạo 1 lần trong `st.cache_resource` + `threading.Lock`.
- Dự phòng thuần Python: `underthesea==9.5.0` (0.46ms/câu) nhưng lệch RDR (18/24 câu mẫu giống hệt, F1 ranh giới 0.967) → phải train/đánh giá lại cùng bộ tách (nhờ r1). Nguyên tắc: cùng segmenter ở train & serve; UIT-VSFC đã lowercase + tách dấu câu → chuẩn hoá y hệt lúc suy luận.

## 5. Lưu model (Q3)
- Repo model public trên HF Hub; tải `hf_hub_download(repo, file, revision=<sha>)` trong `st.cache_resource`; cache tạm → tải lại sau mỗi wake/redeploy (vài giây). Private thì dùng `st.secrets` → env `HF_TOKEN`. Đặt `HF_HOME=/tmp/hf` nếu thư mục home không ghi được.
- Artefact: `model.int8.onnx` + `tokenizer.json` (copy từ `vinai/phobert-base-v2`) + `labels.json`. `tokenizers.Tokenizer.from_file` cho id giống hệt `PhobertTokenizer` trên 3166/3166 câu, nhanh 8×; tự `enable_truncation(256)` + `enable_padding(pad_id=1, pad_token="<pad>")` (PhobertTokenizer không đặt `model_max_length` → câu >256 token lỗi index). Lưu bản fine-tuned dạng safetensors.
- sklearn (đã thử, 1.9.1/skops 0.16.0): `TfidfVectorizer(token_pattern=r"(?u)\S+")`+LogReg → skops 0 kiểu không tin cậy, file ~13× joblib (3.7MB vs 0.27MB), nạp 45ms vs 9ms; `CalibratedClassifierCV` cần `trusted=` 2 kiểu nội bộ; `tokenizer=str.split` dump được nhưng LOAD lỗi → tách từ ngoài pipeline. Chọn skops (joblib/pickle chạy mã tuỳ ý khi load); pin scikit-learn==1.9.1 ở train & app.
- Giấy phép: PhoBERT AGPL-3.0, VnCoreNLP GPL-3.0+, UIT-VSFC học thuật (mirror ura-hcmut CC BY-NC-SA 4.0) → repo public, ghi license + trích dẫn, phi thương mại.

## 6. Hiệu năng PhoBERT-base (Q4)
Đo trên M1, 2 thread, UIT-VSFC test 3166 câu (TB 14.2 từ), checkpoint bên thứ ba `tmt3103/VSFC-sentiment-classify-phoBERT` (PhoBERT-base v1), input tách từ RDR:
| Cấu hình | Thời gian | acc / macro-F1 | Kích thước |
|---|---|---|---|
| torch FP32 | 35.2s (11ms/câu) | 0.9352 / 0.8277 | 540MB |
| ORT FP32 (export dynamo) | 56.4s (17.8ms) | 0.9352 / 0.8277 (0 lệch) | 540MB |
| ORT int8 mặc định (MatMul+Gather, per-tensor) | ~20s (6.4ms) | 0.9340 / 0.8205 | 136MB |
| ORT int8 MatMul-only, per-channel, reduce_range | 21.8s (6.9ms) | 0.9371 / 0.8325 | 284MB |
| torch `quantize_dynamic` (qnnpack) | 92.5s | 0.9352 / 0.8247 | - |
- Ước lượng x86 2 vCPU (CHƯA đo): ≈0.17 GFLOP/token × ~18 token × 1000 câu ≈ 3.1 TFLOP; Philschmid 2022 (c6i.xlarge 4 vCPU, DistilBERT seq128: PyTorch FP32 75.7ms p95 vs ORT int8 26.75ms = 2.83×, giữ 99.72% accuracy) → ~75 GFLOPS/2 vCPU → FP32 ≈40s/1000 câu (×1.5-2 nếu CPU chia sẻ), int8 ≈15-35s; 1 câu FP32 ~70-100ms, int8 ~30-50ms. Community Cloud chỉ đảm bảo 0.078 lõi → nếu bị throttle có thể chậm hơn nhiều: đo thật, hiển thị progress bar, trần ~2000 dòng/lần.
- **Đơn giản nhất đạt mục tiêu:** ORT int8: `torch.onnx.export(dynamo=True, dynamic_shapes)` (8.6s, sai khác ~1e-6) → `quant_pre_process` (BẮT BUỘC với bản dynamo, không thì ShapeInferenceError) → `quantize_dynamic(QInt8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=True)`; batch 16-32 sắp theo độ dài; `intra_op_num_threads=2`. Cổng chấp nhận: macro-F1 dev giảm ≤1 điểm so FP32. Dự phòng: PyTorch FP32 + batching (không cần export).
- Loại: `optimum[onnxruntime]` (optimum-onnx 0.1.0 ghim `transformers<4.58`; issue #114/#141 còn mở) và `torch.ao.quantization.quantize_dynamic` (2.14.1 vẫn in "deprecated, will be removed in 2.10", dtype quantized bị deprecate pytorch#184982, macOS cần engine qnnpack, chậm 2.6× FP32 trên M1).
- Độ chính xác đo trên ARM, 1 task, lớp neutral chỉ 167 mẫu (±0.5 điểm F1 là nhiễu); x86 không VNNI dùng U8S8 có thể bão hoà (ORT docs) → giữ `reduce_range`+per-channel, đo lại trên host. XAI occlusion tốn (số từ+1) lần forward: câu 100 từ ≈25-30s FP32 → giới hạn ≤40 từ; XAI gradient cần torch.

## 7. Streamlit 1.65.0 (Q5)
- 1.65.0 (2026-10-02, Python ≥3.10); server mặc định Starlette/Uvicorn từ 1.57. Đã gỡ `st.cache` (1.62), `use_column_width` của `st.image` (1.61). Đủ cho spec: `st.navigation`/`st.Page`, `st.file_uploader(max_upload_size=)`, `st.data_editor`+`column_config.SelectboxColumn`, `st.download_button(on_click="ignore")`, `st.progress`/`st.status`, `st.session_state`.
- `st.cache_resource` cho model/JVM (thread-safe); `st.cache_data` mặc định global giữa mọi user → KHÔNG cache dữ liệu upload (hoặc `scope="session"`, 1.53+). Biểu đồ: `st.bar_chart`/`st.altair_chart` (Altair 6.3.0 là phụ thuộc cứng); Plotly là extra, vừa lên 7.0 → tránh. Tiếng Việt do font trình duyệt (chưa kiểm bằng trình duyệt).
- Excel/CSV: `pd.read_excel(engine="openpyxl")` (openpyxl 3.1.5); xuất CSV `encoding="utf-8-sig"`; chống CSV/Formula Injection (ô bắt đầu `= + - @`, cả full-width → thêm `'`; OWASP). Community Cloud chạy ở `*.streamlit.app` (không iframe) nên upload không dính lỗi XSRF của HF.
- Nếu sau này dùng HF Docker: bắt buộc cờ CLI `--server.enableXsrfProtection=false` (docs `spaces-cookie-limitations`; forum 166694: `config.toml` không được đọc trong Docker); forum 172442 (01/2026) báo upload >1MB lỗi 413 trên Space Streamlit (chưa xác nhận).

## 8. Đóng gói & mẫu cấu hình (Q6)
- Python 3.12 (numpy 2.5.4, scipy 1.18.1 yêu cầu ≥3.12); mọi pin có wheel cp312 manylinux x86_64 (đã dry-run `uv pip install --python-platform x86_64-manylinux_2_28 --only-binary :all:`, trừ py_vncorenlp sdist thuần Python). uv: `[dependency-groups]` (`app`/`train`) + `uv.lock` cho dev; sinh file deploy: `uv export --only-group app --no-hashes --no-emit-project --emit-index-url -o app/requirements.txt` (đã thử). Wheel torch CPU 196MB vs PyPI mặc định 555MB + gói CUDA → nếu cần torch trên Linux PHẢI dùng index CPU.
- Bố cục repo cho Community Cloud: `app/streamlit_app.py` + `app/requirements.txt` (cạnh entrypoint), `vncorenlp/` (jar+2 file), `.streamlit/config.toml` (chỉ đọc ở gốc), `packages.txt` KHÔNG cần. Deploy = push GitHub rồi tạo app trên share.streamlit.io (chọn nhánh, entrypoint, Python 3.12). Model đẩy bằng `hf upload <user>/edupulse-phobert-onnx ./artifacts`.
```text
streamlit==1.65.0  pandas==3.0.6  openpyxl==3.1.5  scikit-learn==1.9.1  skops==0.16.0
onnxruntime==1.31.0  tokenizers==0.23.3  huggingface-hub==2.2.0  py-vncorenlp==0.1.4  pyjnius==1.9.0  jdk4py==25.0.2.1
# kéo theo: numpy 2.5.4, scipy 1.18.1, pyarrow 25.0.1, altair 6.3.0. Train/export (không vào app): torch 2.14.1, transformers 5.19.0, onnx 1.23.2, onnxscript 0.7.2
```
- Nếu có PRO/credits (HF Docker): `hf repos create <user>/edupulse --type space --sdk docker [--flavor cpu-upgrade --sleep-time 1800]`, `hf upload <user>/edupulse <thư mục> . --repo-type=space --delete="*"`. README: `sdk: docker`, `app_port: 8501`, `title`, `tags`. Dockerfile đã kiểm từng phần (JRE+pip chạy uid 1000):
```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV JAVA_HOME=/usr/lib/jvm/default-java PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
```

## 9. Riêng tư / đạo đức (Q7)
- Streamlit giữ file upload trong RAM theo session, "có thể" dọn sau khi ngắt kết nối (`server.disconnectedSessionTTL`=120s); không ghi đĩa/log nội dung; không cache global. Community Cloud ép `gatherUsageStats=true`: engineer Streamlit nói telemetry gỡ thông tin cá nhân và họ không thấy chuỗi người dùng nhập (forum 87353; 1 nguồn) → ghi rõ trong disclaimer. Chính sách riêng tư HF (2023-03) không nêu Spaces.
- Che PII (SĐT/email/MSSV bằng regex; tên riêng không bắt được nếu không có NER → "best effort") TRƯỚC khi hiển thị/xuất. Disclaimer đề xuất: "Chỉ dùng dữ liệu mẫu/đã ẩn danh; kết quả do mô hình dự đoán, có thể sai, cần người duyệt, không dùng để đánh giá/kỷ luật cá nhân; dự án học thuật, phi thương mại." Luật BVDLCN 91/2025/QH15 hiệu lực 01/01/2026 (không phải tư vấn pháp lý).

## 10. Bảng xếp hạng (miễn phí trước)
| # | Phương án | Chi phí/thẻ | RAM (CPU) | Cold start | Độ khó | Rủi ro chính |
|---|---|---|---|---|---|---|
| 1 | **Streamlit Community Cloud** | 0 / không | ≤2.7GB (≤2 lõi), cần ~0.7-1.3GB | build vài phút; wake sau 12h: không có số, nạp app ước 5-20s | Thấp | RAM/CPU chia sẻ, giới hạn đổi không báo trước, jdk4py trên Linux chưa thử |
| 2 | Laptop + Cloudflare Quick Tunnel (**dự phòng**) | 0 / không | M1 16GB | 0 (chạy sẵn) | Rất thấp | không SLA, máy phải bật, URL đổi mỗi lần |
| 3 | HF ZeroGPU + Gradio | 0 / không | GPU 48GB + CPU host | hàng đợi/cold start không có số | Cao (2-4 ngày) | tài khoản ≥30 ngày, quota khách 2-5 phút/ngày, data editor yếu |
| 4 | HF Docker `cpu-upgrade` + credits | ~$0.03/h chỉ lúc chạy / cần thẻ | 8 vCPU, 32GB | container + nạp model | Thấp | cần thẻ; "chỉ cần billing" mới dựa docs |
| 5 | HF PRO $9 (cpu-basic) | $9 prorate, hủy trước 01/11 / cần thẻ | 2 vCPU, 16GB | ngủ sau 48h | Thấp | Space sau khi hết PRO: không rõ |
Khuyến nghị: #1 làm chính, #2 làm dự phòng; chuyển khỏi #1 nếu smoke test ngày 2 (jdk4py, RAM, 1000 câu <2 phút, upload 5MB) thất bại. Smoke test: in cờ CPU, nạp JVM+ORT+tokenizer từ HF Hub, chạy 1000 câu, upload CSV/XLSX 5MB, `data_editor`+download, đo RSS và thời gian wake.

## 11. Giới hạn & câu hỏi chưa giải quyết
Giới hạn: chưa deploy thật; chưa đo x86/Linux (RAM, tốc độ); số Community Cloud là của docs 02/2024 + forum; độ chính xác lượng tử hoá đo trên checkpoint bên thứ ba + ARM + 1 task; font tiếng Việt trong chart chưa kiểm; thời gian wake/queue ZeroGPU không có số công bố.
1. Repo GitHub của nhóm: ai là admin, public hay private (giới hạn 1 app private)?
2. Tài khoản HF của người dùng đã >30 ngày và xác minh email chưa (chỉ cần nếu chọn ZeroGPU)?
3. Community Cloud có cấp đủ CPU cho 1000 câu <2 phút và jdk4py chạy trên Linux của họ không (smoke test)?
4. Có nên nộp form xin tăng tài nguyên (giáo dục) từ đầu?
5. XAI dùng perturbation hay gradient (quyết định có cần torch lúc chạy)?
6. r1: chuẩn hoá đầu vào (lowercase, tách dấu câu) và segmenter cuối có khớp train–serve không?
