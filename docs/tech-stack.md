# EduPulse — Tech stack (chốt 2026-10-15)

Bằng chứng và số đo: `plans/reports/researcher-261014-1913-*.md` (phiên bản xác minh trên PyPI ngày 2026-10-14) và `plans/261015-0121-edupulse-implementation/research/researcher-02-architecture.md` (web, API; xác minh trên npm/PyPI ngày 2026-10-15).

> **Chốt 2026-10-15:** bỏ Streamlit. Giao diện là **Next.js** (export tĩnh) do **FastAPI** phục vụ cùng origin trong 1 container; triển khai bằng **Docker + docker compose trên VPS**, build ngay trên VPS. Phần ML (dữ liệu, tách từ, mô hình, ONNX int8) giữ nguyên. Kế hoạch triển khai: `plans/261015-0121-edupulse-implementation/`.

| Lớp | Lựa chọn | Phiên bản | Lý do |
|---|---|---|---|
| Ngôn ngữ, môi trường | Python + uv (dependency groups `app` / `demo` / `train` / `dev`, `uv.lock`) | Python 3.12, uv 0.11.22 | Khớp image Docker; numpy 2.5 và scipy 1.18 cần Python ≥ 3.12 |
| Dữ liệu | UIT-VSFC từ nhánh parquet `refs/convert/parquet`. pandas + pyarrow chỉ ở huấn luyện (`train`); runtime đọc CSV bằng `csv`, XLSX bằng `python-calamine`. Dữ liệu demo tính lúc build image | pandas 3.0.6, pyarrow 26.0.0, python-calamine 0.8.3 | Repo HF chỉ có loading script, `datasets` ≥ 4 không chạy script. Bỏ pandas ở runtime tiết kiệm ~40–70 MB RSS |
| Tách từ | RDRSegmenter: `py_vncorenlp` + `pyjnius` + JRE `default-jre-headless` (apt trong Docker; máy dev dùng JRE hệ thống) | 0.1.4 / 1.9.0 | PhoBERT pretrain bằng RDR; hơn không tách khoảng 2 macro-F1 |
| Baseline | scikit-learn: TF-IDF (word 1–2 + char_wb 2–5) + LinearSVC / LogisticRegression / Naive Bayes; lưu bằng `skops` | 1.9.1 / 0.16.0 | Chuẩn, nhanh; skops an toàn hơn pickle khi load |
| Mô hình chính | `vinai/phobert-base-v2`, 1 encoder + 2 head (đa nhiệm); 2 mô hình đơn nhiệm làm đối chứng | torch 2.14.1, transformers 5.19.0 | Đa nhiệm ngang đơn nhiệm về độ chính xác, tốn ½ RAM và latency |
| Huấn luyện | PyTorch MPS (M1 Pro) fp32; Colab T4 dự phòng | — | Khoảng 12 phút mỗi run |
| Export, suy luận | ONNX (`torch.onnx.export(dynamo=True)`) → `onnxruntime` int8 dynamic (MatMul, per-channel, reduce_range), giữ FP32 cho FFN-ra của 6 lớp đầu và 2 đầu phân loại; chạy từng câu (batch=1) ở mọi đường + `tokenizers` | onnxruntime 1.31.0, tokenizers 0.23.3 | Model 333MB (FP32 545MB), ~9 ms/câu trên M1 so với 17,8 ms của FP32; macro-F1 val giảm ≤0,5 điểm. Int8 toàn phần (291MB) trượt cổng ≤1 điểm. Chạy theo lô làm nhãn phụ thuộc câu cùng lô nên không dùng. Không cần torch khi chạy app |
| Lưu model | Hugging Face Hub (repo model public) qua `huggingface-hub` | 2.2.0 | Miễn phí, không cần token khi tải |
| API | FastAPI + Uvicorn, 1 worker; batch là job nền, giao diện hỏi tiến độ mỗi giây | fastapi 0.143.1, uvicorn 0.54.0, python-multipart 0.0.32 | Trạng thái batch nằm trong RAM nên bắt buộc 1 worker; endpoint tốn CPU dùng `def` để chạy trong threadpool |
| Giao diện | Next.js `output: "export"` + React + Tailwind CSS v4 + shadcn/ui (Base UI); biểu đồ tự vẽ bằng CSS grid/flex và SVG; FastAPI phục vụ thư mục `out/` cùng origin | next 16.4.1, react 19.3.1, tailwindcss 4.3.3, shadcn 4.21.5, TypeScript 5.9.3 | 1 tiến trình, không CORS, không cần Node lúc chạy. Ghim TypeScript 5.9.3 vì `openapi-typescript` hỏng với 7.x |
| Hợp đồng API | OpenAPI do FastAPI sinh → `openapi-typescript` + `openapi-fetch`; `schema.d.ts` commit sẵn | 7.13.0 / 0.17.0 | Kiểu TypeScript khớp API; pytest so snapshot OpenAPI để bắt lệch |
| Công cụ web | Node + pnpm | Node 24, pnpm 12.11.2 | Chỉ cần khi dev và ở stage build web của image |
| Ẩn dữ liệu cá nhân | regex + luật tên "xưng hô + Viết hoa" + `underthesea` NER (tuỳ chọn, bật bằng `EDUPULSE_NER=1`, chạy trên chữ gốc) | 9.5.0 | NER chỉ nhận tên khi còn chữ hoa; tên viết thường không bắt được nên giao diện ghi rõ giới hạn |
| Triển khai | Docker (image nhiều stage: build web bằng Node, runtime Python + JRE) + `docker compose` trên VPS; Caddy làm HTTPS (tuỳ chọn) | — | VPS 2 vCPU / 2GB RAM + swap ≥ 2GB chạy 24/7; không phụ thuộc nhà cung cấp; build trên VPS nên image khớp kiến trúc máy chạy |
| Kiểm thử | pytest + httpx (TestClient), Vitest + Testing Library, Playwright; dữ liệu thật, không mock | pytest 9.1.1, httpx 0.28.1, Vitest 5.0.4, Playwright 1.64.1 | Playwright chạy trên site đã export do uvicorn phục vụ nên kiểm luôn cấu trúc `out/` |
| Lint | ruff (Python); ESLint + `tsc --noEmit` (web) | ruff 0.17.0, ESLint 10.12.0 | — |

## Không dùng

| Thứ | Lý do |
|---|---|
| Streamlit, Streamlit Community Cloud | Đổi sang Next.js + FastAPI chạy trong Docker (quyết định 2026-10-15) |
| HF Spaces (Docker/Gradio) | Từ ~07/2026, tạo Space mới loại này cần gói PRO |
| pandas ở runtime | Thêm ~40–70 MB RSS; `csv` + `python-calamine` đủ cho CSV/XLSX |
| TanStack Table, TanStack Query | Bảng duyệt cố định thứ tự, không sort/lọc cột; một hook polling là đủ |
| SSE | Polling 1 Hz đủ cho tiến độ batch; SSE thêm `sse-starlette`, reconnect và buffering của proxy mà không thêm giá trị |
| `rewrites` của Next ở chế độ dev | Với `trailingSlash: true`, Next redirect `/api/upload` sang `/api/upload/` và POST 9,5 MB không tới FastAPI; dev gọi thẳng `NEXT_PUBLIC_API_BASE` kèm CORS |
| `optimum-onnx` | Ghim `transformers` < 4.58, không chạy với 5.x |
| `torch.ao.quantization.quantize_dynamic` | Deprecated, chậm hơn FP32 trên M1 |
| LIME, SHAP | Vượt ngân sách 2 giây/câu |
| Attention, Gradient×Input làm XAI | Độ trung thực đo được chỉ ngang chọn ngẫu nhiên |
| Plotly, WordCloud | Thêm phụ thuộc; WordCloud cần font phía server |
| Recharts | Các biểu đồ đều đơn giản: cột chồng có vạch mức nền từng hàng và heatmap dựng bằng CSS, biểu đồ độ tin cậy bằng SVG; gỡ ngày 2026-10-19 vì không trang nào dùng |

## Lưu ý khi code

- `transformers` 5.x đã bỏ `warmup_ratio` và `group_by_length` → dùng `warmup_steps=0.1` và `train_sampling_strategy="group_by_length"`. Code mẫu cho bản 4 sẽ lỗi.
- Model đa nhiệm tự viết phải đặt `base_model_prefix="roberta"` và thuộc tính `self.roberta`; sai tên thì encoder bị khởi tạo ngẫu nhiên mà không báo lỗi.
- `py_vncorenlp`: constructor tự đổi thư mục làm việc; `download_model()` gọi `wget` → tải tay 3 file cần cho tách từ.
- Pin chính xác mọi phiên bản. Trên Linux, nếu cần torch CPU thì lấy từ index `https://download.pytorch.org/whl/cpu`.
- Image chỉ cài group `app`: `uv sync --frozen --no-default-groups --group app --no-editable`, nên không có torch, pandas, pyarrow. `tests/test_package.py` canh điều này trên `uv.lock`.
- File nặng hoặc tái tạo được (data, models, outputs, artifacts/<phiên bản>, demo-data, khoá MPS) nằm dưới `EDUPULSE_HOME`; để trống thì dùng thư mục repo. `make` đọc cấu hình từ `.env` (mẫu: `.env.example`).

## License

PhoBERT-base-v2: AGPL-3.0 · VnCoreNLP: GPL-3.0+ · UIT-VSFC: học thuật, phi thương mại → repo public theo AGPL-3.0 (`LICENSE`), ghi nguồn và trích dẫn trong app (`THIRD_PARTY_NOTICES.md`).
