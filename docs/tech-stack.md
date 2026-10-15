# EduPulse — Tech stack (duyệt 2026-10-15)

Bằng chứng và số đo: `plans/reports/researcher-261014-1913-*.md`. Phiên bản xác minh trên PyPI ngày 2026-10-14.

> **Thay đổi 2026-10-15 (quyết định của người dùng):** bỏ Streamlit; giao diện dùng **Next.js** (frontend) gọi **API Python** (backend). Các dòng "Giao diện" và "Host" bên dưới đã lỗi thời; frontend, framework API và nơi host miễn phí sẽ được chốt trong kế hoạch triển khai (`plans/`). Phần ML (dữ liệu, tách từ, mô hình, ONNX int8) giữ nguyên.

| Lớp | Lựa chọn | Phiên bản | Lý do |
|---|---|---|---|
| Ngôn ngữ, môi trường | Python + uv (dependency groups `train` / `app`, `uv.lock`) | Python 3.12 | Khớp host; scikit-learn 1.9 cần Python ≥ 3.11 |
| Dữ liệu | pandas + pyarrow; UIT-VSFC từ nhánh parquet `refs/convert/parquet` | pandas 3.0.6 | Repo HF chỉ có loading script, `datasets` ≥ 4 không chạy script |
| Tách từ | RDRSegmenter: `py_vncorenlp` + `pyjnius` + JRE (trên host dùng `jdk4py`) | 0.1.4 / 1.9.0 / 25.0.2.1 | PhoBERT pretrain bằng RDR; hơn không tách khoảng 2 macro-F1 |
| Baseline | scikit-learn: TF-IDF (word 1–2 + char_wb 2–5) + LinearSVC / LogisticRegression / Naive Bayes; lưu bằng `skops` | 1.9.1 / 0.16.0 | Chuẩn, nhanh; skops an toàn hơn pickle khi load |
| Mô hình chính | `vinai/phobert-base-v2`, 1 encoder + 2 head (đa nhiệm); 2 mô hình đơn nhiệm làm đối chứng | torch 2.14.1, transformers 5.19.0 | Đa nhiệm ngang đơn nhiệm về độ chính xác, tốn ½ RAM và latency |
| Huấn luyện | PyTorch MPS (M1 Pro) fp32; Colab T4 dự phòng | — | Khoảng 12 phút mỗi run |
| Export, suy luận | ONNX (`torch.onnx.export(dynamo=True)`) → `onnxruntime` int8 dynamic (MatMul, per-channel, reduce_range) + `tokenizers` | onnxruntime 1.31.0, tokenizers 0.23.3 | Model 284MB, nhanh ~2,6×, không cần torch khi chạy app |
| Lưu model | Hugging Face Hub (repo model public) qua `huggingface-hub` | 2.2.0 | Miễn phí, không cần token khi tải |
| Giao diện | Streamlit (`st.navigation`, `st.data_editor`) + Altair | 1.65.0 / 6.3.0 | Đủ multipage, bảng sửa nhãn, biểu đồ |
| Ẩn dữ liệu cá nhân | regex + `underthesea` NER (tuỳ chọn, chạy trên chữ gốc) | 9.5.0 | NER chỉ nhận tên khi còn chữ hoa |
| Host | Streamlit Community Cloud; dự phòng local + Cloudflare Quick Tunnel | — | Miễn phí, không cần thẻ |
| Kiểm thử | pytest | — | — |

## Không dùng

| Thứ | Lý do |
|---|---|
| HF Spaces (Docker/Gradio) | Từ ~07/2026, tạo Space mới loại này cần gói PRO |
| `optimum-onnx` | Ghim `transformers` < 4.58, không chạy với 5.x |
| `torch.ao.quantization.quantize_dynamic` | Deprecated, chậm hơn FP32 trên M1 |
| LIME, SHAP | Vượt ngân sách 2 giây/câu |
| Attention, Gradient×Input làm XAI | Độ trung thực đo được chỉ ngang chọn ngẫu nhiên |
| Plotly, WordCloud | Thêm phụ thuộc; WordCloud cần font phía server |

## Lưu ý khi code

- `transformers` 5.x đã bỏ `warmup_ratio` và `group_by_length` → dùng `warmup_steps=0.1` và `train_sampling_strategy="group_by_length"`. Code mẫu cho bản 4 sẽ lỗi.
- Model đa nhiệm tự viết phải đặt `base_model_prefix="roberta"` và thuộc tính `self.roberta`; sai tên thì encoder bị khởi tạo ngẫu nhiên mà không báo lỗi.
- `py_vncorenlp`: constructor tự đổi thư mục làm việc; `download_model()` gọi `wget` → tải tay 3 file cần cho tách từ.
- Pin chính xác mọi phiên bản. Trên Linux, nếu cần torch CPU thì lấy từ index `https://download.pytorch.org/whl/cpu`.

## License

PhoBERT-base-v2: AGPL-3.0 · VnCoreNLP: GPL-3.0+ · UIT-VSFC: học thuật, phi thương mại → repo public, ghi nguồn và trích dẫn trong app.
