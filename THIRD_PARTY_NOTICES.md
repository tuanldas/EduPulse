# Thành phần bên thứ ba

EduPulse theo [GNU AGPL-3.0](LICENSE). Các mô hình, dữ liệu và mã dưới đây giữ giấy phép riêng của chúng.

## Mô hình, dữ liệu, font và phần mềm đi kèm

| Thành phần | Giấy phép | Cách dùng | Nguồn |
|---|---|---|---|
| PhoBERT-base-v2 (`vinai/phobert-base-v2`) | AGPL-3.0 | Mô hình nền được tinh chỉnh; trọng số sau tinh chỉnh là sản phẩm phái sinh | https://huggingface.co/vinai/phobert-base-v2 |
| VnCoreNLP (RDRSegmenter) | GPL-3.0 trở lên | Tách từ. Jar và file mô hình nằm ở `vendor/vncorenlp/`, kèm giấy phép gốc | https://github.com/vncorenlp/VnCoreNLP |
| UIT-VSFC | Học thuật, phi thương mại | Huấn luyện và đánh giá. Repo, bundle và Hugging Face không chứa câu gốc | https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback |
| Be Vietnam Pro | SIL OFL-1.1 | Font giao diện | https://fonts.google.com/specimen/Be+Vietnam+Pro |
| JetBrains Mono | SIL OFL-1.1 | Font mã | https://www.jetbrains.com/lp/mono/ |
| Material Symbols | Apache-2.0 | Biểu tượng giao diện | https://fonts.google.com/icons |

Theo AGPL-3.0 §13, người dùng ứng dụng qua mạng phải truy cập được mã nguồn tương ứng. Ứng dụng hiển thị đường dẫn lấy từ biến `EDUPULSE_SOURCE_URL` ở trang "Mô hình & đạo đức".

## Thư viện Python chạy khi triển khai (group `app`)

| Thư viện | Giấy phép |
|---|---|
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| ONNX Runtime | MIT |
| tokenizers | Apache-2.0 |
| NumPy | BSD-3-Clause (kèm các giấy phép của thành phần đóng gói theo wheel) |
| py_vncorenlp | MIT |
| PyJNIus | MIT |
| python-calamine | MIT |
| scikit-learn | BSD-3-Clause |
| skops | MIT |
| underthesea | Apache-2.0 |

Phiên bản chính xác nằm trong `pyproject.toml` và `uv.lock`. Thư viện huấn luyện (group `train`) và thư viện web có giấy phép riêng, xem metadata của từng gói và `uv.lock`, `apps/web/pnpm-lock.yaml`.

## Trích dẫn

Công trình dùng trong dự án; ứng dụng hiển thị cùng nội dung này ở trang "Mô hình & đạo đức".

```bibtex
@inproceedings{nguyen2018uitvsfc,
  title     = {UIT-VSFC: Vietnamese Students' Feedback
               Corpus for Sentiment Analysis},
  author    = {Nguyen, Kiet Van and Nguyen, Vu Duc and
               Nguyen, Phu X. V. and Truong, Tham T. H.
               and Nguyen, Ngan Luu-Thuy},
  booktitle = {Proc. KSE 2018},
  pages     = {19--24},
  year      = {2018}
}
@inproceedings{nguyen2020phobert,
  title     = {PhoBERT: Pre-trained language models
               for Vietnamese},
  author    = {Nguyen, Dat Quoc and Nguyen, Anh Tuan},
  booktitle = {Findings of EMNLP 2020},
  pages     = {1037--1042},
  year      = {2020}
}
```
