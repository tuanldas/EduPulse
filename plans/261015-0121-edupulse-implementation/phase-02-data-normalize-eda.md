---
title: "Phase 2: Dữ liệu, chuẩn hoá dùng chung & EDA"
status: todo
priority: P1
effort: 10.5h
dates: "2026-10-15 → 2026-10-16"
depends_on: [1]
---

# Phase 2: Dữ liệu, chuẩn hoá dùng chung & EDA
<!-- Updated: Validation Session 1–2 - tên file parquet `0000.parquet` + assert theo tree (F1); ghim SHA VnCoreNLP + LICENSE.md (F17); `data/processed/MANIFEST.json` có normalizer_version (F10); module dùng chung `text/ngrams.py` (F11); luật tên "xưng hô + Viết hoa" vào Must, NER vẫn Should (F12); đếm nền bỏ n-gram xuất hiện 1 lần; đánh dấu bước [smoke] -->

## Context links
- [PDR](../../docs/project-overview-pdr.md): dataset + split chuẩn (L21), EDA cho báo cáo (L22), PII gồm tên (L30).
- [r1 modeling](../reports/researcher-261014-1913-phobert-vsfc-modeling.md): §2 tách từ + bẫy py_vncorenlp (L23–32, nhất là L30); §3 chuẩn hoá + bảng token đặc biệt (L34–50); không dấu −13 acc (L49).
- [r3 intelligence §5](../reports/researcher-261014-1913-intelligent-layer-xai-insights.md): regex PII đã kiểm (L69), tên: NER chỉ chạy trên chữ có hoa + luật "cue + Viết Hoa" khi bỏ NER (L70), 2 view PII (L78); §4 đếm nền + mức nền (L59, L62).
- [r2 deploy §4](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): RDR trong container, `-Xmx512m`, Lock (L26–30).
- [design-guidelines](../../docs/design-guidelines.md): cảnh báo không dấu + placeholder PII (L119), thuật ngữ (L153).
- Báo cáo kiểm chứng: [F1, F10, F11, F12, F17](./reports/validation-verification.md).
- Prototype (scratchpad phiên, có thể đã bị dọn — chỉ tham khảo): `scratchpad/r1/normalize_proto.py`, `r1/skew_test.py`, `r3/scripts/pii_fix.py`, `r3/scripts/ner_test.py`.

## Overview
- Ngày: 2026-10-15 → 2026-10-16 · Effort: 10.5h (lõi 8h + EDA 2.5h) · Priority: P1 (đường găng) · Trạng thái: chưa bắt đầu.
- Tải UIT-VSFC ghim commit; module chuẩn hoá DÙNG CHUNG train ↔ serve (NFC → PII → lowercase → dấu thanh → token đặc biệt → gộp ký tự lặp → RDRSegmenter); module n-gram dùng chung; cache dữ liệu đã tách từ có đóng dấu phiên bản chuẩn hoá; EDA sinh hình/bảng báo cáo + đếm nền cho insight.
- Bước 1–8 là lõi, chặn phase 3 (bước [smoke] làm trước, 2 luồng: chuẩn hoá/PII ‖ tải dữ liệu/RDR). Bước 9–10 (EDA) chạy song song phase 3.

## Key Insights
- Parquet ở `refs/convert/parquet` = commit `2d76906157232f87e9883fa37ebfc91ffdee1f83`; tên file THẬT là `default/{train,validation,test}/0000.parquet` (474.785 / 63.327 / 133.749 byte, kiểm bằng HF tree API 2026-10-15; `0.parquet` là 404). Liệt kê tree rồi assert đúng 3 file trước khi tải; ghim SHA vì HF có thể tái tạo nhánh convert.
- VnCoreNLP ghim commit `62bbc58fe5d113c898eae112656be97dcf50b3a0` (2023-02-11, "v1.2 released"): `VnCoreNLP-1.2.jar`, `models/wordsegmenter/vi-vocab` (526.544 byte), `models/wordsegmenter/wordsegmenter.rdr` (128.508 byte), license ở `LICENSE.md` (GPL-3.0+); đã kiểm 4 URL raw trả 200.
- Dữ liệu gốc đã lowercase, NFC, dấu câu tách bằng space, tên → `wzjwz<N>`; 284 câu (1,8%) dấu thanh kiểu cũ trong khi vocab PhoBERT-v2 chỉ có kiểu mới → bảng 15 cặp VinAI (r1 L36).
- Thứ tự là hợp đồng: PII chạy trên chữ gốc (luật tên + NER cần chữ hoa) TRƯỚC lowercase; map emoticon TRƯỚC RDR (RDR tách ":)" làm 2 câu); `word_segment` trả list câu → join bằng space (r1 L30, L35).
- Bỏ lowercase −0,68 mF1; gõ không dấu −13 acc/−17 mF1 → cần detector không dấu để luôn gắn cờ (r1 L49). `doubledot` lệch nhãn (86% tiêu cực) nhưng vẫn map để khớp train (r1 L48); gộp ký tự lặp chỉ với chữ cái (≥3 lần → 1), không với số; "!!!" → "!".
- py_vncorenlp: constructor `os.chdir(save_dir)` vĩnh viễn → bọc lưu/khôi phục cwd; `download_model()` gọi `wget` và tải cả NER/parse → tự tải đúng 3 file; JVM 1 lần/process → singleton + `threading.Lock`.
- PII 2 view (r3 L78): hiển thị/xuất dùng `[TÊN] [SĐT] [EMAIL] [MSSV] [CCCD] [URL]`; đầu vào mô hình: tên → `wzjwz<k>` (k phân biệt theo tên trong câu), PII khác → xoá. Tên: luật rẻ "từ xưng hô (thầy, cô, anh, chị, bạn, em, ông, bà, giảng viên, gv, ths, ts…) + 1–3 âm tiết Viết Hoa" là Must (r3 L70); NER underthesea là Should; tên viết thường KHÔNG bắt được ⇒ tiêu chí "best effort + disclaimer". Dữ liệu thật có 0 PII ngoài tên → kiểm regex bằng chuỗi tổng hợp (22 positive / 25 negative của r3).
- `NORMALIZER_VERSION` phải đi theo dữ liệu: `data/processed/MANIFEST.json` ghi version lúc tiền xử lý; train/export/build_artifacts assert khớp hằng trong code (F10).
- N-gram (uni+bigram trên từ RDR) là 1 hàm dùng chung `text/ngrams.py` cho đếm nền (đây) và insight (phase 6) ⇒ prior log-odds không lệch cách tách token (F11). Mức nền tiêu cực theo chủ đề (train) GV 35,6% · CT 78,0% · CSVC 95,4% · Khác 40,2% + tỉ lệ cảm xúc toàn train là thống kê dữ liệu ⇒ sinh ở đây, đóng vào bundle ở phase 3; bỏ n-gram đếm = 1 (gọn, không giữ cụm hiếm gần nguyên văn).

## Requirements
- `prepare(raw) -> Prepared(display_text, model_text, pii_types, no_diacritics, n_words)`; `prepare_many(texts)`; hằng `NORMALIZER_VERSION`.
- Detector không dấu: tỉ lệ âm tiết chữ có dấu thanh/mũ < ngưỡng (khởi điểm 0,15, cần ≥3 âm tiết; bỏ qua `wzjwz<k>`), hiệu chỉnh trên val.
- Tên: luật "xưng hô + Viết Hoa" luôn bật; NER underthesea chỉ khi `EDUPULSE_NER=1` và câu có chữ hoa; import lười.
- `text/ngrams.py`: `ngrams(text_rdr) -> list[str]` (uni+bigram, bỏ token chỉ là dấu câu, giữ `_`), dùng cho cả đếm nền và insight.
- Cache `data/processed/{train,validation,test}.parquet` (`sentence, sentiment, topic, text_rdr, n_words, no_diacritics`) + `data/processed/MANIFEST.json` (`normalizer_version`, `dataset_revision`, sha256 từng file).
- Đếm nền insight `outputs/data/insight_background.json` (n-gram count ≥2, mức nền, tỉ lệ cảm xúc toàn train, tỉ trọng chủ đề, α0=1000) sinh ở bước lõi vì bundle v0 cần; EDA chỉ còn hình + bảng tiếng Việt có dấu trong `results/`.
- Test bằng câu thật (val, tải lúc chạy, không commit) + chuỗi PII tổng hợp; không mock.

## Architecture
```
raw ─NFC─► pii.mask (regex · luật tên xưng hô+Viết Hoa · NER tuỳ chọn) ─┬─► display_text  ([TÊN] [SĐT] … , giữ hoa/thường)
                                                                       └─► model_pre     (tên → wzjwz<k>; PII khác → xoá)
model_pre ─► is_no_diacritics (trước lowercase)
model_pre ─lower─► TONE_MAP ─► special_tokens ─► collapse_repeats ─► Segmenter.segment ─► model_text
data/raw (0000.parquet ×3) ─► preprocess (prepare_many) ─► data/processed/*.parquet + MANIFEST.json{normalizer_version}
                                                       └─► ngrams() ─► outputs/data/insight_background.json
```
- `text/normalize.py`: `TONE_MAP` 15 cặp; `map_special_tokens`: emoticon trước (":))"→`colonsmilesmile`, ":)"→`colonsmile`, ":("→`colonsad`, ":D"→`colonbigsmile`, "<3"/":*"→`colonlove`, ":v"→`coloncontemn`, ":o"→`colonsurprise`), emoji khác → xoá, "v.v"→`vdotv`, "..."/"…"→`dotdotdot`, "11:55"→`11doubledot55`, ":" đứng riêng→`doubledot`, "." giữa chữ/số ("3.0", "uit.edu.vn")→`dot` (r1 bảng L39–46); `collapse_repeats`; `is_no_diacritics`.
- `text/pii.py`: regex email/URL/SĐT di động + cố định/CCCD/MSSV theo ngữ cảnh + chuỗi số 8–12 độc lập (r3 L69); luật tên; NER tuỳ chọn; trả `PiiResult(display, model, types)`.
- `text/segment.py`: `Segmenter` singleton (`EDUPULSE_VNCORENLP_DIR`, mặc định `vendor/vncorenlp`; `JAVA_HOME` từ env/`/usr/libexec/java_home`), `annotators=["wseg"]`, `max_heap_size="-Xmx512m"`, khôi phục cwd, Lock.
- `text/pipeline.py`: điểm vào DUY NHẤT cho `ml/` và `api/`.

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/text/__init__.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/text/normalize.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/text/pii.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/text/segment.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/text/ngrams.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/text/pipeline.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/vendor/vncorenlp/VnCoreNLP-1.2.jar`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/vendor/vncorenlp/models/wordsegmenter/vi-vocab`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/vendor/vncorenlp/models/wordsegmenter/wordsegmenter.rdr`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/vendor/vncorenlp/LICENSE.md`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/vendor/vncorenlp/SHA256SUMS`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/download_data.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/preprocess.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/eda.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/text/test_normalize.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/text/test_pii.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/text/test_segment.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/text/test_ngrams.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/text/test_pipeline.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/figures/eda_*.png` (sinh, commit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/tables/eda_*.csv` (sinh, commit; chỉ số đếm, không chứa câu)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/metrics/data_summary.json` (sinh, commit)

Sinh ra, không commit: `data/raw/*.parquet` + `data/raw/MANIFEST.json`, `data/processed/*.parquet` + `data/processed/MANIFEST.json`, `outputs/data/insight_background.json`.

## Implementation Steps
1. [smoke] Tải 3 file VnCoreNLP từ `https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/62bbc58fe5d113c898eae112656be97dcf50b3a0/…` bằng `curl`; ghi `SHA256SUMS` + `LICENSE.md` vào `vendor/vncorenlp/`.
2. [smoke] `ml/download_data.py`: `list_repo_tree(repo_type="dataset", revision=<SHA>)` → assert đúng `default/{train,validation,test}/0000.parquet` → `hf_hub_download` → `data/raw/`; `MANIFEST.json` (SHA, sha256 từng file, số dòng); assert 11.426/1.583/3.166 và cột `sentence, sentiment, topic`.
3. [smoke] `normalize.py` + `tests/text/test_normalize.py`: golden từng quy tắc (dấu thanh, emoticon, "11:55", "v.v", "...", gộp lặp chữ nhưng giữ "2000").
4. [smoke] `pii.py` (regex + luật tên) + `tests/text/test_pii.py`: 22 positive / 25 negative tổng hợp; marker `data`: 0 hit trên 16.175 câu thật; luật tên: "Thầy Nam dạy hay" → `[TÊN]`, "thầy nam dạy hay" → không che (giới hạn đã ghi); NER (Should, marker `slow`) bắt "Nguyễn Văn An" trong câu có hoa, rỗng khi toàn chữ thường.
5. [smoke] `segment.py` + `tests/text/test_segment.py` (marker `java`): cwd không đổi sau khởi tạo; ":)" đã map không tách câu; 4 luồng × 200 câu không lỗi, kết quả giống chạy tuần tự.
6. [smoke] `pipeline.py` + `tests/text/test_pipeline.py`: (a) parity "biến thể thô" — 200 câu val chuyển về dạng người dùng gõ (viết hoa đầu câu, dấu câu dính chữ, ":)" ":" "v.v" dạng thật) cho `model_text` giống câu gốc; (b) detector không dấu trên val gốc và val đã bỏ dấu (NFD + bỏ dấu + đ→d).
7. `ngrams.py` + `tests/text/test_ngrams.py` (đơn vị + parity: đếm lại từ `data/processed/train.parquet` khớp `insight_background.json`).
8. [smoke] `ml/preprocess.py`: `prepare_many` cho 3 split → `data/processed/` + `MANIFEST.json{normalizer_version}`; in thống kê (số câu sửa dấu thanh, token đặc biệt, không dấu); ghi `outputs/data/insight_background.json`.
9. `ml/eda.py` (song song phase 3): `eda_label_dist.png` (2 task × 3 split), `eda_topic_sentiment.png` + `.csv` (crosstab + mức nền, tỉ trọng), `eda_length.png` + `eda_length_percentiles.csv` (âm tiết, từ RDR, token PhoBERT qua `tokenizer.json` của `vinai/phobert-base-v2` @ `86cd7fd4…`), `eda_special_tokens.csv`, `eda_tone_old_style.csv`, `eda_duplicates.csv`, `eda_top_logodds.csv`, `data_summary.json` (số câu, IAA thẻ dataset: cảm xúc 91%, chủ đề 71%).
10. Đối chiếu số với research: phân bố nhãn (r1 L53), 1,8% dấu cũ, p95 token = 30 (r1 L81), mức nền (r3 L62); lệch → điều tra trước khi sang phase 3. Commit (`feat: add shared text normalization and data pipeline`, `feat: add eda figures and tables`).

## Todo list
- [x] Vendor VnCoreNLP @ `62bbc58f…` (3 file + SHA256SUMS + LICENSE.md)
- [x] `ml/download_data.py`: assert tree `0000.parquet`, MANIFEST, số dòng đúng
- [x] `text/normalize.py` + test golden
- [x] `text/pii.py`: regex + luật tên (Must) + NER (Should) + test 22/25 + 0 hit câu thật
- [x] `text/segment.py` + test JVM/cwd/đa luồng
- [x] `text/pipeline.py` + `NORMALIZER_VERSION` + test parity + test không dấu
- [x] `text/ngrams.py` + test đơn vị/parity
- [x] `ml/preprocess.py` → parquet + `MANIFEST.json` + `insight_background.json`
- [x] `ml/eda.py` → hình/bảng báo cáo
- [x] Đối chiếu số với research + commit

## Success Criteria
- `data/raw/MANIFEST.json` ghi SHA `2d769061…` + sha256; số dòng 11.426 / 1.583 / 3.166; `data/processed/MANIFEST.json` có `normalizer_version` = hằng trong code.
- `uv run pytest tests/text` pass: PII 22/22 positive, 0/25 negative, 0 hit trên 16.175 câu thật; luật tên che đúng ca có Viết Hoa.
- Parity biến thể thô: ≥98% (≥196/200) câu cho `model_text` giống hệt; mọi câu lệch được giải thích. Parity n-gram: đếm lại = đếm đã lưu.
- Không dấu: bắt ≥95% câu val đã bỏ dấu (≥3 âm tiết); trên val gốc mọi câu bị đánh dấu đều đúng là gõ không dấu (kiểm tay danh sách).
- `prepare_many` 1.000 câu < 2 s trên M1 sau khi JVM nạp.
- ≥6 hình/bảng EDA trong `results/` (không chứa câu gốc); mức nền khớp r3 (±0,1 điểm).

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| pyjnius/JVM lỗi với Java 25 trên macOS | Thấp × Cao | Đặt `JAVA_HOME` tường minh; kiểm sớm trong Docker (phase 4); Plan B pyvi chỉ khi bắt buộc (phải train lại; paper không dùng PhoBERT đo pyvi ≈ RDR −0,1 mF1, với PhoBERT chưa thử — r1 L26) |
| Luật tên che nhầm ("Cô Giáo" viết hoa) | TB × Thấp | Che thừa an toàn hơn sót; ghi giới hạn; test ca điển hình |
| Bảng emoticon suy luận sai | Thấp × Thấp | Chỉ 1–2% câu; giữ, ghi vào hạn chế |
| Ngưỡng không dấu sai | TB × TB | Hiệu chỉnh bằng 2 phép đo ở bước 6; danh sách bắt nhầm kiểm tay |
| HF tái tạo nhánh convert / đổi tên file | Thấp × TB | Ghim SHA, assert tree + sha256 trong MANIFEST |

Rollback: `data/` tái tạo bằng `make data`; code `git revert`. Đổi logic chuẩn hoá sau phase 3 ⇒ tăng `NORMALIZER_VERSION`, chạy lại `make data` và train lại (train/build/API đều từ chối khi lệch version).

## Security Considerations
- Xử lý văn bản trong RAM; không log nội dung câu (kể cả trong test, chỉ log số đếm).
- Vendor jar từ nguồn chính thức, ghim commit + sha256.
- Dataset học thuật, phi thương mại: chỉ tải về local, không commit `data/`; test đọc dữ liệu đã tải lúc chạy; bảng EDA chỉ có số đếm.

## Next steps
- Phase 3 (cần bước 1–8; bước [smoke] trước). EDA (bước 9–10) chạy song song phase 3. Bảng/hình EDA vào gói báo cáo ở phase 9.
