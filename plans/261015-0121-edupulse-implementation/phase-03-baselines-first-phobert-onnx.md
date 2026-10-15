---
title: "Phase 3: Baseline, PhoBERT run đầu, ONNX int8, bundle v0"
status: todo
priority: P1
effort: 13.5h
dates: "2026-10-16 → 2026-10-17"
depends_on: [2]
---

# Phase 3: Baseline, PhoBERT run đầu, ONNX int8, bundle v0
<!-- Updated: Validation Session 1–2 - bundle chỉ chứa model + cấu hình, bỏ sample.csv/demo_results (S1 #3); `artifacts/<version>/` (F8); schema Pydantic dùng chung `bundle_schema.py` có `bundle_version`/`stage` (F7); luật cờ/τ_B trong `scoring.py` dùng chung với engine (F11); assert normalizer_version theo `data/processed/MANIFEST.json` (F10); golden 20 câu tự viết + `golden_predictions.json` (F8); guard không lọt câu gốc vào bundle; chia 2 luồng + đánh dấu bước [smoke] -->

## Context links
- [r1 modeling](../reports/researcher-261014-1913-phobert-vsfc-modeling.md): §5 MTL + bẫy `base_model_prefix` (L64–75, L74); §6 siêu tham số, MPS, bộ nhớ (L78–87); §7 baseline (L89–101); §9 cấu hình copy được (L109–126).
- [r2 deploy](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): `tokenizer.json` + skops (L34–35); công thức ONNX int8 + cổng (L48); loại optimum (L49).
- [r3 intelligence §2](../reports/researcher-261014-1913-intelligent-layer-xai-insights.md): T-scaling trên logits của bản triển khai (L39), baseline `method="temperature"` (L40), ngân sách duyệt (L41–48).
- [tech-stack.md](../../docs/tech-stack.md): bẫy transformers 5.x (L35), MTL (L36).
- Báo cáo kiểm chứng: [F7, F8, F10, F11](./reports/validation-verification.md).
- Prototype (scratchpad phiên, có thể đã bị dọn): `scratchpad/r1/ft.py` (vòng train tay STL/MTL đã chạy), `r1/mtl_saveload_test.py`, `r1/baselines.py`, `r2/export_dynamo.py`, `r2/eval_dynamo_q.py`, `r2/tokjson_test.py`, `r3/scripts/review_budget.py`.

## Overview
- Ngày: 2026-10-16 → 2026-10-17 · Effort: 13.5h (+12 phút MPS) · Priority: P1 (đường găng tới smoke test) · Trạng thái: chưa bắt đầu.
- Viết toàn bộ code mô hình và chạy end-to-end MỘT lần: 3 baseline × 2 task; PhoBERT đa nhiệm seed 42 (4 epoch); export ONNX int8 qua cổng ≤1 điểm; calibration bản đầu; đóng `artifacts/v0-smoke/` (model thật, chỉ model + cấu hình) cho smoke test phase 4. Phase 5 dùng lại đúng các script này.
- 2 luồng agent: (A) train → export → tokenizer [smoke]; (B) baseline → `scoring.py` + calibrate → `bundle_schema.py` + build_artifacts → golden. Đường găng ≈ 8h.

## Key Insights
- Baseline: 1 pipeline chung word 1–2 (min_df 2, sublinear, `token_pattern=(?u)\b\w+\b`) ⊕ char_wb 2–5 (min_df 3); LinearSVC C∈{0.3,1,3}, LR C∈{3,10,30}, ComplementNB α∈{0.1,0.3,1}; tune trên val; C nhỏ làm sập lớp trung tính (r1 L98–99). Xác suất: `CalibratedClassifierCV(FrozenEstimator(clf), method="temperature")` fit trên val; báo cả bản raw (F1) lẫn calibrated (r1 L101, r3 L40).
- skops: tách từ NGOÀI pipeline (input là `text_rdr`); `CalibratedClassifierCV` cần `trusted=[...]` 2 kiểu nội bộ khi load (r2 L35) → test roundtrip; xoá `stop_words_` của vectorizer trước khi lưu (gọn, không giữ n-gram hiếm).
- Mô hình: 1 lớp `PreTrainedModel` heads cấu hình được (`{"sentiment":3,"topic":4}` = MTL; 1 head = STL) ⇒ cùng kiến trúc head, 1 code path. BẮT BUỘC `base_model_prefix="roberta"` + `self.roberta`, sai là encoder khởi tạo ngẫu nhiên im lặng (r1 L74) → có test.
- Vòng train tay (port `r1/ft.py`) thay Trainer: đã chạy thật trên MPS, tránh API v5 đổi tên. Cấu hình r1 §9: lr 2e-5, bs 32 gộp theo độ dài, warmup 10% + linear, wd 0,01 (trừ bias/LayerNorm), clip 1,0, fp32, max_len 128, pad_to_multiple_of=8, AdamW fused, chọn epoch theo mF1 val (MTL: trung bình 2 task).
- MPS: 1 run 4 epoch ≈ 12 phút; peak 7–9GB; `torch.mps.clear_graph_cache()` + `empty_cache()` mỗi 100 step; KHÔNG 2 job song song và tắt Docker Desktop khi train (r1 L82–85; F14) → khoá `outputs/.mps.lock`.
- ONNX: wrapper `forward(input_ids, attention_mask) → (sentiment, topic)`; `torch.onnx.export(dynamo=True, dynamic_shapes)` với `torch.export.Dim` → `quant_pre_process` (BẮT BUỘC với dynamo) → `quantize_dynamic(QInt8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=True)`; ≈284MB; export trên CPU (r2 L48).
- `vinai/phobert-base-v2` @ `86cd7fd4c148980922ac11a2cf5e257f2ba639e1` có `tokenizer.json`; `tokenizers` cho id giống `PhobertTokenizer` 3166/3166 (r2 L34). Serve truncation 128 = `max_len` train (p99,9 = 72 token, r1 L81), pad id 1.
- Luật cờ + τ_B + bảng hiệu quả nằm trong `src/edupulse/scoring.py` (numpy-only), `ml/calibrate.py` và `engine/flags.py` cùng gọi ⇒ số "hiệu quả ước tính" khớp luật chạy thật (F11). T fit trên logits val của CHÍNH bản int8 (r3 L39).
- Bundle KHÔNG chứa câu gốc UIT-VSFC (Validation S1 #3): file mẫu, câu mẫu trang 1 và demo dashboard tính lúc build image (phase 6). Golden cho test hồi quy = 20 câu TỰ VIẾT (`tests/fixtures/golden_sentences.json`, nội dung của dự án, AGPL) + `golden_predictions.json` trong bundle chỉ có id → nhãn/xác suất (F8).

## Requirements
- `ml/common.py`: đường dẫn, `set_seed`, `mps_lock()` (fcntl, không chờ → lỗi rõ), `assert_normalizer()` (MANIFEST dữ liệu = hằng code), metrics (acc, mF1, wF1, P/R/F1 từng lớp, CM chuẩn hoá hàng, ECE 15 bin, NLL), lưu JSON/npz.
- `ml/train_baseline.py`: 3 model × 2 task, tune trên val, `models/baseline/*.skops`, `results/metrics/val_baselines.json` + `results/tables/val_baselines.md`.
- `ml/phobert_model.py` + `ml/train_phobert.py --heads {mtl,sentiment,topic} --seed --epochs --device {mps,cuda,cpu}`: best checkpoint (`save_pretrained`, safetensors) + `val_logits.npz` + `history.json` (mF1 từng epoch, peak MPS MB, thời gian, `normalizer_version`); checkpoint epoch 1 lưu riêng.
- `ml/export_onnx.py --run <dir|best>`: parity FP32 ONNX vs torch, int8, đánh giá val FP32 vs int8 cả 2 task, `val_logits_int8.npz` + `onnx_gate.json`, `golden_predictions.json`; exit ≠0 nếu giảm >1 điểm mF1.
- `src/edupulse/scoring.py`: `softmax_t`, `quantile_rank`, `flag_score = min(q_s, q_t)`, `threshold_for_budget`, `rank_batch(scores, pct, forced)`, `efficiency(flags, ok_s, ok_t)`.
- `ml/calibrate.py`: T mỗi head (tối ưu NLL), CDF val 101 điểm, τ_B cho 5/10/20/30%, bảng hiệu quả val → `calibration.json` (qua `scoring.py`).
- `src/edupulse/bundle_schema.py` (Pydantic, dùng chung ml + API): `Manifest` (`bundle_version`, `stage` smoke|release, `model_version`, `created_at`, `git_commit`, `normalizer_version`, `max_len`, `labels`, `files{name: sha256}`, `dataset`, `base_model`), `Calibration`, `ModelCard` (khối chỉ số có `split` val|test; bảng từng lớp, CM, reliability bins, so sánh baseline, MTL vs STL, ngân sách duyệt kèm mức 10%, CheckList, giới hạn, ghi chú x86/demo), `GoldenPredictions`, `DemoManifest`, `DemoResults` (dùng ở phase 6).
- `ml/build_artifacts.py --version <v>` → `artifacts/<v>/`: whitelist tên file; validate mọi JSON bằng schema; assert `normalizer_version` (code = MANIFEST dữ liệu = `history.json`); guard: 200 câu ngẫu nhiên của `data/processed` không xuất hiện trong bất kỳ JSON nào của bundle.

## Architecture
```
data/processed (+ MANIFEST) ─► train_baseline ─► models/baseline/{task}_{svc,lr,cnb}.skops ─► results/metrics/val_baselines.json
                           └► train_phobert (MPS + lock) ─► models/phobert/mtl-s42/{model.safetensors, config.json, val_logits.npz, history.json}
models/phobert/mtl-s42 ─► export_onnx (CPU) ─► outputs/onnx/mtl-s42/{model.fp32.onnx, model.int8.onnx, val_logits_int8.npz, onnx_gate.json, golden_predictions.json}
val_logits_int8 ─► calibrate (scoring.py) ─► outputs/calibration/calibration.json
build_artifacts (bundle_schema) ─► artifacts/v0-smoke/{manifest.json, model.int8.onnx, tokenizer.json, calibration.json, model_card.json,
                                                      insight_background.json, golden_predictions.json, baseline/{sentiment,topic}_linearsvc.skops}
```
- Luật cờ (dùng chung 1 câu và batch): q = phân vị của độ tin cậy (max-prob sau T) trên CDF val của task; điểm cờ = min(q_s, q_t). Ngân sách B% ⇔ τ_B sao cho trên val tỉ lệ câu có điểm < τ_B = B%. 1 câu: cờ nếu q_s < τ_10 hoặc q_t < τ_10; batch: cờ ⌈B·n⌉ câu điểm thấp nhất (phase 6). Lý do = task có q dưới ngưỡng. Bảng "Hiệu quả ước tính" (lỗi CX bắt · lỗi CĐ bắt · phần không cờ đúng cả 2) tính theo luật này (số ở design L124 từ luật tích của r3 nên sẽ lệch nhẹ).
- Bundle v0: `stage: "smoke"`, `model_card.json` chỉ có khối `split: "val"`; trang 4 hiển thị nhãn "bản smoke · số trên val" cho tới v1.
- `tests/fixtures/golden_sentences.json`: 20 câu tự viết phủ 3 cảm xúc × 4 chủ đề + ca đặc biệt (không dấu, có SĐT, "Thầy Nam" viết hoa, ":)", ">40 từ", "11:55", dấu thanh cũ "hòa"); ghi rõ nguồn "tự viết cho kiểm thử, không trích UIT-VSFC".

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/common.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/train_baseline.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/phobert_model.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/train_phobert.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/export_onnx.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/calibrate.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/build_artifacts.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/scoring.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/src/edupulse/bundle_schema.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/fixtures/golden_sentences.json`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/test_scoring.py` (đơn vị + parity τ_B/hiệu quả: đường calibrate = đường `rank_batch`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/test_bundle_schema.py` (v0 + v1 hợp lệ; thiếu trường → lỗi)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_common_metrics.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_phobert_model.py` (marker `train`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_baseline_roundtrip.py` (marker `data`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_build_artifacts.py` (whitelist, schema, guard câu gốc, normalizer)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/metrics/val_baselines.json` (sinh, commit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/tables/val_baselines.md` (sinh, commit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/metrics/val_phobert_first_run.json` (sinh, commit)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/metrics/onnx_gate_v0.json` (sinh, commit)

Sinh ra, không commit: `models/`, `outputs/onnx/`, `outputs/calibration/`, `artifacts/v0-smoke/`.

## Implementation Steps
1. [smoke] `ml/common.py` + `tests/ml/test_common_metrics.py` (ECE/mF1 trên mảng biết trước đáp án).
2. [smoke · luồng A] `ml/phobert_model.py` + test bẫy khởi tạo (1 tensor encoder bằng đúng trọng số gốc; `save_pretrained`/`from_pretrained` cho logits giống hệt).
3. [smoke · luồng A] `ml/train_phobert.py` (port `r1/ft.py`): batch gộp theo độ dài, collator pad 8, AdamW fused, scheduler, clip, `clear_graph_cache` mỗi 100 step, eval mỗi epoch, lưu best + epoch 1; `--device cuda` cho Colab; assert normalizer. Chạy thử 256 mẫu × 1 epoch.
4. [smoke · luồng A] Run MTL seed 42, 4 epoch (khoá MPS; tắt Docker Desktop + app nặng; `caffeinate -i`). Sanity với r1 L70.
5. [smoke · luồng A] `ml/export_onnx.py` theo r2 L48; parity max |Δlogit| < 1e-3 trên 64 câu val; int8; đánh giá val; cổng ≤1 điểm; `golden_predictions.json` trên 20 câu tự viết.
6. [smoke · luồng A] Tải `tokenizer.json` (`vinai/phobert-base-v2` @ `86cd7fd4…`); kiểm id giống `PhobertTokenizer` trên 1.583 câu val.
7. [luồng B] `ml/train_baseline.py`; lưới C/α trên val; skops (xoá `stop_words_`) + test roundtrip (`trusted` liệt kê tường minh).
8. [smoke · luồng B] `src/edupulse/scoring.py` + `tests/test_scoring.py`; `ml/calibrate.py` (T, CDF, τ_B, hiệu quả val).
9. [smoke · luồng B] `src/edupulse/bundle_schema.py` + `tests/test_bundle_schema.py`; `tests/fixtures/golden_sentences.json` (20 câu tự viết).
10. [smoke] `ml/build_artifacts.py --version v0-smoke` → `artifacts/v0-smoke/` + `tests/ml/test_build_artifacts.py`; kiểm nạp được trong env chỉ group app (onnxruntime + tokenizers, không torch).
11. Ghi `results/metrics/*` + commit (`feat: add baseline and phobert training scripts`, `feat: add onnx export, calibration and bundle schema`).

## Todo list
- [x] `ml/common.py` (metrics, ECE, khoá MPS, assert normalizer) + test
- [x] `ml/phobert_model.py` + test bẫy `base_model_prefix`
- [x] `ml/train_phobert.py` (vòng tay, chạy thử 256 mẫu)
- [x] Run MTL seed 42 (4 epoch, Docker tắt) + sanity
- [x] `ml/export_onnx.py`: parity FP32, int8, cổng ≤1 điểm, golden predictions
- [x] `tokenizer.json` ghim sha + kiểm id 1.583/1.583
- [x] `ml/train_baseline.py` + tune val + skops roundtrip
- [x] `scoring.py` + `ml/calibrate.py` + test parity
- [x] `bundle_schema.py` + golden 20 câu tự viết + test schema
- [x] `ml/build_artifacts.py` → `artifacts/v0-smoke/` + guard + test; ghi metrics + commit

## Success Criteria
- Baseline val gần r1 L94: LinearSVC mF1 cảm xúc ≈79,6, chủ đề ≈77,2 (±1,5); thứ hạng LinearSVC > LR > CNB.
- MTL seed 42: mF1 val ≥84 (cảm xúc) và ≥81 (chủ đề) (r1: 85,55 / 82,72).
- Test bẫy khởi tạo, skops roundtrip, parity `scoring.py`, schema v0 đều pass.
- ONNX FP32 parity < 1e-3; int8 giảm mF1 val ≤1 điểm mỗi task ở batch=1; file ≈333MB với cấu hình chọn theo tìm kiếm đăng ký trước (`results/metrics/onnx_gate_search.json`; int8 toàn phần 291MB trượt cổng).
<!-- Updated: Session 5 (lead) - kích thước model và chế độ batch=1 theo kết quả tìm kiếm lượng tử hoá -->
- `tokenizer.json` khớp id 100% val; `artifacts/v0-smoke/manifest.json` hợp lệ (schema + sha256), `stage: "smoke"`, nạp được không cần torch; guard: 0 câu gốc trong bundle.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Export dynamo lỗi với model tuỳ biến 2 output | TB × Cao | Export trên CPU, wrapper thuần tensor; thử exporter cũ `dynamo=False` (còn ở torch 2.14.1 hay không: [UNVERIFIED]); bí thì dừng mốc M6 hỏi người dùng — chạy torch CPU sẽ vi phạm ràng buộc "runtime không torch" |
| MPS OOM/swap, máy sleep giữa run | TB × TB | Khoá MPS, tắt Docker, eval bs 64, `caffeinate -i`; Colab T4 dự phòng (tách từ local, upload parquet) |
| Cổng int8 không qua | Thấp × Cao | MatMul-only per-channel + reduce_range (r2 đo +0,5 mF1); dự phòng ONNX FP32 (≈540MB, RAM cao hơn) — đo lại ở smoke |
| Schema chưa đủ trường cho trang 4 | TB × TB | Lấy danh sách trường từ design L140–141 + wireframe 04; v1 thêm trường thì bump schema có test |
| Số val lệch r1 nhiều | Thấp × TB | Kiểm parity chuẩn hoá, LR, seed; so `history.json` với r1 L80 |

Rollback: run hỏng xoá thư mục run; bundle theo phiên bản (`artifacts/v0-smoke/` giữ nguyên khi tạo `artifacts/v1.0/`); code `git revert`.

## Security Considerations
- skops: chỉ `trusted` danh sách kiểu tường minh, không `trusted=True`; chỉ nạp file có sha256 khớp manifest.
- Tải model/tokenizer từ HF ghim commit SHA; không commit `models/`, `artifacts/`.
- Bundle không chứa câu gốc dataset (guard tự động); không log văn bản câu trong log huấn luyện (chỉ số đếm, metric).

## Next steps
- Phase 4 dùng `artifacts/v0-smoke/` cho smoke test (mốc M1). Phase 5 dùng lại script cho 9 run + đánh giá cuối (sau smoke, Docker tắt khi train).
