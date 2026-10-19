---
title: "Phase 5: PhoBERT đầy đủ, đánh giá cuối, calibration, bundle v1"
status: todo
priority: P1
effort: 12h (+~3h MPS chạy nền)
dates: "2026-10-17 → 2026-10-21"
depends_on: [3, 4]
---

# Phase 5: PhoBERT đầy đủ, đánh giá cuối, calibration, bundle v1
<!-- Updated: Validation Session 1–2 - bundle v1 chỉ model + cấu hình, đẩy HF không chứa câu gốc (S1 #3); target `make report` + `--version` (F6); model card theo schema, thêm ngân sách duyệt trên test (gồm 10%), ghi "chưa đo x86" + ghi chú demo tính lúc build (F7); `artifacts/v1.0/` (F8); `run-experiments.sh` kiểm Docker tắt (F14); mốc dừng M2 (kết quả test, xong khối ML) + M3 (trước khi đẩy HF); tên repo HF mặc định `edupulse-phobert-mtl` -->

## Context links
- [PDR](../../docs/project-overview-pdr.md): bảng kết quả + kiểm định (L52–53), cổng int8 (L54), model card/CheckList (L57), test chỉ dùng 1 lần (L36).
- [r1 modeling](../reports/researcher-261014-1913-phobert-vsfc-modeling.md): quy tắc MTL vs STL (L75); epoch 5–6 cho chủ đề (L80); giao thức đánh giá, McNemar + bootstrap, error analysis (L103–107); Colab (L86); swap khi chạy song song (L85).
- [r3 intelligence](../reports/researcher-261014-1913-intelligent-layer-xai-insights.md): T-scaling + cross-fit (L39), ngân sách duyệt (L41–48), bias + CheckList (L84–86), model card (L82).
- [design-guidelines](../../docs/design-guidelines.md): nội dung trang 4 (L140–141), "Hiệu quả ước tính" trang 2 (L124), số wireframe chỉ minh hoạ (L166).
- Báo cáo kiểm chứng: [F2, F6, F7, F8, F14 + "Dòng chảy tập test"](./reports/validation-verification.md).
- Prototype (scratchpad phiên): `scratchpad/r1/qrun.sh`, `r1/paired_vs_baseline.py`, `r3/scripts/calib_phobert_val.py`, `r3/scripts/bias_phobert_val.py`.

## Overview
<!-- Updated: Session 6 - bỏ train thêm; chỉ đánh giá + phân tích trên CPU -->
- **Session 6 (2026-10-17): KHÔNG train thêm.** Phạm vi còn lại (chỉ CPU, không MPS/Docker): dùng MTL seed 42 làm bản deploy; freeze → đánh giá test ĐÚNG 1 lần cho 3 baseline + PhoBERT (FP32 và int8 batch=1); bootstrap CI 95% + McNemar/paired bootstrap so với LinearSVC; ECE + reliability trên test; hiệu quả ngân sách duyệt trên test; error analysis, bias, CheckList; model card; build `artifacts/v1.0/`; dừng M2. Các bước 9 run/3 seed/STL bên dưới không còn áp dụng; so sánh MTL/STL lấy thí nghiệm sơ bộ của research (ghi rõ sơ bộ). Effort còn ~6h.
- Ngày: 2026-10-19 → 2026-10-20 · Priority: P1 · Trạng thái: đang làm.
- Chạy đủ thực nghiệm (MTL + 2 STL × 3 seed), so sánh trên val, chọn bản deploy, export int8 cuối, calibration + ngưỡng ngân sách, ĐÓNG BĂNG rồi đánh giá test đúng 1 lần; kiểm định, lỗi, bias, CheckList; đóng `artifacts/v1.0/` (model + cấu hình). **Mốc dừng M2** = kết quả test + xong khối ML; **M3** = hỏi trước khi đẩy HF Hub.
- Song song với phase 6, 7 (máy chạy nền); chỉ chạy MPS sau M1 và khi Docker Desktop tắt.

## Key Insights
- Ngân sách máy: 1 run ≈12 phút (4 epoch) / ≈18 phút (6 epoch). MTL seed 42 từ phase 3 dùng lại nếu E* = 4 ⇒ còn 8 run ≈1,6–2,7h. Hàng đợi tuần tự + khoá MPS + `caffeinate -i` (r1 từng mất run do máy sleep, L71); Docker Desktop TẮT trong lúc chạy (r1 L85 từng swap thrash khi có VM ~7GB) — `run-experiments.sh` kiểm `docker info`, đang chạy thì dừng kèm hướng dẫn.
- Chọn E*/siêu tham số trên val với 1 seed, rồi 3 seed 42/43/44; MPS không tái lập từng bit → báo mean ± std (r1 L104).
- Quy tắc r1 L75: MTL thấp hơn STL >1,5 mF1 (trung bình 3 seed, val) ở task nào → DỪNG (M6), báo người dùng. Quyết định deploy MTL đã chốt, không tự đổi.
- Bản deploy = seed MTL có mean mF1 val cao nhất (luật cố định, đọc `val_experiments.json`); bảng chính báo mean ± std 3 seed, số bản deploy báo riêng.
- T fit trên logits val của bản int8 deploy; cross-fit không giảm ECE/NLL thì giữ T=1 (r3 L39). Bảng hiệu quả ngân sách qua `scoring.py` (cùng luật engine).
- Test chạm 1 lần: `results/freeze.json` (sha256 model + bundle, T, τ_B, E*, seed deploy, siêu tham số baseline, git commit) phải có trước; `evaluate --split test --final` lưu dự đoán MỌI model vào `outputs/test_preds/` rồi khoá. Phân tích sau (McNemar, bootstrap, lỗi, bias, ngân sách trên test) đọc file này, không suy luận lại. CheckList + mô phỏng không dấu dựng từ câu val. Demo dashboard (phase 6) chỉ HIỂN THỊ dự đoán tính lúc build, không tính chỉ số ⇒ không vi phạm.
- McNemar exact `binomtest(b, b+c, 0.5)` + paired bootstrap ΔmF1 (2.000 vòng, seed cố định) (r1 L105).
- Bundle v1 và repo HF chỉ chứa model + cấu hình (Validation S1 #3); guard của `build_artifacts` chạy lại trước khi đẩy. `results/tables/error_samples.csv` (≤30 câu) là "vài câu ví dụ" cho báo cáo — được phép giữ trong repo.
- Số trên M1 (arm64). Model card ghi "chưa đo trên x86" cho tới khi người dùng chạy `smoke_test.py --url` trên VPS; ghi chú demo tính lại lúc build trên kiến trúc VPS nên có thể lệch nhẹ so với `test_preds` của M1.
- Đẩy HF Hub là hành động ra ngoài (M3): repo mặc định `<user>/edupulse-phobert-mtl`, người dùng tự đăng nhập (`hf auth login`); chưa đẩy được thì bundle local vẫn đủ cho phase 6–8.

## Requirements
- `ml/run-experiments.sh`: hàng đợi tuần tự, bỏ qua run đã có (idempotent), log `outputs/logs/`, kiểm Docker tắt (cờ `--allow-docker` để bỏ qua có chủ đích).
- Seed deploy theo luật cố định; `export_onnx --run best`, `calibrate`, `build_artifacts` tự dùng luật này ⇒ `make export calibrate evaluate report artifacts` tái lập không cần tham số tay.
- `ml/evaluate.py`: `--split val` (tự do), `--freeze` (ghi `results/freeze.json`) và `--split test --final` (cần freeze; từ chối nếu đã có `outputs/test_preds/.done`); metrics đủ PDR L52 cho 2 task × (LinearSVC, LR, CNB [raw + calibrated], PhoBERT MTL ×3, STL ×3, MTL-int8 deploy).
- `make report` = `ml/significance.py` (deploy vs LinearSVC 2 task; MTL vs STL theo cặp cùng seed; int8 vs FP32) + `ml/error_analysis.py` (cặp nhầm lẫn, 30 lỗi tự tin nhất, lỗi theo chủ đề/độ dài >40 từ/token đặc biệt, giao lỗi TF-IDF ∩ PhoBERT; bias: recall theo ô chủ đề × cảm xúc, baseline chỉ-dùng-chủ-đề, lát "thầy" vs "cô" — bản chi tiết là Should) + `ml/checklist.py` + `ml/calibrate.py --report test`.
- `ml/checklist.py` (câu val): INV — viết hoa/thường, dấu câu/khoảng trắng thừa, thầy↔cô, đổi k của `wzjwz`, chèn SĐT/email/MSSV; DIR — thêm "không" phủ định (P tích cực phải giảm), thêm ":(" (P tiêu cực không giảm); bỏ dấu toàn câu (mức giảm acc + tỉ lệ bị cờ).
- `ml/build_artifacts.py --version v1.0` → `artifacts/v1.0/`: `stage: "release"`; `model_card.json` (schema `ModelCard`, khối `split: "test"`: acc/mF1/wF1, từng lớp, CM, reliability bins, so sánh baseline, MTL vs STL, ngân sách duyệt trên test 5/10/20/30 với 10% làm mặc định, CheckList, giới hạn, ghi chú x86 + demo, phiên bản).
- `ml/push_hub.py`: chạy lại guard → upload `artifacts/v1.0/` + README model card HF (license agpl-3.0, base_model, dataset, trích dẫn, giới hạn, cấm dùng đánh giá cá nhân) → in commit SHA để ghim.

## Architecture
```
run-experiments.sh (khoá MPS, Docker tắt) ─► models/phobert/{mtl,stl-sentiment,stl-topic}-s{42,43,44}/ (+ val_logits.npz, history.json)
evaluate --split val ─► results/metrics/val_experiments.json · tables/val_mtl_vs_stl.md · figures/val_curves.png
seed deploy (luật) ─► export_onnx --run best (cổng ≤1) ─► calibrate (scoring.py: T, CDF, τ_B, hiệu quả val, reliability val)
evaluate --freeze ─► results/freeze.json ─► evaluate --split test --final (1 lần) ─► outputs/test_preds/*.npz + .done
make report ─► significance · error_analysis (+ bias) · checklist (câu val) · calibrate --report test ─► results/{tables,figures,metrics}
build_artifacts v1.0 ─► artifacts/v1.0/ (model + cấu hình) ─► [M2] ─► [M3 duyệt] push_hub ─► HF <user>/edupulse-phobert-mtl@<sha>
```
Đầu ra commit vào `results/`: `tables/test_main.{md,csv}` (bảng PDR), `tables/test_per_class.csv`, `figures/cm_{task}_{model}.png`, `figures/reliability_{task}.png`, `tables/significance.md`, `tables/review_budget_{val,test}.md`, `tables/error_analysis.md` + `error_samples.csv` (≤30 câu), `tables/bias_cells.csv`, `tables/checklist.md`, `metrics/test_all.json`, `metrics/onnx_gate_v1.json`, `freeze.json`.

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/run-experiments.sh`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/evaluate.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/significance.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/error_analysis.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/checklist.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/push_hub.py`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_evaluate_guard.py` (từ chối khi thiếu freeze / đã chạy)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/tests/ml/test_significance.py` (McNemar/bootstrap trên mảng biết trước)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/freeze.json` và các file `results/` liệt kê ở trên (sinh, commit)

Modify:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/calibrate.py` (reliability, `--report test`)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/build_artifacts.py` (model card đầy đủ theo schema, stage release)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.env.example` (`EDUPULSE_HF_REPO`, `EDUPULSE_HF_REVISION` sau khi đẩy)

## Implementation Steps
1. (Should) Thăm dò: MTL seed 42, 6 epoch, giữ checkpoint epoch tốt nhất → E* = 4 nếu epoch tốt nhất ≤4, ngược lại 6.
2. `ml/run-experiments.sh`: MTL × 3, STL-sentiment × 3, STL-topic × 3 với E*; chạy đêm 17/10 (sau M1, Docker tắt).
3. `evaluate --split val` → mean ± std, MTL vs STL; kiểm quy tắc 1,5 điểm (vi phạm → dừng M6, hỏi người dùng).
4. Seed deploy → `export_onnx --run best` (cổng ≤1 điểm, golden predictions) → `calibrate` (T cross-fit, CDF, τ_B, hiệu quả val, reliability val). Baseline: calibrated `method="temperature"` trên val.
5. `python -m ml.evaluate --freeze` → `results/freeze.json`; commit trước khi chạy test (bằng chứng thứ tự).
6. `evaluate --split test --final` (1 lần, giữ khoá MPS cho phần torch FP32) → `outputs/test_preds/` + bảng/hình chính.
7. `make report` (significance, error analysis + bias, CheckList, ngân sách trên test).
8. Viết mục phân tích nếu PhoBERT không thắng baseline tốt nhất (PDR L53); ghi rõ val > test ~3–4 mF1 (r1 L21).
9. `make artifacts BUNDLE_VERSION=v1.0`; chạy lại `pytest -m model` + `engine.bench` với `artifacts/v1.0/` (không cần Docker).
10. **Dừng M2**: trình bảng test + MTL vs STL + cổng int8 cho người dùng.
11. **Dừng M3**: hỏi người dùng → `push_hub.py` → ghi `EDUPULSE_HF_REPO`/`EDUPULSE_HF_REVISION` vào `.env.example`.
12. Commit (`feat: add experiment runner and final evaluation`, `docs: add evaluation results`).

## Todo list
- [ ] (Should) Thăm dò 6 epoch → chốt E*
- [ ] `ml/run-experiments.sh` (kiểm Docker tắt) + chạy 8–9 run (khoá MPS, `caffeinate`)
- [ ] `evaluate --split val`: mean ± std, MTL vs STL, kiểm quy tắc 1,5 điểm
- [ ] Seed deploy, export int8 (cổng), calibrate cuối, baseline calibrated
- [ ] `evaluate --freeze` → `results/freeze.json` + commit
- [ ] `evaluate --split test --final` (đúng 1 lần)
- [ ] `make report`: significance · error analysis + bias · CheckList · ngân sách trên test
- [ ] `artifacts/v1.0/` (schema, guard) + chạy lại test model + bench
- [ ] Mốc M2 (kết quả test) + M3 (duyệt) đẩy HF Hub, ghim revision
- [ ] Commit kết quả

## Success Criteria
- 9 run PhoBERT có log + `val_logits.npz`; bảng val mean ± std cho MTL, STL-sentiment, STL-topic.
- `freeze.json` có commit trước `outputs/test_preds/.done`; test chạy đúng 1 lần (timestamp duy nhất trong `test_all.json`).
- `results/tables/test_main.md` đủ 2 task × (3 baseline + MTL + STL): acc, mF1, wF1, F1 từng lớp, ECE; CM chuẩn hoá hàng; McNemar + bootstrap cho ≥3 cặp; `make report` sinh đủ file liệt kê.
- Bản deploy int8: giảm mF1 val ≤1 điểm cả 2 task (`onnx_gate_v1.json`). ECE test kỳ vọng ≤0,04 (r1 L119) — không đạt thì báo trung thực, không là cổng.
- `artifacts/v1.0/` qua schema (`stage: "release"`, model card có ngân sách duyệt test gồm 10%, ghi chú x86 + demo); guard 0 câu gốc; `pytest -m model` pass; (sau M3) repo HF public có revision ghim.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Máy sleep/treo giữa đêm, mất run | TB × TB | `caffeinate -i`; hàng đợi idempotent chạy tiếp được |
| Docker/VM chạy cạnh train gây swap | TB × TB | Kiểm `docker info` trong script; quy tắc độc quyền ở plan |
| MTL kém STL >1,5 mF1 | Thấp × TB | Dừng M6, đưa số + lựa chọn cho người dùng (không tự đổi quyết định) |
| Lỡ chạm test trước khi đóng băng | Thấp × Cao | Guard trong `evaluate.py` + test; freeze commit trước |
| Bug phát hiện SAU khi chạy test | Thấp × Cao | Sửa, chạy lại có ghi lý do vào `freeze.json["reruns"]` và nêu trong báo cáo (không giấu) |
| Trễ mốc M2 (20–21/10) | TB × Cao | Cắt Should trước (thăm dò epoch, error/bias chi tiết); STL 3 seed giữ (tiêu chí PDR) |
| Không đẩy được HF (thiếu token/duyệt) | TB × TB | Bundle local cho phase 6–8; phase 9 dùng `scp artifacts/v1.0/` |

Rollback: `artifacts/v0-smoke/` vẫn còn; HF ghim theo revision (lùi = đổi revision); `freeze.json` không xoá, chỉ thêm lần chạy lại kèm lý do.

## Security Considerations
- Token HF chỉ qua `hf auth login` do người dùng chạy (gợi ý `! hf auth login`) hoặc env tạm; không ghi vào repo/log; không có target `make` đẩy HF.
- Repo HF + bundle không chứa câu gốc UIT-VSFC (guard trước khi đẩy).
- Model card ghi rõ: AI gợi ý, cần người duyệt; không dùng để đánh giá/kỷ luật cá nhân; giới hạn dữ liệu (1 trường, 2013–2016, nhãn chủ đề nhiễu); tên người che best effort.
- `error_samples.csv` chỉ chứa câu dataset (tên đã là `wzjwz`), ≤30 câu để trích trong báo cáo.

## Next steps
- Phase 6/7 tự nhận số liệu thật qua `/api/model-card` (v1) và demo tính lúc build (không sửa code). Phase 8 đo hiệu năng cuối với `artifacts/v1.0/`. Phase 9 gom `results/` thành chỉ mục báo cáo.
