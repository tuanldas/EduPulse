---
title: "EduPulse — triển khai end-to-end (ML + FastAPI + Next.js + Docker)"
description: "Phân loại cảm xúc/chủ đề góp ý SV tiếng Việt bằng PhoBERT đa nhiệm + XAI, cờ duyệt, insight, ẩn PII; 1 container cho VPS."
status: pending
priority: P1
effort: 128h
branch: main
tags: [ml, nlp, phobert, onnx, fastapi, nextjs, docker, xai]
created: 2026-10-15
---

# EduPulse — kế hoạch triển khai

## Overview
Đồ án cuối kỳ môn AI, 2026-10-14 → ~2026-10-28. UIT-VSFC → baseline + PhoBERT-base-v2 đa nhiệm → ONNX int8 → FastAPI phục vụ API + Next.js export tĩnh trong 1 image Docker (VPS 2 vCPU/2GB, 24/7). Nguồn chuẩn: [PDR](../../docs/project-overview-pdr.md), [design](../../docs/design-guidelines.md), [wireframes](../../docs/wireframes/index.html), research ở [plans/reports](../reports/) và [research/](./research/). Quyết định đã chốt + Validation Session 1–2 ghi trong từng phase, không bàn lại. Repo/bundle/HF không chứa câu gốc UIT-VSFC: file mẫu, câu mẫu, demo tính lúc build image.
Năng lực: 1 người + agent AI, làm cả cuối tuần ~7h/ngày ≈ 98h lịch đến 28/10. Ước lượng lại **128h** (126,5h + 1,5h retrain đầy đủ ở Session 3; +37h so với 91h: ~20h phạm vi mới từ validation, ~15h kéo web/API về kịch bản "Thường" của r6 — F13) > 98h ⇒ chỉ kịp khi ≥2 luồng agent song song (16–17/10: P3 hai luồng ‖ P4 phần A ‖ EDA; 18–24/10: P5 nền ‖ P6 ‖ P7 hai agent). Đường găng ≈ 67h (gồm ~15% chờ duyệt mốc) ⇒ đệm ~30h trên giấy, nhưng web/API trượt thêm 30–50% (kịch bản "Xấu" của r6) là hết đệm ⇒ **rủi ro tiến độ: CAO**.

## Goals
| # | Mục tiêu (đo được) | Priority |
|---|---|---|
| 1 | Tái lập bằng `make`: dữ liệu → EDA → train → đánh giá → báo cáo → export, seed cố định | P1 |
| 2 | Bảng test 2 task × (3 baseline, PhoBERT MTL, STL; 3 seed mean ± std) + McNemar/bootstrap | P1 |
| 3 | 4 tính năng A–D chạy trong 4 trang tiếng Việt đúng wireframe đã duyệt | P1 |
| 4 | int8 giảm mF1 val ≤1 điểm; 1 câu+XAI <2 s; 1.000 câu <2 phút; container ≤1,5GB RAM | P1 |
| 5 | 1 image + `docker-compose.yml` build/chạy trên VPS Linux bất kỳ, có tài liệu deploy | P1 |

## Phases
| # | Phase | Ngày (10/2026) | Effort | Phụ thuộc | Status |
|---|---|---|---|---|---|
| 1 | [Thiết lập repo & tooling](./phase-01-start.md) | 15 | 4h | — | Completed |
| 2 | [Dữ liệu, chuẩn hoá dùng chung, EDA](./phase-02-data-normalize-eda.md) | 15–16 | 10.5h | 1 | Completed |
| 3 | [Baseline, PhoBERT run đầu, ONNX int8, bundle v0](./phase-03-baselines-first-phobert-onnx.md) | 16–17 | 13.5h | 2 (bước 1–8) | Completed |
| 4 | [API skeleton, Docker, smoke test giới hạn tài nguyên](./phase-04-api-skeleton-docker-smoke-test.md) | 16–17 (≤ sáng 18) | 9.5h | 1 (phần A); 3 (phần B) | Completed |
| 5 | [PhoBERT đầy đủ, đánh giá cuối, calibration, bundle v1](./phase-05-phobert-experiments-eval-calibration.md) | 17–21 | 12h (+3h MPS nền) | 3, 4 (sau M1) | In progress (chờ M2, M3) |
| 6 | [Backend đủ tính năng](./phase-06-backend-features.md) | 18–22 | 24h | 4 | In progress (chờ build Docker, M4) |
| 7 | [Frontend 4 trang](./phase-07-frontend-four-pages.md) | 18–24 | 32.5h | 4, 6 (bước 1) | In progress |
| 8 | [Kiểm thử, review, hiệu năng](./phase-08-testing-review-performance.md) | 23–26 | 12h | 5, 6, 7 | Pending |
| 9 | [Deploy VPS, docs, gói artifact báo cáo](./phase-09-vps-deploy-docs-report-artifacts.md) | 25–27 (+28 dự phòng) | 10h | 8 | Pending |

## Dependencies
- Đường găng: 1 → 2 (bước [smoke]) → 3 (luồng A) → 4 phần B (M1 smoke: mục tiêu 17/10, muộn nhất sáng 18/10) → 6 (hợp đồng, rồi endpoint batch/insight) → 7 (trang 2–3) → 8 → 9.
- Song song: EDA của 2 ‖ 3 · 3 chia 2 luồng · 4 phần A ‖ 3 · 5 (train nền) ‖ 6 ‖ 7 · docs của 9 ‖ 8. Trong 7: sau nền tảng, agent A làm trang 1 + 4, agent B làm trang 2 + 3. Bước đánh dấu [smoke] ở phase 1–4 làm trước mọi việc khác.
- 7 cần hợp đồng OpenAPI (6 bước 1, sáng 18/10) và `make demo-data` (6 bước 2). Trang 4 + dashboard chạy với bundle v0, tự đúng khi 5 ra v1.
- Tài nguyên độc quyền: MPS chỉ 1 job (khoá `outputs/.mps.lock`): run đầu (3) → 9 run (5) → retrain PhoBERT thử (6, sau 5). Train MPS ⇔ Docker Desktop tắt; đo hiệu năng/Docker ⇔ không train (`smoke_test.py` từ chối khi khoá đang giữ).
- Sở hữu file: `pyproject.toml`/`uv.lock`/`Makefile` thuộc 1 (sửa sau phải báo lead); `scoring.py`, `bundle_schema.py`, golden 20 câu thuộc 3; `text/ngrams.py` thuộc 2; `Dockerfile` do 4 tạo, 6 thêm stage `demo`/`final`, 9 hoàn thiện (tuần tự); `apps/web/openapi.json` do 6 sinh; phase song song chỉ sửa file trong "Related code files" của mình.
- Quy trình duyệt (Validation S1 #1): agent chạy liên tục, commit conventional không chờ duyệt từng commit; chỉ DỪNG ở mốc: M1 smoke go/no-go (17/10) · M2 kết quả test + xong khối ML (20–21/10) · M3 trước khi đẩy HF Hub (21/10) và GitHub (27/10) · M4 xong khối API (22/10) · M5 xong khối UI (24/10, xem ảnh 4 trang) · M6 khi cần cắt phạm vi hoặc chạm quyết định đã chốt.
- Môi trường (Session 4): repo nằm trong thư mục Syncthing và team code chạy trên nhiều git worktree ⇒ file nặng/tái tạo được (data, models, outputs, artifacts, demo-data, khoá MPS) đặt ở `EDUPULSE_HOME=$HOME/.cache/edupulse` (khai trong `.env`, dùng chung giữa các worktree). Không đặt biến thì mặc định là thư mục repo (xem phase 1).
- Tương thích: dự án mới, không dữ liệu cũ; hợp đồng giữa các phần = `data/processed/MANIFEST.json` + `artifacts/<version>/manifest.json` (`normalizer_version`, nhãn, sha256, schema `bundle_schema`) + snapshot OpenAPI.

## Ưu tiên & cắt giảm (MoSCoW)
- **Must:** `make` tái lập (gồm `report`); chuẩn hoá + PII regex + luật tên "xưng hô + Viết Hoa" dùng chung; 3 baseline; PhoBERT MTL 1 run (seed 42, đã train, không train thêm — S6); test 1 lần + McNemar/bootstrap/CI 95%/ECE; int8 + cổng; T-scaling + ngân sách duyệt; XAI occlusion; insight Wilson + từ khoá + khuyến nghị; dữ liệu demo tính lúc build; heartbeat/close batch; 4 trang lõi; export CSV an toàn; Docker/compose/deployment.md; smoke + perf + kiểm nhãn amd64 (QEMU); pytest + E2E vàng; model card, disclaimer, CheckList.
- **Must (bảo vệ — chỉ cắt khi người dùng quyết ở M6):** so sánh PhoBERT↔LinearSVC (mặc định tắt) · script retrain · nhập XLSX · heatmap nhóm × chủ đề.
- **Should** (~4,5h; cắt khi trễ mốc, theo thứ tự): thăm dò 6 epoch → error/bias chi tiết → Vitest đầy đủ → lọc câu tiêu biểu theo từ khoá → NER underthesea.
- **Could** (cắt đầu tiên, phần lớn chưa tính vào effort): sparkline hiệu quả · polish mobile 390 · axe a11y · đo faithfulness XAI · Caddy HTTPS · CI/GHCR.
- **Won't:** đăng nhập/DB/lưu dữ liệu · lưu nhãn sửa phía server · LLM · khía cạnh cụm từ · SSE · TanStack · đa worker.
- Mốc lịch: M1 17/10 · M2 20–21/10 · M4 22/10 · M5 24/10 → đóng băng tính năng · E2E + perf cuối 26/10 · deploy + docs 27/10 · 28/10 dự phòng. Trễ mốc ≥1 ngày → cắt Could rồi Should theo thứ tự trên; vẫn trễ → dừng M6.

## Success Criteria
- [ ] PDR 1: clone sạch → `make setup data eda baseline train export calibrate freeze` → commit `results/freeze.json` → `make evaluate report hitl artifacts` (+ `make demo-data`) chạy lại được. (Session 6: `experiments` ngoài phạm vi.)
- [x] PDR 2–3: `results/tables/test_main.md` đủ ô, khoảng tin cậy 95% bootstrap (thay mean ± std, Session 6), kiểm định ghép cặp (`significance.md`); PhoBERT thắng baseline tốt nhất nên không cần mục phân tích thua.
- [ ] PDR 4: `results/metrics/onnx_gate_v1.json` giảm mF1 val ≤1 điểm cả 2 task (M1); QEMU amd64: golden 20/20 và ≥98% của 200 câu val cùng nhãn; model card ghi "chưa đo x86" tới khi đo trên VPS.
- [ ] PDR 5: trên compose cuối (2 CPU/1,5GB, M1, hệ số an toàn ×3): 1 câu+XAI p95 ≤0,6 s; 1.000 câu gồm auto-XAI ≤30 s; RAM ≤1,2GB; upload 5MB chạy xong; không OOM.
- [ ] PDR 6: pytest + Vitest + Playwright pass trên image cuối; `docker compose up` từ clone sạch chạy; link public khi người dùng dựng VPS.
- [ ] PDR 7: trang 4 đủ model card, disclaimer, bảng từng lớp, CheckList, quyền riêng tư (tên: best effort + disclaimer); nhóm của file mẫu mang nhãn "dữ liệu minh hoạ".
- [ ] Dữ liệu: repo, bundle, HF không chứa câu gốc UIT-VSFC (guard `build_artifacts`; chỉ ≤30 câu ví dụ trong `results/`); HF lỗi lúc build → demo ẩn kèm thông báo, build không fail.

## Nhật ký mốc
### M1 — 2026-10-15 (sớm 2 ngày so với kế hoạch): ĐẠT CÓ ĐIỀU KIỆN
- Bundle `v0-smoke`: PhoBERT MTL seed 42, int8 `ffn_out_early6_fp32+heads_fp32` (333MB), chọn qua 2 vòng tìm kiếm đăng ký trước (`results/metrics/onnx_gate_search.json`). Val (batch=1): cảm xúc 94,8 / 85,3, chủ đề 89,8 / 81,0 (acc / mF1); LinearSVC mF1 78,1 / 76,4. Phục vụ từng câu ở mọi đường (int8 động phụ thuộc lô).
- Smoke chính thức (`results/perf/smoke-2026-10-15-arm64.md`, container 2 CPU/1,5GB): RAM đỉnh 618MB, khởi động 2,3 s, 1.000 câu 27,3 s, API/log sạch: ĐẠT. 1 câu + giải thích p95 1,09 s > 0,6 s: TRƯỢT. Nguyên nhân: phần FP32 chạy chậm ~2,9× trong VM Docker trên Mac (trên máy thật ~1,3×), chưa có số x86.
- Người dùng chọn phương án A: nhãn hiện ngay, giải thích tải sau; `XAI_MAX_WORDS` mặc định 25 (phủ ~90% câu val, chỉnh bằng env), chốt sau khi đo trên VPS (phase 9). Tiêu chí PDR 5 tách: nhãn p95 ≤0,5 s, giải thích p95 ≤2 s. Giữ nguyên cấu hình lượng tử hoá (đạt cổng chất lượng).
- Sửa lỗi phát hiện ở M1: token id ngoài bảng embedding (≥64001) → `<unk>`; giải thích chạy lô 1 để nhất quán với nhãn.

### Thay đổi phạm vi — Session 6 (2026-10-17): bỏ train thêm
- Người dùng duyệt bỏ 8 run MPS còn lại của phase 5 (không tắt Docker, không chạy đêm). Thay bằng: bootstrap CI 95% trên test thay cho mean ± std 3 seed; so sánh MTL/STL dùng thí nghiệm sơ bộ của research (val, 1 seed, p=0,31/0,46) và ghi rõ là sơ bộ; vòng người duyệt (HITL) chạy trọn với LinearSVC trên CPU + smoke retrain PhoBERT 20 bước trên CPU. Bản deploy = MTL seed 42 (`v0-smoke` → đánh giá test 1 lần → `v1.0`).
- Bối cảnh: mạng mất 2 lần (tối 15/10) làm toàn bộ agent dừng tới 19/10; mất ~3,5 ngày lịch. Docker Desktop của người dùng đang tắt (không do team).

### M2 — 2026-10-19 (sớm 1–2 ngày): ĐẠT (người dùng duyệt)
- Test (n = 3.166) chấm đúng 1 lần: freeze commit `bac46b8` có trước dự đoán test, `freeze.reruns` rỗng. Bảng: `results/tables/test_main.md`, `significance.md`, `review_budget_test.md`, `checklist.md`, `error_analysis.md`, `hitl_retrain.md`.
- Bản triển khai (int8, batch=1): cảm xúc 94,0 / 82,1 [79,3–84,6], chủ đề 90,0 / 81,3 [79,0–83,4] (acc / mF1 [KTC 95%]); ECE 0,014 / 0,023. Hơn LinearSVC mF1 +8,6 / +8,0 điểm (McNemar p < 0,0001); không khác FP32 (p = 0,58 / 0,35). Khớp paper (≈ 94/83, 89/80).
- Ngân sách duyệt 10% trên test bắt 45,3% lỗi cảm xúc và 34,4% lỗi chủ đề; phần không cờ đúng cả hai 89,8% (không gắn cờ: 85,5%). Lớp yếu: trung tính (recall 43,1%), Khác (48,4%).
- Giới hạn ghi trong thẻ mô hình: luật ẩn tên che nhầm câu Viết Hoa Chữ Đầu (CheckList 86,6%); thêm "không" 83,2%. HITL mô phỏng (80 câu cờ) không cải thiện có ý nghĩa.
- Bundle `v1.0` (stage release) ở `$EDUPULSE_HOME/artifacts/v1.0`, mặc định từ 2026-10-19; golden 20/20, guard câu gốc 0/200.

### M4 — 2026-10-19 (sớm 3 ngày): ĐẠT (người dùng duyệt; build Docker làm tiếp)
- 17 operation (`apps/web/openapi.json`): export CSV (BOM, chống formula injection, `all`/`corrected`), so sánh LinearSVC nạp lười, gzip, `model_url` (env `EDUPULSE_MODEL_URL`). Host Mac: 1.000 câu qua HTTP kèm tự giải thích 20 câu 10,4 s; `/api/analyze` giữa batch ≤27 ms.
- Lỗi tích hợp sửa khi gộp: file "nhãn đã sửa" của API (`label_sentiment`/`label_topic`) không khớp cột `ml/retrain.py` đọc (`label_s`/`label_t`); retrain giờ đọc đúng `CORRECTED_HEADER` và có test đọc ngược từ API thật. Trùng tên module test `test_stats.py` → đổi `tests/ml/test_bootstrap_stats.py`.
- Còn lại: build stage `demo`/`final` + smoke trong container (cần Docker); `scripts/smoke_test.py` còn cổng cũ `single_p95_s=0.6`, cần tách nhãn ≤0,5 s / giải thích ≤2 s.

### Quyết định sau M2/M4 — 2026-10-19
- M3 Hugging Face: team viết `ml/push_hub.py` và chạy thử không đẩy; chỉ đẩy repo public sau khi người dùng tự đăng nhập (`uv run hf auth login`) và báo.
- Docker: agent được tự bật Docker Desktop (`open -a Docker`) khi cần build, smoke, kiểm amd64, và tắt lại khi xong (không còn train nên không tranh tài nguyên).
- Đo RAM: `import skops.io` kéo theo torch và pandas khi chúng có trong môi trường (dev: mô hình cơ sở +~500MB RSS); image chỉ có group `app` nên số chính thức đo trong container, có bật so sánh.
- Bỏ Recharts (không trang nào dùng): cột chồng, heatmap bằng CSS; biểu đồ độ tin cậy bằng SVG.

### Review bảo mật và quyết định — 2026-10-19 chiều
- Báo cáo: `plans/reports/code-reviewer-261019-1430-security-review.md` (2 High, 3 Medium, 7 Low; log, CORS, CSV injection, zip bomb, XXE, static, non-root, XSS đạt).
- Sửa trước khi deploy (dev-api): H1 XLSX ô xa (quét toạ độ ô trước calamine, trần giải nén 30MB), H2 trần thân 512KB cho route ngoài upload, M1 semaphore parse + kiểm sức chứa trước khi đọc, L2/L4–L7; L3 hardening đi cùng compose cuối.
- Người dùng chọn sửa luật ẩn tên (thầy/cô giáo, hiệu trưởng, MSSV chữ+số, SĐT bàn) → `NORMALIZER_VERSION` 2, bundle v1.1 cùng mô hình. Bằng chứng: 0/16.175 câu UIT-VSFC có chữ hoa hay khớp mẫu mới ⇒ đầu vào mô hình không đổi, số test M2 giữ nguyên (dev-ml ghi `results/metrics/normalizer_v2_invariance.json`). HF sẽ đẩy v1.1.
- Người dùng chọn gỡ câu gốc UIT-VSFC khỏi repo: `error_samples.csv` chỉ giữ mã câu + nhãn (bản đầy đủ ở `$EDUPULSE_HOME/outputs/`), wireframe dùng câu tự viết; lịch sử commit làm sạch lúc đẩy GitHub (cách làm chốt ở M3).
- `EDUPULSE_NER` mặc định tắt ở production (RAM, tải mô hình lúc chạy). Không giả định có proxy: giới hạn thân/upload nằm trong app.

### M5 — 2026-10-19 (sớm 5 ngày): ĐẠT (người dùng duyệt)
- Review ảnh 4 trang (1440/390 × sáng/tối, 20 ảnh): https://claude.ai/artifact/Lhct9n8hHA8X92vpqQf19x. Playwright 94/94 trên bản export do FastAPI phục vụ, axe 0 vi phạm, không cuộn ngang từ 320px. Đóng băng tính năng.
- Sửa nhỏ không cần duyệt lại: bảng so sánh trang 4 ghi "± 0,0" cho hàng 1 seed; khối quyền riêng tư theo luật ẩn tên v1.1.
- Smoke chính thức trong Docker (`results/perf/smoke-2026-10-19-arm64.md`): nhãn p95 0,046 s, giải thích p95 0,655 s, RAM không thu hồi ~650MB (kể cả khi bật so sánh; page cache file mô hình ~620MB tính riêng), upload 5MB, khởi động 2,5 s; amd64 qua Rosetta: golden 20/20, 197/200 câu cùng nhãn (`results/perf/amd64-check-2026-10-19.md`).
- Cổng "1.000 câu ≤ 30 s trên Mac" trượt 30,8 s do máy (A/B: image ngày 15/10 hôm nay cũng 32–34 s). Người dùng chọn **đạt có điều kiện**: thước đo quyết định là tiêu chí PDR (dưới 2 phút trên VPS), đo lại ở bước nghiệm thu VPS.

### Gián đoạn — 2026-10-19 17:30: agent hết hạn mức tuần
- dev-ml và dev-api (model Sonnet) dừng vì hạn mức tuần của tài khoản, mở lại 23:00 ngày 21/10; cây làm việc của cả hai sạch, phần đã commit đã gộp vào main (bundle v1.1, chuẩn hoá v2, sửa H1/H2/M1/L2). Lead tự làm việc nhỏ trong lúc chờ; hẹn khôi phục agent lúc 23:17 ngày 21/10.
- Lead đã làm (2026-10-20 sáng): gộp v1.1 + H1/H2/M1/L2; tự sửa L5 (Location tương đối), L6 (nhịp tim theo TTL), L7 (đọc kết quả batch một lần); sửa "± 0,0" và khối quyền riêng tư trang 4; dựng lại demo-data cho v1.1; quét câu gốc trên repo = 0; pytest 2.102 pass; `push_hub --dry-run` trên v1.1 đạt (10 tệp, 340MB, 0 câu gốc).
- Còn lại cho agent: L4 CSP + ghim Swagger (kèm chạy lại E2E), nới `test_batch_performance`, L1 tuỳ chọn; compose cuối + hardening L3 + đo smoke cuối trong Docker; docs/deployment.md, README cuối, results/README.md + model_card.md; tái lập trên clone sạch; M3 (HF sau khi người dùng đăng nhập; GitHub với lịch sử sạch).

## Câu hỏi chưa giải quyết
1. VPS x86_64 hay arm64, có domain cho HTTPS không? (quyết định Caddy, cách build, mức tin của kiểm QEMU).
2. ~~Chấp nhận mốc smoke 17/10?~~ Đã chốt: chấp nhận (Session 3); smoke chạy 15/10.
3. ~~Retrain PhoBERT đầy đủ hay chỉ smoke?~~ Đã chốt Session 6: chỉ smoke 20 bước CPU; HITL đầy đủ chạy bằng LinearSVC.
4. `docs/wireframes/*.html` có vài câu ví dụ có thể trích UIT-VSFC: giữ như "vài câu ví dụ" (mặc định) hay thay bằng câu tự viết?

## Validation Log

### Session 1 — 2026-10-15
**Trigger:** phỏng vấn xác nhận sau khi tạo plan (`/ak:plan --hard`, validation mode=prompt). **Questions asked:** 4.

1. **[Process]** Khi triển khai, bạn muốn duyệt thế nào? — Dừng ở mốc (Khuyến nghị) | Duyệt sau từng phase → **Dừng ở mốc**. Lý do: cook chạy liên tục, chỉ dừng ở smoke go/no-go, kết quả test, trước khi đẩy HF/GitHub, xong khối ML/API/UI, hoặc khi cần cắt phạm vi.
2. **[Scope]** Nếu trễ, mục Should nào giữ lại? — So sánh mô hình | Script retrain | Nhập XLSX | Heatmap nhóm × chủ đề → **giữ cả 4**. Lý do: 4 mục thành "Must (bảo vệ)"; Should còn cắt được: lọc câu tiêu biểu theo từ khoá, NER underthesea, error/bias chi tiết, thăm dò 6 epoch, Vitest đầy đủ.
3. **[Data]** Câu gốc UIT-VSFC với repo public? — Tính sẵn lúc build (Khuyến nghị) | Commit vào repo | Tải lúc khởi động → **Tính sẵn lúc build**. Lý do: repo không chứa câu gốc; build image tải dataset (revision `refs/convert/parquet`) rồi tính sẵn dự đoán demo; build cần mạng; vài câu ví dụ trong artifact báo cáo vẫn giữ.
4. **[Product]** File mẫu có cột "môn học"? — Không gán giả (Khuyến nghị) | Gán ngẫu nhiên, ghi minh hoạ → **Gán ngẫu nhiên, ghi minh hoạ**. Lý do: cột nhóm gán ngẫu nhiên (seed cố định), UI và model card ghi rõ "dữ liệu minh hoạ" ở heatmap/cảnh báo theo nhóm.

**Mặc định chốt (không hỏi):** repo GitHub `edupulse` + HF `edupulse-phobert-mtl` dưới tài khoản người dùng, token HF do người dùng cấp lúc đẩy; model card = trang 4 + README HF + `results/model_card.md`; image build ngay trên VPS (base image đa kiến trúc), không hỏi lại về cloud/thẻ; cập nhật PDR mục 5, 7 theo kiến trúc mới.

**Action items:**
- [x] MoSCoW: 4 mục bảo vệ → Must; thứ tự cắt mới; đánh giá lại rủi ro tiến độ.
- [x] Dữ liệu demo tính lúc build image; `.gitignore`/`.dockerignore` không để lọt câu gốc vào repo.
- [x] File mẫu có cột nhóm gán ngẫu nhiên (seed cố định) + nhãn "minh hoạ" trên UI.
- [x] Ghi cách duyệt "dừng ở mốc" cho bước cook.
- [x] Cập nhật PDR mục 5, 7.

### Session 2 — 2026-10-15
**Trigger:** kết quả kiểm chứng độc lập ([reports/validation-verification.md](./reports/validation-verification.md)): 317 claim, Verified 291 · Failed 17 (High 5 · Medium 8 · Low 4) · Unverified 9. **Questions asked:** 3.

1. **[Capacity]** Có chạy phiên làm việc + duyệt mốc cuối tuần 17–18, 24–25/10? — Cả 4 ngày (Khuyến nghị) | Chỉ 17–18/10 | Không → **Cả 4 ngày** (~98h ≥ ước lượng).
2. **[Flow]** Duyệt/sửa nhãn quá 30'? — Giữ khi tab mở (Khuyến nghị) | TTL 120' | Xuất CSV trên trình duyệt → **Giữ khi tab mở**: heartbeat 60 s khi tab mở, `sendBeacon` DELETE khi đóng, TTL 30' chỉ là lưới an toàn.
3. **[Product]** Tên cột nhóm minh hoạ? — Nhóm A–E (Khuyến nghị) | Tên môn giả → **Nhóm A–E**.

**Lead tự chốt (kỹ thuật, không hỏi):** F1 dùng `0000.parquet` (hoặc liệt kê tree rồi assert); F3 middleware chặn theo `Content-Length` + đếm byte stream; F4 tắt access log mặc định, log theo route template, `group` không đi qua query string; F9 `source_url` qua `/api/config` (env runtime); F12 thêm luật rẻ "thầy/cô + Viết hoa" vào Must, tiêu chí tên = best effort + disclaimer; F15 ngưỡng 1.000 câu gồm auto-XAI ≤30 s trên M1; build image trên VPS (swap ≥2GB, cần mạng HF), HF lỗi lúc build → ẩn demo kèm thông báo chứ không fail; kiểm x86 = QEMU kiểm nhãn trước, đo thật bằng `smoke_test.py --url` trên VPS của người dùng, model card ghi "chưa đo x86" tới khi đo; các F còn lại sửa theo đề xuất trong report.

**Action items:** áp Session 1–2 + F1–F17 vào phase; ước lượng lại effort; Whole-Plan Consistency Sweep; `ak plan validate` exit 0.

### Verification Results
- Nguồn: [reports/validation-verification.md](./reports/validation-verification.md) · Tier: Full (rút gọn cho greenfield) · Claims: 317 · Verified 291 · Failed 17 (High 5 · Medium 8 · Low 4) · Unverified 9. Cả 17 Failed đã xử lý trong phase (2026-10-15):

| F | Mức | Xử lý (phase) |
|---|---|---|
| F1 | High | `default/{split}/0000.parquet` + assert theo tree, ghi kích thước (P2) |
| F2 | High | Session 1 lan vào phase: Must (bảo vệ) (plan, P6, P7); demo build-time + bundle chỉ model/cấu hình (P3, P5, P6, P7, P9); nhóm minh hoạ + `illustrative_groups` (P6, P7); dừng ở mốc M1–M6 (plan, P1, P4–P7, P9) |
| F3 | High | middleware `upload_limit.py` (Content-Length + đếm byte), file tạm trong tmpfs `/tmp`, test chunked 50MB (P6, P9) |
| F4 | High | `--no-access-log` + `access_log.py` theo route template; insights nhận nhóm qua POST body; E2E chuỗi đánh dấu ở văn bản + cột nhóm (P4, P6, P8) |
| F5 | High | heartbeat 60 s + `POST …/close` qua sendBeacon + TTL 30' lưới an toàn; E2E với TTL ngắn (P6, P7, P8) |
| F6 | Medium | Makefile `report`, `demo-data`, `artifacts --version $(BUNDLE_VERSION)`, `-include .env` (P1, P5) |
| F7 | Medium | `bundle_schema.py` (Manifest có `bundle_version`/`stage`; ModelCard có ngân sách duyệt trên test gồm 10%) dùng chung ml + API (P3, P5, P6) |
| F8 | Medium | `artifacts/<version>/` + `EDUPULSE_ARTIFACTS_DIR`; golden 20 câu tự viết + `golden_predictions.json` (P3, P4, P9) |
| F9 | Medium | `source_url` qua `/api/config` từ `EDUPULSE_SOURCE_URL` runtime (P1, P6, P7, P9) |
| F10 | Medium | `data/processed/MANIFEST.json{normalizer_version}`, assert ở train/export/build/API (P2, P3) |
| F11 | Medium | `edupulse/scoring.py` + `text/ngrams.py` dùng chung + test parity (P2, P3, P4, P6) |
| F12 | Medium | luật "xưng hô + Viết Hoa" thành Must; lọc câu nghi còn tên; tiêu chí tên = best effort + disclaimer (P2, P6, P7) |
| F13 | Medium | ước lượng lại 91h → 126,5h; làm cuối tuần (S2 #1); rủi ro CAO, cần ≥2 luồng agent (plan, P6, P7) |
| F14 | Low | Docker tắt khi train; `run-experiments.sh` kiểm `docker info`; `smoke_test.py` từ chối khi khoá MPS đang giữ (P3, P4, P5) |
| F15 | Low | 1.000 câu gồm auto-XAI ≤30 s trên M1 (×3 = 90 s, còn biên 30 s) (plan, P4, P6, P8) |
| F16 | Low | chạy lại smoke + E2E vàng trên compose cuối (P9) |
| F17 | Low | BibTeX → `04-mo-hinh-dao-duc.html:563–580`; mục trạng thái 01 L550 / 02 L549 / 03 L543; P4 `depends_on: [1, 3]`; VnCoreNLP @ `62bbc58f…` + `LICENSE.md` (P1, P2, P4, P7) |
- Unverified: #1–2 (tốc độ/cổng int8 trên x86) → QEMU so nhãn (P8) + đo VPS bằng `smoke_test.py --url` (P9), model card ghi "chưa đo x86"; #3 `dynamo=False` giữ [UNVERIFIED] trong rủi ro P3; #4 shadcn/`cn` kiểm khi `init` (P7); #5 NER dưới read-only kiểm ở P9 bước 2; #6 skops + temperature có test roundtrip (P3); #7 zip header giả → giải nén có trần thật + test (P6); #8 giấy phép UIT-VSFC → không phát hành câu gốc (S1 #3); #9 build trên VPS 2GB → swap ≥2GB, đo thời gian build trên Mac, `EDUPULSE_BUILD_DEMO=0` (P6, P9).

### Whole-Plan Consistency Sweep
- Files reread: `plan.md` + `phase-01` … `phase-09` (10 file) sau khi sửa; script kiểm 12 mục đúng thứ tự, checkbox chỉ ở Todo list, marker `Updated: Validation Session 1–2` có ở cả 9 phase; bảng Phases khớp frontmatter (effort, ngày, phụ thuộc; tổng 126,5h).
- Decision deltas checked: S1 #1–4 + "Mặc định chốt", S2 #1–3, 10 mục "Lead tự chốt", F1–F17, Unverified 1–9.
- Reconciled stale references (grep 30 mẫu trên 10 file, còn 0 chỗ lỗi thời): `0.parquet` → `0000.parquet`; `sample.csv`/`demo_results` trong bundle/manifest → `demo-data/` build-time; "Should" cho 4 mục bảo vệ → "Must (bảo vệ)"; "(người dùng duyệt)" từng commit → mốc M1–M6; "≤30 s + auto-XAI ≤10 s"/"≤40 s" → "≤30 s gồm auto-XAI"; `?group=` → POST body; `NEXT_PUBLIC_SOURCE_URL` → `EDUPULSE_SOURCE_URL`; `engine/calibration.py` → `scoring.py`; "20 câu val khớp `demo_results.json`" → golden 20 câu tự viết; `artifacts/` phẳng → `artifacts/<version>/`; `depends_on: [1, 2, 3]` → `[1, 3]`; "access log chỉ có path" và "không ghi file upload ra đĩa" (sai) → đã sửa; mốc smoke 16/10 → 17/10. Lần nhắc tên cũ còn lại đều có chủ đích: marker mô tả thay đổi, ghi chú "`0.parquet` là 404" (P2), nguyên văn Session 2.
- Chỉnh kỹ thuật theo thực tế: S2 #2 ghi "`sendBeacon` DELETE" nhưng `navigator.sendBeacon` chỉ gửi POST ⇒ thêm `POST /api/batches/{id}/close` (xử lý như DELETE); DELETE giữ cho gọi trực tiếp.
- Unresolved contradictions: ~~1~~ → **0** sau Session 3 — yêu cầu gốc "smoke test ngày 2–3" mâu thuẫn với phạm vi đã duyệt (đường găng tới smoke ≈ 19h công); người dùng chấp nhận mốc M1 17/10, muộn nhất sáng 18/10 (vẫn trước mọi việc UI).

### Session 3 — 2026-10-15
**Trigger:** chốt plan cuối sau consistency sweep. **Questions asked:** 3.
1. **[Schedule]** Chấp nhận M1 smoke 17/10 (muộn nhất sáng 18/10)? — Chấp nhận 17/10 (Khuyến nghị) | Đo sơ bộ 16/10 bằng PhoBERT chưa fine-tune → **Chấp nhận 17/10**.
2. **[Scope]** Mức chứng minh script retrain cho báo cáo? — 1 lần retrain đầy đủ (Khuyến nghị) | Chỉ chứng minh script chạy → **1 lần retrain đầy đủ** mô phỏng người duyệt (phase 6 bước 10b, +~1–2h, không dùng test).
3. **[Process]** Bắt đầu triển khai thế nào? — Phiên mới + /ak:cook (Khuyến nghị) | Bắt đầu ngay trong phiên này | Cần chỉnh → **Custom:** "/ak:team cook" (triển khai bằng Agent Team chạy song song).

**Mặc định giữ cho câu hỏi mở còn lại:** VPS kiến trúc/domain do người dùng tự xử lý sau (build trên VPS, Caddy tuỳ chọn trong docs); vài câu ví dụ trong `docs/wireframes/` giữ như trích minh hoạ.

<!-- slug: edupulse-implementation -->
