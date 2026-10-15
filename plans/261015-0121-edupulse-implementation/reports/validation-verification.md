# Kiểm chứng plan EduPulse (v1-plan-verifier, 2026-10-15)
Lưu ý: `plan.md` đổi lúc 09:09 (thêm Validation Log Session 1); các `phase-*.md` không đổi từ 08:59. Báo cáo kiểm bản hiện hành (stat lại 09:27, không đổi).
Chỉ đọc plan/docs; nguồn ngoài (PyPI, npm, HF API, GitHub raw) chỉ tra cứu, không cài gì. Ký hiệu `phase-NN:dòng` = file phase tương ứng.

### Verification Results
- Tier: Full (rút gọn cho greenfield)
- Claims checked: 317 (179 trích dẫn dòng · 53 pin/tag · 11 SHA/URL/tên file · 8 lock/wheel/closure · 6 số học/ngày · 14 quyết định · 46 hợp đồng/luồng/phạm vi)
- Verified: 291 | Failed: 17 (High 5 · Medium 8 · Low 4) | Unverified: 9
- Xác minh nổi bật:
  - 179/179 trích dẫn đúng dòng (lệch ngữ nghĩa: F17; r1 L75 plan chủ ý đổi "chuyển STL" thành "dừng, hỏi người dùng").
  - Mọi pin tồn tại (29 PyPI, 21 npm). Wheel cp312 có cho macOS arm64 + Linux x86_64/aarch64 (riêng `py-vncorenlp` là sdist thuần Python). `transformers 5.19.0` ↔ `tokenizers 0.23.3` ↔ `huggingface-hub 2.2.0` giải được chung.
  - Closure 52 gói của group `app`: 0 torch/pandas (xấp xỉ: bỏ qua marker, bản mới nhất cho phụ thuộc bắc cầu). sklearn 1.9.1 có `method="temperature"` + `FrozenEstimator`.
  - SHA dataset `2d769061…` và model `86cd7fd4…` (có `tokenizer.json`, agpl-3.0) đúng; 3 file VnCoreNLP tồn tại; `uv 0.11.22` + cờ `uv sync/tree` đúng; `ghcr.io/astral-sh/uv:0.11.22` có; Node 24.15.0 khớp `jsdom ^24.15.0`.
  - Tổng 91h, 98h (có cuối tuần) / 70h (không), thứ trong tuần (17/10 = Thứ Bảy) đều đúng.
- Quyết định lead đã chốt: 10/10 khớp ở mức số liệu/phạm vi.
  - Ngân sách 5/10/20/30 mặc định 10, không dấu luôn cờ, >40 từ chỉ bỏ XAI: phase-01:100, phase-03:50, phase-04:36, phase-06:26. XAI top-20: phase-01:100, phase-06:27.
  - 10MB/2.000: phase-06:40,49 (cơ chế chặn: F3). So sánh mặc định tắt: phase-06:61, phase-07:35. Runtime không torch/pandas: phase-04:127. Export tĩnh 1 container, dev CORS không rewrites: phase-04:26,33.
  - Docker VPS bất kỳ: phase-09:21. AGPL-3.0: phase-01:35. MPS 1 job: phase-03:28, plan.md:44. Test 1 lần: phase-05:37,111.
  - Validation Log Session 1: 0/4 đã áp dụng vào phase (plan.md:85–89 còn `[ ]`) → F2.

#### Failures
**High**
- **F1 [Fact Checker]** `phase-02:26` ghi file `default/{train,validation,test}/0.parquet`. SHA đúng nhưng tên file thật là `0000.parquet` (HF API siblings; HEAD `0.parquet`=404, `0000.parquet`=302).
  `ml/download_data.py` (phase-02:82) và bước tính demo lúc build (F2) sẽ 404. Sửa: ghi `0000.parquet` hoặc `list_repo_tree` rồi assert tên.
- **F2 [Contract Verifier]** Validation Log (plan.md:72–89) chưa lan vào phase (mtime phase 08:59 < 09:09):
  - #2 (4 mục → Must): plan.md:51 vẫn là Should.
  - #3 (tính demo lúc build, repo không chứa câu gốc): phase-03:47–51 và phase-05:42 đưa `sample.csv`/`demo_results.json` (câu val/test) vào bundle + manifest; phase-05:43 `push_hub` đẩy cả bundle; phase-06:37,47,48 lấy câu mẫu/sample/demo từ bundle; phase-04:52 Dockerfile chưa có stage tính demo; phase-07:81 fixtures commit câu thật.
  - Hệ quả của #3: stage build cần mạng HF + `pyarrow` (không thuộc group `app`) + JVM/ORT; `bundle.py` kiểm sha256 theo manifest (phase-04:36) nên file sinh lúc build phải tách khỏi manifest; cross-build amd64 qua QEMU (phase-09:26) không thực tế cho bước suy luận; thiếu target dev `make demo-data`.
  - #4 (nhóm giả): `sample.csv` chưa có cột nhóm (phase-03:51); cần cờ `illustrative_groups` trong hợp đồng batch/insights (phase-06:42–45) để nhãn "minh hoạ" không mất khi mở lại `?batch=`.
  - #1 (dừng ở mốc): phase-01:108 "commit (người dùng duyệt)" mâu thuẫn "cook chạy liên tục".
  - Sửa: 1 lượt cập nhật phase trước khi cook; build thiếu HF thì fallback "không demo" thay vì fail.
- **F3 [Scope Auditor]** phase-06:49 "đếm byte khi đọc → 413" và :143 "không ghi file upload ra đĩa (giữ trong RAM)" sai với FastAPI mặc định.
  Starlette 1.8.0 `formparsers.py:151,234`: spool 1MB rồi ghi file tạm; `max_part_size` (:187–189) chỉ áp cho field thường, file part không có trần; FastAPI parse hết multipart trước khi handler chạy ⇒ 413 đến sau khi toàn bộ upload đã vào đĩa/tmpfs (tmpfs tính vào `mem_limit`, phase-09:33). Compose mặc định bind 0.0.0.0, không Caddy (phase-09:33).
  Sửa: ASGI middleware kiểm `Content-Length` + đếm byte stream; test chunked 50MB → 413; sửa câu "không ghi đĩa" thành "tạm thời trong /tmp, xoá khi xong".
- **F4 [Scope Auditor]** phase-04:160 "uvicorn access log chỉ có path" sai: uvicorn 0.54.0 `h11_impl.py:477–481` ghi `get_path_with_query_string`.
  `GET /api/batches/{id}/insights?group=<giá trị cột>` (phase-06:45) lộ giá trị cột nhóm + batch id vào log, trái phase-06:146. Grep log (phase-08:26,40) chỉ tìm "câu đã nhập" với file mẫu không có cột nhóm nên lọt.
  Sửa: đưa `group` vào POST body hoặc `--no-access-log` + log theo route template; E2E dùng cột nhóm chứa chuỗi đánh dấu.
- **F5 [Flow Tracer]** TTL 30' từ lần truy cập cuối (phase-06:25) + nhãn sửa chỉ ở client (phase-06:29, phase-07:128) + export cần batch trong RAM (phase-06:46) + polling dừng khi `done` (phase-07:49)
  ⇒ duyệt >30' (chỉ đổi Select, không gọi server) thì batch hết hạn, export 404, mất công duyệt. Sửa: heartbeat 60 s khi tab mở + `sendBeacon DELETE` khi đóng, hoặc export phía client (Q2).

**Medium**
- **F6 [Contract Verifier]** Makefile (phase-01:63–73) không có target cho `ml.significance`, `ml.error_analysis`, `ml.checklist`, `ml.calibrate --report test` (phase-05:38–40,81); `artifacts` thiếu `--version` (phase-05:42).
  Chuỗi plan.md:57 / phase-05:36 không sinh CheckList + MTL-vs-STL + kiểm định cho `model_card.json` ⇒ PDR 1 chưa đạt. `make api/web` không nạp `.env` (CORS dev, `NEXT_PUBLIC_*`: phase-01:100,105; `config.py` chỉ đọc env). Sửa: thêm `report`, `demo-data`; Makefile `include .env`.
- **F7 [Contract Verifier]** Không có schema cho `model_card.json`, `demo_results.json`, `calibration.json`: v0 (val, `"stage":"smoke"`, phase-03:51) khác v1 (test, phase-05:42) trong khi `/api/model-card` trả "JSON bundle" (phase-06:47) và trang 4 render cố định (phase-07:38).
  Trang 4 cần "lỗi bắt ở ngân sách 10%" trên test mà phase-05:42 không liệt kê. Manifest (phase-03:39) thiếu `bundle_version`/`stage` dù `/api/health` trả `bundle_version` (phase-04:37). Sửa: Pydantic model dùng chung trong `edupulse`, `build_artifacts` validate, thêm test.
- **F8 [Contract Verifier]** `build_artifacts` ghi phẳng vào `artifacts/` (phase-03:47) nhưng rollback giả định v0 "giữ nguyên khi tạo v1" (phase-03:111, phase-05:116); test "nhãn 20 câu val khớp `demo_results.json`" (phase-04:124) vỡ khi demo đổi val→test (phase-05:42).
  Sửa: `artifacts/<version>/` + `EDUPULSE_ARTIFACTS_DIR`; fixture golden 20 câu riêng.
- **F9 [Contract Verifier]** `NEXT_PUBLIC_SOURCE_URL` (link AGPL §13) là biến build-time (phase-01:105; phase-09:74 "build lại image") nhưng không phase nào khai `ARG`/`build.args` (grep = 0), `.dockerignore` chặn `.env*` (phase-04:121), compose chỉ `env_file` runtime (phase-09:33) ⇒ link nguồn rỗng/sai.
  Sửa: trả `source_url` qua `/api/config` (env runtime) hoặc khai ARG + build.args.
- **F10 [Flow Tracer]** `NORMALIZER_VERSION` lấy từ hằng trong code lúc build (phase-02:36), `data/processed` không đóng dấu phiên bản (phase-02:39) ⇒ parquet cũ + hằng mới vẫn qua cổng `bundle.py` (phase-04:36, phase-02:118).
  Sửa: ghi `data/processed/MANIFEST.json` {normalizer_version, sha}; train/`build_artifacts` đọc từ đó và assert khớp code.
- **F11 [Flow Tracer]** Logic cài 2 nơi: (a) luật cờ/τ_B ở `ml/calibrate.py` (phase-03:38) vs `engine/flags.py` (phase-04:36) trong khi `ml/` chỉ import text|labels|config (phase-01:48) ⇒ bảng "hiệu quả ước tính" có thể lệch luật chạy thật;
  (b) đếm uni+bigram nền ở `ml/preprocess.py` (phase-02:87) vs `insights/stats.py` (phase-06:28,60) ⇒ prior log-odds lệch cách tách token. Sửa: module numpy-only dùng chung (`edupulse/scoring.py`, `edupulse/text/ngrams.py`) + test parity.
- **F12 [Scope Auditor]** PDR:30 (D) liệt kê "tên", nhưng NER là Should (plan.md:51) và phase-02:38–51 không có quy tắc "thầy/cô + chữ Viết hoa" (r3 L70) ⇒ tiêu chí "không có tên người" (phase-06:128) không kiểm chứng được với file người dùng; nguyên tắc 2 (design:7) bị đe doạ ở "câu tiêu biểu".
  Sửa: thêm quy tắc rẻ vào Must + loại câu có token Viết hoa sau thầy/cô khỏi câu tiêu biểu; đổi tiêu chí thành "best effort + disclaimer".
- **F13 [Fact Checker]** Ước lượng web+API (phase 4+6+7+8 = 8+13+24+9 = 54h ≈ 7,7 ngày công @7h, plan.md:16) ≈ kịch bản "Tốt" của r6 (L95: 8 ngày; "Thường" 13), trong khi r6 L96 khuyên cắt toggle/XLSX/sparkline và Session 1 #2 đưa 4 mục về Must.
  Phần còn cắt được giờ < 12h (plan.md:16,51). Sửa: ước lượng lại; chốt cuối tuần (Q1).

**Low**
- **F14 [Flow Tracer]** Khoá MPS chỉ chặn 2 job train; Docker Desktop VM ≥4GB (phase-04:31) và `next build` vẫn chạy cạnh 9 run (phase-05:22) trong khi r1 L85 ghi VM ~7GB từng gây swap thrash.
  Sửa: runbook tắt Docker khi train; `run-experiments.sh` kiểm `docker info`; `smoke_test.py` từ chối đo khi `.mps.lock` đang giữ.
- **F15 [Fact Checker]** Ngưỡng go cộng lại (30 s + 10 s) ×3 = 120 s = đúng ngưỡng PDR:55 "<2 phút", không còn biên (phase-04:67–68 viết "vẫn < 2 phút"; phase-06:126). Sửa: ≤30 s gồm XAI, hoặc hệ số ×2,5.
- **F16 [Contract Verifier]** E2E/perf chạy trên compose/`docker run` cơ bản (phase-08:38,68); bản phát hành thêm `read_only` + tmpfs + `cap_drop` sau đó (phase-09:33,68) ⇒ cấu hình cuối chưa qua E2E/smoke. Sửa: chạy lại `smoke_test.py` + golden path trên compose cuối.
- **F17 [Fact Checker]** Lệch nhỏ: phase-01:104 dẫn BibTeX ở design L141 (L141 chỉ nhắc khối bibtex; nội dung ở cuối `wireframes/04-mo-hinh-dao-duc.html`); phase-07:108 "Các trạng thái khác của 4 wireframe" nhưng chỉ wireframe 01 (:550) và 02 (:549) có, 03 là "Trạng thái mặc định" (:543), 04 không có;
  `depends_on: [1,2,3]` (phase-04:7) ≠ plan.md:33 "1; smoke cần 3"; phase-02:81 "ghim commit" VnCoreNLP nhưng chưa nêu SHA.

#### Dòng chảy tập test (phục vụ lead)
- Chạm test: P2 `prepare_many` + thống kê mô tả EDA (phase-02:87–88; không chọn mô hình) · P5 `--final` đúng 1 lần, guard `.done` + `freeze.json` (phase-05:37,111) · P5 phân tích sau chỉ đọc `outputs/test_preds` (phase-05:30).
- Không chạm: P3 run đầu (val; demo v0 = val, phase-03:51) · P4 smoke, P8 perf/E2E (1.000 câu val, sample 200 val: phase-04:57, phase-08:32) · P6 retrain (val + audit, loại câu trùng test: phase-06:31).
- Demo dashboard dùng 3.166 câu test (phase-05:42, phase-06:48): chỉ HIỂN THỊ dự đoán đã lưu, không tính chỉ số ⇒ không vi phạm "đánh giá 1 lần" nếu không dùng để chỉnh gì và số model card lấy từ lần `--final`.
- Wireframe 03:543 ghi "nhãn gốc" nhưng chú thích UI :435 ghi "nhãn dự đoán"; plan theo :435 (nhất quán). Nếu tính lại lúc build (Session 1 #3) vẫn chỉ hiển thị, nhưng chạy trên x86 nên lệch nhẹ so với `test_preds` của M1: ghi chú vào model card.

#### Câu hỏi đề xuất cho phỏng vấn (không lặp 4 câu Session 1)
1. **Năng lực sau khi 4 mục về Must (F13, F2):** có làm cuối tuần 17–18 và 24–25/10 không?
   - (a) Làm cả 4 ngày, 98h ≥ 91h — **khuyến nghị**.
   - (b) Chỉ 17–18/10, 84h: cần cắt ≥7h trong 5 mục Should còn lại (plan không tách giờ).
   - (c) Không làm cuối tuần, 70h: lùi hạn ≥3 ngày làm việc (~2/11) hoặc xem lại #2.
2. **Phiên duyệt dài hơn TTL 30' (F5):**
   - (a) Heartbeat 60 s khi tab mở + xoá ngay khi đóng tab, TTL 30' chỉ là lưới an toàn — **khuyến nghị** (không giữ dữ liệu lâu hơn khi rời trang).
   - (b) Tăng TTL cố định lên 120' (đơn giản, giữ dữ liệu người dùng lâu hơn).
   - (c) Xuất CSV hoàn toàn trên trình duyệt, server xoá sớm (thêm `text_model` vào payload).
3. **Nơi build image khi demo tính lúc build (F2):**
   - (a) Build trên VPS với swap 2GB, tải artifacts trước, cần mạng HF — **khuyến nghị** (đúng kiến trúc đích, không QEMU).
   - (b) Build ở máy dev rồi `docker save | ssh docker load` — chỉ khi VPS cùng kiến trúc (arm64).
   - (c) CI GitHub Actions + GHCR (đang là Could, thêm hạ tầng).
4. **Kiểm trên x86 thật (Unverified 1–2):**
   - (a) Thuê VPS x86 2 vCPU/2GB theo giờ trước 26/10 để chạy `smoke_test.py --url` + so nhãn 200 câu val với M1 — **khuyến nghị**.
   - (b) Chạy `--platform linux/amd64` trên Mac (QEMU) chỉ để kiểm nhãn, không đo tốc độ.
   - (c) Chỉ đo M1 ×3, ghi rõ "chưa kiểm x86" trong model card + deployment.md.

#### Unverified
1. Tốc độ/RAM trên x86 2 vCPU (r2 L47 "CHƯA đo"): mọi ngưỡng ×3 là ước lượng.
2. Cổng int8 ≤1 điểm trên x86 (PDR:54 chỉ đo M1; r2 L50 cảnh báo U8S8; plan.md:52 xếp Could).
3. `torch.onnx.export(dynamo=False)` còn ở torch 2.14.1 (phase-03:106 tự đánh [UNVERIFIED]).
4. shadcn 4.21.5/Base UI, `cn@^0.4.0`, TS 7 làm `openapi-typescript` sập (r6 L6,45): không cài/chạy lại.
5. underthesea NER dưới rootfs read-only (phase-09:68 chỉ kiểm JVM/ORT/RDR).
6. Roundtrip skops cho `CalibratedClassifierCV(FrozenEstimator, temperature)` và danh sách `trusted` (phase-03:25): chưa chạy.
7. Kiểm zip bomb theo header `file_size` (phase-06:49) với calamine khi header bị giả.
8. Giấy phép dữ liệu UIT-VSFC (HF "unknown"; r2 L36 mirror CC BY-NC-SA 4.0) và pháp lý VN (r3 L83 nguồn thứ cấp).
9. Thời gian/RAM build trên VPS 2GB khi có bước tính demo (Session 1 #3).

#### Câu hỏi chưa giải quyết
- VPS x86_64 hay arm64, có domain không (plan.md:69): quyết định Q3/Q4 và việc kiểm int8.
- Tên nhóm minh hoạ của file mẫu (Session 1 #4): "Nhóm A–E" (đề xuất, tránh gợi liên hệ thật) hay tên môn giả?
- Khi HF không truy cập được lúc build: fail build hay fallback ẩn demo (đề xuất fallback)?
