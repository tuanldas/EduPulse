# Review bảo mật EduPulse (backend + quét web) - 2026-10-19

Commit 5a3aefb (HEAD f63a447 chỉ đổi docs). Chỉ đọc, repo không đổi. Đã chạy: pytest hẹp (148 + 469 pass: upload/parsing/export/meta/static/heartbeat/log/jobs/errors/text/insights/scoring/demo),
`ruff check src` sạch, snippet python, 1 server uvicorn thật (đúng cờ Dockerfile, cổng 18765, đã tắt). Không bật Docker.
Mô hình đe doạ: public, không auth/cookie/bí mật. Rủi ro chính: DoS, lộ dữ liệu qua log/batch, injection file xuất, XSS.
Kết luận: log, CORS, header, static, CSV injection, zip bomb, XXE, non-root, XSS web ĐẠT. Có 2 lỗi High (1 request rẻ làm OOM container), 3 Medium, 7 Low.

| ID | Mức | Tóm tắt |
|---|---|---|
| H1 | High | XLSX ~5KB khiến calamine cấp phát ma trận đặc theo ô xa nhất: OOM/abort |
| H2 | High | Thân JSON của mọi route ngoài `/api/batches` không bị giới hạn: 1 request ~250MB làm OOM |
| M1 | Medium | Upload song song không có trần; parse xong mới kiểm sức chứa; CPU khuếch đại |
| M2 | Medium | PII sót tên sau "thầy giáo/cô giáo/hiệu trưởng", MSSV chữ-số, SĐT bàn có ngoặc; model card nói nhẹ hơn thực tế |
| M3 | Medium | Repo chứa câu gốc UIT-VSFC (trái README:36) |
| L1-L7 | Low | Chiếm chỗ batch; purge chỉ khi có request; hardening container; CSP/Swagger CDN; Host/forwarded; heartbeat vs TTL; vặt |

## H1 - XLSX: file nhỏ, dải ô khổng lồ (High)
- Bằng chứng: `api/parsing.py:251-261` (`check_xlsx_size` chỉ chặn byte giải nén; `get_sheet_by_name` nạp cả hình chữ nhật dòng×cột thành ma trận đặc TRƯỚC mọi kiểm tra);
  `MAX_COLUMNS` chỉ xét ở `build_table` (`parsing.py:327`) sau khi đã cấp phát; `limits.py:10` trần giải nén 100MB quá cao.
- Kịch bản: XLSX chỉ có A1 và một ô ở (dòng 1.048.576, cột 50) -> ~52 triệu ô x ~31B = ~1,6GB > `mem_limit` 1536m (compose:29); cực đại Excel 17 tỉ ô (suy ra: abort).
- Đo (gọi `parse_upload`, cùng đường `POST /api/batches`, file 4,9KB, calamine không từ chối): 20.000x100 ô = +62MB/0,32s; 1.048.576x3 = +97MB/0,72s; tuyến tính ~31B/ô.
  Không đẩy tới OOM để khỏi ảnh hưởng máy đo hiệu năng.
- Tác động: 1 request không xác thực giết process, mất mọi batch đang giữ; `restart: unless-stopped` chỉ khởi động lại, kẻ tấn công lặp lại được.
- Sửa: trước khi gọi calamine, quét sheet XML bằng bộ giải nén có trần sẵn có (`_inflated_size`) lấy max dòng/cột từ thuộc tính `r="…"` của `<c>` (KHÔNG tin thẻ `<dimension>`);
  từ chối nếu dòng > ~50.000, cột > `MAX_COLUMNS` hoặc dòng x cột > ~3 triệu. Hạ `XLSX_MAX_UNCOMPRESSED_MB` 100 -> 30. Thêm test hồi quy (file sinh bằng openpyxl, ô xa).

## H2 - Thân JSON không giới hạn (High)
- Bằng chứng: `api/main.py:154-169` chỉ gắn `UploadLimitMiddleware`, middleware chỉ áp cho `POST /api/batches` (`upload_limit.py:24,59-64`);
  `tests/api/test_upload_limit.py:140-155` còn KHẲNG ĐỊNH `/api/analyze`, `/run` không bị giới hạn. FastAPI đọc nguyên thân vào RAM trước khi validate/gọi dependency.
- Kịch bản: `POST /api/analyze` chunked, thân 100MB toàn chữ `a` (hoặc Content-Length lớn).
- Đo: ASGI trực tiếp, thân 100MB đọc hết, RSS đỉnh +288MB (0,25s), kết quả 503/422 chỉ sau khi đọc xong. Server thật: thân 50MB +207MB; 100MB thêm +229MB (~4x thân, `ps` macOS).
  Nền ~600MB (smoke: memory.peak 573-618MB) + trần 1.536MB => thân ~250MB là đủ OOM; 40MB bị 422 `text_too_long` sau khi đã đọc hết.
- Sửa: dùng tham số `path`/`max_bytes` có sẵn của middleware: giới hạn 512KB cho mọi POST/DELETE ngoài `/api/batches` (ExportRequest tối đa 2.000 override ~140KB),
  đổi test 140-155 thành khẳng định 413. Thêm lớp thứ hai ở proxy (`request_body max_size` của Caddy / `client_max_body_size`).

## M1 - Upload song song, kiểm sức chứa sau khi parse (Medium)
- Bằng chứng: `routes/batches.py:87-95` đọc + `parse_upload` rồi mới `store.create` (`jobs.py:269-270` trả 429); `upload_batch` là `def` nên tới ~40 luồng threadpool chạy song song; không semaphore.
- Đo (server thật, 6 upload song song, mỗi file hợp lệ 9,35MB/1.400 dòng): RSS đỉnh +286MB, giữ lại +279MB (~46MB/batch; 8 batch ~370MB cố định); đóng batch chỉ trả ~18MB (macOS).
  ~20 upload song song vượt trần; khi kho đầy vẫn parse đầy đủ rồi mới 429. CPU: CSV toàn dòng trống ~1,2µs/dòng (`choose_delimiter` quét 3 lần không giới hạn vì `widths` không tới 100)
  => file 10MB (~5 triệu dòng) ~6s CPU giữ GIL cho 1 request; 10 triệu dấu phẩy trên 1 dòng: 1,45s + 155MB.
- Sửa: `threading.BoundedSemaphore(2)` quanh phần đọc+parse; kiểm `store.count() >= max_batches` TRƯỚC khi đọc file; `choose_delimiter` dừng sau ~2.000 dòng thô;
  `build_table` từ chối khi dòng trống > MAX_ROWS.

## M2 - PII sót (Medium)
- Bằng chứng: `text/pii.py:69-72` (`_CUE` không có "thầy giáo/cô giáo/hiệu trưởng"), `:76-79` (`_NOT_NAMES` chứa "giáo" nên chặn luôn chuỗi), `:121-140`.
  `insights/build.py:53-56,197` đã tự vá riêng cho câu trích dẫn ("thầy giáo|cô giáo") nên đội đã biết lỗ hổng, nhưng `display_text`, file xuất, bảng giải thích (token của `model_text`) không được vá.
- Đo (`mask()`): KHÔNG ẩn: "Thầy giáo Minh", "Cô giáo Lan", "thầy hiệu trưởng Hùng", "mã sinh viên B20DCCN123", "mã 1712345", "(028) 3812 3456", "instagram.com/x", "github.com/x/y", "@handle".
  Ẩn đúng: email, +84/84/dấu chấm/cách số, CCCD, CMND, MSSV 8 số, URL http/www/fb/zalo, "thầy Nam", "Cô Nguyễn Thị Hương", "TS./GV Tên", "bạn Lan", "anh Tuấn".
- Tác động: tên giảng viên lọt vào cột "Câu (đã ẩn thông tin cá nhân)", dashboard, XAI. Model card (`limitations`) chỉ nói "tên viết thường thì sót" nên người dùng tin rằng "Cô giáo Lan" đã được ẩn.
- Sửa: thêm `(?:thầy|cô)\s+giáo`, `hiệu\s+trưởng`, `trưởng\s+(?:khoa|bộ\s+môn)`, `trợ\s+giảng`, `cố\s+vấn` vào `_CUE` và cho `_cue_name_spans` bỏ qua từ chức danh trước tên;
  thêm MSSV chữ+số (`\b[A-Z]\d{2}[A-Z]{2,4}\d{3,4}\b`), SĐT bàn có `(0xx)`. pii.py dùng chung với train => phải tăng `NORMALIZER_VERSION` (`text/pipeline.py:23`) và dựng
  lại bundle (quyết định của lead).
  Nếu không kịp: sửa câu trong model card + `privacy-blocks.tsx:35` nêu rõ các dạng trên sót. ReDoS: không (chuỗi xấu nhất 500 ký tự <= 1ms).

## M3 - Câu gốc UIT-VSFC nằm trong git (Medium, tuân thủ)
- Bằng chứng: README.md:36 "repo không chứa câu gốc". Quét 8-gram + khớp nguyên câu (>= 7 từ) với train/val/test (15,8K câu) trên mọi file đã track:
  `results/tables/error_samples.csv` (30 dòng, 24 câu test nguyên văn kèm cột `sentence`), `docs/wireframes/01-phan-tich-cau.html` (6), `02-phan-tich-hang-loat.html` (8), `index.html` (1),
  1 câu trong `plans/reports/researcher-261014-1913-intelligent-layer-xai-insights.md`. Trùng ở `tests/`, `apps/web/tests` là cụm câu ngắn phổ biến, không kết luận.
- Tác động: giấy phép UIT-VSFC chỉ học thuật/phi thương mại; repo AGPL công khai sẽ phát tán câu gốc. Image KHÔNG bị (xem mục ĐẠT 15).
- Sửa: bỏ cột `sentence` khỏi `error_samples.csv` (giữ split,row,nhãn), thay câu wireframe bằng câu tự viết; nếu đã push public, cân nhắc viết lại lịch sử.

## Low
- L1 Chiếm chỗ batch: `config.py:28` MAX_JOBS=8, `jobs.py:269-270`. Đo: 8 file nhỏ => batch thứ 9 và `/api/batches/sample` đều 429 (`Retry-After: 5`); `GET`/heartbeat gia hạn TTL (`jobs.py:227`)
  nên giữ chỗ vô hạn; `/api/analyze` vẫn chạy. Tương tự `MAX_QUEUE=3` bị 3 batch 2.000 dòng chiếm liên tục. Đã chấp nhận (không rate-limit theo IP). Giảm thiểu rẻ: TTL 5
  phút cho batch `uploaded` chưa chạy + `limit_req` ở proxy.
- L2 Purge lười: `jobs.py:245-252` chỉ chạy trong get/create/run/count (223,242,268,293); worker `_work:361-374` không timer, `/api/health` không đụng kho. Lúc vắng
  request, `table` thô của batch `uploaded`/`failed`
  ở lại RAM quá TTL (<= 8 x ~46MB). Sửa: `self._queue.get(timeout=60)` rồi `_purge_locked()`.
- L3 Container: `Dockerfile:71-73` `COPY --chown=1000` làm mã vendor/bundle/web ghi được bởi user app (chỉ cần đọc; dùng `--chmod=a+rX`); `docker-compose.yml` thiếu
  `read_only: true` + `tmpfs: /tmp:size=64m`
  (Starlette spool upload > 1MB ra đĩa lớp ghi, trái "không ghi đĩa"), `cap_drop: [ALL]`, `no-new-privileges`, `pids_limit`; `ports: "8000:8000"` (compose:17-18) mở
  0.0.0.0 HTTP; nếu Caddy cùng host thì bind `127.0.0.1`.
  uvicorn không có `--limit-concurrency`/`--timeout-keep-alive` (slowloris); `python:3.12-slim-trixie` chưa ghim digest.
- L4 `security_headers.py:13-17` không có CSP (Next export cần `script-src 'unsafe-inline'`; vẫn nên `default-src 'self'; connect-src 'self'; frame-ancestors 'none';
  base-uri 'none'`, loại `/api/docs`).
  `/api/docs` (GIỮ theo quyết định) nạp `swagger-ui-dist@5` từ cdn.jsdelivr.net (không SRI, chỉ ghim major) + favicon fastapi.tiangolo.com, cùng origin app. Tác động thấp
  (không cookie, sessionStorage theo tab); ghim bản đầy đủ qua `swagger_js_url`.
- L5 Redirect `/batch` -> `/batch/` dựng `Location` từ header Host: `Host: evil.example` => `307 Location: http://evil.example/batch/` (đo). `--proxy-headers` chỉ tin
  127.0.0.1 nên sau proxy qua cầu Docker
  `X-Forwarded-Proto` bị bỏ và redirect hạ xuống http. Sửa: `FORWARDED_ALLOW_IPS` = mạng proxy, hoặc `TrustedHostMiddleware`.
- L6 `config.py:27` HEARTBEAT_S=60 cố định, còn TTL cấu hình (`config.py:183`); plan phase-08 E2E đặt TTL=1 phút => hết hạn đúng lúc nhịp tim (điều kiện `>` ở
  `jobs.py:247`), E2E "tab mở >90s" chập chờn.
  Sửa: `heartbeat_s = max(5, min(60, int(ttl_s/3)))` ở `routes/meta.py:75`.
- L7 Vặt: `.gitignore.swp` (untracked, chưa bị ignore; chứa tên máy/đường dẫn) -> xoá + thêm `*.swp`; `components/model/model-page.tsx:66` `decodeURIComponent(hash)` ném
  URIError với `/model/#%E0%A4%A` (tự làm vỡ trang, cần try/catch);
  `routes/batches.py:182-184` `assert batch.result is not None` sau `_finished`: đóng tab đúng lúc => 500 (cửa sổ micro-giây), nên đọc `result` một lần.

## Đã kiểm và ĐẠT
1. Content-Length khai 100MB => 413 ngay, không đọc thân (live: gửi 112 byte, 0,00s, `Connection: close`). 2. Chunked 12MB => 413, RSS không đổi. 3. CL khai gian nhỏ +
   gửi 3MB => h11 trả 400, không smuggling.
4. Multipart 10MB+1 => 413 `file_too_large`; > 2.000 dòng => 422 `too_many_rows`; ô > 131K ký tự => 422; UTF-16/NUL => 422; cp1258 fallback; CR-only, `;`/TAB/`|` không gây 500.
5. Zip bomb 150MB/400MB nhiều tệp/khai gian file_size => 422 trong 0,03s (đếm byte thực, `parsing.py:177-232`). XML entity lồng + `SYSTEM "file:///etc/hosts"` => 422, không mở rộng, không đọc file.
6. CSV injection: `= + - @`, toàn góc `＝＋－＠`, TAB, CR, tiêu đề cột nhóm, giá trị nhóm, `text_model`: 0 ô chưa bảo vệ ở cả `all` và `corrected` (live;
   `export.py:29,65,122-127`). Mọi cột (kể cả tiêu đề) qua `guard_cell`, không cột nào miễn trừ;
   web không tự sinh CSV (chỉ `saveBlob` file server; tên file qua `fileStem`). Đánh đổi: ô bị thêm `'` hiển thị nguyên trong Excel (đã có `unguard_cell`).
7. CORS: không có `Access-Control-Allow-Origin` cho request thường lẫn preflight (live); `_origins` (`config.py:98-108`) từ chối `*`; compose mặc định rỗng. Header
   nosniff/no-referrer/XFO DENY trên API và trang tĩnh, `Cache-Control: no-store` cho `/api`.
8. Batch id `secrets.token_urlsafe(16)` (128 bit); hết hạn và không tồn tại cùng 404. Khoá luôn theo thứ tự store -> batch, không giữ khoá khi chạy mô hình; `table=None`
   sau khi xong (`jobs.py:453`); `_discard` đặt `cancel`.
9. 429: batch thứ 9 -> 429 + `Retry-After`; hàng đợi chạy có test (`test_jobs.py` pass). TTL/heartbeat/close có test thời gian thật (`test_heartbeat.py` pass).
10. Log: chạy uvicorn thật với `--no-access-log`, gửi 12 chuỗi đánh dấu (văn bản, nhóm, tên cột, tên file, batch id THẬT, query, path, User-Agent, Referer, header lạ, request HTTP hỏng, method lạ):
    0 lần xuất hiện trong toàn bộ stdout/stderr (kể cả uvicorn, JVM). pytest hiện chỉ bắt logger `edupulse*`; lần chạy thật bao phủ cả `uvicorn.*`.
11. Static: 14 biến thể traversal (`..`, `%2e%2e`, `..%2f`, `%252e`, `//`, `%5c`, NUL, `..;`, đường dẫn 9.000 ký tự) => 404, không rò; `follow_symlink=False`; `/api`,
    `/api/x` mọi method => JSON 404; sai method => 405 JSON.
12. Non-root: `Dockerfile:75` `USER app` (uid 1000) ở stage `runtime`, `final` kế thừa; stage `demo` (root) không vào image. Không secret/ARG nhạy cảm; `.dockerignore`
    allowlist loại `.env`, data, results, tests, ml, docs.
13. Quét git: không token/khoá. `ruff check src` sạch.
14. XSS web: không có `dangerouslySetInnerHTML`/`innerHTML`/`eval` trong `apps/web/src` (grep); `xai-sentence.tsx` render text node qua JSX; 3 liên kết `target=_blank` đều `rel="noopener noreferrer"`;
    `href` động chỉ gồm `sourceUrl`/`modelUrl` (cấu hình, regex `^https?://\S+$` loại `javascript:`) và `?batch=` đã `encodeURIComponent`. `web/out` không gọi tài nguyên
  bên thứ ba, không source map/dotfile.
15. Image không chứa câu gốc: quét 8-gram trên bundle (json + `.skops` giải nén) và `web/out` = 0 trùng; `/app` chỉ có src cài sẵn, vendor, artifacts, web, demo-data
    (sinh lúc build). Skops baseline có sha256 trong manifest, nạp với `trusted` hẹp.

## Chưa kiểm được
Image thật (`docker history`, quyền file), CVE phụ thuộc (không có công cụ/mạng), hành vi OOM trong cgroup 1,5GB (chỉ ngoại suy từ RSS macOS; số Linux có thể khác).

## Câu hỏi chưa giải quyết
1. Có reverse proxy (Caddy) trước container không? Quyết định nơi đặt giới hạn thân/tốc độ, `FORWARDED_ALLOW_IPS` và bind `127.0.0.1`.
2. M2: sửa `_CUE` (tăng `NORMALIZER_VERSION`, dựng lại bundle) hay chỉ chỉnh model card/UI cho đúng thực tế?
3. M3: giấy phép UIT-VSFC có cho hiển thị công khai câu gốc qua demo (`/api/sample.csv`, câu trích dẫn, `demo_results.json`) không? Có xoá `error_samples.csv`/câu wireframe khỏi lịch sử git không?
4. `EDUPULSE_NER` có bật ở production không? Mặc định 0 (tên không có từ xưng hô không bao giờ được ẩn); bật thì underthesea tải mô hình lúc chạy và tốn RAM.
