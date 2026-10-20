---
title: "Phase 9: Đóng gói deploy VPS, cập nhật docs, gói artifact báo cáo"
status: todo
priority: P1
effort: 10h
dates: "2026-10-25 → 2026-10-27 (dự phòng 28/10)"
depends_on: [8]
---

# Phase 9: Đóng gói deploy VPS, cập nhật docs, gói artifact báo cáo
<!-- Updated: Validation Session 1–2 - cập nhật PDR mục 5, 7 + ràng buộc host là việc chắc chắn (S1 mặc định chốt, (e)); chạy lại smoke + E2E vàng trên compose cuối (F16); build ngay trên VPS (swap ≥2GB, cần mạng HF cho stage demo; HF lỗi → ẩn demo), `docker save | load` chỉ khi VPS cùng kiến trúc; `EDUPULSE_SOURCE_URL` là env runtime, không cần build lại (F9); tmpfs `/tmp` đủ cho file upload tạm (F3); tải artifacts vào `artifacts/<version>/` (F8); model card giữ "chưa đo x86" tới khi đo trên VPS; repo GitHub mặc định `edupulse`, đẩy ở mốc M3; ước lượng lại 10h -->

## Context links
- [PDR](../../docs/project-overview-pdr.md): mục 5 + 7 và ràng buộc host còn ghi Streamlit/Community Cloud (L25, L31, L37); tiêu chí 1, 6 (L51, L56).
- [r5 hosting](./research/researcher-01-hosting.md): nướng model vào image, cold start, chặn trần chi phí (L65–70) — host cloud nay ngoài phạm vi, chỉ giữ ý kỹ thuật.
- [r6 kiến trúc](./research/researcher-02-architecture.md): `--proxy-headers` cho redirect tuyệt đối (L78).
- [r2 deploy](../reports/researcher-261014-1913-hf-spaces-streamlit-deploy.md): U8S8 trên x86 (L50); riêng tư, disclaimer, Luật BVDLCN 91/2025/QH15 (L81–83).
- [Phase 4](./phase-04-api-skeleton-docker-smoke-test.md) (Dockerfile, compose cơ bản) · [Phase 6](./phase-06-backend-features.md) (stage `demo`/`final`) · [Phase 8](./phase-08-testing-review-performance.md) (`smoke_test.py --url`, QEMU). Báo cáo kiểm chứng: [F3, F9, F16, Unverified 9](./reports/validation-verification.md).

## Overview
- Ngày: 2026-10-25 → 2026-10-27, dự phòng 28/10 · Effort: 10h · Priority: P1 · Trạng thái: chưa bắt đầu.
- Hoàn thiện compose cho VPS Linux bất kỳ chạy 24/7 (không gắn nhà cung cấp), Caddy HTTPS tuỳ chọn, chạy lại smoke + E2E vàng trên đúng cấu hình phát hành, script tải artifacts ghim revision, `docs/deployment.md` + checklist nghiệm thu VPS; README cuối; cập nhật tech-stack + PDR (mục 5, 7, ràng buộc host); chỉ mục `results/`; thử tái lập trên clone sạch; **mốc M3** trước khi tạo/push repo GitHub public.
- Phần docs bắt đầu trong lúc phase 8 chạy; phần compose cuối chờ E2E phase 8 pass.

## Key Insights
- Chưa có VPS (người dùng dựng sau) ⇒ kiểm compose trên Mac (arm64); tài liệu ghi rõ bước nào đã kiểm thật, bước nào chỉ theo docs; người dùng chạy checklist nghiệm thu khi có VPS.
- Build NGAY TRÊN VPS (base image đa kiến trúc): cần mạng tới HF (stage `demo` tải dataset; `fetch-artifacts.sh` tải bundle) và swap ≥2GB (`next build` + stage demo có thể vượt 2GB RAM). HF lỗi lúc build → build vẫn xong, demo bị ẩn kèm thông báo; build lại khi có mạng. `docker save | ssh … docker load` chỉ khi VPS cùng kiến trúc với máy build (chạy suy luận demo qua QEMU không thực tế).
- Artifacts tải từ HF Hub bằng revision ghim (curl, không cần Python trên VPS) vào `artifacts/<version>/`, kiểm sha256 theo `manifest.json`; chưa đẩy HF → `scp artifacts/v1.0/`.
- 24/7: `restart: unless-stopped`, healthcheck, log rotation `json-file` 10m × 3, `mem_limit` 1536m (chừa RAM cho OS), stateless (không volume dữ liệu, không cần backup). `read_only: true` + tmpfs `/tmp` (≥64MB: file upload tạm của Starlette ≤3 × 10,5MB + file tạm JVM; tmpfs tính vào `mem_limit`).
- Caddy (profile `https`) tự cấp cert khi domain trỏ về VPS; `request_body max_size 11MB`; dùng Caddy thì bind cổng app vào `127.0.0.1` và đặt `FORWARDED_ALLOW_IPS=*` (chỉ Caddy tới được app); không Caddy thì giữ mặc định để header giả mạo bị bỏ qua.
- AGPL §13: link "Mã nguồn" lấy `EDUPULSE_SOURCE_URL` lúc chạy qua `/api/config` ⇒ đổi URL chỉ cần sửa `.env` + restart, không build lại.
- x86: QEMU đã so nhãn (phase 8); tốc độ/RAM thật chỉ đo được trên VPS người dùng bằng `smoke_test.py --url`; tới lúc đó model card + deployment.md ghi "chưa đo trên x86".

## Requirements
- `docker-compose.yml` cuối: service `edupulse` (build `.` target `final`, args `BUNDLE_VERSION`, `EDUPULSE_BUILD_DEMO`; image `edupulse:${EDUPULSE_VERSION}`), cổng `${EDUPULSE_BIND:-0.0.0.0}:${EDUPULSE_PORT:-8000}:8000`, `env_file: .env`, `mem_limit`/`memswap_limit` 1536m, healthcheck, logging rotation, `read_only: true` + tmpfs `/tmp`, `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`; service `caddy` (`caddy:2-alpine`, profile `https`, volume cert).
- `deploy/Caddyfile` (`{$DOMAIN}`, gzip/zstd, giới hạn body, reverse_proxy, HSTS).
- `scripts/fetch-artifacts.sh`: đọc `EDUPULSE_HF_REPO`/`EDUPULSE_HF_REVISION`/`BUNDLE_VERSION` từ `.env` → `manifest.json` → từng file → kiểm sha256 (`sha256sum` Linux / `shasum -a 256` macOS) → `artifacts/<version>/`.
- `docs/deployment.md` (tiếng Việt): cấu hình tối thiểu (2 vCPU / 2GB RAM + swap ≥2GB / 20GB SSD; Ubuntu 24.04 LTS hoặc Debian 12+; x86_64 hoặc arm64; khuyến nghị 4GB nếu bật NER và nhiều người dùng), chuẩn bị máy (user + SSH key, ufw 22/80/443 hoặc 8000, unattended-upgrades, swap, Docker Engine + compose plugin theo docs chính thức), lấy mã + artifacts, build ngay trên VPS (cần mạng HF; `EDUPULSE_BUILD_DEMO=0` khi offline) hoặc `docker save/load` khi cùng kiến trúc, kiểm tra + checklist nghiệm thu bằng `smoke_test.py --url`, cập nhật ghi chú x86 sau khi đo, HTTPS tuỳ chọn, vận hành (cập nhật, rollback theo tag, dọn image, giám sát `docker stats` + `/api/health`), xử lý sự cố (OOM, JVM, 413, 404 HTML, CPU steal, demo bị ẩn), bảo mật + riêng tư + rủi ro chấp nhận (không rate-limit theo IP, tên viết thường có thể sót), số đo tham chiếu từ phase 8 + thời gian build đo trên Mac.
- README cuối: giới thiệu + ảnh, kiến trúc, quick start dev (`make setup`, `make demo-data`, `make api`, `make web`), prod (`docker compose`), tái lập ML (`make data eda baseline experiments export calibrate evaluate report artifacts`), test (`make lint test e2e`), retrain từ file nhãn sửa, nguồn dữ liệu demo (tính lúc build, không có trong repo), license + ghi nguồn + trích dẫn, link HF model + deployment.
- `docs/project-overview-pdr.md`: mục 5 → "Web Next.js 4 trang + API FastAPI (1 container)"; mục 7 → "Docker + compose trên VPS người dùng; model trên HF Hub; dữ liệu demo tính lúc build"; ràng buộc host → "VPS 2 vCPU/2GB, runtime ONNX int8 không torch"; bỏ banner tạm.
- `results/README.md`: chỉ mục hình/bảng/metric ↔ chương báo cáo + lệnh tái tạo; `results/model_card.md` sinh từ `model_card.json` (mặc định đã chốt: trang 4 + README HF + file này).

## Architecture
```
VPS (Docker Engine, swap ≥2GB)                     HF Hub (public)
┌──────────────────────────────────────┐   fetch-artifacts.sh @ revision ──curl+sha256──► ./artifacts/v1.0/
│ docker compose build (target final)  │◄── stage demo tải dataset @ 2d769061… (HF lỗi → demo ẩn)
│ caddy (profile https) :80/443 ──► edupulse:8000 (uvicorn 1 worker: /api + web tĩnh, RAM ≤1,5GB, read-only, tmpfs /tmp)
└──────────────────────────────────────┘
```
Rollback VPS: giữ image tag cũ → đặt `EDUPULSE_VERSION=<tag cũ>` → `docker compose up -d --no-build`; artifacts lùi theo revision HF + `BUNDLE_VERSION`.

## Related code files
Create:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/deploy/Caddyfile`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/scripts/fetch-artifacts.sh`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docs/deployment.md`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/README.md`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/model_card.md` (sinh)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/results/perf/final-compose-<ngày>-arm64.json` (sinh, commit)

Modify:
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docker-compose.yml`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/Dockerfile` (chỉ nếu cần cho read-only/tmpfs)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/.env.example` (`DOMAIN`, `EDUPULSE_VERSION`, `EDUPULSE_BIND`, `EDUPULSE_PORT`, `FORWARDED_ALLOW_IPS`, `EDUPULSE_BUILD_DEMO`, `EDUPULSE_SOURCE_URL` thật)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/README.md`
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docs/tech-stack.md` (rà lại phiên bản thực dùng)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/docs/project-overview-pdr.md` (mục 5, 7, ràng buộc host)
- `/Users/admin/Desktop/Sync/Codes/tuanldas/EduPulse/ml/build_artifacts.py` (render `results/model_card.md`)

## Implementation Steps
1. `docker-compose.yml` cuối + `deploy/Caddyfile` + biến `.env.example`; chạy thử trên Mac: `docker compose up -d --build` → healthy; `docker compose --profile https up -d` với `DOMAIN=localhost` (cert nội bộ Caddy) để kiểm proxy + upload 10MB qua proxy.
2. Kiểm container `read_only` + tmpfs: JVM, ORT, VnCoreNLP, NER (khi bật), đọc `demo-data/`, file upload tạm đều chạy (cần ghi chỗ khác → thêm tmpfs, ghi lý do).
3. Chạy lại trên đúng compose cuối (F16): `smoke_test.py` (bảng go phase 4) → `results/perf/final-compose-*.json`; Playwright `golden-path.spec.ts`.
4. `scripts/fetch-artifacts.sh` + thử với revision HF (nếu đã đẩy) vào thư mục tạm, so sha256.
5. `docs/deployment.md` (đánh dấu "đã kiểm trên Mac" / "theo docs, chưa kiểm trên VPS"); kèm số đo phase 8 + thời gian build đo trên Mac.
6. README cuối; rà `docs/tech-stack.md`; sửa PDR mục 5, 7, ràng buộc host.
7. `results/README.md` + `results/model_card.md`; kiểm mọi đường dẫn trong chỉ mục tồn tại.
8. Tái lập trên clone sạch (thư mục tạm): `make setup data` → `make test` (test cần model skip kèm lý do) → `make fetch-artifacts` hoặc copy `artifacts/v1.0/` → `make demo-data` → `make test` đủ; ghi thời gian từng bước vào README.
9. **Dừng M3**: hỏi người dùng tạo repo GitHub public `<user>/edupulse` + push (`gh repo create`); đặt `EDUPULSE_SOURCE_URL` trong `.env` (restart, không build lại); tag `v1.0.0`.
10. Commit (`docs: add vps deployment guide`, `docs: finalize readme, pdr and results index`, `build: harden compose for production`).

## Todo list
- [ ] Compose cuối (restart, healthcheck, log rotation, giới hạn RAM, read-only, tmpfs, cap_drop) + Caddy profile
- [ ] Kiểm read-only (JVM, ORT, NER, demo-data, upload tạm) + upload 10MB qua Caddy trên Mac
- [ ] Chạy lại smoke + E2E vàng trên compose cuối
- [ ] `scripts/fetch-artifacts.sh` (revision ghim + sha256 → `artifacts/<version>/`)
- [ ] `docs/deployment.md` + checklist nghiệm thu VPS
- [ ] README cuối; rà tech-stack; cập nhật PDR mục 5, 7, ràng buộc host
- [ ] `results/README.md` + `results/model_card.md`
- [ ] Tái lập trên clone sạch
- [x] Mốc M3: public repo GitHub (https://github.com/tuanldas/EduPulse, lịch sử đã viết lại), `EDUPULSE_SOURCE_URL` — 2026-10-20
- [ ] Tag `v1.0.0` khi nộp bản cuối

## Success Criteria
- Clone sạch + `.env` + artifacts → `docker compose up -d --build` → `/api/health` 200, `healthy`, `demo_available=true` (Mac, có mạng).
- Trên compose cuối: bảng go phase 4 đạt và Playwright golden path pass (F16).
- `docs/deployment.md` đủ các mục ở Requirements; mọi lệnh đã chạy thử được đánh dấu; checklist VPS gọi `smoke_test.py --url`.
- README + PDR cập nhật; mọi lệnh README đã chạy thử trên clone sạch; `results/README.md` không có đường dẫn chết.
- (Sau M3) repo GitHub public; link "Mã nguồn · AGPL-3.0" trong app trỏ đúng repo qua env runtime.

## Risk Assessment
| Rủi ro | Khả năng × Tác động | Giảm thiểu |
|---|---|---|
| Không kiểm được trên VPS thật trước hạn | Cao × TB | Checklist nghiệm thu + `smoke_test.py --url`; ghi rõ phần chưa kiểm; tiêu chí "link public" phụ thuộc người dùng dựng VPS |
| Build OOM/lâu trên VPS 2GB (next build + stage demo) | TB × TB | Swap ≥2GB; `EDUPULSE_BUILD_DEMO=0` khi cần build nhanh; `docker save/load` nếu cùng kiến trúc |
| VPS không tới được HF lúc build | Thấp × TB | Demo ẩn kèm thông báo; artifacts qua `scp`; build lại khi có mạng |
| Kiến trúc VPS khác Mac (amd64) làm lệch int8 | TB × TB | QEMU đã so nhãn (phase 8); đo lại trên VPS; per-channel + reduce_range cho U8S8 (r2 L50) |
| read-only rootfs làm JVM/ORT/NER lỗi | Thấp × Thấp | tmpfs bổ sung hoặc bỏ `read_only` (ghi lý do) |
| Trễ sang 28/10 | TB × TB | Docs tối giản trước; Could (Caddy HTTPS, CI) bỏ đầu tiên |

Rollback: compose/Caddy là file cấu hình, `git revert`; trên VPS lùi theo tag image; artifacts lùi theo revision HF.

## Security Considerations
- Không secret trong repo/image; `.env` chỉ trên VPS, quyền 600; token HF không cần trên VPS (repo public).
- Container non-root, read-only, `cap_drop: ALL`, `no-new-privileges`; ufw chỉ mở cổng cần thiết; SSH key, tắt đăng nhập mật khẩu (khuyến nghị trong docs).
- HTTPS qua Caddy khi có domain; HSTS; giới hạn body 11MB khớp giới hạn app.
- Rủi ro chấp nhận ghi trong docs: không rate-limit theo IP (đồ án demo), không xác thực, tên viết thường có thể sót; khuyến cáo chỉ dùng dữ liệu đã ẩn danh; disclaimer pháp lý (không phải tư vấn luật). Repo không chứa câu gốc UIT-VSFC ngoài ≤30 câu ví dụ trong `results/`.

## Next steps
- Sau khi người dùng dựng VPS: chạy checklist nghiệm thu, cập nhật số đo VPS vào `results/perf/` + `docs/deployment.md`, bỏ ghi chú "chưa đo x86" trong model card nếu đạt. Người dùng viết báo cáo dựa trên `results/README.md`.
