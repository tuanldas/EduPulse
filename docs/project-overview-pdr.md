# EduPulse — Tổng quan dự án & yêu cầu (PDR)

> Hệ thống thông minh phân tích ý kiến phản hồi của sinh viên tiếng Việt. Đồ án cuối kỳ, khoảng 2 tuần từ 2026-10-14.

## Mục tiêu

Phân loại từng câu góp ý theo **cảm xúc** (tiêu cực / trung tính / tích cực) và **chủ đề** (giảng viên / chương trình / cơ sở vật chất / khác), rồi tổng hợp thành dashboard hỗ trợ khoa/trường ra quyết định.

Không dừng ở một classifier: hệ thống nhận góp ý → suy luận → giải thích → gắn cờ cho người duyệt → rút insight & khuyến nghị → học lại từ nhãn đã sửa.

## Người dùng

- Cán bộ khoa / phòng đảm bảo chất lượng: tải file góp ý, xem dashboard, duyệt câu bị gắn cờ, xuất kết quả.
- Giảng viên, sinh viên (demo): thử phân tích một câu.
- Hội đồng chấm đồ án: xem demo, model card, so sánh mô hình.

> **Thay đổi 2026-10-15:** giao diện chuyển từ Streamlit sang **Next.js (frontend) + API Python (backend)**. Thiết kế thị giác và nội dung 4 trang trong `docs/wireframes/` vẫn giữ. Nơi host miễn phí sẽ chốt lại trong kế hoạch; mục 5 và 7 bên dưới hiểu theo kiến trúc mới.

## Phạm vi (chốt 2026-10-14/15)

1. Dữ liệu UIT-VSFC (`uitnlp/vietnamese_students_feedback`, revision `refs/convert/parquet`), giữ split chuẩn 11.426 / 1.583 / 3.166.
2. EDA, sinh hình và bảng cho báo cáo.
3. Baseline: TF-IDF + LinearSVC / LogisticRegression / Naive Bayes.
4. Mô hình chính: fine-tune `vinai/phobert-base-v2`, 1 encoder + 2 head (bản deploy); 2 mô hình đơn nhiệm làm đối chứng.
5. App Streamlit 4 trang: phân tích 1 câu · phân tích hàng loạt (CSV/XLSX) · dashboard & insight · mô hình & đạo đức.
6. Bốn tính năng thông minh:
   - (A) XAI: tô các từ quyết định dự đoán (occlusion bỏ-từng-từ).
   - (B) Xác suất đã hiệu chỉnh, cờ "cần xem xét", sửa nhãn và xuất file để huấn luyện lại.
   - (C) Insight & khuyến nghị theo luật (từ khoá đặc trưng, câu tiêu biểu, cảnh báo so với mức nền từng chủ đề).
   - (D) Ẩn dữ liệu cá nhân (tên, email, SĐT, MSSV, CCCD, URL); không lưu dữ liệu người dùng.
7. Deploy miễn phí: Streamlit Community Cloud từ repo GitHub public; model trên Hugging Face Hub (repo public). Dự phòng: chạy local + Cloudflare Quick Tunnel.

## Ràng buộc

- Khoảng 2 tuần. Máy dev Apple M1 Pro 16GB (MPS), dự phòng Google Colab.
- Tập test chỉ dùng cho lần đánh giá cuối.
- Host miễn phí (RAM ~0,7–2,7GB, CPU dùng chung) → runtime ONNX Runtime int8, không torch.
- Không dùng LLM / agent (tránh trùng đề tài TaskMate).
- Giao diện tiếng Việt.
- License: PhoBERT-base-v2 AGPL-3.0, VnCoreNLP GPL-3.0+ → repo public, có link mã nguồn trong app. UIT-VSFC chỉ dùng học thuật, phi thương mại.

## Ngoài phạm vi

- Viết báo cáo cuối kỳ (chỉ sinh hình, bảng, metric, phân tích lỗi).
- Đăng nhập, cơ sở dữ liệu, đa người dùng, lưu trữ bền dữ liệu người dùng.
- Chatbot / LLM.
- Trích xuất khía cạnh mức cụm từ (chỉ phân loại cấp câu).

## Tiêu chí nghiệm thu

1. Tái lập bằng vài lệnh, seed cố định: tải dữ liệu → EDA → train → đánh giá → export.
2. Bảng kết quả trên test cho 2 task × (3 baseline + PhoBERT đa nhiệm bản deploy, cả FP32 và int8): accuracy, macro-F1, weighted-F1, F1 từng lớp, confusion matrix, ECE, kèm khoảng tin cậy 95% bằng bootstrap; so sánh với baseline bằng McNemar + paired bootstrap. So sánh đa nhiệm với đơn nhiệm chỉ ở mức thí nghiệm sơ bộ (val, 1 seed, bước research). Không train thêm (đổi ngày 2026-10-17, xem plan).
3. So sánh trung thực. Kỳ vọng PhoBERT vượt baseline tốt nhất về macro-F1; nếu không, phân tích nguyên nhân. Tham chiếu paper: cảm xúc ≈ 94 / 83, chủ đề ≈ 89 / 80 (accuracy / macro-F1).
4. Model int8: macro-F1 trên val giảm không quá 1 điểm so với FP32.
5. App trên host: nhãn của 1 câu p95 ≤ 0,5 giây; phần giải thích (tải sau nhãn) p95 ≤ 2 giây cho câu không quá `XAI_MAX_WORDS` từ (mặc định 25, chỉnh bằng biến môi trường, chốt sau khi đo trên VPS); 1.000 câu < 2 phút; upload file 5MB chạy được; RAM nằm trong giới hạn host. (Đổi ngày 2026-10-15 tại mốc M1, xem plan.)
6. Link app public chạy được; test tự động (pytest) pass.
7. Có model card, disclaimer, bảng chỉ số từng lớp, test hành vi (CheckList), mục quyền riêng tư.

## Tài liệu liên quan

- Công nghệ: [tech-stack.md](tech-stack.md)
- Research (bằng chứng, số đo): `plans/reports/researcher-261014-1913-phobert-vsfc-modeling.md`, `...-hf-spaces-streamlit-deploy.md`, `...-intelligent-layer-xai-insights.md`
