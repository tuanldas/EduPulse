# EduPulse

Phân loại cảm xúc (tiêu cực, trung tính, tích cực) và chủ đề (giảng viên, chương trình, cơ sở vật chất, khác) của góp ý sinh viên bằng tiếng Việt. Mô hình là PhoBERT-base-v2 đa nhiệm huấn luyện trên UIT-VSFC, chạy bằng ONNX Runtime int8. Ứng dụng giải thích từng dự đoán, gắn cờ câu cần xem xét, tổng hợp insight theo chủ đề và ẩn thông tin cá nhân. Đồ án cuối kỳ môn AI.

> Trạng thái: đang phát triển. README đầy đủ (kết quả, hướng dẫn deploy) sẽ có ở cuối dự án.

## Bắt đầu

Cần [uv](https://docs.astral.sh/uv/) (tự tải Python 3.12) và `make`. Phần web cần thêm Node 24 và pnpm.

```bash
make setup   # cài môi trường Python và phụ thuộc web
make help    # liệt kê mọi lệnh
make test    # chạy kiểm thử
```

## Cấu hình

Sao `.env.example` thành `.env` rồi sửa. `make` đọc `.env` và truyền biến cho mọi lệnh con.

Dữ liệu, mô hình, kết quả huấn luyện, bundle và dữ liệu demo đều tái tạo được nên không nằm trong git. Chúng nằm dưới `EDUPULSE_HOME`; để trống thì dùng thư mục repo. Đặt `EDUPULSE_HOME` ra ngoài repo khi repo nằm trong thư mục đồng bộ hoặc khi nhiều bản làm việc (git worktree) dùng chung dữ liệu:

```
$EDUPULSE_HOME/{data, models, outputs, artifacts/<version>, demo-data}
```

## Tài liệu

- [Tổng quan, yêu cầu](docs/project-overview-pdr.md)
- [Công nghệ](docs/tech-stack.md)
- [Hướng dẫn thiết kế](docs/design-guidelines.md) và [wireframe](docs/wireframes/index.html)
- [Kế hoạch triển khai](plans/261015-0121-edupulse-implementation/plan.md)

## Giấy phép

Mã nguồn theo [GNU AGPL-3.0](LICENSE) vì PhoBERT-base-v2 dùng AGPL-3.0. Thành phần bên thứ ba và cách trích dẫn nằm ở [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). UIT-VSFC chỉ dùng cho học thuật, phi thương mại; repo không chứa câu gốc của bộ dữ liệu.
