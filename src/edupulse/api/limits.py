"""Giới hạn riêng của tầng API (giới hạn dùng chung với phần còn lại nằm ở `edupulse.config`)."""

MAX_GROUPS = 30  # số giá trị khác nhau tối đa của cột nhóm (heatmap và bộ lọc nhóm)
MAX_GROUP_CHARS = 100  # độ dài tối đa của một tên nhóm; dài hơn bị cắt
MAX_COLUMN_CHARS = 200  # độ dài tối đa của tên cột
MAX_COLUMNS = 200  # số cột tối đa của file
MAX_CELL_CHARS = 20_000  # ô dài hơn bị cắt khi đọc file (câu còn bị cắt tiếp ở MAX_CHARS)
PREVIEW_ROWS = 5  # số dòng xem trước sau khi tải file
PREVIEW_CELL_CHARS = 200  # độ dài tối đa của một ô trong phần xem trước
XLSX_MAX_UNCOMPRESSED_MB = 100  # tổng dung lượng giải nén tối đa của một file XLSX
XLSX_MAX_MEMBERS = 2000  # số tệp con tối đa trong một file XLSX
DEMO_BATCH_ID = "demo"  # batch chỉ đọc nạp sẵn từ dữ liệu demo
