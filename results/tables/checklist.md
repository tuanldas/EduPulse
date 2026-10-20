# CheckList: kiểm thử hành vi của mô hình trên validation

Validation n = 1.583 câu · run `mtl-s42` · mô hình ONNX int8, từng câu một (batch 1) · seed 0 · nhiệt độ T: cảm xúc 1,00, chủ đề 1,00.

| Kiểm thử | Loại | n | Tỉ lệ đạt | Ghi chú |
|---|---|---:|---:|---|
| Viết HOA toàn câu | Bất biến | 1.583 | 100,0% | Đạt khi nhãn cảm xúc và chủ đề đều không đổi; hỏng 0/1.583 câu. |
| Viết Hoa Chữ Đầu Mỗi Từ | Bất biến | 1.583 | 86,4% | Đạt khi nhãn cảm xúc và chủ đề đều không đổi; hỏng 215/1.583 câu. |
| Dấu câu và khoảng trắng thừa | Bất biến | 1.583 | 96,4% | Đạt khi nhãn cảm xúc và chủ đề đều không đổi; hỏng 57/1.583 câu (thêm ' .' 43, thêm ' !!' 46, gấp đôi dấu cách 0, đệm dấu cách đầu/cuối 0). |
| Đổi thầy ↔ cô | Bất biến | 520 | 99,4% | Đạt khi nhãn cảm xúc và chủ đề đều không đổi; hỏng 3/520 câu (thầy→cô 3/371, cô→thầy 0/149). |
| Đổi chỉ số tên ẩn (wzjwz + số) | Bất biến | 18 | 100,0% | Đạt khi nhãn cảm xúc và chủ đề đều không đổi; hỏng 0/18 câu. |
| Chèn thông tin cá nhân | Bất biến | 1.583 | 100,0% | Đạt khi nhãn cảm xúc và chủ đề đều không đổi; hỏng 0/1.583 câu (SĐT 0, email 0, MSSV 0). |
| Thêm "không" phủ định | Có hướng | 809 | 83,2% | Xét câu gốc dự đoán tích cực (P ≥ 0,5). Đạt khi P(tích cực) giảm; hỏng 136/809 câu. |
| Thêm ":(" cuối câu | Có hướng | 1.583 | 96,3% | Đạt khi P(tiêu cực) không giảm quá 0,01; hỏng 58/1.583 câu. |
| Bỏ dấu toàn câu | Bền vững | 1.579 | 98,4% | Đạt khi hệ thống gắn cờ không dấu (lọt cờ 25/1.579 câu). Accuracy: cảm xúc giảm 13,6 điểm (94,8% → 81,2%), chủ đề giảm 13,0 điểm (89,9% → 76,9%). |

Bất biến: nhãn cảm xúc và chủ đề cùng không đổi sau một biến đổi vô hại. Có hướng: xác suất phải đổi đúng chiều. Bền vững: đo mức suy giảm khi đầu vào xấu đi. `n` là số câu đủ điều kiện; mỗi biến thể sinh từ câu gốc rồi qua đúng đường tiền xử lý và suy luận lúc phục vụ.
