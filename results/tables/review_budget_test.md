Hiệu quả ngân sách duyệt trên tập test (n = 3.166) với calibration đóng băng từ validation (T, phân vị độ tin cậy, ngưỡng τ). Không fit lại gì trên test.

### Luật xếp hạng cả lô (cờ ⌈B·n/100⌉ câu điểm thấp nhất)

| Ngân sách | Tỉ lệ cờ | Lỗi cảm xúc bị bắt | Lỗi chủ đề bị bắt | Chính xác phần không cờ (đúng cả hai) | Chính xác khi không cờ |
|---:|---:|---:|---:|---:|---:|
| 5% | 5,0% | 27,9% | 16,7% | 87,6% | 85,5% |
| 10% (mặc định) | 10,0% | 45,3% | 34,4% | 89,8% | 85,5% |
| 20% | 20,0% | 64,2% | 57,7% | 92,5% | 85,5% |
| 30% | 30,0% | 72,6% | 70,0% | 93,7% | 85,5% |

### Luật ngưỡng từng câu (cờ khi điểm < τ, τ lấy từ validation)

| Ngân sách | Ngưỡng τ | Tỉ lệ cờ thực | Lỗi cảm xúc bị bắt | Lỗi chủ đề bị bắt | Chính xác phần không cờ (đúng cả hai) |
|---:|---:|---:|---:|---:|---:|
| 5% | 0,0243 | 4,7% | 25,8% | 15,1% | 87,5% |
| 10% | 0,0534 | 10,2% | 45,3% | 35,0% | 89,8% |
| 20% | 0,1134 | 20,4% | 65,3% | 58,7% | 92,6% |
| 30% | 0,1796 | 30,2% | 72,6% | 70,0% | 93,7% |

### Hiệu chuẩn xác suất trên cùng tập

| Tác vụ | T | ECE trước T | ECE sau T | NLL trước T | NLL sau T |
|---|---:|---:|---:|---:|---:|
| Cảm xúc | 1,000 | 0,0138 | 0,0138 | 0,2203 | 0,2203 |
| Chủ đề | 1,000 | 0,0232 | 0,0232 | 0,3218 | 0,3218 |
