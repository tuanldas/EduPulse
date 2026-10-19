# Phân tích lỗi và thiên lệch

Tập test, n = 3.166. Mô hình chính: PhoBERT đa nhiệm · int8 (bản triển khai), lấy từ run `mtl-s42`; đối chứng: TF-IDF + LinearSVC (nhãn thô). Các bảng chỉ chứa số đếm, tỉ lệ và nhãn.

## 1. Cặp nhầm lẫn

Mỗi dòng là một cặp (nhãn thật → nhãn dự đoán), xếp theo số lỗi giảm dần. "% tổng lỗi" là phần của cặp trong mọi lỗi của tác vụ; "Recall lớp thật" là tỉ lệ câu của lớp thật được đoán đúng.

### Cảm xúc

**PhoBERT đa nhiệm · int8 (bản triển khai)**: 190 lỗi trên 3.166 câu (đúng 94,0%).

| Nhãn thật → dự đoán | Số lỗi | % tổng lỗi | Số câu lớp thật | Recall lớp thật |
|---|---:|---:|---:|---:|
| Trung tính → Tích cực | 60 | 31,6% | 167 | 43,1% |
| Tích cực → Tiêu cực | 46 | 24,2% | 1.590 | 96,5% |
| Trung tính → Tiêu cực | 35 | 18,4% | 167 | 43,1% |
| Tiêu cực → Tích cực | 25 | 13,2% | 1.409 | 97,2% |
| Tiêu cực → Trung tính | 15 | 7,9% | 1.409 | 97,2% |
| Tích cực → Trung tính | 9 | 4,7% | 1.590 | 96,5% |

**TF-IDF + LinearSVC**: 335 lỗi trên 3.166 câu (đúng 89,4%).

| Nhãn thật → dự đoán | Số lỗi | % tổng lỗi | Số câu lớp thật | Recall lớp thật |
|---|---:|---:|---:|---:|
| Tích cực → Tiêu cực | 106 | 31,6% | 1.590 | 91,9% |
| Tiêu cực → Tích cực | 66 | 19,7% | 1.409 | 93,8% |
| Trung tính → Tích cực | 61 | 18,2% | 167 | 28,7% |
| Trung tính → Tiêu cực | 58 | 17,3% | 167 | 28,7% |
| Tiêu cực → Trung tính | 22 | 6,6% | 1.409 | 93,8% |
| Tích cực → Trung tính | 22 | 6,6% | 1.590 | 91,9% |

### Chủ đề

**PhoBERT đa nhiệm · int8 (bản triển khai)**: 317 lỗi trên 3.166 câu (đúng 90,0%).

| Nhãn thật → dự đoán | Số lỗi | % tổng lỗi | Số câu lớp thật | Recall lớp thật |
|---|---:|---:|---:|---:|
| Chương trình → Giảng viên | 97 | 30,6% | 572 | 79,5% |
| Giảng viên → Chương trình | 92 | 29,0% | 2.290 | 95,2% |
| Khác → Giảng viên | 45 | 14,2% | 159 | 48,4% |
| Khác → Chương trình | 34 | 10,7% | 159 | 48,4% |
| Chương trình → Khác | 18 | 5,7% | 572 | 79,5% |
| Giảng viên → Khác | 16 | 5,0% | 2.290 | 95,2% |
| Cơ sở vật chất → Giảng viên | 4 | 1,3% | 145 | 95,2% |
| Giảng viên → Cơ sở vật chất | 3 | 0,9% | 2.290 | 95,2% |
| Cơ sở vật chất → Chương trình | 3 | 0,9% | 145 | 95,2% |
| Khác → Cơ sở vật chất | 3 | 0,9% | 159 | 48,4% |
| Chương trình → Cơ sở vật chất | 2 | 0,6% | 572 | 79,5% |

**TF-IDF + LinearSVC**: 433 lỗi trên 3.166 câu (đúng 86,3%).

| Nhãn thật → dự đoán | Số lỗi | % tổng lỗi | Số câu lớp thật | Recall lớp thật |
|---|---:|---:|---:|---:|
| Chương trình → Giảng viên | 146 | 33,7% | 572 | 71,5% |
| Giảng viên → Chương trình | 121 | 27,9% | 2.290 | 93,8% |
| Khác → Giảng viên | 62 | 14,3% | 159 | 28,9% |
| Khác → Chương trình | 49 | 11,3% | 159 | 28,9% |
| Giảng viên → Khác | 15 | 3,5% | 2.290 | 93,8% |
| Chương trình → Khác | 14 | 3,2% | 572 | 71,5% |
| Cơ sở vật chất → Chương trình | 9 | 2,1% | 145 | 89,7% |
| Giảng viên → Cơ sở vật chất | 6 | 1,4% | 2.290 | 93,8% |
| Cơ sở vật chất → Giảng viên | 5 | 1,2% | 145 | 89,7% |
| Chương trình → Cơ sở vật chất | 3 | 0,7% | 572 | 71,5% |
| Khác → Cơ sở vật chất | 2 | 0,5% | 159 | 28,9% |
| Cơ sở vật chất → Khác | 1 | 0,2% | 145 | 89,7% |

## 2. Lỗi tự tin nhất (kiểm nhãn)

Mỗi tác vụ lấy 15 lỗi có độ tin cậy cao nhất của PhoBERT đa nhiệm · int8 (bản triển khai), tối đa 30 dòng, ghi cùng câu gốc ở `error_samples.csv` để người xem đối chiếu lại nhãn. Đó là tệp duy nhất chứa câu gốc. Độ tin cậy là xác suất softmax lớn nhất của logits bản triển khai ở T = 1 (chưa áp nhiệt độ hiệu chuẩn); chỉ dùng để xếp hạng lỗi.

| Tác vụ | Số dòng | Độ tin cậy cao nhất | Độ tin cậy thấp nhất trong nhóm |
|---|---:|---:|---:|
| Cảm xúc | 15 | 0,9831 | 0,9741 |
| Chủ đề | 15 | 0,9859 | 0,9700 |

## 3. Lỗi theo lát

Mỗi ô là số lỗi (tỉ lệ lỗi trong lát). Ở lát chủ đề thật, tỉ lệ lỗi chủ đề của một chủ đề bằng 1 trừ recall của chủ đề ấy. Lát rỗng (n = 0) không có tỉ lệ.

### 3.1 Theo chủ đề thật

| Lát | n | Cảm xúc · PhoBERT int8 | Cảm xúc · LinearSVC | Chủ đề · PhoBERT int8 | Chủ đề · LinearSVC |
|---|---:|---:|---:|---:|---:|
| Toàn tập | 3.166 | 190 (6,0%) | 335 (10,6%) | 317 (10,0%) | 433 (13,7%) |
| Giảng viên | 2.290 | 101 (4,4%) | 196 (8,6%) | 111 (4,8%) | 142 (6,2%) |
| Chương trình | 572 | 53 (9,3%) | 85 (14,9%) | 117 (20,5%) | 163 (28,5%) |
| Cơ sở vật chất | 145 | 2 (1,4%) | 6 (4,1%) | 7 (4,8%) | 15 (10,3%) |
| Khác | 159 | 34 (21,4%) | 48 (30,2%) | 82 (51,6%) | 113 (71,1%) |

### 3.2 Theo độ dài (số từ sau tách, ngưỡng 40)

| Lát | n | Cảm xúc · PhoBERT int8 | Cảm xúc · LinearSVC | Chủ đề · PhoBERT int8 | Chủ đề · LinearSVC |
|---|---:|---:|---:|---:|---:|
| Toàn tập | 3.166 | 190 (6,0%) | 335 (10,6%) | 317 (10,0%) | 433 (13,7%) |
| ≤ 40 từ | 3.128 | 188 (6,0%) | 333 (10,6%) | 313 (10,0%) | 429 (13,7%) |
| > 40 từ | 38 | 2 (5,3%) | 2 (5,3%) | 4 (10,5%) | 4 (10,5%) |

### 3.3 Theo token đặc biệt (wzjwz, colon..., doubledot, ...)

| Lát | n | Cảm xúc · PhoBERT int8 | Cảm xúc · LinearSVC | Chủ đề · PhoBERT int8 | Chủ đề · LinearSVC |
|---|---:|---:|---:|---:|---:|
| Toàn tập | 3.166 | 190 (6,0%) | 335 (10,6%) | 317 (10,0%) | 433 (13,7%) |
| Có token đặc biệt | 138 | 11 (8,0%) | 17 (12,3%) | 10 (7,2%) | 17 (12,3%) |
| Không có | 3.028 | 179 (5,9%) | 318 (10,5%) | 307 (10,1%) | 416 (13,7%) |

### 3.4 Theo gõ không dấu

| Lát | n | Cảm xúc · PhoBERT int8 | Cảm xúc · LinearSVC | Chủ đề · PhoBERT int8 | Chủ đề · LinearSVC |
|---|---:|---:|---:|---:|---:|
| Toàn tập | 3.166 | 190 (6,0%) | 335 (10,6%) | 317 (10,0%) | 433 (13,7%) |
| Có dấu | 3.164 | 189 (6,0%) | 334 (10,6%) | 317 (10,0%) | 433 (13,7%) |
| Gõ không dấu | 2 | 1 (50,0%) | 1 (50,0%) | 0 (0,0%) | 0 (0,0%) |

## 4. Giao lỗi TF-IDF ∩ PhoBERT

Mỗi câu rơi vào đúng một trong bốn ô, tuỳ PhoBERT int8 và LinearSVC (nhãn thô) có đoán đúng hay không.

| Tác vụ | Cả hai sai | Chỉ PhoBERT int8 sai | Chỉ LinearSVC sai | Cả hai đúng | Lỗi PhoBERT int8 cũng bị LinearSVC sai |
|---|---:|---:|---:|---:|---:|
| Cảm xúc | 148 | 42 | 187 | 2.789 | 148 / 190 (77,9%) |
| Chủ đề | 234 | 83 | 199 | 2.650 | 234 / 317 (73,8%) |

## 5. Thiên lệch

### 5.1 Recall cảm xúc theo ô chủ đề × cảm xúc

Mô hình: PhoBERT đa nhiệm · int8 (bản triển khai). Chủ đề và cảm xúc đều là nhãn thật; recall là tỉ lệ câu được đoán đúng cảm xúc trong ô. Ô hoặc lát có n < 10 gắn cờ "mẫu quá ít để kết luận".

| Chủ đề thật | Cảm xúc thật | n | Recall | Ghi chú |
|---|---|---:|---:|---|
| Giảng viên | Tiêu cực | 791 | 97,5% |  |
| Giảng viên | Trung tính | 74 | 35,1% |  |
| Giảng viên | Tích cực | 1.425 | 97,7% |  |
| Chương trình | Tiêu cực | 416 | 97,4% |  |
| Chương trình | Trung tính | 47 | 36,2% |  |
| Chương trình | Tích cực | 109 | 89,0% |  |
| Cơ sở vật chất | Tiêu cực | 138 | 100,0% |  |
| Cơ sở vật chất | Trung tính | 2 | 50,0% | mẫu quá ít để kết luận |
| Cơ sở vật chất | Tích cực | 5 | 80,0% | mẫu quá ít để kết luận |
| Khác | Tiêu cực | 64 | 85,9% |  |
| Khác | Trung tính | 44 | 63,6% |  |
| Khác | Tích cực | 51 | 82,4% |  |

### 5.2 Baseline chỉ-dùng-chủ-đề

Quy tắc: chủ đề thật là giảng viên → tích cực, còn lại → tiêu cực, không đọc câu. Ô theo prior: chủ đề giảng viên + cảm xúc tích cực, hoặc chủ đề khác + cảm xúc tiêu cực (chỗ quy tắc chỉ-dùng-chủ-đề đoán đúng). Ô ngược prior: mọi câu còn lại, gồm cả trung tính.

| Cách đoán cảm xúc | Accuracy | macro-F1 |
|---|---:|---:|
| Chỉ dùng chủ đề thật | 64,5% | 42,5% |
| PhoBERT đa nhiệm · int8 (bản triển khai) | 94,0% | 82,1% |
| TF-IDF + LinearSVC | 89,4% | 73,4% |

| Accuracy cảm xúc của | Ô theo prior | Ô ngược prior | Chênh lệch (điểm) |
|---|---:|---:|---:|
| PhoBERT đa nhiệm · int8 (bản triển khai) | 97,4% (n = 2.043) | 87,8% (n = 1.123) | +9,6 |
| TF-IDF + LinearSVC | 93,3% (n = 2.043) | 82,4% (n = 1.123) | +10,9 |

### 5.3 Lát "thầy" so với "cô"

Cảm xúc của PhoBERT đa nhiệm · int8 (bản triển khai). Token tách theo khoảng trắng và so khớp nguyên token. Câu có cả "thầy" lẫn "cô" (n = 15) bị bỏ khỏi cả hai lát. Lát thầy / cô chỉ là proxy vì dữ liệu không có thuộc tính nhân khẩu nào; chênh lệch chưa qua kiểm định thống kê. macro-F1 của một lát tính trên các lớp có mặt trong nhãn thật hoặc dự đoán của lát; toàn tập có đủ các lớp nên trùng macro-F1 của bảng chính.

| Lát | n | Tiêu cực thật | Tiêu cực dự đoán | Accuracy | macro-F1 |
|---|---:|---:|---:|---:|---:|
| Có token "thầy" | 759 | 35,0% | 36,8% | 94,7% | 75,1% |
| Có token "cô" | 284 | 19,7% | 21,1% | 94,0% | 72,5% |
| Hiệu số (thầy − cô), điểm |  | +15,3 | +15,6 | +0,7 | +2,7 |
