# Mô phỏng vòng người duyệt (HITL) trên validation

> **Đây là mô phỏng.** Nhãn gốc của UIT-VSFC đóng vai nhãn do người duyệt sửa (coi như đúng); không có người duyệt thật. Không dùng tập test.

## Thiết kế

- Dữ liệu: train 11.426 câu, validation 1.583 câu (chuẩn hoá bản 1). Không dùng tập test.
- Chia validation cố định (seed 0, phân tầng theo nhãn cảm xúc): nửa duyệt **R** = 791 câu, nửa đánh giá **E** = 792 câu, rời nhau.
- Mô hình «trước»: TF-IDF + LinearSVC với siêu tham số đã đóng băng của bản triển khai (cảm xúc C = 3 · chủ đề C = 0,3), fit chỉ trên train bằng `ml.retrain.retrain_baseline` với phản hồi rỗng; nhãn dự đoán trên toàn validation (1.583 câu) trùng 100% với bản đã lưu `models/baseline/*_linearsvc.skops`.
- Gắn cờ trên R theo ngân sách 10%: ⌈10% · 791⌉ = 80 câu điểm thấp nhất. Độ tin cậy là xác suất lớn nhất của softmax(decision_function); điểm cờ là phân vị thấp hơn của hai tác vụ, phân vị tính ngay trong R (`edupulse.scoring`).
- Người duyệt mô phỏng: 80 câu cờ nhận nhãn gốc của cả hai tác vụ.
- Ba nhánh «sau», mỗi nhánh retrain bằng đúng `retrain_baseline` và cùng siêu tham số: **cờ** (train + câu cờ); **ngẫu nhiên** (đối chứng: train + cùng số câu chọn ngẫu nhiên trong R, lặp 20 lần với seed 0..19); **trần trên** (train + toàn bộ R).
- Đánh giá mọi nhánh trên E (n = 792); kiểm định ghép cặp: McNemar chính xác (độ chính xác) và bootstrap 2.000 vòng (hiệu macro-F1).

## Kết quả trên nửa đánh giá E (n = 792)

Phần trăm. Acc = accuracy, mF1 = macro-F1, wF1 = weighted-F1; nhánh ngẫu nhiên là trung bình ± độ lệch chuẩn mẫu của các lần lặp.

| Nhánh | Cảm xúc · Acc | Cảm xúc · mF1 | Cảm xúc · wF1 | F1 trung tính | Chủ đề · Acc | Chủ đề · mF1 | Chủ đề · wF1 | F1 «khác» |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Trước (chỉ train) | 90,5 | 76,6 | 90,2 | 45,2 | 87,5 | 77,6 | 86,9 | 53,2 |
| Sau · câu cờ (80 câu) | 90,9 | 77,1 | 90,6 | 45,9 | 87,5 | 77,3 | 86,8 | 51,3 |
| Sau · ngẫu nhiên (80 câu, TB ± ĐLC, 20 lần) | 90,5 ± 0,2 | 76,7 ± 0,3 | 90,2 ± 0,2 | 45,4 ± 0,9 | 87,6 ± 0,2 | 77,5 ± 0,5 | 86,9 ± 0,2 | 52,6 ± 1,1 |
| Trần trên · duyệt hết R (791 câu) | 90,7 | 77,1 | 90,5 | 46,2 | 87,0 | 76,5 | 86,3 | 51,3 |

## Kiểm định trên E

Δ macro-F1 = macro-F1(A) − macro-F1(B), tính bằng điểm phần trăm; khoảng tin cậy 95% theo phân vị của 2.000 lần bootstrap ghép cặp (cùng chỉ số câu cho hai mô hình); McNemar chính xác hai phía trên đúng/sai từng câu (b = A đúng, B sai; c = A sai, B đúng).

| Tác vụ | So sánh (A so với B) | Δ macro-F1 (điểm) | KTC 95% của Δ | p bootstrap | A đúng, B sai | A sai, B đúng | p McNemar |
|---|---|---:|---:|---:|---:|---:|---:|
| Cảm xúc | Cờ so với trước | +0,5 | [-0,1; +1,3] | 0,079 | 4 | 1 | 0,375 |
| Cảm xúc | Trần trên so với trước | +0,5 | [-1,3; +2,9] | 0,701 | 7 | 6 | 1,000 |
| Chủ đề | Cờ so với trước | -0,4 | [-1,5; +0,4] | 0,603 | 2 | 2 | 1,000 |
| Chủ đề | Trần trên so với trước | -1,2 | [-2,8; -0,1] | 0,028 | 1 | 5 | 0,219 |

Nhánh cờ so với đối chứng ngẫu nhiên cùng cỡ (p thực nghiệm một phía = (1 + số lần ngẫu nhiên ≥ nhánh cờ) / (số lần + 1)):

| Tác vụ | mF1 nhánh cờ | mF1 ngẫu nhiên (TB ± ĐLC) | Nhỏ nhất – lớn nhất | Hiệu cờ − ngẫu nhiên (điểm) | Số lần cờ ≥ ngẫu nhiên | p thực nghiệm (một phía) |
|---|---:|---:|---:|---:|---:|---:|
| Cảm xúc | 77,1 | 76,7 ± 0,3 | 76,3 – 77,5 | +0,4 | 18/20 | 0,143 |
| Chủ đề | 77,3 | 77,5 ± 0,5 | 76,3 – 78,1 | -0,2 | 5/20 | 0,810 |

## Hiệu quả gắn cờ trên R (n = 791)

| Chỉ số trên R (mô hình «trước») | Cảm xúc | Chủ đề | Cả hai tác vụ |
|---|---:|---:|---:|
| Accuracy phần được cờ | 65,0 | 65,0 | 47,5 |
| Accuracy phần còn lại | 93,2 | 90,4 | 84,8 |
| Số câu sai trong R | 76 | 96 | - |
| Số câu sai nằm trong phần cờ | 28 | 28 | - |
| Lỗi bị bắt (tỉ lệ lỗi của R nằm trong phần cờ) | 36,8 | 29,2 | - |
| Tỉ lệ câu sai ít nhất một tác vụ: phần cờ / phần còn lại | - | - | 52,5 / 15,2 |

## Cách đọc

Đây là mô phỏng nhỏ: R chỉ có 791 câu, ngân sách 10% là 80 câu (bằng 0,7% số câu train), nên hiệu ứng kỳ vọng rất nhỏ và có thể không có ý nghĩa thống kê. Kết quả được nêu đúng như đo được, kể cả khi không có cải thiện.

- **Cảm xúc**, nhánh cờ so với «trước»: macro-F1 tăng 0,5 điểm (KTC 95% [-0,1; +1,3], p bootstrap 0,079; McNemar p 0,375): không có ý nghĩa thống kê.
- **Chủ đề**, nhánh cờ so với «trước»: macro-F1 giảm 0,4 điểm (KTC 95% [-1,5; +0,4], p bootstrap 0,603; McNemar p 1,000): không có ý nghĩa thống kê.
- **Cảm xúc**, cờ so với ngẫu nhiên cùng cỡ: macro-F1 nhánh cờ cao hơn trung bình ngẫu nhiên 0,4 điểm; nhánh cờ không thấp hơn ngẫu nhiên ở 18/20 lần, p thực nghiệm một phía 0,143. Độ lệch chuẩn giữa các lần ngẫu nhiên là 0,3 điểm, cho biết nhiễu của chính phép đo.
- **Chủ đề**, cờ so với ngẫu nhiên cùng cỡ: macro-F1 nhánh cờ thấp hơn trung bình ngẫu nhiên 0,2 điểm; nhánh cờ không thấp hơn ngẫu nhiên ở 5/20 lần, p thực nghiệm một phía 0,810. Độ lệch chuẩn giữa các lần ngẫu nhiên là 0,5 điểm, cho biết nhiễu của chính phép đo.
- **Cảm xúc**, trần trên (duyệt hết R): macro-F1 tăng 0,5 điểm (KTC 95% [-1,3; +2,9], p bootstrap 0,701; McNemar p 1,000): không có ý nghĩa thống kê. Trần trên dùng toàn bộ nhãn của R nên là mốc tham chiếu cho nhánh cờ.
- **Chủ đề**, trần trên (duyệt hết R): macro-F1 giảm 1,2 điểm (KTC 95% [-2,8; -0,1], p bootstrap 0,028; McNemar p 0,219): chỉ kiểm định bootstrap có ý nghĩa, bằng chứng yếu. Trần trên không cải thiện ở tác vụ này, nên mô phỏng không cho cơ sở kỳ vọng nhánh cờ cải thiện.
- Quy tắc nhận mô hình mới của `ml.retrain`, áp lên số đo trên E (nửa validation chưa dùng để huấn luyện) thay vì toàn validation, với nhánh cờ: TỪ CHỐI (chủ đề tụt 0,0037 (0,7763 -> 0,7727), vượt dung sai 0,0000).
- Gắn cờ 10% số câu của R bắt được 36,8% lỗi cảm xúc và 29,2% lỗi chủ đề của mô hình «trước» (chọn ngẫu nhiên kỳ vọng bắt khoảng 10%).

## Giới hạn

- Nhãn gốc thay người duyệt: người thật có thể sửa sai hoặc không nhất quán, còn nhãn gốc của UIT-VSFC cũng có nhiễu (nhất là chủ đề «khác»); mô phỏng này vì vậy là ước lượng lạc quan về chất lượng nhãn.
- Chỉ LinearSVC (baseline TF-IDF) được retrain đầy đủ; mô hình triển khai là PhoBERT và phần PhoBERT chỉ smoke 20 bước, không phải retrain đầy đủ.
- Một lần chia R/E (seed 0) và một vòng duyệt; chưa lặp qua nhiều cách chia. E chỉ có 792 câu, trong đó lớp trung tính 36 câu và lớp «khác» 53 câu, nên F1 của hai lớp này nhiễu mạnh.
- Siêu tham số baseline đã được chọn trên toàn bộ validation (gồm cả E), nên số của «trước» hơi lạc quan; mọi nhánh dùng chung E nên so sánh trước/sau vẫn cùng điều kiện.
- Câu cờ được chọn theo độ tin cậy của LinearSVC (softmax của điểm quyết định, chưa hiệu chuẩn), không phải của PhoBERT đã hiệu chuẩn đang triển khai, nên tập câu cờ khác tập mà API sẽ cờ; vì chỉ xếp hạng theo phân vị trong R nên chỉ thứ hạng mới có nghĩa.
- Mô hình mới chỉ được chấm trên validation; tập test không được dùng ở đây.

## Smoke retrain PhoBERT (CPU, 20 bước)

Chạy `retrain_phobert` từ run đã triển khai `mtl-s42` với 80 câu cờ (nhãn gốc) làm phản hồi. Mỗi batch 32 câu gồm 16 phản hồi (lấy có hoàn lại) và 16 câu phát lại từ train; AdamW lr 2e-5, weight decay 0,01, clip 1,0. Dự đoán là FP32 bằng torch trên CPU (cả trước lẫn sau), không phải bản int8 ONNX đang triển khai; đo trên E (n = 792).

| Tác vụ | Acc trước | Acc sau | mF1 trước | mF1 sau | Δ mF1 (điểm) | Số nhãn đổi trên E |
|---|---:|---:|---:|---:|---:|---:|
| Cảm xúc | 95,3 | 94,8 | 86,2 | 86,0 | -0,2 | 17 |
| Chủ đề | 90,0 | 89,6 | 82,4 | 80,4 | -2,0 | 29 |

Thời gian: huấn luyện 29,5 giây; cả bước (gồm lưu checkpoint và dự đoán trước/sau) 43,5 giây. Loss tổng hai tác vụ: bước đầu 1,017, bước cuối 0,435.

Quy tắc nhận mô hình mới của `ml.retrain` áp lên số đo trên E (không phải toàn validation): TỪ CHỐI (cảm xúc tụt 0,0016 (0,8620 -> 0,8604); chủ đề tụt 0,0196 (0,8236 -> 0,8040), vượt dung sai 0,0000).

Đây chỉ là smoke chứng minh đường chạy `python -m ml.retrain --model phobert` chạy trọn trên CPU; 20 bước không kỳ vọng thay đổi đáng kể, và phép đo này không dùng để kết luận mô hình tốt lên hay xấu đi.
