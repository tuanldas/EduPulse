Kết quả trên tập test (n = 3.166); phần trăm. Trong ngoặc vuông là khoảng tin cậy 95% theo bootstrap (2.000 vòng chọn lại câu, hạt giống cố định).

### Cảm xúc

| Mô hình | Acc | mF1 | wF1 | F1 Tiêu cực | F1 Trung tính | F1 Tích cực | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 89,4 [88,3–90,5] | 73,4 [70,7–75,9] | 88,8 [87,6–89,9] | 91,3 | 37,1 | 92,0 | 0,011 |
| TF-IDF + LogisticRegression | 88,5 [87,5–89,7] | 74,6 [72,2–77,0] | 88,7 [87,5–89,9] | 90,6 | 41,1 | 91,9 | 0,015 |
| TF-IDF + ComplementNB | 85,8 [84,6–87,1] | 65,1 [62,6–67,8] | 84,4 [83,0–85,8] | 87,0 | 19,6 | 88,8 | 0,017 |
| PhoBERT đa nhiệm · FP32 (seed 42) | 93,9 [93,1–94,8] | 81,8 [79,1–84,3] | 93,4 [92,5–94,4] | 95,6 | 54,2 | 95,6 | 0,015 |
| PhoBERT đa nhiệm · int8 (bản triển khai) | 94,0 [93,1–94,8] | 82,1 [79,3–84,6] | 93,5 [92,6–94,5] | 95,8 | 54,8 | 95,6 | 0,014 |

### Chủ đề

| Mô hình | Acc | mF1 | wF1 | F1 Giảng viên | F1 Chương trình | F1 Cơ sở vật chất | F1 Khác | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 86,3 [85,2–87,5] | 73,2 [70,8–75,8] | 85,7 [84,5–86,9] | 92,4 | 70,5 | 90,9 | 39,1 | 0,015 |
| TF-IDF + LogisticRegression | 86,1 [84,9–87,2] | 73,6 [71,2–75,9] | 85,7 [84,5–86,9] | 92,3 | 70,0 | 89,9 | 42,4 | 0,010 |
| TF-IDF + ComplementNB | 81,8 [80,5–83,1] | 67,3 [64,6–69,8] | 81,7 [80,4–83,0] | 89,1 | 65,5 | 83,7 | 31,0 | 0,044 |
| PhoBERT đa nhiệm · FP32 (seed 42) | 89,8 [88,7–90,8] | 80,6 [78,4–82,8] | 89,5 [88,5–90,6] | 94,3 | 78,6 | 94,2 | 55,2 | 0,026 |
| PhoBERT đa nhiệm · int8 (bản triển khai) | 90,0 [89,0–91,0] | 81,3 [79,0–83,4] | 89,7 [88,6–90,8] | 94,4 | 78,7 | 94,8 | 57,0 | 0,023 |

Baseline: F1 từ nhãn thô, ECE từ xác suất đã hiệu chuẩn nhiệt độ trên validation. PhoBERT FP32 tính ECE ở T = 1; bản int8 dùng T đã đóng băng. Chỉ có một run PhoBERT đa nhiệm (seed 42) nên không có trung bình theo seed: sai khác giữa các mô hình xem khoảng tin cậy và kiểm định ghép cặp ở significance.md.
