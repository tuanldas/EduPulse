| Mô hình | Cảm xúc · Acc | Cảm xúc · mF1 | F1 trung tính | Chủ đề · Acc | Chủ đề · mF1 | F1 «khác» | ECE cảm xúc | ECE chủ đề |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + LinearSVC | 90.5 | 78.1 | 50.0 | 87.7 | 76.4 | 46.7 | 0.013 | 0.016 |
| TF-IDF + LogisticRegression | 89.5 | 76.8 | 46.6 | 87.2 | 75.4 | 47.9 | 0.019 | 0.017 |
| TF-IDF + ComplementNB | 87.3 | 65.7 | 18.8 | 83.1 | 67.0 | 29.7 | 0.018 | 0.048 |

Acc/mF1/F1 từng lớp là bản thô (không hiệu chuẩn). ECE ở bảng là của bản hiệu chuẩn, đo trên chính tập val dùng để fit nhiệt độ nên hơi lạc quan; số cuối cùng đo trên test.
