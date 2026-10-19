Kiểm định ghép cặp trên tập test (n = 3.166). McNemar chính xác kiểm accuracy trên các câu hai mô hình bất đồng; ΔmF1 là hiệu macro-F1 (A − B, điểm phần trăm) với khoảng tin cậy 95% theo bootstrap ghép cặp (2.000 vòng, hạt giống 0, cùng chỉ số câu cho hai mô hình); 'có ý nghĩa' nghĩa là p < 0,05 (McNemar) hoặc khoảng tin cậy không chứa 0 (bootstrap).

### Bản triển khai (int8) so với baseline TF-IDF

| Tác vụ | So sánh (A so với B) | Acc A | Acc B | A đúng, B sai | A sai, B đúng | p McNemar | ΔmF1 (điểm) | KTC 95% | p bootstrap | Có ý nghĩa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cảm xúc | PhoBERT đa nhiệm · int8 (bản triển khai) so với TF-IDF + LinearSVC | 94,0 | 89,4 | 187 | 42 | < 0,0001 | 8,61 | [5,63; 11,50] | < 0,0001 | acc: có · mF1: có |
| Chủ đề | PhoBERT đa nhiệm · int8 (bản triển khai) so với TF-IDF + LinearSVC | 90,0 | 86,3 | 199 | 83 | < 0,0001 | 8,02 | [5,55; 10,55] | < 0,0001 | acc: có · mF1: có |
| Cảm xúc | PhoBERT đa nhiệm · int8 (bản triển khai) so với TF-IDF + LogisticRegression | 94,0 | 88,5 | 221 | 48 | < 0,0001 | 7,48 | [4,66; 10,17] | < 0,0001 | acc: có · mF1: có |
| Chủ đề | PhoBERT đa nhiệm · int8 (bản triển khai) so với TF-IDF + LogisticRegression | 90,0 | 86,1 | 212 | 89 | < 0,0001 | 7,61 | [5,06; 10,25] | < 0,0001 | acc: có · mF1: có |
| Cảm xúc | PhoBERT đa nhiệm · int8 (bản triển khai) so với TF-IDF + ComplementNB | 94,0 | 85,8 | 303 | 44 | < 0,0001 | 16,92 | [13,49; 20,15] | < 0,0001 | acc: có · mF1: có |
| Chủ đề | PhoBERT đa nhiệm · int8 (bản triển khai) so với TF-IDF + ComplementNB | 90,0 | 81,8 | 347 | 88 | < 0,0001 | 13,93 | [11,39; 16,61] | < 0,0001 | acc: có · mF1: có |

### Bản triển khai (int8) so với PhoBERT FP32 cùng run

| Tác vụ | So sánh (A so với B) | Acc A | Acc B | A đúng, B sai | A sai, B đúng | p McNemar | ΔmF1 (điểm) | KTC 95% | p bootstrap | Có ý nghĩa |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Cảm xúc | PhoBERT đa nhiệm · int8 (bản triển khai) so với PhoBERT đa nhiệm · FP32 (seed 42) | 94,0 | 93,9 | 8 | 5 | 0,5811 | 0,24 | [-0,75; 1,35] | 0,5860 | acc: không · mF1: không |
| Chủ đề | PhoBERT đa nhiệm · int8 (bản triển khai) so với PhoBERT đa nhiệm · FP32 (seed 42) | 90,0 | 89,8 | 24 | 17 | 0,3489 | 0,67 | [-0,15; 1,57] | 0,1220 | acc: không · mF1: không |
