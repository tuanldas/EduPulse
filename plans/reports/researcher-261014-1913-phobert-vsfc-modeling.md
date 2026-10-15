# Nghiên cứu mô hình: PhoBERT-base-v2 trên UIT-VSFC (EduPulse)

**Khuyến nghị:** fine-tune PhoBERT-base-v2 với RDRSegmenter (py_vncorenlp), fp32, lr 2e-5, bs 32, 4 epoch, max_len 128, CE thường (trọng số lớp hoà → bỏ); 1 encoder + 2 head (MTL) để deploy, STL 2 model làm đối chứng; baseline TF-IDF(word1-2 + char2-5) + LinearSVC. Kỳ vọng test ≈ sentiment 94 / 83, topic 89 / 80 (acc / macro-F1). Bắt buộc lowercase + chuẩn hoá trước khi tách từ, và guard cho input không dấu.

Ngày 2026-10-14. "val" = tôi tự đo trên M1 Pro (train→val, chưa chạm test; 1 seed trừ khi ghi khác). "test" = số trích từ paper. Script/log: scratchpad `r1/` (ngoài repo). Bẫy chính: §2 (py_vncorenlp: wget/chdir/":)"), §3 (dấu thanh, emoticon trước segment, lowercase, không dấu), §6 (transformers v5 API, MPS), §10 (AGPL-3.0).
Logits val thật (cho hiệu chuẩn/ngưỡng/AOPC của R3): `/private/tmp/claude-501/-Users-admin-Desktop-Sync-Codes-tuanldas-EduPulse/6fd0f54f-0fd0-432c-8694-e708c2ec6026/scratchpad/r1/runs/{s,t,m}_rdr_none_42.val_logits.npz` (khoá `sentiment`/`topic` + `y_*`, thứ tự = val chuẩn). Checkpoint chưa giữ (12 phút/run nếu cần).

## 1. Benchmark đã công bố (Q1)
Split chuẩn 11426/1583/3166, số **test**. "F1" của các paper = macro-F1 (acc≈94 mà F1≈83; Doan&Luu 2022 Bảng VII: XLM-R acc 94.13 / macro 83.09 / weighted 93.74).

| Model (nguồn, năm) | Sentiment acc / mF1 | Topic acc / mF1 |
|---|---|---|
| PhoBERT-base (BamiBERT, arXiv 2607.02259, 2026) | 94.10 / 83.27 | 89.24 / 79.80 |
| BamiBERT · XLM-R-base · ViSoBERT · ViDeBERTa (cùng paper) | 93.86/83.41 · 93.56/82.20 · 93.15/81.49 · 87.86/70.74 | 89.34/79.90 · 89.18/79.56 · 88.80/79.86 · 83.94/66.49 |
| CafeBERT · XLM-R-large · PhoBERT-large · mBERT (VLUE, arXiv 2403.15882, 2024) | 94.16/84.29 · 94.13/83.70 · 93.43/82.81 · 91.88/78.67 | 89.07/79.82 · 88.57/79.20 · 88.22/78.08 · 87.93/77.28 |
| TF-IDF 1-3gram SVM · NB (arXiv 2301.00418, TB 100 lần) | 89.4/73.0 · 87.0/60.3 | không có |
| Paper gốc 2018 (Maxent) | 87.94 = weighted-F1 (Doan&Luu, arXiv 2210.02063) | ">84% F1", không rõ loại |

- Kỳ vọng thực tế PhoBERT-v2: sentiment ≈ 94 / 83±1, topic ≈ 89 / 80±1 (các encoder mạnh chụm trong ~1 điểm, trong nhiễu). TF-IDF+SVM thấp hơn ~4-5 điểm acc, ~10 điểm macro-F1 (lớp trung tính F1 ~0.5 vs ~0.65).
- SMTCE (arXiv 2209.10482) KHÔNG chứa UIT-VSFC (4 task khác). Các paper dùng PhoBERT-base; chưa thấy số publish cho phobert-base-v2.
- Số val của tôi cao hơn test ~3-4 điểm mF1 (87.0 vs ~83) → đừng so val với bảng này. Trần topic do nhiễu nhãn (IAA topic 71%, sentiment 91% theo README dataset).

## 2. Word segmentation (Q2): chọn RDRSegmenter (py_vncorenlp 0.1.4)
Xếp hạng: 1) RDRSegmenter · 2) pyvi (không Java; đồng thuận 98.7% với RDR; CHƯA fine-tune thử) · 3) không tách từ (không cần gì thêm; đã đo −0.6 acc / −2 mF1) · 4) underthesea (không hơn pyvi, chậm hơn 5x).
- VinAI model card v2: "INPUT TEXT MUST BE ALREADY WORD-SEGMENTED"; pretrain đã chuẩn hoá dấu thanh + tách từ bằng RDRSegmenter → nên dùng đúng segmenter này. v2: 135M param, 20GB wiki/news + 120GB OSCAR-2301, tối đa 256 token.
- Paper (UIT-VSFC split chuẩn, LM S4+BPE pretrain khớp segmenter, TB 100 lần; arXiv 2301.00418): mF1 77.84 RDR · 77.74 pyvi · 77.26 underthesea · 77.17 uitnlp · 75.82 không tách; với TF-IDF+SVM/NB tách từ không giúp (SVM 72.96 RDR vs 73.48 không). Không phải PhoBERT.
- Tôi fine-tune PhoBERT-v2 (sentiment, val, seed 42, 4 epoch): RDR 95.26 / 87.01 · không tách (câu gốc dạng âm tiết, đã áp bảng dấu thanh) 94.63 / 84.91 (neutral F1 68.2 vs 62.9) · McNemar p=0.14 (b=14, c=24) → cùng chiều paper (+~2 mF1) nhưng chưa đạt ý nghĩa trên val.
- Đo local 16,175 câu: RDR 6.7k câu/s (JVM khởi động 0.9s), pyvi 15k, underthesea 3.2k → tốc độ không là vấn đề. Biên từ khớp RDR: pyvi 98.7% F1, underthesea 97.8%. `<unk>` sau BPE (~217k token): RDR 2, pyvi 43, underthesea 47.
- RDRSegmenter tự đổi dấu thanh hòa→hoà, khóa→khoá (đã kiểm chứng); pyvi/underthesea/không tách thì không → tự áp bảng VinAI (mục 3).
- Đóng gói RDR: JRE≥8 (headless đủ) + pyjnius 1.9.0 (wheel cp311–cp314, manylinux_2_28 và macOS universal2) + `VnCoreNLP-1.2.jar` 27MB + `models/wordsegmenter/{vi-vocab,wordsegmenter.rdr}` 0.65MB (chỉ cần wseg). Bẫy đã gặp: `download_model()` gọi `wget` qua os.system (máy này không có wget; image slim cũng thiếu) → tự curl 3 file; `VnCoreNLP()` chdir sang save_dir (làm hỏng đường dẫn tương đối trong script của tôi); JVM chỉ khởi 1 lần/process → `@st.cache_resource`; `word_segment()` trả LIST câu (":)" làm tách đôi câu) → join bằng space. Đã thử 8 thread×200 câu + 20 thread ngắn: không crash.
- Không có port RDR thuần Python (đã tìm PyPI/GitHub). Dự phòng không Java: pyvi 0.1.1 (2021; thuần Python + scikit-learn + sklearn-crfsuite; SyntaxWarning trên Py3.12).
- Tokenizer: AutoTokenizer trả PhobertTokenizer chậm (transformers 5.19 không dùng tokenizer.json; 16k câu = 0.6s); `model_max_length`=1e30 → luôn truyền `max_length`; pad id 1; emoji → `<unk>`.

## 3. Chuẩn hoá input thô & token đặc biệt (Q3)
Thứ tự (đã thử với RDRSegmenter, prototype `r1/normalize_proto.py`): NFC → PII (tên→`wzjwz1`, URL/email/SĐT xoá) → lowercase → bảng dấu thanh VinAI → map emoticon/dấu → gộp ký tự lặp → RDRSegmenter (tự tách dấu câu dính chữ: "hay." → "hay ."; join câu bằng space; mỗi câu 1 dự đoán là khớp đơn vị huấn luyện).
- Data: 100% lowercase, NFC, dấu câu tách bằng space, không URL/email, val không trùng câu nào với train. 284 câu (1.8%) dùng dấu thanh kiểu cũ (hòa 94, họa 52, khóa 48, khỏe 13…) trong khi vocab PhoBERT-v2 chỉ có kiểu mới (khoá/hoà/khoẻ/thuỷ…, 0 entry kiểu cũ) → áp bảng 15 cặp VinAI: òa→oà óa→oá ỏa→oả õa→oã ọa→oạ òe→oè óe→oé ỏe→oẻ õe→oẽ ọe→oẹ ùy→uỳ úy→uý ủy→uỷ ũy→uỹ ụy→uỵ.
- Bảng ánh xạ, SUY LUẬN từ ngữ cảnh + nhãn (không có tài liệu gốc). Token đặc biệt chỉ xuất hiện ở 1-2% câu (train 1.7%, val 1.1%, test 1.9%) → ánh xạ lệch ít ảnh hưởng (xem thử skew bên dưới):

| Raw | Token trong data (số câu) | Tin cậy |
|---|---|---|
| ":" đứng riêng; "11:55" | `doubledot` (117, 86% tiêu cực); `11doubledot55` | cao |
| "v.v" | `vdotv` (12) | cao |
| "..." ; "uit.edu.vn", "3.0", ".net" | `dotdotdot` (6) ; `uitdotedudotvn`, `3dot0`, `dotnet` | cao |
| ":)" · ":))" · ":(" | `colonsmile` (39) · `colonsmilesmile` (18) · `colonsad` (13, 100% tiêu cực) | trung bình |
| ":D" · "<3" ":*" · ":v" · ":o" | `colonbigsmile` (7) · `colonlove` (13) · `coloncontemn` (11) · `colonsurprise` (2) | thấp |
| tên người | `wzjwz<N>`, mỗi tên một số riêng (326 loại, 302 câu) | cao |
- Emoticon/emoji khác → xoá (emoji thành `<unk>`). Giữ teencode (ko, j, z… có sẵn trong data). Gộp ký tự lặp chỉ với chữ cái ("quáaaa"), KHÔNG với số ("2000"); gộp "!!!" thành "!".
- `doubledot` mang lệch nhãn (86% tiêu cực vì ":" thường mở đầu danh sách phàn nàn): vẫn map để khớp train, đừng xoá.
- Thử skew (val sentiment, checkpoint CE trọng số √ seed 42, giả lập raw từ câu val: viết hoa đầu câu, dấu câu dính chữ, ":)" ":" "v.v" dạng thật; `r1/skew_test.py`): pipeline đầy đủ cho kết quả GIỐNG HỆT dạng train (acc 95.33 / mF1 86.76); bỏ lowercase −0.38 acc / −0.68 mF1 (6 câu đổi nhãn); bỏ map emoticon/":"/"v.v" → 0 câu đổi nhãn (val chỉ có 17 câu đặc biệt, thiếu sức mạnh thống kê) → lowercase là bước bắt buộc, map token rẻ nên vẫn giữ. Gõ KHÔNG DẤU: 82.25 / 69.58 (−13 acc / −17 mF1) → cần guard (tỉ lệ ký tự có dấu quá thấp → cảnh báo / gắn cờ "cần xem xét").
- XAI: giải thích theo từ RDR (có "_") cho highlight gọn; đổi "_"→" " khi hiển thị.

## 4. Mất cân bằng lớp (Q4): dùng CE thường, không cần trọng số
Train: sentiment 5325/458/5643 (neutral 4.0%); topic 8166/2201/497/562 (CSVC 4.3%, khác 4.9%). Lớp khó thật sự là neutral (F1 0.65-0.68) và "khác" (0.56-0.62); CSVC hiếm nhưng dễ (F1 0.93-0.95) → trọng số theo tần suất không nhắm đúng chỗ. Đo thật (val, seed 42, 4 epoch; acc / mF1 / F1 lớp khó / ECE):

| Loss | Sentiment (neutral) | Topic ("khác") |
|---|---|---|
| CE | 95.26 / 87.01 / 68.2 / 0.028 | 90.02 / 81.54 / 59.1 / 0.050 |
| CE trọng số w∝√(N/(K·n_c)): sent [0.85, 2.88, 0.82], topic [0.59, 1.14, 2.40, 2.25] | 95.33 / 86.76 / 67.2 / 0.028 | 89.83 / 81.74 / 59.4 / 0.045 |
McNemar CE vs trọng số: p=1.00 (sentiment), p=0.74 (topic) → hoà. Trọng số chỉ giúp hội tụ nhanh hơn ở epoch 1 (sentiment mF1 85.3 vs 81.6; topic 80.7 vs 79.2) rồi bằng nhau ở epoch 4.
- Chưa thử focal loss (thiết kế cho tỉ lệ ~1:1000 và thêm siêu tham số γ; ở đây 1:12) và oversampling (lặp mẫu, dễ overfit); survey Henning et al. EACL 2023 (arXiv 2210.04675) không có phương pháp thắng chung → phải benchmark; với kết quả hoà ở trên thì không đáng thêm độ phức tạp (YAGNI). Trọng số "balanced" mạnh (8x cho neutral) chưa thử.
- Chỉnh bias lớp hậu kỳ trên val (logit adjustment, Menon ICLR 2021) KHÔNG đáng tin ở cỡ val này: cross-fit 2 nửa val cho −1.4…+1.3 mF1 tuỳ run → nhiễu, bỏ. Sklearn: class_weight xem mục 7.
- Hiệu chuẩn: CE fine-tune đã gần hiệu chuẩn (ECE 0.007–0.05); temperature scaling 1 tham số fit trên logits val (T 1.0–1.25) đưa ECE về 0.006–0.037 (cross-fit). Trọng số √ không làm xấu ECE.

## 5. Single-task vs multi-task (Q5): chọn MTL để deploy, giữ STL làm đối chứng
Đo thật (val, seed 42, cùng siêu tham số; acc / mF1):

| | Sentiment | Topic |
|---|---|---|
| STL (2 model riêng) | 95.26 / 87.01 | 90.02 / 81.54 |
| MTL (1 encoder + 2 head tuyến tính trên `<s>`, loss = CE_s + CE_t, chọn epoch theo mean mF1) | 94.88 / 85.55 | 90.40 / 82.72 |
(Chỉ 1 seed mỗi ô, bootstrap CI 95% của mF1 ≈ ±3 điểm/ô; lượt seed 43 bị huỷ khi máy sleep nên chưa có std.) McNemar exact STL vs MTL: p=0.31 (sentiment), p=0.46 (topic) → không khác biệt có ý nghĩa; chênh ±1.5 mF1 ngược chiều nhau ở 2 task; MTL hiệu chuẩn tốt hơn (ECE 0.007 / 0.027 vs STL 0.028 / 0.050). Bằng chứng ngoài: lợi/hại của MTL phụ thuộc độ liên quan task, dung lượng chia sẻ, tối ưu (Hazy Research 2020; Standley ICML 2020: chia sẻ giảm chi phí suy luận nhưng có thể giảm accuracy) → phải tự benchmark với STL, đúng như vừa làm. Không tìm thấy paper MTL trên UIT-VSFC.
- Chi phí (đo): train MTL 11.7 phút vs STL 2 task ≈ 25 phút. CPU 2 thread (M1 Pro): 52 ms/câu/model (bs1), 196 câu/s (bs32) → STL 104 ms vs MTL 52 ms/câu. RAM macOS (RSS): 1 model ≈ 0.8GB, 2 model ≈ 1.15GB (+80MB JVM); Linux dự kiến cao hơn ~30-50% (chưa đo). Artifact MTL 537.7MB vs 2×~540MB. torch int8 dynamic (qnnpack): 24 ms bs1 nhưng chậm hơn ở bs32 (hướng ORT int8 và accuracy: xem hai gạch cuối).
- XAI occlusion theo từ (R3: n_từ+1 lần forward): MTL cho cả 2 head trong 1 forward → rẻ gấp đôi STL; export ONNX 1 lần (2 output) cho hướng ORT int8 của báo cáo deploy.
- Độ phức tạp: STL = `AutoModelForSequenceClassification` + Trainer (~30 dòng), checkpoint riêng từng task. MTL = ~35 dòng PreTrainedModel + vòng train tay (~60 dòng) hoặc Trainer + `compute_loss`. BẪY đã gặp: PreTrainedModel tuỳ biến phải đặt `base_model_prefix="roberta"` VÀ thuộc tính `self.roberta` — tên khác thì TOÀN BỘ encoder bị khởi tạo ngẫu nhiên IM LẶNG (chỉ có bảng LOAD REPORT "MISSING"). Sau khi sửa: `save_pretrained`/`from_pretrained` chạy đúng (model.safetensors 537.7MB, đầu ra giống hệt sau reload).
- Xếp hạng khi host free chỉ có RAM ~1–3GB, CPU yếu (Streamlit Community Cloud: 0.69–2.7GB, ≤2 lõi; HF Space free không còn khả thi): 1) MTL + ORT int8 (trọng số ~284MB; RSS Linux ước 0.7–0.9GB, số của r2) · 2) MTL fp32 torch (540MB; ~1.0–1.3GB) · 3) STL×2 int8 · 4) STL×2 fp32 (1.08GB trọng số, ước ~1.5–1.8GB, sát trần 2.7GB và latency/chi phí XAI gấp đôi). Quy tắc: deploy MTL; nếu 3 seed trên val cho thấy MTL thấp hơn STL >1.5 mF1 ở task nào thì chuyển task đó sang STL.
- int8 dynamic KHÔNG làm giảm macro-F1 đáng kể trên dữ liệu tương tự: r2 đo trên UIT-VSFC sentiment test 3166 câu, checkpoint bên thứ ba (PhoBERT-base v1), ARM: FP32 0.8277 → ORT int8 MatMul-only per-channel 0.8325, ORT int8 mặc định 0.8205, torch `quantize_dynamic` 0.8247 (|Δ| ≤0.7, trong nhiễu ±0.5 vì neutral chỉ 167 mẫu). Tôi chưa tự đo trên checkpoint của mình, và x86 chưa đo → giữ cổng "mF1 dev giảm ≤1 điểm so với FP32".

## 6. Hyperparameter, Apple MPS, phiên bản (Q6)
- Tham chiếu: PhoBERT paper (Findings EMNLP 2020): AdamW lr 1e-5, bs 32, ≤30 epoch, dừng sớm sau 5 epoch không tiến bộ, chọn best val. BamiBERT paper: lưới lr {1e-5,2e-5,5e-5}, bs 32, 30 epoch, best val F1. HF card BaoNhan (split khác): lr 2e-5, bs 8, 3 epoch → mF1 test 83.25±0.73, acc 94.91 (3 seed).
- Tôi dùng: lr 2e-5 (không tune), bs 32, 4 epoch, warmup 10% + linear decay, wd 0.01, clip 1.0, fp32, max_len 128. Val mF1 theo epoch: sentiment STL 81.6→86.7→86.1→87.0; topic STL 79.2→79.4→79.6→81.5 (topic còn tăng ở epoch 4 → thử 5-6 epoch nếu còn thời gian).
- max_len: token PhoBERT sau RDR (gồm `<s>` `</s>`): mean 13.4, p95 30, p99 45, p99.9 72, max 136; >64: 0.22%, >128: 0.01% (2 câu) → max_len 128 + dynamic padding (không tốn thêm).
- Tốc độ M1 Pro (MPS, bs32, torch 2.14.1): batch ngẫu nhiên 0.63 s/step = 4.2 phút/epoch; gộp theo độ dài + pad_to_multiple_of=8 + fused AdamW 0.46 s/step ≈ 2.8–3.2 phút/epoch (357 step). Run thật 4 epoch: 11.7–12.5 phút/run (STL 1 task hoặc MTL). Chi phí chính là optimizer 135M tham số, không phải độ dài chuỗi.
- Mixed precision: autocast fp16 0.63 s/step (= fp32), bf16 0.76 (chậm hơn) → dùng fp32. PyTorch #196067 (đóng 2026-09-12; chưa xác minh bản vá có trong 2.14.1): `lerp_` sai với tensor fp16/bf16 + weight fp32 trên MPS → đừng ép model sang half/bf16 khi train bằng AdamW trên MPS.
- HF Trainer 5.19 trên MPS chạy được (smoke 256 mẫu: fp32/fp16/bf16 + group-by-length, tự nhận device mps). Không thấy issue mở về MPS+Trainer cho BERT-classification. v5 BỎ `warmup_ratio` → `warmup_steps=0.1`; BỎ `group_by_length` → `train_sampling_strategy="group_by_length"`.
- Bộ nhớ: peak MPS driver 7.0–9.1GB/run (bs32 train + bs128 eval), phẳng giữa các epoch khi pad_to_multiple_of=8 (≤16 shape) + `torch.mps.clear_graph_cache()` mỗi 100 step (torch≥2.13; Trainer: `torch_empty_cache_steps`; HF doc: MPS cache kernel theo shape, không evict). KHÔNG chạy 2 job song song trên 16GB: tôi đã gây swap thrash cả máy ~5 phút (một VM Virtualization đang chiếm ~7GB + 11GB nén). Đóng app khác hoặc giảm bs/eval bs nếu thiếu RAM.
- Colab T4 dự phòng: chưa đo; ước tính <1 phút/epoch với fp16 AMP (bug lerp_ chỉ ở MPS). Tách từ ở local rồi upload parquet → Colab không cần Java.
- Phiên bản: dùng Python 3.12 (numpy 2.5.4 và scipy 1.18.1 yêu cầu ≥3.12; pyjnius 1.9.0 ≥3.11; torch/transformers ≥3.10 → với 3.11 phải hạ numpy/scipy). Danh sách pin ở mục 9.

## 7. Baseline TF-IDF (Q7)
Lưới trên val (train→val; 7 bộ đặc trưng × 4 họ mô hình × 2 task, hyperparameter chọn trên val): LinearSVC > LR > ComplementNB ≈ MultinomialNB.

| Model (cấu hình tốt nhất) | Sentiment acc / mF1 (F1 neutral) | Topic acc / mF1 (F1 "khác") |
|---|---|---|
| LinearSVC (sent: word1-2+char2-5, C=1; topic: word1-3, C=1, balanced) | 91.79 / 79.56 (0.52) | 87.49 / 77.15 (0.49) |
| LogisticRegression (C=30; C=10 balanced) | 90.90 / 78.18 | 84.65 / 75.20 |
| ComplementNB (α=0.1 trên rdr; α=1.0) | 89.58 / 72.74 | 83.45 / 68.62 |
| MultinomialNB (α=0.03) | 88.44 / 71.76 | 83.51 / 69.86 |
- TF-IDF: word 1-2gram, min_df=2, sublinear_tf=True, token_pattern `(?u)\b\w+\b` (giữ từ 1 ký tự); thêm char_wb 2-5 (min_df=3): sentiment +1.3~1.8 mF1, topic −0.6 (trong nhiễu) → dùng 1 pipeline chung cho cả 2 task (DRY). Chênh do sublinear on/off, 1-2 vs 1-3gram, văn bản tách từ RDR vs âm tiết ≤0.8 điểm → không đáng tune.
- Siêu tham số quan trọng nhất là C: C nhỏ làm sập lớp thiểu số (neutral F1 0.10 ở C=0.03, 0.52 ở C=1). class_weight="balanced" làm đường cong phẳng hơn theo C nhưng sau khi tune C thì không hơn (sent 78.1 vs 79.6; topic 75.4 vs 76.6) → tune C∈{0.3,1,3} trên val, cw tuỳ chọn.
- SVC rbf: sentiment 92.10 / 76.13 (C=10), topic 87.24 / 74.49; ~90 giây/lần fit (LinearSVC <1 giây) và sập lớp neutral → bỏ. SVC kernel linear: chậm hơn, không hơn.
- Xác suất cho SVM: `CalibratedClassifierCV(LinearSVC(C=1), method="sigmoid"|"isotonic", cv=5)`: sentiment NLL 0.252/0.259, ECE 0.024/0.013 (LR C=30: 0.311/0.038; MNB ECE 0.094, CNB 0.082); topic NLL 0.37, ECE 0.03-0.04 (LR 0.48/0.068; NB 0.05-0.14). Hiệu chuẩn đổi ranh giới quyết định: mF1 sentiment 79.6 → 77.3–77.7. → báo cáo cả bản raw (cho F1) lẫn bản calibrated (cho xác suất).

## 8. Giao thức đánh giá (Q8)
- Đa seed khả thi: 4 epoch ≈ 12 phút/run trên M1 → 3 seed × 2 task ≈ 75 phút (STL) hoặc 38 phút (MTL). Đề xuất: chọn epoch/siêu tham số trên val (1 seed), rồi 3 seed (42,43,44) cho cấu hình cuối, báo cáo mean±std; baseline TF-IDF tất định (1 lần). Không retrain train+val (giữ câu chuyện sạch, test chạm 1 lần). MPS không đảm bảo tái lập từng bit → báo cáo phân phối, không báo 1 số.
- Độ nhiễu: bootstrap 95% CI của mF1 sentiment val cho 1 model = ±3 điểm (neutral chỉ 73 mẫu val / 167 test); std giữa seed ≈ 0.7 (HF card BaoNhan, 3 seed). So sánh model dùng kiểm định ghép cặp: McNemar exact (`scipy.stats.binomtest(b, b+c, 0.5)` trên số mẫu bất đồng b,c) + paired bootstrap cho ΔmF1 (McNemar chỉ kiểm accuracy). Ví dụ val: PhoBERT vs LinearSVC sentiment +7.5 mF1 [3.7, 11.2], p=2e-9; topic +5.0 [2.3, 7.4], p=4e-4; STL vs MTL p=0.31/0.46 (hoà); RDR vs không tách p=0.14.
- Báo cáo cuối: accuracy, macro-F1, weighted-F1, F1 từng lớp, confusion matrix (chuẩn hoá theo hàng), ECE; nêu rõ val > test ~3-4 mF1.
- Error analysis rẻ nhưng hữu ích (đã thử trên sentiment val, STL): neutral recall 60% (44/73; lỗi 12→neg, 17→pos), precision 79%; nhiều nhãn neutral đọc như lời khen ("sự nhiệt tình .", "cẩn thận .") → nhiễu nhãn; lỗi theo chủ đề: giảng viên 43, chương trình 23, khác 9, CSVC 0/70; câu >40 từ acc 90.9% (n=33) vs ~95%. Thêm: top lỗi tự tin cao (kiểm nhãn), nhóm câu có `doubledot`/emoticon, so lỗi TF-IDF vs PhoBERT, tương quan chéo task (CSVC→tiêu cực).

## 9. Đề xuất cấu hình (copy được)
```
segmenter    : py_vncorenlp==0.1.4 (annotators=["wseg"], max_heap_size="-Xmx512m") + pyjnius==1.9.0 + JRE>=8; tách từ 1 lần, cache parquet
preprocess   : NFC → PII → lower → bảng dấu thanh VinAI → map emoticon/":"/"v.v"/"..." (mục 3) → gộp ký tự lặp → RDR → " ".join(câu)
model        : vinai/phobert-base-v2; head = Dropout(0.1) → Linear(768, n) trên hidden state <s>
architecture : MTL (1 encoder + head_sentiment[3] + head_topic[4]), loss = CE_s + CE_t; STL 2 model làm đối chứng trong báo cáo
max_len      : 128 (truncation; dynamic padding, pad_to_multiple_of=8)
optimizer    : AdamW fused, lr 2e-5, weight_decay 0.01 (trừ bias/LayerNorm), warmup 10% + linear decay, clip_grad_norm 1.0
train        : bs 32 gộp theo độ dài, 4 epoch (thử 5-6 cho topic), fp32; chọn epoch theo macro-F1 val (MTL: trung bình 2 task); seeds 42/43/44
loss         : CrossEntropy thường (không class weight, không label smoothing)
calibration  : temperature scaling 1 tham số fit trên logits val (T≈1.0–1.25, ECE → ≤0.04)
review flag  : p_max sau T-scaling (STL, val): sentiment ≤0.95 → gắn cờ ~10% câu, bắt ~53% lỗi; topic ≤0.81/≤0.87 → ~10%/~15% câu, bắt 46%/61% lỗi (sentiment: ngưỡng quen dùng 0.6-0.7 chỉ gắn cờ 0.8-1.6% câu, bắt ≤16% lỗi)
baseline     : Tfidf(word 1-2, min_df=2, sublinear_tf, token_pattern=r"(?u)\b\w+\b") ⊕ Tfidf(char_wb 2-5, min_df=3, sublinear_tf) → LinearSVC(C=1)
               [+ CalibratedClassifierCV(method="sigmoid", cv=5) cho xác suất]; đối chứng LR(C=30), ComplementNB(alpha=0.1)
pins (đã chạy) : python 3.12.13; torch==2.14.1; transformers==5.19.0; tokenizers==0.23.3; accelerate==1.15.0; safetensors==0.8.0; huggingface-hub==2.2.0;
               numpy==2.5.4; pandas==3.0.6; pyarrow==26.0.0; scikit-learn==1.9.1; scipy==1.18.1; py-vncorenlp==0.1.4; pyjnius==1.9.0
serve CPU    : báo cáo deploy chọn Streamlit Community Cloud + ONNX Runtime int8 (không torch). Nếu serve bằng torch: torch==2.14.1+cpu (pip --index-url https://download.pytorch.org/whl/cpu; wheel cp312 x86_64 = 196MB); transformers 5.19.0 cần torch>=2.7 khi không có GPU
```

## 10. Rủi ro & giới hạn
- License: phobert-base-v2 = AGPL-3.0 (nguồn: https://huggingface.co/vinai/phobert-base-v2 — metadata `license: agpl-3.0` + file LICENSE "GNU Affero General Public License v3"; v1 https://huggingface.co/vinai/phobert-base = MIT). Chạy bản fine-tune qua mạng (Space/Streamlit) kích hoạt điều khoản nguồn của AGPL §13 (chưa rõ áp dụng thế nào cho trọng số fine-tune; không phải tư vấn pháp lý) → repo public + link nguồn trong app, hoặc dùng phobert-base v1 (MIT; chưa đo chênh accuracy).
- Plan B nếu Java/AGPL gây khó: (a) không tách từ (đã đo: −0.6 acc / −2 mF1 trên val, chưa có ý nghĩa thống kê); (b) pyvi + bảng dấu thanh; (c) xlm-roberta-base (MIT, raw text; BamiBERT paper: −0.5/−1.1 mF1 sentiment, −0.06/−0.24 topic so với PhoBERT-base; 278M tham số ≈ 1.1GB fp32); (d) BamiBERT (103M, raw text, ngang PhoBERT; `AutoTokenizer` lỗi trên transformers 5.19 → dùng `PreTrainedTokenizerFast`; license BSD-3-Clear + Qualcomm Responsible AI "research/education", chưa đọc được điều khoản).
- Churn: transformers 5.x ra ~2 minor/tháng và đã BỎ tham số (warmup_ratio, group_by_length) → pin chính xác; code mẫu v4 hoặc do LLM sinh ra dễ lỗi (v4 dừng ở 4.57.6). `tokenizers` 1.0.0rc* chỉ là pre-release → tránh.
- PhoBERT ở chế độ bảo trì (README trỏ sang BamiBERT; repo chỉ có pytorch_model.bin, 4 PR safetensors chưa merge) — rủi ro thấp trong 2 tuần.
- Dữ liệu: 1 trường, 2013–2016 (lệch miền với phản hồi 2026), nhãn topic nhiễu (IAA 71%), license dataset "unknown" trên HF → nêu trong báo cáo; mô hình vẫn tự tin sai với input ngoài miền (chưa có OOD detector).
- Giới hạn nghiên cứu này: 1–2 seed, chỉ train→val, lr/epoch không tune, chưa đo CPU Linux/x86 và accuracy int8 trên checkpoint của mình (chỉ có số của r2), chưa thử focal/oversampling, chưa fine-tune bản pyvi, Colab T4 chưa đo.

## 11. Câu hỏi chưa giải quyết
- Báo cáo deploy chọn Streamlit Community Cloud (RAM 0.69–2.7GB, JRE qua jdk4py thay apt, chưa thử trên Linux): cần kiểm accuracy int8/ONNX trên CHECKPOINT fine-tune của mình (họ chỉ đo checkpoint bên thứ ba) và RAM thật trên host; RAM sát trần thì MTL (1 model) là lựa chọn an toàn.
- Giữ phobert-base-v2 (AGPL) hay v1 (MIT)? Chưa có số v1 vs v2 trên VSFC (~12 phút/run để đo).
- Cờ "cần xem xét": đặt ngưỡng theo tỷ lệ gắn cờ hay theo recall lỗi? (số liệu mục 4).
