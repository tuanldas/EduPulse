"""Xuất run PhoBERT đa nhiệm sang ONNX int8 và kiểm cổng chất lượng.

    python -m ml.export_onnx --run best        # make export

Các bước (đều trên CPU): xuất FP32 bằng `torch.onnx.export(dynamo=True)` với trục batch và seq
động -> kiểm khớp torch (|Δlogit| < 1e-3) -> `quant_pre_process` (bắt buộc với bản dynamo) ->
lượng tử hoá int8 chỉ MatMul, từng kênh, reduce_range -> chấm val FP32 và int8 -> ghi cổng.
Cổng: macro-F1 val của int8 không được thấp hơn FP32 quá 1 điểm ở MỘT trong hai tác vụ, ở cả hai
chế độ chạy (từng câu một và lô 32), nếu không tiến trình thoát với mã 1. Int8 động phụ thuộc lô,
nên logits val dùng để hiệu chuẩn (`val_logits_int8.npz`) và `golden_predictions.json` đều chạy
từng câu một (batch 1), đúng điều kiện của phân tích một câu; mức lệch so với lô nằm trong
`onnx_gate.json["batch_dependence"]`.

Đầu ra trong `outputs/onnx/<run>/`: model.fp32.onnx, model.int8.onnx, val_logits_int8.npz,
onnx_gate.json, golden_predictions.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from edupulse.bundle_schema import GoldenItem, GoldenPredictions
from edupulse.config import REPO_ROOT, load_settings
from edupulse.labels import LABELS, TASKS
from edupulse.scoring import softmax_t
from edupulse.text import NORMALIZER_VERSION, prepare_many
from ml.common import (
    load_processed,
    read_json,
    resolve_run,
    settings,
    task_metrics,
    write_json,
)
from ml.phobert_model import (
    BASE_MODEL,
    BASE_REVISION,
    MAX_LEN,
    MODEL_VOCAB_SIZE,
    PAD_ID,
    UNK_ID,
    MultiHeadPhoBERT,
    clamp_ids,
    expand_word_embeddings,
    load_tokenizer,
)

INPUT_NAMES = ("input_ids", "attention_mask")
OUTPUT_NAMES = {"sentiment": "sentiment_logits", "topic": "topic_logits"}
PARITY_SAMPLES = 64
PARITY_TOLERANCE = 1e-3
# Câu tự viết có mảnh BPE nằm ngoài bảng embedding gốc ("đạt_", id 65771): luôn thử khi xuất.
RARE_PIECE_TEXTS = (
    "thầy truyền_đạt_kiến thức rất tốt .",
    "cô truyền_đạt_kiến_thức dễ hiểu .",
)
MAX_DATASET_RARE = 16  # thêm tối đa ngần này câu của dữ liệu có id ngoài bảng (chỉ dùng trong RAM)
GATE_MAX_DROP_POINTS = 1.0
EVAL_BATCH = 32
GOLDEN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "golden_sentences.json"
QUANTIZATION = "int8 động, chỉ MatMul, từng kênh, reduce_range (ONNX Runtime)"


class _Export(torch.nn.Module):
    """Bọc mô hình thành hàm thuần tensor -> tuple, đúng thứ tự đầu ra của manifest."""

    def __init__(self, model: MultiHeadPhoBERT) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return tuple(logits[task] for task in TASKS)


def export_fp32(model: MultiHeadPhoBERT, sample: list[list[int]], path: Path) -> None:
    longest = max(len(s) for s in sample)
    ids = torch.full((len(sample), longest), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(sample), longest), dtype=torch.long)
    for row, sequence in enumerate(sample):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, : len(sequence)] = 1
    batch = torch.export.Dim("batch", min=1, max=256)
    seq = torch.export.Dim("seq", min=2, max=256)
    with torch.no_grad():
        program = torch.onnx.export(
            _Export(model).eval(),
            (ids, mask),
            dynamo=True,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAMES[t] for t in TASKS],
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
        )
    program.save(str(path))


def quantize_int8(fp32_path: Path, int8_path: Path) -> None:
    prepared = fp32_path.with_name("model.preprocessed.onnx")
    try:
        quant_pre_process(str(fp32_path), str(prepared))
        quantize_dynamic(
            str(prepared),
            str(int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul"],
            per_channel=True,
            reduce_range=True,
        )
    finally:
        prepared.unlink(missing_ok=True)


def make_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def run_logits(
    session: ort.InferenceSession, sequences: list[list[int]], batch_size: int = EVAL_BATCH
) -> dict[str, np.ndarray]:
    """Logits từng head theo thứ tự đầu vào; lô sắp theo độ dài, đệm bằng PAD_ID."""
    names = [o.name for o in session.get_outputs()]
    out = {task: np.zeros((len(sequences), len(LABELS[task])), dtype=np.float32) for task in TASKS}
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    for start in range(0, len(order), batch_size):
        index = order[start : start + batch_size]
        longest = max(len(sequences[i]) for i in index)
        ids = np.full((len(index), longest), PAD_ID, dtype=np.int64)
        mask = np.zeros((len(index), longest), dtype=np.int64)
        for row, i in enumerate(index):
            ids[row, : len(sequences[i])] = sequences[i]
            mask[row, : len(sequences[i])] = 1
        results = session.run(names, {"input_ids": ids, "attention_mask": mask})
        for task, result in zip(TASKS, results, strict=True):
            out[task][index] = result
    return out


def encode(texts: list[str]) -> list[list[int]]:
    """Id THÔ của tokenizer.json (có thể >= 64.001): đúng thứ mà phía phục vụ đưa vào ONNX."""
    tokenizer = load_tokenizer(MAX_LEN)
    return [e.ids for e in tokenizer.encode_batch(texts)]


def unk_alias_gap(session: ort.InferenceSession, rows: int, unk_id: int = UNK_ID) -> float:
    """Chênh lệch logits lớn nhất giữa id cuối bảng nới và `<unk>`: phải bằng 0 (hàng sao chép)."""
    last = run_logits(session, [[0, rows - 1, 2]], batch_size=1)
    unk = run_logits(session, [[0, unk_id, 2]], batch_size=1)
    return max(float(np.abs(last[t] - unk[t]).max()) for t in TASKS)


def find_rare_piece_texts(limit: int = MAX_DATASET_RARE) -> list[str]:
    """Câu của train/val có id ngoài bảng embedding gốc; chỉ giữ trong RAM để thử parity."""
    found: list[str] = []
    for split in ("train", "validation"):
        texts = load_processed(split)["text_rdr"].tolist()
        for text, ids in zip(texts, encode(texts), strict=True):
            if max(ids) >= MODEL_VOCAB_SIZE:
                found.append(text)
    return found[:limit]


def torch_logits(model: MultiHeadPhoBERT, sequences: list[list[int]]) -> dict[str, np.ndarray]:
    out = {task: [] for task in TASKS}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(sequences), 16):
            chunk = sequences[start : start + 16]
            longest = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), longest), PAD_ID, dtype=torch.long)
            mask = torch.zeros((len(chunk), longest), dtype=torch.long)
            for row, sequence in enumerate(chunk):
                ids[row, : len(sequence)] = torch.tensor(sequence)
                mask[row, : len(sequence)] = 1
            logits = model(input_ids=ids, attention_mask=mask)
            for task in TASKS:
                out[task].append(logits[task].numpy())
    return {task: np.concatenate(parts) for task, parts in out.items()}


def batch_dependence(single: np.ndarray, batched: np.ndarray) -> dict[str, float]:
    """Độ lệch logits của cùng một câu giữa chạy riêng và chạy trong lô (int8 động)."""
    diff = np.abs(single.astype(np.float64) - batched.astype(np.float64))
    return {
        "median_abs_logit_diff": float(np.median(diff)),
        "p95_abs_logit_diff": float(np.percentile(diff, 95)),
        "max_abs_logit_diff": float(diff.max()),
        "label_flips": int((single.argmax(1) != batched.argmax(1)).sum()),
    }


def summary(logits: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> dict[str, Any]:
    result = {}
    for task in TASKS:
        metrics = task_metrics(logits[task], targets[task], LABELS[task])
        result[task] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "ece": metrics["ece"],
            "per_class_f1": [c["f1"] for c in metrics["per_class"]],
        }
    return result


def golden_predictions(int8_path: Path, threads: int) -> GoldenPredictions:
    """Dự đoán thô của từng câu vàng, mỗi câu chạy riêng một lần (batch 1, không đệm)."""
    items = read_json(GOLDEN_FIXTURE)["items"]
    prepared = prepare_many([item["text"] for item in items], use_ner=False)
    session = make_session(int8_path, threads)
    sequences = encode([p.model_text for p in prepared])
    golden = []
    for item, p, sequence in zip(items, prepared, sequences, strict=True):
        logits = run_logits(session, [sequence], batch_size=1)
        probs = {task: softmax_t(logits[task][0], 1.0) for task in TASKS}
        golden.append(
            GoldenItem(
                id=item["id"],
                model_text=p.model_text,
                sentiment=LABELS["sentiment"][int(probs["sentiment"].argmax())],
                topic=LABELS["topic"][int(probs["topic"].argmax())],
                p_sentiment=[float(x) for x in probs["sentiment"]],
                p_topic=[float(x) for x in probs["topic"]],
            )
        )
    return GoldenPredictions(normalizer_version=NORMALIZER_VERSION, items=golden)


def export(run: Path, out_dir: Path, threads: int) -> dict[str, Any]:
    history = read_json(run / "history.json")
    if history.get("normalizer_version") != NORMALIZER_VERSION:
        raise RuntimeError(
            f"run huấn luyện bằng bản chuẩn hoá {history.get('normalizer_version')!r}, mã đang là "
            f"{NORMALIZER_VERSION!r}: huấn luyện lại"
        )
    model = MultiHeadPhoBERT.from_pretrained(run).eval()
    if tuple(model.config.task_heads) != TASKS:
        raise RuntimeError(
            f"chỉ xuất được mô hình đa nhiệm, run này có đầu {list(model.config.task_heads)}"
        )

    # Bảng embedding của bản xuất được nới tới đủ số mục của tokenizer.json (hàng thừa = <unk>);
    # `model` giữ bảng gốc 64.001 hàng làm tham chiếu torch (nhận id đã đổi mảnh lạ thành <unk>).
    rows = load_tokenizer(MAX_LEN).get_vocab_size()
    if rows < MODEL_VOCAB_SIZE:
        raise RuntimeError(f"tokenizer.json chỉ có {rows} mục, ít hơn bảng embedding gốc")
    export_model = MultiHeadPhoBERT.from_pretrained(run).eval()
    expand_word_embeddings(export_model, rows)

    val = load_processed("validation")
    sequences = encode(val["text_rdr"].tolist())
    targets = {task: val[task].to_numpy() for task in TASKS}
    rare_texts = [*RARE_PIECE_TEXTS, *find_rare_piece_texts()]
    rare_sequences = encode(rare_texts)
    if not any(max(ids) >= MODEL_VOCAB_SIZE for ids in rare_sequences[: len(RARE_PIECE_TEXTS)]):
        raise RuntimeError("câu thử mảnh hiếm không còn tạo id ngoài bảng embedding: cập nhật câu")
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_path, int8_path = out_dir / "model.fp32.onnx", out_dir / "model.int8.onnx"
    timing: dict[str, float] = {}

    started = time.time()
    export_fp32(export_model, sequences[:2], fp32_path)
    timing["export_fp32_s"] = round(time.time() - started, 1)

    fp32_session = make_session(fp32_path, threads)
    parity_sequences = sequences[:PARITY_SAMPLES]
    reference = torch_logits(model, [clamp_ids(s) for s in parity_sequences])
    onnx_ref = run_logits(fp32_session, parity_sequences, batch_size=16)
    max_diff = max(float(np.abs(reference[t] - onnx_ref[t]).max()) for t in TASKS)
    if max_diff >= PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX FP32 lệch torch {max_diff:.2e} >= {PARITY_TOLERANCE:.0e}")
    # Câu có mảnh hiếm: ONNX nhận id THÔ, torch nhận id đã đổi thành <unk>; hai bên phải khớp.
    rare_reference = torch_logits(model, [clamp_ids(s) for s in rare_sequences])
    rare_onnx = run_logits(fp32_session, rare_sequences, batch_size=16)
    rare_diff = max(float(np.abs(rare_reference[t] - rare_onnx[t]).max()) for t in TASKS)
    if rare_diff >= PARITY_TOLERANCE:
        raise RuntimeError(f"ONNX lệch torch ở câu mảnh hiếm: {rare_diff:.2e}")
    alias_fp32 = unk_alias_gap(fp32_session, rows)
    if alias_fp32 > 1e-5:
        raise RuntimeError(
            f"id {rows - 1} không hành xử như <unk> trong ONNX FP32: {alias_fp32:.2e}"
        )

    started = time.time()
    quantize_int8(fp32_path, int8_path)
    timing["quantize_s"] = round(time.time() - started, 1)

    started = time.time()
    fp32_val = run_logits(fp32_session, sequences)
    timing["val_fp32_s"] = round(time.time() - started, 1)
    int8_session = make_session(int8_path, threads)
    alias_int8 = unk_alias_gap(int8_session, rows)
    if alias_int8 > 1e-5:
        raise RuntimeError(
            f"id {rows - 1} không hành xử như <unk> trong ONNX int8: {alias_int8:.2e}"
        )
    # Int8 động tính thang kích hoạt theo CẢ tensor nên logits của một câu phụ thuộc các câu cùng
    # lô (lệch cỡ 0,1 logit). Chuẩn tham chiếu là từng câu một (batch 1): đúng điều kiện của phân
    # tích một câu và của câu vàng, nên hiệu chuẩn và cổng dựa trên đó. Lô 32 (đường phân tích
    # hàng loạt) được đo thêm để biết mức lệch và cũng phải qua cổng.
    started = time.time()
    int8_val = run_logits(int8_session, sequences, batch_size=1)
    timing["val_int8_single_s"] = round(time.time() - started, 1)
    started = time.time()
    int8_batched = run_logits(int8_session, sequences)
    timing["val_int8_batched_s"] = round(time.time() - started, 1)

    fp32_summary, int8_summary = summary(fp32_val, targets), summary(int8_val, targets)
    batched_summary = summary(int8_batched, targets)
    deltas = {
        t: round((int8_summary[t]["macro_f1"] - fp32_summary[t]["macro_f1"]) * 100, 3)
        for t in TASKS
    }
    deltas_batched = {
        t: round((batched_summary[t]["macro_f1"] - fp32_summary[t]["macro_f1"]) * 100, 3)
        for t in TASKS
    }
    flips = {t: int((fp32_val[t].argmax(1) != int8_val[t].argmax(1)).sum()) for t in TASKS}
    dependence = {t: batch_dependence(int8_val[t], int8_batched[t]) for t in TASKS}
    passed = all(
        -delta <= GATE_MAX_DROP_POINTS for delta in (*deltas.values(), *deltas_batched.values())
    )

    np.savez(
        out_dir / "val_logits_int8.npz",
        **{t: int8_val[t] for t in TASKS},
        **{f"y_{t}": targets[t] for t in TASKS},
    )
    golden = golden_predictions(int8_path, threads)
    write_json(out_dir / "golden_predictions.json", golden.model_dump(mode="json"))

    gate = {
        "run": run.name,
        "normalizer_version": NORMALIZER_VERSION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "n_val": len(sequences),
        "parity": {
            "n": PARITY_SAMPLES,
            "max_abs_logit_diff": max_diff,
            "tolerance": PARITY_TOLERANCE,
            "ok": True,
        },
        # Id ngoài bảng embedding gốc: tokenizer.json có `embedding_rows` mục nhưng checkpoint chỉ
        # `base_vocab_size` hàng. ONNX được nới bảng, hàng thừa = <unk>; train dùng id đã đổi.
        "embedding": {
            "base_vocab_size": MODEL_VOCAB_SIZE,
            "unk_id": UNK_ID,
            "embedding_rows": rows,
            "extra_rows": rows - MODEL_VOCAB_SIZE,
            "unk_alias_max_diff": {"fp32": alias_fp32, "int8": alias_int8},
            "rare_piece_parity": {
                "n": len(rare_sequences),
                "n_out_of_range": sum(max(ids) >= MODEL_VOCAB_SIZE for ids in rare_sequences),
                "max_abs_logit_diff": rare_diff,
                "tolerance": PARITY_TOLERANCE,
                "ok": True,
            },
        },
        "quantization": QUANTIZATION,
        "fp32": fp32_summary,
        "int8": int8_summary,
        "int8_batched": batched_summary,
        "delta_macro_f1_points": deltas,
        "delta_macro_f1_points_batched": deltas_batched,
        "label_flips": flips,
        "batch_dependence": dependence,
        "gate": {
            "max_drop_points": GATE_MAX_DROP_POINTS,
            "passed": passed,
            "modes": ["batch=1", f"batch={EVAL_BATCH}"],
        },
        "files": {"fp32_bytes": fp32_path.stat().st_size, "int8_bytes": int8_path.stat().st_size},
        "versions": {"onnxruntime": ort.__version__, "torch": torch.__version__},
        "threads": threads,
        "timing": timing,
    }
    write_json(out_dir / "onnx_gate.json", gate)
    return gate


def run_summary(run: Path, out_dir: Path) -> dict[str, Any]:
    """Tóm tắt run huấn luyện (epoch, thời gian, bộ nhớ) cùng số val của bản FP32 tốt nhất."""
    history = read_json(run / "history.json")
    gate = read_json(out_dir / "onnx_gate.json")
    return {
        "run": run.name,
        "normalizer_version": history["normalizer_version"],
        "device": history["device"],
        "args": {k: history["args"][k] for k in ("heads", "seed", "epochs", "lr", "batch_size")},
        "best_epoch": history["best_epoch"],
        "total_s": history["total_s"],
        "peak_mb": history["peak_mb"],
        "epochs": history["epochs"],
        "val_fp32_onnx": gate["fp32"],
        "val_int8_onnx": gate["int8"],
        "val_int8_batched_onnx": gate["int8_batched"],
        "batch_dependence": gate["batch_dependence"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="best", help="'best', tên run hoặc thư mục checkpoint")
    parser.add_argument("--threads", type=int, default=None, help="mặc định EDUPULSE_ORT_THREADS")
    parser.add_argument("--gate-copy", type=Path, help="chép onnx_gate.json tới đường dẫn này")
    parser.add_argument(
        "--run-summary", type=Path, help="ghi tóm tắt run huấn luyện tới đường dẫn này"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run = resolve_run(args.run)
    out_dir = settings().outputs_dir / "onnx" / run.name
    threads = args.threads or load_settings().ort_threads
    gate = export(run, out_dir, threads)
    if args.gate_copy:
        args.gate_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_dir / "onnx_gate.json", args.gate_copy)
    if args.run_summary:
        write_json(args.run_summary, run_summary(run, out_dir))
    keys = (
        "run",
        "delta_macro_f1_points",
        "delta_macro_f1_points_batched",
        "label_flips",
        "batch_dependence",
        "gate",
    )
    print(json.dumps({k: gate[k] for k in keys}))
    if not gate["gate"]["passed"]:
        print(
            f"CỔNG KHÔNG QUA: int8 giảm macro-F1 val quá {GATE_MAX_DROP_POINTS} điểm "
            f"(từng câu {gate['delta_macro_f1_points']}, "
            f"lô {gate['delta_macro_f1_points_batched']})",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
