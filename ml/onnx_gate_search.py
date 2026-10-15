"""Chạy tìm kiếm cấu hình int8 đã ĐĂNG KÝ TRƯỚC và áp luật chọn đã ghi.

    python -m ml.onnx_gate_search          # ghi kết quả vào results/metrics/onnx_gate_search.json

Danh sách cấu hình, luật chọn và cổng nằm trong `results/metrics/onnx_gate_search.json` (phần
`registration`, có sha256) và được commit TRƯỚC khi chạy; script từ chối chạy nếu phần đó bị sửa.
Chỉ CPU, chỉ validation (không chạm test). Mỗi cấu hình đo: macro-F1 hai tác vụ, F1 lớp trung tính
và lớp "khác", kích thước tệp, số nhãn đổi so với FP32 và ms/câu ở batch 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import onnxruntime as ort

from edupulse.config import REPO_ROOT, load_settings
from edupulse.labels import LABELS, TASKS
from ml.common import docker_running, load_processed, settings, write_json
from ml.export_onnx import (
    EVAL_BATCH,
    GATE_MAX_DROP_POINTS,
    QUANT_CONFIGS,
    encode,
    make_session,
    quantize_int8,
    run_logits,
    summary,
)

REGISTRATION_PATH = REPO_ROOT / "results" / "metrics" / "onnx_gate_search.json"
TIMING_SENTENCES = 200
TIMING_REPEATS = 3
NEUTRAL = LABELS["sentiment"].index("neutral")
OTHERS = LABELS["topic"].index("others")


def registration_digest(registration: dict[str, Any]) -> str:
    canonical = json.dumps(registration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_registration(path: Path = REGISTRATION_PATH) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if registration_digest(document["registration"]) != document["registration_sha256"]:
        raise RuntimeError("phần `registration` đã bị sửa sau khi đăng ký (sha256 không khớp)")
    if document["registration"]["gate"]["max_drop_points"] != GATE_MAX_DROP_POINTS:
        raise RuntimeError("ngưỡng cổng trong đăng ký khác ngưỡng của mã")
    unknown = [
        c["name"] for c in document["registration"]["configs"] if c["name"] not in QUANT_CONFIGS
    ]
    if unknown:
        raise RuntimeError(f"cấu hình đã đăng ký nhưng mã không có: {unknown}")
    return document


def timed_ms_per_sentence(
    session: ort.InferenceSession, sequences: list[list[int]]
) -> dict[str, Any]:
    run_logits(session, sequences[:10], batch_size=1)  # khởi động
    runs = []
    for _ in range(TIMING_REPEATS):
        started = time.perf_counter()
        run_logits(session, sequences[:TIMING_SENTENCES], batch_size=1)
        runs.append((time.perf_counter() - started) / TIMING_SENTENCES * 1000)
    return {"median": round(statistics.median(runs), 2), "runs": [round(r, 2) for r in runs]}


def evaluate(
    name: str,
    path: Path,
    sequences: list[list[int]],
    targets: dict[str, Any],
    reference: dict[str, Any],
    reference_logits: dict[str, Any],
    threads: int,
    n_excluded: int | None,
) -> dict[str, Any]:
    session = make_session(path, threads)
    single = run_logits(session, sequences, batch_size=1)
    batched = run_logits(session, sequences, batch_size=EVAL_BATCH)
    single_summary, batched_summary = summary(single, targets), summary(batched, targets)

    def delta(candidate: dict[str, Any]) -> dict[str, float]:
        return {
            t: round((candidate[t]["macro_f1"] - reference[t]["macro_f1"]) * 100, 3) for t in TASKS
        }

    delta_single, delta_batched = delta(single_summary), delta(batched_summary)
    return {
        "name": name,
        "n_matmul_kept_fp32": n_excluded,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "size_bytes": path.stat().st_size,
        "batch1": {
            t: {k: single_summary[t][k] for k in ("accuracy", "macro_f1", "weighted_f1", "ece")}
            for t in TASKS
        },
        "neutral_f1_sentiment": single_summary["sentiment"]["per_class_f1"][NEUTRAL],
        "others_f1_topic": single_summary["topic"]["per_class_f1"][OTHERS],
        "delta_macro_f1_points_batch1": delta_single,
        "delta_macro_f1_points_batch32_info": delta_batched,
        "label_flips_vs_fp32_batch1": {
            t: int((reference_logits[t].argmax(1) != single[t].argmax(1)).sum()) for t in TASKS
        },
        "ms_per_sentence_batch1": timed_ms_per_sentence(session, sequences),
        "passes_gate_batch1": all(-d <= GATE_MAX_DROP_POINTS for d in delta_single.values()),
    }


def run(threads: int) -> dict[str, Any]:
    document = load_registration()
    if document.get("results") is not None:
        raise RuntimeError(
            "tìm kiếm đã chạy xong; không chạy lại để khỏi chỉnh sau khi xem kết quả"
        )
    registration = document["registration"]
    out_dir = settings().outputs_dir / "onnx" / "mtl-s42"
    fp32_path = out_dir / "model.fp32.onnx"
    work = settings().outputs_dir / "onnx" / "mtl-s42-search"
    work.mkdir(parents=True, exist_ok=True)

    val = load_processed("validation")  # chỉ validation, không chạm test
    sequences = encode(val["text_rdr"].tolist())
    targets = {task: val[task].to_numpy() for task in TASKS}
    fp32_session = make_session(fp32_path, threads)
    reference_logits = run_logits(fp32_session, sequences)
    reference = summary(reference_logits, targets)
    reference_row = {
        "name": "fp32 (mốc so sánh)",
        "size_mb": round(fp32_path.stat().st_size / 1e6, 1),
        "val": {t: {k: reference[t][k] for k in ("accuracy", "macro_f1", "ece")} for t in TASKS},
        "neutral_f1_sentiment": reference["sentiment"]["per_class_f1"][NEUTRAL],
        "others_f1_topic": reference["topic"]["per_class_f1"][OTHERS],
        "ms_per_sentence_batch1": timed_ms_per_sentence(fp32_session, sequences),
    }

    rows: dict[str, dict[str, Any]] = {}
    for name in ["all_int8", *(c["name"] for c in registration["configs"])]:
        path = work / f"{name}.int8.onnx"
        excluded = quantize_int8(fp32_path, path, name)
        rows[name] = evaluate(
            name, path, sequences, targets, reference, reference_logits, threads, excluded
        )
        row = rows[name]
        print(
            f"{name:14s} {row['size_mb']:6.1f}MB  d_mF1(b1) {row['delta_macro_f1_points_batch1']}  "
            f"flips {row['label_flips_vs_fp32_batch1']}  "
            f"{row['ms_per_sentence_batch1']['median']} ms  "
            f"{'PASS' if row['passes_gate_batch1'] else 'FAIL'}",
            flush=True,
        )
        path.unlink(missing_ok=True)

    candidates = [rows[c["name"]] for c in registration["configs"]]
    passing = [c for c in candidates if c["passes_gate_batch1"]]
    selected = (
        min(passing, key=lambda c: (c["size_bytes"], c["ms_per_sentence_batch1"]["median"]))["name"]
        if passing
        else None
    )
    results = {
        "fp32_reference": reference_row,
        "all_int8_reference": rows["all_int8"],
        "candidates": candidates,
        "passing": [c["name"] for c in passing],
        "selected": selected,
        "selection_rule_applied": registration["selection_rule"],
        "environment": {
            "threads": threads,
            "onnxruntime": ort.__version__,
            "loadavg_1m": round(os.getloadavg()[0], 1),
            "docker_running_during_timing": docker_running(),
            "note": "ms/câu đo khi máy còn chạy container khác, chỉ nên so sánh tương đối",
        },
    }
    document.update(status="completed", completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    document["results"] = results
    write_json(REGISTRATION_PATH, document)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threads", type=int, default=None, help="mặc định EDUPULSE_ORT_THREADS")
    args = parser.parse_args(argv)
    results = run(args.threads or load_settings().ort_threads)
    print(json.dumps({k: results[k] for k in ("passing", "selected")}, ensure_ascii=False))
    return 0 if results["selected"] else 1


if __name__ == "__main__":
    sys.exit(main())
