"""Chạy tìm kiếm cấu hình int8 đã ĐĂNG KÝ TRƯỚC (hai vòng) và áp luật chọn đã ghi.

    python -m ml.onnx_gate_search          # chạy vòng 2, ghi kết quả vào onnx_gate_search.json

Mỗi vòng có một khối đăng ký (cấu hình, luật chọn, cách đo) kèm sha256, được commit TRƯỚC khi chạy:
vòng 1 ở gốc tệp JSON (đã chạy bằng bản mã cũ, xem lịch sử git), vòng 2 ở khoá `round2`. Script từ
chối chạy nếu khối bị sửa hoặc vòng đã có kết quả. Chỉ CPU, chỉ validation (không chạm test).

Vòng 2 đo mọi ứng viên trong cùng một phiên: macro-F1 hai tác vụ, F1 lớp trung tính và lớp "khác",
kích thước tệp, số nhãn đổi so với FP32 (batch 1) và ms/câu (batch 1). ms/câu đo XEN KẼ giữa các
mô hình, nhiều vòng, thứ tự xoay, để tải của máy (container khác còn chạy) tác động như nhau lên
mọi ứng viên.
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
TIMING_WARMUP = 10
TIMING_ROUNDS = 5
SPEED_TIE_FRACTION = 0.05  # luật v2: ứng viên chậm hơn bản nhanh nhất <= 5% được xét theo nhãn đổi
NEUTRAL = LABELS["sentiment"].index("neutral")
OTHERS = LABELS["topic"].index("others")


def registration_digest(registration: dict[str, Any]) -> str:
    canonical = json.dumps(registration, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def round_block(document: dict[str, Any], round_number: int) -> dict[str, Any]:
    """Khối chứa `registration`, `registration_sha256` và `results` của một vòng."""
    if round_number == 1:
        return document
    if round_number == 2:
        return document["round2"]
    raise ValueError(f"không có vòng {round_number}")


def registered_names(registration: dict[str, Any], round_number: int) -> list[str]:
    if round_number == 1:
        return [c["name"] for c in registration["configs"]]
    pool = [entry["name"] for entry in registration["pool"]]
    return [*pool, *(entry["name"] for entry in registration["reference_configs"])]


def load_registration(path: Path = REGISTRATION_PATH, round_number: int = 1) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    block = round_block(document, round_number)
    registration = block["registration"]
    if registration_digest(registration) != block["registration_sha256"]:
        raise RuntimeError("phần `registration` đã bị sửa sau khi đăng ký (sha256 không khớp)")
    if registration["gate"]["max_drop_points"] != GATE_MAX_DROP_POINTS:
        raise RuntimeError("ngưỡng cổng trong đăng ký khác ngưỡng của mã")
    unknown = [n for n in registered_names(registration, round_number) if n not in QUANT_CONFIGS]
    if unknown:
        raise RuntimeError(f"cấu hình đã đăng ký nhưng mã không có: {unknown}")
    if round_number == 2:
        timing = registration["measurement"]["timing"]
        in_code = {
            "rounds": TIMING_ROUNDS,
            "sentences": TIMING_SENTENCES,
            "warmup_sentences": TIMING_WARMUP,
        }
        if {key: timing[key] for key in in_code} != in_code:
            raise RuntimeError("cách đo ms/câu trong đăng ký khác mã")
        if registration["speed_tie_fraction"] != SPEED_TIE_FRACTION:
            raise RuntimeError("ngưỡng hoà tốc độ trong đăng ký khác mã")
    return document


def total_flips(row: dict[str, Any]) -> int:
    return sum(row["label_flips_vs_fp32_batch1"].values())


def select_config(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Luật v2 (thứ tự trong `candidates` là thứ tự đăng ký, dùng để phá hoà cuối cùng).

    Giữ cấu hình đạt cổng; lấy cấu hình có ms/câu thấp nhất; mọi cấu hình đạt cổng chậm hơn nó không
    quá `SPEED_TIE_FRACTION` họp thành nhóm hoà và nhóm này chọn cấu hình ít TỔNG nhãn đổi nhất
    (còn hoà thì ms/câu thấp hơn, rồi thứ tự đăng ký). Kích thước tệp không tham gia.
    """
    passing = [c for c in candidates if c["passes_gate_batch1"]]
    if not passing:
        return {"passing": [], "fastest": None, "within_tie_band": [], "selected": None}

    def ms(row: dict[str, Any]) -> float:
        return row["ms_per_sentence_batch1"]["median"]

    fastest = min(passing, key=ms)
    band = [c for c in passing if (ms(c) - ms(fastest)) / ms(fastest) <= SPEED_TIE_FRACTION + 1e-12]
    order = {c["name"]: index for index, c in enumerate(candidates)}
    chosen = min(band, key=lambda c: (total_flips(c), ms(c), order[c["name"]]))
    return {
        "passing": [c["name"] for c in passing],
        "fastest": {"name": fastest["name"], "ms_per_sentence_batch1": ms(fastest)},
        "within_tie_band": [c["name"] for c in band],
        "selected": chosen["name"],
    }


def quality_row(
    name: str,
    path: Path,
    sequences: list[list[int]],
    targets: dict[str, Any],
    reference: dict[str, Any],
    reference_logits: dict[str, Any],
    threads: int,
    n_kept_fp32: int,
) -> dict[str, Any]:
    session = make_session(path, threads)
    single = run_logits(session, sequences, batch_size=1)
    batched = run_logits(session, sequences, batch_size=EVAL_BATCH)
    single_summary, batched_summary = summary(single, targets), summary(batched, targets)

    def delta(candidate: dict[str, Any]) -> dict[str, float]:
        return {
            t: round((candidate[t]["macro_f1"] - reference[t]["macro_f1"]) * 100, 3) for t in TASKS
        }

    delta_single = delta(single_summary)
    flips = {t: int((reference_logits[t].argmax(1) != single[t].argmax(1)).sum()) for t in TASKS}
    return {
        "name": name,
        "heads_fp32": QUANT_CONFIGS[name].heads_fp32,
        "n_matmul_kept_fp32": n_kept_fp32,
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "size_bytes": path.stat().st_size,
        "batch1": {
            t: {k: single_summary[t][k] for k in ("accuracy", "macro_f1", "weighted_f1", "ece")}
            for t in TASKS
        },
        "neutral_f1_sentiment": single_summary["sentiment"]["per_class_f1"][NEUTRAL],
        "others_f1_topic": single_summary["topic"]["per_class_f1"][OTHERS],
        "delta_macro_f1_points_batch1": delta_single,
        "delta_macro_f1_points_batch32_info": delta(batched_summary),
        "label_flips_vs_fp32_batch1": flips,
        "total_label_flips_batch1": sum(flips.values()),
        "passes_gate_batch1": all(-d <= GATE_MAX_DROP_POINTS for d in delta_single.values()),
    }


def interleaved_timing(
    paths: dict[str, Path], sequences: list[list[int]], threads: int
) -> tuple[dict[str, dict[str, Any]], list[float]]:
    """ms/câu ở batch 1 cho mọi mô hình, đo xen kẽ: mỗi vòng đo lần lượt tất cả mô hình.

    Thứ tự xoay một bước mỗi vòng (mô hình nào cũng có lúc đứng đầu, giữa và cuối), mỗi lần tạo
    phiên ORT mới rồi khởi động. Trả trung vị các vòng cùng tải máy (loadavg 1 phút) đầu mỗi vòng.
    """
    names = list(paths)
    runs: dict[str, list[float]] = {name: [] for name in names}
    load: list[float] = []
    for round_index in range(TIMING_ROUNDS):
        shift = round_index % len(names)
        load.append(round(os.getloadavg()[0], 1))
        for name in names[shift:] + names[:shift]:
            session = make_session(paths[name], threads)
            run_logits(session, sequences[:TIMING_WARMUP], batch_size=1)
            started = time.perf_counter()
            run_logits(session, sequences[:TIMING_SENTENCES], batch_size=1)
            runs[name].append((time.perf_counter() - started) / TIMING_SENTENCES * 1000)
            del session
    timings = {
        name: {
            "median": round(statistics.median(values), 2),
            "runs": [round(v, 2) for v in values],
        }
        for name, values in runs.items()
    }
    return timings, load


def reproduction_of_round_one(
    rows: dict[str, dict[str, Any]], round_one: dict[str, Any]
) -> dict[str, dict[str, bool]]:
    """Các cấu hình vòng 1 đo lại ở vòng 2 phải cho lại đúng delta và số nhãn đổi (tất định)."""
    previous = {row["name"]: row for row in round_one["results"]["candidates"]}
    return {
        name: {
            "deltas_equal": rows[name]["delta_macro_f1_points_batch1"]
            == previous[name]["delta_macro_f1_points_batch1"],
            "flips_equal": rows[name]["label_flips_vs_fp32_batch1"]
            == previous[name]["label_flips_vs_fp32_batch1"],
        }
        for name in previous
        if name in rows
    }


def run(threads: int, round_number: int = 2) -> dict[str, Any]:
    document = load_registration(REGISTRATION_PATH, round_number)
    block = round_block(document, round_number)
    if block["results"] is not None:
        raise RuntimeError(
            f"tìm kiếm vòng {round_number} đã chạy xong; không chạy lại để khỏi chỉnh sau khi xem "
            "kết quả"
        )
    if round_number != 2:
        raise RuntimeError("vòng 1 chỉ chạy được bằng bản mã cũ (xem lịch sử git)")
    registration = block["registration"]
    if threads != registration["measurement"]["timing"]["threads"]:
        raise RuntimeError(
            f"đăng ký đo với {registration['measurement']['timing']['threads']} luồng, "
            f"không phải {threads}"
        )

    pool = [entry["name"] for entry in registration["pool"]]
    references = [entry["name"] for entry in registration["reference_configs"]]
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
    del fp32_session

    rows: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {"fp32": fp32_path}
    try:
        for name in [*pool, *references]:
            path = work / f"{name}.int8.onnx"
            kept = quantize_int8(fp32_path, path, name)
            paths[name] = path
            rows[name] = quality_row(
                name, path, sequences, targets, reference, reference_logits, threads, kept
            )
            row = rows[name]
            print(
                f"{name:34s} {row['size_mb']:6.1f}MB  "
                f"d_mF1(b1) {row['delta_macro_f1_points_batch1']}  "
                f"flips {row['label_flips_vs_fp32_batch1']}  "
                f"{'PASS' if row['passes_gate_batch1'] else 'FAIL'}",
                flush=True,
            )

        timings, load_per_round = interleaved_timing(paths, sequences, threads)
        for name, row in rows.items():
            row["ms_per_sentence_batch1"] = timings[name]
            print(f"{name:34s} {timings[name]['median']:6.2f} ms/câu {timings[name]['runs']}")
    finally:
        for name, path in paths.items():
            if name != "fp32":
                path.unlink(missing_ok=True)

    candidates = [rows[name] for name in pool]
    outcome = select_config(candidates)
    results = {
        "fp32_reference": {
            "name": "fp32 (mốc so sánh)",
            "size_mb": round(fp32_path.stat().st_size / 1e6, 1),
            "val": {
                t: {k: reference[t][k] for k in ("accuracy", "macro_f1", "ece")} for t in TASKS
            },
            "neutral_f1_sentiment": reference["sentiment"]["per_class_f1"][NEUTRAL],
            "others_f1_topic": reference["topic"]["per_class_f1"][OTHERS],
            "ms_per_sentence_batch1": timings["fp32"],
        },
        "reference_rows": [rows[name] for name in references],
        "candidates": candidates,
        "reproduction_of_round_one": reproduction_of_round_one(rows, document),
        **outcome,
        "selection_rule_applied": registration["selection_rule"],
        "environment": {
            "threads": threads,
            "onnxruntime": ort.__version__,
            "loadavg_1m_at_start_of_each_timing_round": load_per_round,
            "docker_running_during_timing": docker_running(),
            "note": "ms/câu đo khi máy còn chạy container khác; xen kẽ giữa các mô hình nên chỉ "
            "so sánh tương đối trong cùng phiên",
        },
    }
    block["results"] = results
    block.update(status="completed", completed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    write_json(REGISTRATION_PATH, document)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, choices=(1, 2), default=2, dest="round_number")
    parser.add_argument("--threads", type=int, default=None, help="mặc định EDUPULSE_ORT_THREADS")
    args = parser.parse_args(argv)
    results = run(args.threads or load_settings().ort_threads, args.round_number)
    print(json.dumps({k: results[k] for k in ("passing", "selected")}, ensure_ascii=False))
    return 0 if results["selected"] else 1


if __name__ == "__main__":
    sys.exit(main())
