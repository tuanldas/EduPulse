"""Đánh giá cuối: đóng băng cấu hình, chấm test đúng một lần, diễn tập trên validation.

    python -m ml.evaluate --freeze                 # ghi results/freeze.json (rồi commit nó)
    python -m ml.evaluate --split test --final     # đúng một lần: dự đoán test của MỌI mô hình
    python -m ml.evaluate --split test --tables    # sinh lại bảng/hình từ dự đoán đã lưu
    python -m ml.evaluate --rehearsal [--tables]   # diễn tập đường chấm cuối trên validation
    python -m ml.evaluate --record-rerun "lý do"   # ghi lý do chạy lại test sau khi sửa lỗi mã

Các mô hình được chấm: ba baseline TF-IDF, run PhoBERT đa nhiệm được triển khai (FP32, torch CPU)
và bản int8 của nó (ONNX Runtime, từng câu một). `--final` chỉ chạy khi mọi chốt chặn của
`ml.freeze.assert_final_allowed` thoả. Dự đoán của mọi mô hình được ghi vào `outputs/test_preds/`
(xem `ml.preds`) trước khi in bất kỳ số nào; từ đó mọi phân tích chỉ đọc thư mục này. `--rehearsal`
chạy đúng các bước ấy trên validation, ghi vào `outputs/rehearsal/` (không vào `results/`), để bắt
lỗi mã trước khi chạm test.

Mã thoát: 0 xong; 6 một chốt chặn của lần chấm cuối chưa thoả hoặc chưa đủ điều kiện đóng băng.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from edupulse.config import load_settings
from edupulse.labels import TASKS
from ml import eval_tables
from ml import freeze as freeze_lib
from ml.common import best_run, read_json, sha256_file, write_json
from ml.freeze import (
    GuardError,
    assert_final_allowed,
    build_freeze,
    calibration_path,
    git_dirty,
    git_head,
    record_rerun,
    write_freeze,
)
from ml.preds import (
    BASELINES,
    DEPLOY_KEY,
    DONE_FILE,
    META_FILE,
    EvalContext,
    context,
    final_keys,
    is_done,
    read_meta,
    save_preds,
)

EXIT_GUARD = 6


# --- đóng băng ----------------------------------------------------------------------------------


def cmd_freeze() -> int:
    ctx = context()
    if ctx.preds_dir.exists():
        print(
            f"evaluate: {ctx.preds_dir} đã có: test đã được chấm, không đóng băng lại. Dùng "
            "--record-rerun nếu phải chấm lại.",
            file=sys.stderr,
        )
        return EXIT_GUARD
    try:
        freeze = build_freeze()
    except (GuardError, FileNotFoundError) as error:
        print(f"evaluate: chưa đủ điều kiện đóng băng: {error}", file=sys.stderr)
        return EXIT_GUARD
    write_freeze(freeze)
    deploy = freeze["deploy"]["run"]
    print(f"-> {freeze_lib.FREEZE_PATH}")
    print(f"bản triển khai {deploy}; {len(freeze['files'])} tệp đã băm sha256.")
    print(
        "Bước tiếp: commit results/ (gồm freeze.json) rồi chạy "
        "`python -m ml.evaluate --split test --final`."
    )
    return 0


# --- chấm cuối và diễn tập ------------------------------------------------------------------------


def available_for_rehearsal(deploy_run: str) -> list[str]:
    """Khoá mô hình đã có tệp để diễn tập."""
    from ml.predictors import baseline_path, deploy_onnx_path, phobert_run_dir

    keys = []
    for key in final_keys(deploy_run):
        if key in BASELINES:
            ok = all(baseline_path(task, key).is_file() for task in TASKS)
        elif key == DEPLOY_KEY:
            ok = deploy_onnx_path(deploy_run).is_file()
        else:
            ok = (phobert_run_dir(key) / "model.safetensors").is_file()
        if ok:
            keys.append(key)
    return keys


def render_outputs(ctx: EvalContext) -> list[Path]:
    """Bảng, hình và test_all.json từ dự đoán đã lưu; không đọc dữ liệu hay mô hình."""
    meta = read_meta(ctx.preds_dir)
    deploy_run = meta["deploy_run"]
    calibration = read_json(calibration_path(deploy_run))
    temperatures = {t: calibration[t]["temperature"] for t in TASKS}
    collected = eval_tables.collect_metrics(ctx.preds_dir, temperatures)
    rows = eval_tables.main_rows(collected)
    models = collected["models"]
    written: list[Path] = []

    document = {
        "split": ctx.split,
        "rehearsal": ctx.rehearsal,
        "evaluated_at": meta["created_at"],
        "freeze_sha256": meta.get("freeze_sha256"),
        "git_commit": meta.get("git_commit"),
        "deploy_run": deploy_run,
        **collected,
    }
    written.append(ctx.metrics_path("test_all.json"))
    write_json(written[-1], document)

    note = (
        "Baseline: F1 từ nhãn thô, ECE từ xác suất đã hiệu chuẩn nhiệt độ trên validation. PhoBERT "
        "FP32 tính ECE ở T = 1; bản int8 dùng T đã đóng băng. Chỉ có một run PhoBERT đa nhiệm "
        "(seed 42) nên không có trung bình theo seed: sai khác giữa các mô hình xem khoảng tin "
        "cậy và kiểm định ghép cặp ở significance.md."
    )
    label = "test" if ctx.split == "test" else "validation (diễn tập, KHÔNG phải test)"
    markdown = ctx.table_path("test_main.md")
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(
        eval_tables.main_markdown(rows, n=collected["n"], split_label=label, note=note),
        encoding="utf-8",
    )
    written.append(markdown)
    eval_tables.write_main_csv(rows, ctx.table_path("test_main.csv"))
    eval_tables.write_per_class_csv(models, ctx.table_path("test_per_class.csv"))
    written += [ctx.table_path("test_main.csv"), ctx.table_path("test_per_class.csv")]

    for task in TASKS:
        for key in models:
            written.append(
                eval_tables.plot_confusion(
                    ctx.figure_path(f"cm_{task}_{key}.png"),
                    models[key][task]["confusion"],
                    task,
                    f"{eval_tables.TASK_VI[task]} · {eval_tables.model_display(key)}",
                )
            )
        series = {
            "PhoBERT MTL int8": models[DEPLOY_KEY][task],
            "TF-IDF + LinearSVC": models["linearsvc"][task],
        }
        written.append(
            eval_tables.plot_reliability(
                ctx.figure_path(f"reliability_{task}.png"), series, task, "Độ tin cậy"
            )
        )
    return written


def cmd_final(rehearsal: bool, threads: int) -> int:
    ctx = context(rehearsal)
    freeze = None
    if not rehearsal:
        try:
            freeze = assert_final_allowed(preds_dir=ctx.preds_dir)
        except GuardError as error:
            print(f"evaluate: không được chấm test: {error}", file=sys.stderr)
            return EXIT_GUARD
        deploy_run = freeze["deploy"]["run"]
        keys = final_keys(deploy_run)
    else:
        deploy_run = best_run().name
        keys = available_for_rehearsal(deploy_run)
        shutil.rmtree(ctx.preds_dir, ignore_errors=True)

    from ml.predictors import compute_preds

    frame = ctx.load_split()  # mở test chỉ khi split == 'test', tức là đã qua mọi chốt chặn trên
    mode = "diễn tập" if rehearsal else "CHÍNH THỨC"
    print(f"Chấm {len(frame)} câu {ctx.split} cho {len(keys)} mô hình ({mode}):")
    started = time.time()
    results, descriptions = compute_preds(keys, frame, deploy_run, threads=threads)

    building = ctx.preds_dir.with_name(f"{ctx.preds_dir.name}.building")
    shutil.rmtree(building, ignore_errors=True)
    for preds in results:
        save_preds(building, preds)
    write_json(
        building / META_FILE,
        {
            "split": ctx.split,
            "n": len(frame),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "rehearsal": rehearsal,
            "deploy_run": deploy_run,
            "git_commit": git_head(),
            "git_dirty": git_dirty() if rehearsal else [],
            "freeze_sha256": None if freeze is None else sha256_file(freeze_lib.FREEZE_PATH),
            "seconds": round(time.time() - started, 1),
            "models": descriptions,
        },
    )
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (building / DONE_FILE).write_text(stamp + "\n", encoding="utf-8")
    building.rename(ctx.preds_dir)  # chỉ từ đây mới có thể thấy số: dự đoán đã nằm trọn trên đĩa
    return print_outputs(ctx)


def print_outputs(ctx: EvalContext) -> int:
    written = render_outputs(ctx)
    print(ctx.table_path("test_main.md").read_text(encoding="utf-8"))
    print(f"{len(written)} tệp trong {ctx.results_dir}")
    return 0


def cmd_tables(rehearsal: bool) -> int:
    ctx = context(rehearsal)
    if not is_done(ctx.preds_dir):
        print(f"evaluate: chưa có {ctx.preds_dir / DONE_FILE}: chưa chấm xong", file=sys.stderr)
        return EXIT_GUARD
    return print_outputs(ctx)


def cmd_record_rerun(reason: str) -> int:
    try:
        archive = record_rerun(reason, context().preds_dir)
    except GuardError as error:
        print(f"evaluate: {error}", file=sys.stderr)
        return EXIT_GUARD
    print(f"đã cất dự đoán cũ ở {archive}; commit results/freeze.json rồi chạy lại --final.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["test"], help="chỉ có test: cần --final hoặc --tables")
    parser.add_argument("--freeze", action="store_true", help="ghi results/freeze.json")
    parser.add_argument("--final", action="store_true", help="chấm test đúng một lần")
    parser.add_argument(
        "--tables", action="store_true", help="sinh lại bảng/hình từ dự đoán đã lưu"
    )
    parser.add_argument("--rehearsal", action="store_true", help="diễn tập trên validation")
    parser.add_argument("--record-rerun", metavar="LÝ DO", help="ghi lý do chạy lại test")
    parser.add_argument("--threads", type=int, default=None, help="luồng ORT (mặc định theo env)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    threads = args.threads or load_settings().ort_threads
    if args.record_rerun:
        return cmd_record_rerun(args.record_rerun)
    if args.freeze:
        return cmd_freeze()
    if args.rehearsal or args.split == "test":
        if args.tables:
            return cmd_tables(args.rehearsal)
        if args.final or args.rehearsal:
            return cmd_final(args.rehearsal, threads)
        parser.error("--split test cần --final hoặc --tables")
    parser.error(
        "chọn một trong --freeze, --split test --final, --split test --tables, --rehearsal"
    )
    return EXIT_GUARD


if __name__ == "__main__":
    sys.exit(main())
