"""Chấm các run PhoBERT và baseline: validation (tự do), đóng băng, và lần chấm test duy nhất.

    python -m ml.evaluate --split val              # bảng val, luật chọn seed, quy tắc MTL/STL
    python -m ml.evaluate --freeze                 # ghi results/freeze.json (rồi commit nó)
    python -m ml.evaluate --split test --final     # đúng một lần: dự đoán test của MỌI mô hình
    python -m ml.evaluate --split test --tables    # sinh lại bảng/hình từ dự đoán đã lưu
    python -m ml.evaluate --rehearsal              # diễn tập đường chấm cuối trên validation
    python -m ml.evaluate --record-rerun "lý do"   # ghi lý do chạy lại test sau khi sửa lỗi mã

`--final` chỉ chạy khi mọi chốt chặn của `ml.freeze.assert_final_allowed` thoả. Dự đoán của mọi
mô hình được ghi vào `outputs/test_preds/` (xem `ml.preds`) trước khi in bất kỳ số nào; từ đó mọi
phân tích chỉ đọc thư mục này. `--rehearsal` chạy đúng các bước ấy trên validation, ghi vào
`outputs/rehearsal/` (không vào `results/`), để bắt lỗi mã trước khi chạm test.

Mã thoát: 0 xong; 2 thiếu run hoặc tham số; 5 MTL kém STL quá 1,5 điểm mF1 ở một tác vụ (DỪNG,
hỏi người dùng); 6 một chốt chặn của lần chấm cuối chưa thoả.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.ticker import MaxNLocator

from edupulse.config import REPO_ROOT, load_settings
from edupulse.labels import LABELS, TASKS
from ml import eval_tables
from ml import freeze as freeze_lib
from ml.common import (
    best_run,
    finished_runs,
    read_json,
    sha256_file,
    task_metrics,
    write_json,
)
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
from ml.plotting import ACCENT, COLORS, MUTED, plt, save_figure_to, setup_style, vn
from ml.preds import (
    BASELINES,
    DEPLOY_KEY,
    DONE_FILE,
    META_FILE,
    SEEDS,
    EvalContext,
    all_keys,
    context,
    is_done,
    read_meta,
    save_preds,
)

EXIT_MISSING, EXIT_RULE, EXIT_GUARD = 2, 5, 6
# Luật r1: MTL thấp hơn STL quá ngần này (mF1, trung bình seed, val) ở tác vụ nào thì dừng.
MTL_LAG_POINTS = 1.5
ARCHES = ("mtl", "stl-sentiment", "stl-topic")
DEPLOY_RULE = (
    "Seed MTL có trung bình macro-F1 hai tác vụ trên validation (ở epoch tốt nhất) cao nhất; "
    "hoà thì lấy seed nhỏ hơn."
)
RESULTS = REPO_ROOT / "results"


class SummaryError(RuntimeError):
    """Thiếu run hoặc cấu hình lẫn lộn: không thể tóm tắt validation cho bước đóng băng."""


# --- validation -------------------------------------------------------------------------------


def run_summary(run: Path) -> dict[str, Any]:
    """Chỉ số val (T = 1) của một run từ logits epoch tốt nhất, cùng đường học theo epoch."""
    history = read_json(run / "history.json")
    data = np.load(run / "val_logits.npz")
    tasks = [task for task in TASKS if task in data.files]
    metrics = {t: task_metrics(data[t], data[f"y_{t}"], LABELS[t]) for t in tasks}
    return {
        "run": run.name,
        "seed": int(run.name.rsplit("-s", 1)[1]),
        "tasks": tasks,
        "n_val": len(data[f"y_{tasks[0]}"]),
        "best_epoch": history["best_epoch"],
        "best_score": history["best_score"],
        "epochs_trained": history["args"]["epochs"],
        "total_s": history.get("total_s"),
        "device": history.get("device"),
        "metrics": {
            t: {
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
                "weighted_f1": m["weighted_f1"],
                "ece": m["ece"],
                "nll": m["nll"],
                "per_class_f1": [c["f1"] for c in m["per_class"]],
            }
            for t, m in metrics.items()
        },
        "curve": [
            {"epoch": e["epoch"], **{t: e["val"][t]["macro_f1"] for t in tasks}}
            for e in history["epochs"]
        ],
    }


def runs_by_arch() -> dict[str, list[dict[str, Any]]]:
    return {
        arch: sorted(
            (run_summary(path) for path in finished_runs(f"{arch}-s")), key=lambda r: r["seed"]
        )
        for arch in ARCHES
    }


def mean_f1(runs: list[dict[str, Any]], task: str) -> float | None:
    values = [r["metrics"][task]["macro_f1"] for r in runs if task in r["metrics"]]
    return float(np.mean(values)) if values else None


def mtl_vs_stl(by_arch: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """So MTL với STL theo từng tác vụ; vi phạm khi STL hơn MTL quá `MTL_LAG_POINTS` điểm mF1."""
    verdict: dict[str, Any] = {"threshold_points": MTL_LAG_POINTS, "tasks": {}, "violations": []}
    for task in TASKS:
        mtl, stl = by_arch["mtl"], by_arch[f"stl-{task}"]
        mtl_mean, stl_mean = mean_f1(mtl, task), mean_f1(stl, task)
        delta = None if mtl_mean is None or stl_mean is None else (mtl_mean - stl_mean) * 100
        violated = delta is not None and delta < -MTL_LAG_POINTS
        verdict["tasks"][task] = {
            "mtl_mean_macro_f1": mtl_mean,
            "stl_mean_macro_f1": stl_mean,
            "mtl_minus_stl_points": delta,
            "n_seeds": {"mtl": len(mtl), "stl": len(stl)},
            "violates": violated,
        }
        if violated:
            verdict["violations"].append(task)
    return verdict


def aggregate_arch(runs: list[dict[str, Any]], task: str) -> dict[str, Any] | None:
    chosen = [r for r in runs if task in r["metrics"]]
    if not chosen:
        return None
    stats = {
        metric: eval_tables.seed_stat([r["metrics"][task][metric] for r in chosen])
        for metric in ("accuracy", "macro_f1", "weighted_f1", "ece")
    }
    return {"seeds": [r["seed"] for r in chosen], **stats}


def build_val_summary(*, require_complete: bool = False) -> dict[str, Any]:
    by_arch = runs_by_arch()
    runs = [r for group in by_arch.values() for r in group]
    if not runs:
        raise SummaryError("chưa có run PhoBERT nào hoàn tất trong models/phobert/")
    problems: list[str] = []
    epochs = {r["epochs_trained"] for r in runs}
    if len(epochs) > 1:
        problems.append(f"các run train với số epoch khác nhau: {sorted(epochs)}")
    for arch in ARCHES:
        seeds = [r["seed"] for r in by_arch[arch]]
        if require_complete and seeds != list(SEEDS):
            problems.append(f"{arch}: có seed {seeds}, cần {list(SEEDS)}")
    if not by_arch["mtl"]:
        problems.append("chưa có run đa nhiệm (mtl-s*) nào")
    if problems:
        raise SummaryError("; ".join(problems))

    deploy = best_run()
    ranking = sorted(by_arch["mtl"], key=lambda r: (-r["best_score"], r["seed"]))
    verdict = mtl_vs_stl(by_arch)
    return {
        "split": "validation",
        "n_val": runs[0]["n_val"],
        "epochs": epochs.pop(),
        "std_note": "độ lệch chuẩn mẫu (ddof = 1) giữa các seed",
        "runs": by_arch,
        "architectures": {
            "mtl": {t: aggregate_arch(by_arch["mtl"], t) for t in TASKS},
            "stl": {t: aggregate_arch(by_arch[f"stl-{t}"], t) for t in TASKS},
        },
        "mtl_vs_stl": verdict,
        "deploy": {
            "run": deploy.name,
            "rule": DEPLOY_RULE,
            "ranking": [{"run": r["run"], "score": r["best_score"]} for r in ranking],
        },
        "baselines": baseline_val(),
    }


def baseline_val() -> dict[str, Any]:
    path = RESULTS / "metrics" / "val_baselines.json"
    if not path.is_file():
        return {}
    data = read_json(path)
    return {task: {name: data["tasks"][task][name]["raw"] for name in BASELINES} for task in TASKS}


def val_markdown(summary: dict[str, Any]) -> str:
    def cell(stat: dict[str, Any] | None, key: str) -> str:
        if not stat:
            return "–"
        item = stat[key]
        text = vn(item["mean"] * 100, 1)
        return f"{text} ± {vn(item['std'] * 100, 1)}" if item["n"] > 1 else text

    arch = summary["architectures"]
    lines = [
        f"Validation (n = {vn(summary['n_val'], 0)}), {summary['epochs']} epoch; "
        f"mean ± {summary['std_note']}.",
        "",
        "| Kiến trúc | Seed | Cảm xúc acc | Cảm xúc mF1 | Chủ đề acc | Chủ đề mF1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, key in (("Đa nhiệm (MTL)", "mtl"), ("Đơn nhiệm (STL)", "stl")):
        seeds = sorted({s for t in TASKS for s in (arch[key][t] or {}).get("seeds", [])})
        sent, topic = arch[key]["sentiment"], arch[key]["topic"]
        lines.append(
            f"| {name} | {', '.join(map(str, seeds))} | {cell(sent, 'accuracy')} "
            f"| {cell(sent, 'macro_f1')} | {cell(topic, 'accuracy')} | {cell(topic, 'macro_f1')} |"
        )
    verdict = summary["mtl_vs_stl"]
    lines += [
        "",
        "| Tác vụ | MTL mF1 | STL mF1 | MTL − STL (điểm) | Vi phạm luật 1,5 điểm |",
        "|---|---:|---:|---:|---|",
    ]
    for task, item in verdict["tasks"].items():
        delta = item["mtl_minus_stl_points"]
        violated = "CÓ" if item["violates"] else "không"

        def percent(value: float | None) -> str:
            return "–" if value is None else vn(value * 100, 1)

        lines.append(
            f"| {eval_tables.TASK_VI[task]} | {percent(item['mtl_mean_macro_f1'])} "
            f"| {percent(item['stl_mean_macro_f1'])} "
            f"| {'–' if delta is None else vn(delta, 2)} | {violated} |"
        )
    lines += [
        "",
        "Theo từng seed:",
        "",
        "| Run | Epoch tốt nhất | Cảm xúc mF1 | Chủ đề mF1 | Điểm chọn |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in summary["runs"].values():
        for run in group:
            sent = run["metrics"].get("sentiment", {}).get("macro_f1")
            topic = run["metrics"].get("topic", {}).get("macro_f1")
            lines.append(
                f"| {run['run']} | {run['best_epoch']} "
                f"| {'–' if sent is None else vn(sent * 100, 2)} "
                f"| {'–' if topic is None else vn(topic * 100, 2)} "
                f"| {vn(run['best_score'] * 100, 2)} |"
            )
    deploy = summary["deploy"]
    lines += ["", f"Bản triển khai: **{deploy['run']}**. Luật: {deploy['rule']}"]
    return "\n".join(lines) + "\n"


def plot_val_curves(summary: dict[str, Any], path: Path) -> Path:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharex=True)
    for ax, task in zip(axes, TASKS, strict=True):
        for arch, style in (("mtl", "-"), (f"stl-{task}", "--")):
            for run in summary["runs"][arch]:
                points = [(c["epoch"], c[task] * 100) for c in run["curve"] if task in c]
                colour = ACCENT if arch == "mtl" else COLORS["training_program"]
                ax.plot(
                    *zip(*points, strict=True),
                    style,
                    marker="o",
                    markersize=3.5,
                    color=colour,
                    alpha=0.8,
                    label=f"{arch} · s{run['seed']}",
                )
        ax.set_title(eval_tables.TASK_VI[task], loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylabel("macro-F1 validation (%)")
        ax.grid(axis="y", color="#DCDAD2", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2, labelcolor=MUTED)
    return save_figure_to(fig, path)


def write_val_outputs(summary: dict[str, Any]) -> list[Path]:
    paths = [RESULTS / "metrics" / "val_experiments.json", RESULTS / "tables" / "val_mtl_vs_stl.md"]
    write_json(paths[0], summary)
    paths[1].parent.mkdir(parents=True, exist_ok=True)
    paths[1].write_text(val_markdown(summary), encoding="utf-8")
    paths.append(plot_val_curves(summary, RESULTS / "figures" / "val_curves.png"))
    return paths


def cmd_val() -> int:
    try:
        summary = build_val_summary()
    except SummaryError as error:
        print(f"evaluate: {error}", file=sys.stderr)
        return EXIT_MISSING
    for path in write_val_outputs(summary):
        print(f"-> {path}")
    print(val_markdown(summary))
    violations = summary["mtl_vs_stl"]["violations"]
    if violations:
        print(
            f"DỪNG: MTL kém STL quá {MTL_LAG_POINTS} điểm mF1 ở {violations}; luật đã chốt là "
            "báo người dùng (không tự đổi quyết định triển khai MTL).",
            file=sys.stderr,
        )
        return EXIT_RULE
    return 0


# --- đóng băng ----------------------------------------------------------------------------------


def cmd_freeze() -> int:
    try:
        summary = build_val_summary(require_complete=True)
    except SummaryError as error:
        print(f"evaluate: chưa đủ điều kiện đóng băng: {error}", file=sys.stderr)
        return EXIT_MISSING
    write_val_outputs(summary)
    if summary["mtl_vs_stl"]["violations"]:
        print(f"DỪNG: MTL kém STL quá {MTL_LAG_POINTS} điểm mF1; không đóng băng.", file=sys.stderr)
        return EXIT_RULE
    context_now = context()
    if context_now.preds_dir.exists():
        print(
            f"{context_now.preds_dir} đã có: test đã được chấm, không đóng băng lại. Dùng "
            "--record-rerun nếu phải chấm lại.",
            file=sys.stderr,
        )
        return EXIT_GUARD
    try:
        freeze = build_freeze(summary)
    except GuardError as error:
        print(f"evaluate: {error}", file=sys.stderr)
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
    """Khoá mô hình đã có tệp để diễn tập (không đòi đủ 9 run)."""
    from ml.predictors import baseline_path, deploy_onnx_path, phobert_run_dir

    keys = []
    for key in all_keys():
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
        "FP32 ECE ở T = 1; bản int8 dùng T đã đóng băng."
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

    deploy_seed = int(deploy_run.rsplit("-s", 1)[1])
    for task in TASKS:
        shown = [*BASELINES, DEPLOY_KEY, f"stl-{task}-s{deploy_seed}"]
        for key in shown:
            if key in models and task in models[key]:
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
        keys = all_keys()
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
    parser.add_argument(
        "--split", choices=["val", "test"], help="val tự do; test cần --final/--tables"
    )
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
    if args.split == "val":
        return cmd_val()
    parser.error("chọn một trong --split val, --freeze, --split test --final, --rehearsal")
    return EXIT_MISSING


if __name__ == "__main__":
    sys.exit(main())
