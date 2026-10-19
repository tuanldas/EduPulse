"""Hàng đợi các run PhoBERT của phase thực nghiệm: tuần tự, chạy lại được, giữ một khoá MPS.

    bash ml/run-experiments.sh                    # make experiments: các run còn thiếu (E* = 4)
    bash ml/run-experiments.sh --dry-run          # chỉ in kế hoạch và ước tính thời gian
    bash ml/run-experiments.sh --explore          # thêm run thăm dò 6 epoch (không tính vào bảng)
    bash ml/run-experiments.sh --epochs 6 --retrain-mismatched   # chốt E* = 6, huấn luyện lại

Thứ tự: MTL seed 43, 44 (đủ 3 seed đa nhiệm trước), rồi từng cặp đơn nhiệm cảm xúc/chủ đề theo
seed, để nếu hàng đợi dừng giữa chừng vẫn có phần so sánh MTL với STL dùng được. Run đã hoàn tất
với đúng siêu tham số thì bỏ qua; run hoàn tất nhưng siêu tham số khác (vd. đổi số epoch) thì từ
chối chạy trừ khi có `--retrain-mismatched`, để không âm thầm trộn hai cấu hình vào một bảng.

Hàng đợi giữ khoá MPS toàn máy suốt cả hàng (các run con chạy với `--lock-held`), nên không job
nào khác chen vào giữa hai run; từ chối chạy khi Docker Desktop đang bật (r1: VM Docker ~7GB từng
làm swap thrash cả máy) trừ khi có `--allow-docker`; giữ máy không ngủ bằng `caffeinate`. Nhật ký
từng run ở `outputs/logs/<run>.log`, trạng thái cả hàng ở `outputs/logs/experiments.status.json`.

Mã thoát: 0 xong, 1 có run lỗi, 2 tham số hoặc run cũ lệch cấu hình, 3 Docker đang chạy, 4 khoá
MPS đang do job khác giữ, 130 bị ngắt.
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from edupulse.config import REPO_ROOT
from ml.common import (
    MpsBusyError,
    docker_running,
    keep_awake,
    mps_lock,
    phobert_run_name,
    phobert_runs_dir,
    read_json,
    settings,
    write_json,
)

SEEDS = (42, 43, 44)
DEFAULT_EPOCHS = 4
EXPLORE_EPOCHS = 6
# Phút cho mỗi epoch trên M1 Pro (run MTL thật: 692 s cho 4 epoch, chiếm hết bộ nhớ MPS).
MINUTES_PER_EPOCH = 2.9
# Siêu tham số mặc định của `train_phobert` mà một run phải khớp mới được coi là cùng cấu hình
# (test so với `train_phobert.build_parser()` để hai nơi không lệch nhau).
EXPECTED_ARGS: dict[str, Any] = {
    "lr": 2e-5,
    "batch_size": 32,
    "max_len": 128,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "clip": 1.0,
    "limit": 0,
}
EXIT_FAILED, EXIT_USAGE, EXIT_DOCKER, EXIT_LOCK, EXIT_INTERRUPTED = 1, 2, 3, 4, 130


@dataclass(frozen=True)
class RunSpec:
    heads: str  # 'mtl' | 'sentiment' | 'topic'
    seed: int
    epochs: int | None = None  # None: theo --epochs của hàng đợi
    name: str | None = None  # None: tên chuẩn của train_phobert

    @property
    def run_name(self) -> str:
        return self.name or phobert_run_name(self.heads, self.seed)


def default_queue() -> list[RunSpec]:
    """9 run của bảng so sánh: MTL ×3, rồi STL cảm xúc và chủ đề ×3 xen kẽ theo seed."""
    stl = [RunSpec(heads, seed) for seed in SEEDS for heads in ("sentiment", "topic")]
    return [*(RunSpec("mtl", seed) for seed in SEEDS), *stl]


def explore_run() -> RunSpec:
    """Thăm dò E*: MTL seed 42 với 6 epoch, ghi riêng để không lẫn vào bảng so sánh."""
    return RunSpec("mtl", 42, epochs=EXPLORE_EPOCHS, name="explore-mtl-e6-s42")


def mismatches(history: dict[str, Any], epochs: int) -> list[str]:
    """Điểm khác giữa cấu hình của run đã hoàn tất và cấu hình hàng đợi muốn."""
    from edupulse.text import NORMALIZER_VERSION

    args = history.get("args", {})
    wanted = {**EXPECTED_ARGS, "epochs": epochs}
    found = [
        f"{key}: đã chạy với {args.get(key)!r}, hàng đợi muốn {value!r}"
        for key, value in wanted.items()
        if args.get(key) != value
    ]
    if history.get("normalizer_version") != NORMALIZER_VERSION:
        found.append(
            f"normalizer_version: đã chạy với {history.get('normalizer_version')!r}, "
            f"mã đang là {NORMALIZER_VERSION!r}"
        )
    return found


def run_state(spec: RunSpec, epochs: int) -> tuple[str, str]:
    """('done' | 'pending' | 'mismatch', ghi chú) của một run, đọc từ history.json."""
    history_path = phobert_runs_dir() / spec.run_name / "history.json"
    if not history_path.is_file():
        return "pending", "chưa có"
    history = read_json(history_path)
    if not history.get("finished"):
        return "pending", "dở dang, chạy lại từ đầu"
    diffs = mismatches(history, spec.epochs or epochs)
    if diffs:
        return "mismatch", "; ".join(diffs)
    return "done", f"epoch tốt nhất {history.get('best_epoch')}, {history.get('total_s', 0):.0f}s"


def estimate_minutes(specs: list[RunSpec], epochs: int) -> float:
    return sum(MINUTES_PER_EPOCH * (spec.epochs or epochs) for spec in specs)


Runner = Callable[..., int]


def run_training(
    spec: RunSpec, *, epochs: int, force: bool, allow_docker: bool, log_path: Path
) -> int:
    """Chạy một run bằng tiến trình con `ml.train_phobert`; nhật ký nối vào `log_path`."""
    command = [
        sys.executable,
        "-m",
        "ml.train_phobert",
        "--heads",
        spec.heads,
        "--seed",
        str(spec.seed),
        "--epochs",
        str(spec.epochs or epochs),
        "--lock-held",
    ]
    if spec.name:
        command += ["--out", str(phobert_runs_dir() / spec.name)]
    if force:
        command.append("--force")
    if allow_docker:
        command.append("--allow-docker")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(command)}\n")
        log.flush()
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=REPO_ROOT)
        try:
            return child.wait()
        finally:
            if child.poll() is None:  # bị ngắt: không để run con mồ côi giữ MPS khi khoá đã nhả
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


@contextmanager
def sigterm_as_interrupt():
    """SIGTERM (vd. `kill`) thành KeyboardInterrupt để khối `finally` dọn run con và nhả khoá."""

    def handler(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} phút"


def run_queue(
    specs: list[RunSpec],
    *,
    epochs: int = DEFAULT_EPOCHS,
    dry_run: bool = False,
    allow_docker: bool = False,
    retrain_mismatched: bool = False,
    runner: Runner = run_training,
    docker_check: Callable[[], bool] = docker_running,
    out: Callable[[str], None] = print,
) -> int:
    states = {spec.run_name: run_state(spec, epochs) for spec in specs}
    out(f"Hàng đợi {len(specs)} run, mỗi run {epochs} epoch (trừ run nêu riêng):")
    for spec in specs:
        state, note = states[spec.run_name]
        out(f"  {spec.run_name:<20} {state:<8} {note}")

    stale = [spec for spec in specs if states[spec.run_name][0] == "mismatch"]
    if stale and not retrain_mismatched:
        out(
            f"{len(stale)} run đã hoàn tất nhưng khác cấu hình hàng đợi; chạy lại cả bộ với "
            "--retrain-mismatched nếu cố ý (vd. chốt E* mới), không thì sửa --epochs"
        )
        return EXIT_USAGE
    pending = [spec for spec in specs if states[spec.run_name][0] != "done"]
    out(
        f"Còn {len(pending)} run, ước tính {minutes(estimate_minutes(pending, epochs) * 60)} "
        f"(~{MINUTES_PER_EPOCH} phút/epoch trên M1 Pro, chưa tính tắt/bật Docker)"
    )
    docker_on = docker_check()
    out(f"Docker Desktop: {'ĐANG CHẠY' if docker_on else 'tắt'}")
    if dry_run:
        return 0
    if not pending:
        out("Không còn run nào cần chạy.")
        return 0
    if docker_on and not allow_docker:
        out(
            "Tắt Docker Desktop trước khi train MPS (VM Docker làm swap thrash máy 16GB), rồi chạy "
            "lại; hoặc thêm --allow-docker nếu chấp nhận rủi ro."
        )
        return EXIT_DOCKER

    log_dir = settings().outputs_dir / "logs"
    status_path = log_dir / "experiments.status.json"
    status: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "epochs": epochs,
        "finished": False,
        "runs": [],
    }
    failed: list[str] = []
    awake = keep_awake()
    try:
        with sigterm_as_interrupt(), mps_lock(note="experiment_queue"):
            for spec in pending:
                started = time.time()
                out(f"[{time.strftime('%H:%M:%S')}] bắt đầu {spec.run_name}")
                code = runner(
                    spec,
                    epochs=epochs,
                    force=states[spec.run_name][0] == "mismatch",
                    allow_docker=allow_docker,
                    log_path=log_dir / f"{spec.run_name}.log",
                )
                seconds = time.time() - started
                status["runs"].append(
                    {"run": spec.run_name, "exit_code": code, "seconds": round(seconds, 1)}
                )
                write_json(status_path, status)
                out(
                    f"[{time.strftime('%H:%M:%S')}] xong {spec.run_name}: mã {code}, "
                    f"{minutes(seconds)}"
                )
                if code in (EXIT_DOCKER, EXIT_LOCK):  # môi trường sai: các run sau cũng sẽ sai
                    out("Dừng hàng đợi: Docker đang chạy hoặc khoá MPS bị job khác giữ.")
                    status["finished"] = False
                    write_json(status_path, status)
                    return code
                if code != 0:
                    failed.append(spec.run_name)
    except MpsBusyError as error:
        out(str(error))
        return EXIT_LOCK
    except KeyboardInterrupt:
        out("Bị ngắt; run đang chạy đã được dừng, khoá MPS đã nhả.")
        return EXIT_INTERRUPTED
    finally:
        if awake is not None:
            awake.terminate()

    status["finished"] = True
    status["failed"] = failed
    write_json(status_path, status)
    if failed:
        out(f"{len(failed)} run lỗi: {', '.join(failed)} (xem outputs/logs/); chạy lại để thử tiếp")
        return EXIT_FAILED
    out("Hàng đợi xong.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="E* (mặc định 4)")
    parser.add_argument("--dry-run", action="store_true", help="chỉ in kế hoạch")
    parser.add_argument("--explore", action="store_true", help="thêm run thăm dò MTL 6 epoch")
    parser.add_argument("--allow-docker", action="store_true", help="chạy dù Docker đang bật")
    parser.add_argument(
        "--retrain-mismatched",
        action="store_true",
        help="chạy lại cả run đã hoàn tất nhưng khác cấu hình (vd. đổi E*)",
    )
    parser.add_argument(
        "--only", help="chỉ các run này, ví dụ mtl-s43,stl-topic-s42 (tên thư mục run)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        print("--epochs phải >= 1", file=sys.stderr)
        return EXIT_USAGE
    specs = default_queue()
    if args.explore:
        specs = [explore_run(), *specs]
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        unknown = wanted - {spec.run_name for spec in specs}
        if unknown:
            print(f"không có run {sorted(unknown)}", file=sys.stderr)
            return EXIT_USAGE
        specs = [spec for spec in specs if spec.run_name in wanted]
    return run_queue(
        specs,
        epochs=args.epochs,
        dry_run=args.dry_run,
        allow_docker=args.allow_docker,
        retrain_mismatched=args.retrain_mismatched,
    )


if __name__ == "__main__":
    sys.exit(main())
