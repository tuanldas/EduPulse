"""Huấn luyện PhoBERT-base-v2 đa nhiệm hoặc đơn nhiệm bằng vòng lặp tay (không dùng Trainer).

Cấu hình đã chạy thật trên MPS: lr 2e-5, batch 32 gộp theo độ dài, warmup 10% rồi giảm tuyến tính,
weight decay 0,01 (trừ bias và LayerNorm), clip 1,0, fp32, max_len 128, CE thường. Chỉ dùng
train để học và validation để chọn epoch; KHÔNG bao giờ đọc test.

    python -m ml.train_phobert --heads mtl --seed 42            # make train
    python -m ml.train_phobert --heads sentiment --seed 43 --epochs 6

Kết quả trong `models/phobert/<heads>-s<seed>/`: `model.safetensors` + `config.json` (epoch tốt
nhất theo macro-F1 trung bình trên val), `epoch1/` (checkpoint sau epoch 1, dùng khi cần đo gấp),
`val_logits.npz` (logits val của epoch tốt nhất), `history.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from edupulse.labels import LABELS
from edupulse.text import NORMALIZER_VERSION
from ml.common import (
    MpsBusyError,
    docker_running,
    load_processed,
    mps_lock,
    pick_device,
    set_seed,
    settings,
    task_metrics,
    write_json,
)
from ml.phobert_model import (
    BASE_MODEL,
    BASE_REVISION,
    MAX_LEN,
    PAD_ID,
    MultiHeadPhoBERT,
    encode_ids,
    head_sizes,
    load_pretrained,
    load_tokenizer,
)

PAD_MULTIPLE = 8  # đệm về bội số của 8 để MPS tái dùng đồ thị tính toán
EVAL_BATCH = 128
LOG_EVERY = 50
CACHE_FLUSH_EVERY = 100  # MPS không tự dọn bộ nhớ đệm đồ thị: dọn định kỳ để RAM phẳng


def run_name(heads: str, seed: int, dry: bool) -> str:
    base = "mtl" if heads == "mtl" else f"stl-{heads}"
    return f"{'dry-' if dry else ''}{base}-s{seed}"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def keep_awake() -> subprocess.Popen[bytes] | None:
    """Giữ máy macOS không ngủ trong suốt tiến trình này (ngủ giữa chừng từng làm mất một run)."""
    if platform.system() != "Darwin" or shutil.which("caffeinate") is None:
        return None
    return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])


def pad_batch(
    sequences: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    longest = max(len(s) for s in sequences)
    width = math.ceil(longest / PAD_MULTIPLE) * PAD_MULTIPLE
    ids = torch.full((len(sequences), width), PAD_ID, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        ids[row, : len(sequence)] = torch.tensor(sequence)
        mask[row, : len(sequence)] = 1
    return ids.to(device), mask.to(device)


def epoch_batches(lengths: list[int], batch_size: int, seed: int, epoch: int) -> list[list[int]]:
    """Xáo trộn, gộp theo độ dài trong từng khúc `50 batch` rồi xáo thứ tự batch.

    Câu cùng độ dài đứng cùng batch nên ít đệm; xáo hai tầng giữ tính ngẫu nhiên của SGD.
    """
    generator = torch.Generator().manual_seed(seed * 1000 + epoch)
    order = torch.randperm(len(lengths), generator=generator).tolist()
    span = batch_size * 50
    batches: list[list[int]] = []
    for start in range(0, len(order), span):
        chunk = sorted(order[start : start + span], key=lambda i: lengths[i], reverse=True)
        batches += [chunk[i : i + batch_size] for i in range(0, len(chunk), batch_size)]
    shuffled = torch.randperm(len(batches), generator=generator).tolist()
    return [batches[i] for i in shuffled]


@torch.no_grad()
def predict(
    model: MultiHeadPhoBERT,
    sequences: list[list[int]],
    device: torch.device,
    tasks: tuple[str, ...],
) -> dict[str, np.ndarray]:
    """Logits từng head theo đúng thứ tự đầu vào."""
    model.eval()
    order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
    outputs = {
        task: np.zeros((len(sequences), len(LABELS[task])), dtype=np.float32) for task in tasks
    }
    for start in range(0, len(order), EVAL_BATCH):
        index = order[start : start + EVAL_BATCH]
        ids, mask = pad_batch([sequences[i] for i in index], device)
        logits = model(input_ids=ids, attention_mask=mask)
        for task in tasks:
            outputs[task][index] = logits[task].float().cpu().numpy()
    model.train()
    return outputs


def optimizer_for(
    model: nn.Module, lr: float, weight_decay: float, fused: bool
) -> torch.optim.Optimizer:
    no_decay = ("bias", "LayerNorm.weight")
    decay, keep = [], []
    for name, parameter in model.named_parameters():
        (keep if any(token in name for token in no_decay) else decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": keep, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, fused=fused)


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "ece": metrics["ece"],
        "per_class_f1": [c["f1"] for c in metrics["per_class"]],
    }


def device_peak_mb(device: str) -> float:
    if device == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def train(args: argparse.Namespace, run_dir: Path, device_name: str) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(device_name)
    tasks = tuple(head_sizes(args.heads))

    train_df = load_processed("train")
    val_df = load_processed("validation")
    if args.limit:
        train_df, val_df = train_df.head(args.limit), val_df.head(max(args.limit // 2, 8))
    tokenizer = load_tokenizer(args.max_len)
    # encode_ids đổi mảnh BPE ngoài bảng embedding (id >= 64.001) thành <unk> như tokenizer gốc.
    train_ids = encode_ids(tokenizer, train_df["text_rdr"].tolist())
    val_ids = encode_ids(tokenizer, val_df["text_rdr"].tolist())
    lengths = [len(s) for s in train_ids]
    targets = {t: torch.tensor(train_df[t].to_numpy()) for t in tasks}
    val_targets = {t: val_df[t].to_numpy() for t in tasks}
    log(f"dữ liệu: train {len(train_ids)} · val {len(val_ids)} · tasks {tasks} · {device_name}")

    model = load_pretrained(args.heads).to(device)
    model.train()
    optimizer = optimizer_for(model, args.lr, args.weight_decay, fused=device_name != "cpu")
    steps_per_epoch = math.ceil(len(train_ids) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(args.warmup_ratio * total_steps), total_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    history: dict[str, Any] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "normalizer_version": NORMALIZER_VERSION,
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "device": device_name,
        "torch": torch.__version__,
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "epochs": [],
        "finished": False,
    }
    best_score = -1.0
    started = time.time()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        for step, batch in enumerate(epoch_batches(lengths, args.batch_size, args.seed, epoch)):
            ids, mask = pad_batch([train_ids[i] for i in batch], device)
            logits = model(input_ids=ids, attention_mask=mask)
            loss = sum(loss_fn(logits[t], targets[t][batch].to(device)) for t in tasks)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            if device_name == "mps" and step % CACHE_FLUSH_EVERY == CACHE_FLUSH_EVERY - 1:
                if hasattr(torch.mps, "clear_graph_cache"):
                    torch.mps.clear_graph_cache()
                torch.mps.empty_cache()
            if step % LOG_EVERY == 0:
                log(f"epoch {epoch + 1} bước {step}/{steps_per_epoch} loss={loss.item():.4f}")
        train_s = time.time() - epoch_start

        val_logits = predict(model, val_ids, device, tasks)
        metrics = {t: task_metrics(val_logits[t], val_targets[t], LABELS[t]) for t in tasks}
        score = float(np.mean([metrics[t]["macro_f1"] for t in tasks]))
        record = {
            "epoch": epoch + 1,
            "train_s": round(train_s, 1),
            "score": score,
            "peak_mb": round(device_peak_mb(device_name)),
            "val": {t: compact(metrics[t]) for t in tasks},
        }
        history["epochs"].append(record)
        summary = " · ".join(f"{t} mF1 {metrics[t]['macro_f1']:.4f}" for t in tasks)
        log(f"epoch {epoch + 1}/{args.epochs}: {summary} · train {train_s:.0f}s")

        if epoch == 0:
            model.save_pretrained(run_dir / "epoch1")
        if score > best_score:
            best_score = score
            model.save_pretrained(run_dir)
            np.savez(
                run_dir / "val_logits.npz",
                **{t: val_logits[t] for t in tasks},
                **{f"y_{t}": val_targets[t] for t in tasks},
            )
            history["best_epoch"] = epoch + 1
            history["best_score"] = score
        write_json(run_dir / "history.json", history)

    history["total_s"] = round(time.time() - started, 1)
    history["peak_mb"] = round(device_peak_mb(device_name))
    history["finished"] = True
    write_json(run_dir / "history.json", history)
    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--heads", choices=["mtl", *LABELS], default="mtl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument(
        "--limit", type=int, default=0, help="chạy thử trên N câu đầu (thư mục dry-*)"
    )
    parser.add_argument("--out", type=Path, help="thư mục run; mặc định models/phobert/<tên>")
    parser.add_argument("--force", action="store_true", help="chạy lại dù run đã hoàn tất")
    parser.add_argument(
        "--allow-docker", action="store_true", help="vẫn train MPS khi Docker Desktop đang chạy"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1 or args.limit < 0:
        print("epochs phải >= 1 và limit >= 0", file=sys.stderr)
        return 2
    device_name = pick_device(args.device)
    run_dir = args.out or settings().models_dir / "phobert" / run_name(
        args.heads, args.seed, dry=bool(args.limit)
    )
    history_path = run_dir / "history.json"
    if history_path.is_file() and not args.force:
        previous = json.loads(history_path.read_text(encoding="utf-8"))
        if previous.get("finished"):
            log(f"đã có run hoàn tất ở {run_dir} (dùng --force để chạy lại)")
            return 0

    if device_name == "mps" and docker_running() and not args.allow_docker:
        print(
            "Docker Desktop đang chạy: train MPS cùng lúc dễ làm máy thrash swap (RAM 16GB). "
            "Tắt Docker Desktop rồi chạy lại, hoặc thêm --allow-docker nếu chấp nhận rủi ro.",
            file=sys.stderr,
        )
        return 3

    # Khoá toàn máy cho mọi lần train thật: MPS chiếm GPU và hàng GB bộ nhớ, còn train CPU chiếm
    # hết lõi nên phép đo hiệu năng cùng lúc sẽ sai. Lần chạy thử (--limit) không cần khoá.
    exclusive = device_name == "mps" or (device_name == "cpu" and not args.limit)
    run_dir.mkdir(parents=True, exist_ok=True)
    awake = keep_awake()
    try:
        with mps_lock(note=f"train_phobert {run_dir.name}") if exclusive else nullcontext():
            history = train(args, run_dir, device_name)
    except MpsBusyError as error:
        print(error, file=sys.stderr)
        return 4
    finally:
        if awake is not None:
            awake.terminate()
    best = history["epochs"][history["best_epoch"] - 1]["val"]
    scores = " · ".join(f"{task} mF1 {values['macro_f1']:.4f}" for task, values in best.items())
    log(
        f"xong {run_dir.name}: epoch tốt nhất {history['best_epoch']} ({scores}), "
        f"tổng {history['total_s']:.0f}s, đỉnh bộ nhớ {history['peak_mb']} MB"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
