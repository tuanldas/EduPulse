"""Đo hiệu năng engine trên câu thật: thời gian từng bước và bộ nhớ.

    python -m edupulse.engine.bench --input cau.txt --n 1000 --xai 20 --json

Các bước tính end-to-end như một batch thật: chuẩn hoá (PII, dấu thanh, tách từ) → ONNX → cờ theo
ngân sách → giải thích tự động cho các câu bị cờ có điểm thấp nhất. Thời gian nạp mô hình không
nằm trong tổng (xem `load_s`). Có `--json` thì stdout chỉ có một dòng JSON, log đi ra stderr.
"""

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from edupulse.config import AUTO_XAI_TOP, DEFAULT_BUDGET, XAI_MAX_WORDS, load_settings
from edupulse.engine.memory import cgroup_peak_mb, current_rss_mb
from edupulse.engine.service import Analyzer


def read_sentences(path: Path, limit: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()][:limit]


def measure(
    analyzer: Analyzer,
    texts: Sequence[str],
    xai_rows: int,
    budget_pct: float,
    batch_size: int | None = None,
) -> dict:
    """Chạy một batch như job thật và trả thời gian (giây) từng bước."""
    steps: dict[str, float] = {}

    started = time.perf_counter()
    prepared = analyzer.prepare_many(texts)
    steps["prepare"] = time.perf_counter() - started
    skipped = sum(p.n_words == 0 for p in prepared)  # câu không còn nội dung thì không có nhãn
    prepared = [p for p in prepared if p.n_words]

    started = time.perf_counter()
    batch_size = batch_size or analyzer.calibration.logits_batch_size
    analyses = analyzer.infer(prepared, batch_size=batch_size)
    steps["predict"] = time.perf_counter() - started

    started = time.perf_counter()
    analyses = analyzer.flag(analyses, budget_pct)
    steps["flags"] = time.perf_counter() - started

    started = time.perf_counter()
    candidates = sorted(
        (a for a in analyses if a.flag.needs_review and a.prepared.n_words <= XAI_MAX_WORDS),
        key=lambda a: a.score,
    )[:xai_rows]
    forwards = sum(analyzer.explain_prepared(a.prepared).n_forwards for a in candidates)
    steps["explain"] = time.perf_counter() - started

    total = sum(steps.values())
    return {
        "n": len(texts),
        "skipped_no_content": skipped,
        "budget_pct": budget_pct,
        "batch_size": batch_size,
        "flagged": sum(a.flag.needs_review for a in analyses),
        "xai_rows": len(candidates),
        "xai_forwards": forwards,
        "steps_s": {name: round(seconds, 3) for name, seconds in steps.items()},
        "total_s": round(total, 3),
        "per_sentence_ms": round(total / max(len(texts), 1) * 1000, 2),
    }


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="file mỗi dòng một câu")
    parser.add_argument("--n", type=int, default=1000, help="số câu đo")
    parser.add_argument("--xai", type=int, default=AUTO_XAI_TOP, help="số câu giải thích tự động")
    parser.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="ngân sách duyệt, %%")
    parser.add_argument(
        "--batch-size", type=int, help="cỡ lô ONNX; mặc định theo logits_batch_size của bundle"
    )
    parser.add_argument("--json", action="store_true", help="in một dòng JSON ra stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    texts = read_sentences(args.input, args.n)
    if len(texts) < args.n:
        print(f"Cần {args.n} câu, {args.input} chỉ có {len(texts)}", file=sys.stderr)
        return 2

    settings = load_settings()
    started = time.perf_counter()
    analyzer = Analyzer.load(settings)
    load_s = time.perf_counter() - started
    print(
        f"Đã nạp bundle {analyzer.bundle.manifest.bundle_version} trong {load_s:.1f} s",
        file=sys.stderr,
    )

    result = measure(analyzer, texts, args.xai, args.budget, args.batch_size)
    result.update(
        load_s=round(load_s, 2),
        rss_mb=current_rss_mb(),
        cgroup_memory_peak_mb=cgroup_peak_mb(),
        ort_threads=settings.ort_threads,
        ner=settings.ner,
        bundle_version=analyzer.bundle.manifest.bundle_version,
        model_version=analyzer.bundle.manifest.model_version,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for key, value in result.items():
            print(f"{key:>22}: {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
