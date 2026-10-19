"""Kiểm tra đường chạy amd64 cho cùng nhãn với arm64 (PDR 4).

Image amd64 chạy giả lập (Rosetta hoặc QEMU) trên máy arm64, nên script chỉ chứng minh nhãn đúng
trên x86 (wheel, JRE, bộ tách từ), không đo hiệu năng. Hai điều kiện:

    1. 20 câu golden tự viết: nhãn (cảm xúc, chủ đề) đúng 20/20 với golden_predictions.json của
       bundle (sinh trên arm64);
    2. 200 câu val chọn cố định theo hạt giống: ít nhất 98% câu có CẢ hai nhãn trùng với image
       arm64.

    uv run python scripts/check_amd64.py --build   # dựng edupulse:<bundle>-amd64 (chậm) rồi kiểm
    uv run python scripts/check_amd64.py           # image đã dựng: chạy hai image và so nhãn

Ghi `results/perf/amd64-check-<ngày>.{json,md}`; thoát mã 1 khi một điều kiện không đạt. Báo cáo
không chép câu val (UIT-VSFC không nằm trong repo): chỉ ghi số dòng trong tập val.
"""

import argparse
import json
import math
import random
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from smoke_test import (
    Container,
    SmokeError,
    call,
    docker,
    free_port,
    git_commit,
    host_info,
    load_sentences,
    table,
    wait_for_health,
)

from edupulse.config import REPO_ROOT, Settings, load_settings

Labels = tuple[str, str]  # (cảm xúc, chủ đề)

GOLDEN_SENTENCES = REPO_ROOT / "tests" / "fixtures" / "golden_sentences.json"
SAMPLE_SIZE = 200
SAMPLE_SEED = 0
AGREEMENT_MIN_PCT = 98
CPU_FLAGS = ("sse4_2", "avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "amx_int8", "asimddp")
CPU_FLAGS_CODE = (
    "import re;"
    "m=re.search(r'^(?:flags|Features)\\s*:(.*)$',open('/proc/cpuinfo').read(),re.M);"
    f"print(*(sorted(set(m.group(1).split()) & set({list(CPU_FLAGS)!r})) if m else []))"
)
OUTCOME = {True: "ĐẠT", False: "KHÔNG ĐẠT"}


class CheckError(SmokeError):
    """Lỗi làm hỏng phép kiểm (không dựng được image, container không lên, phản hồi sai dạng)."""


# --- Logic so khớp (không cần container) --------------------------------------------------------


def pick_indices(total: int, n: int, seed: int = SAMPLE_SEED) -> list[int]:
    """`n` chỉ số khác nhau trong [0, total), cố định theo `seed`, xếp tăng dần."""
    if n < 1:
        raise CheckError("Số câu val phải từ 1 trở lên")
    if n > total:
        raise CheckError(f"Cần {n} câu val, chỉ có {total}")
    return sorted(random.Random(seed).sample(range(total), n))


def labels_of(analysis: dict) -> Labels:
    """(nhãn cảm xúc, nhãn chủ đề) trong phản hồi của /api/analyze."""
    try:
        return analysis["sentiment"]["label"], analysis["topic"]["label"]
    except (KeyError, TypeError) as error:
        raise CheckError(f"Phản hồi /api/analyze thiếu nhãn: {error!r}") from error


def prob_gap(first: dict, second: dict) -> float:
    """Chênh lệch xác suất lớn nhất giữa hai phản hồi cho cùng một câu (cả hai head)."""
    gaps = [0.0]
    for head in ("sentiment", "topic"):
        a, b = first.get(head, {}).get("probs", {}), second.get(head, {}).get("probs", {})
        gaps += [abs(a[label] - b[label]) for label in a if label in b]
    return max(gaps)


def golden_diff(expected: dict[str, Labels], got: dict[str, Labels]) -> list[dict]:
    """Câu golden mà nhãn nhận được khác kỳ vọng (hoặc không nhận được)."""
    return [
        {"id": key, "expected": list(want), "got": list(got[key]) if key in got else None}
        for key, want in expected.items()
        if got.get(key) != want
    ]


def compare_runs(reference: list[Labels], candidate: list[Labels], rows: list[int]) -> dict:
    """So nhãn từng câu val giữa hai lượt chạy; `rows` là số dòng của câu trong tập val."""
    if not len(reference) == len(candidate) == len(rows):
        raise CheckError("Hai lượt chạy không có cùng số câu")
    pairs = list(zip(reference, candidate, strict=True))
    total = len(rows)
    same_both = sum(ref == cand for ref, cand in pairs)
    return {
        "n": total,
        "same_both": same_both,
        "same_sentiment": sum(ref[0] == cand[0] for ref, cand in pairs),
        "same_topic": sum(ref[1] == cand[1] for ref, cand in pairs),
        "needed": math.ceil(total * AGREEMENT_MIN_PCT / 100),
        "rate_pct": round(100 * same_both / total, 2) if total else None,
        "differences": [
            {"val_row": row, "arm64": list(ref), "amd64": list(cand)}
            for row, (ref, cand) in zip(rows, pairs, strict=True)
            if ref != cand
        ],
    }


def evaluate(
    expected: dict[str, Labels], rows: list[int], arm64: list[dict], amd64: list[dict]
) -> dict:
    """Chấm hai lượt phản hồi; mỗi lượt gồm các câu golden (theo thứ tự `expected`) rồi câu val."""
    n_golden = len(expected)
    if not len(arm64) == len(amd64) == n_golden + len(rows):
        raise CheckError("Số phản hồi không khớp số câu golden và val")

    def golden_labels(analyses: list[dict]) -> dict[str, Labels]:
        return {key: labels_of(a) for key, a in zip(expected, analyses[:n_golden], strict=True)}

    gaps = [prob_gap(a, b) for a, b in zip(arm64, amd64, strict=True)]
    return {
        "golden": {
            "total": n_golden,
            "arm64_mismatches": golden_diff(expected, golden_labels(arm64)),
            "amd64_mismatches": golden_diff(expected, golden_labels(amd64)),
        },
        "val": compare_runs(
            [labels_of(a) for a in arm64[n_golden:]],
            [labels_of(a) for a in amd64[n_golden:]],
            rows,
        ),
        "prob_gap": {"max": round(max(gaps), 6), "mean": round(sum(gaps) / len(gaps), 6)},
    }


def verdict_rows(result: dict) -> list[dict]:
    golden, val = result["golden"], result["val"]
    good = golden["total"] - len(golden["amd64_mismatches"])
    rate = f"{val['rate_pct']}% ({val['same_both']}/{val['n']})"
    return [
        {
            "metric": "Golden: nhãn amd64 trùng golden_predictions.json",
            "threshold": f"{golden['total']}/{golden['total']}",
            "value": f"{good}/{golden['total']}",
            "ok": not golden["amd64_mismatches"],
        },
        {
            "metric": f"{val['n']} câu val: cả hai nhãn amd64 trùng arm64",
            "threshold": f"≥ {AGREEMENT_MIN_PCT}%",
            "value": rate,
            "ok": val["same_both"] >= val["needed"],
        },
    ]


def is_go(verdict: list[dict]) -> bool:
    return all(row["ok"] for row in verdict)


# --- Dữ liệu vào và chạy container --------------------------------------------------------------


def load_golden(bundle_dir: Path) -> tuple[dict[str, Labels], list[str]]:
    """Nhãn kỳ vọng theo id (từ bundle) và câu gốc tương ứng (từ fixture của test)."""
    stored = json.loads((bundle_dir / "golden_predictions.json").read_text(encoding="utf-8"))
    fixture = json.loads(GOLDEN_SENTENCES.read_text(encoding="utf-8"))
    sentences = {item["id"]: item["text"] for item in fixture["items"]}
    expected = {item["id"]: (item["sentiment"], item["topic"]) for item in stored["items"]}
    missing = [key for key in expected if key not in sentences]
    if missing:
        raise CheckError(f"Fixture golden thiếu câu {missing}")
    return expected, [sentences[key] for key in expected]


def fetch_analyses(base: str, texts: list[str], timeout: float) -> list[dict]:
    analyses = []
    for text in texts:
        reply = call(base, "POST", "/api/analyze", {"text": text}, timeout=timeout)
        if reply.status != 200:
            raise CheckError(f"POST /api/analyze trả {reply.status}: {reply.body[:200]!r}")
        analyses.append(reply.json())
    return analyses


def container_facts(container: Container) -> dict:
    def run_python(code: str) -> str:
        return docker("exec", container.name, "python", "-c", code, check=False).strip()

    return {
        "machine": run_python("import platform;print(platform.machine())"),
        "cpu_flags": run_python(CPU_FLAGS_CODE).split(),
        "memory_peak_mb": container.memory_peak_mb(),
    }


def run_side(label: str, image: str, platform: str, texts: list[str], args) -> dict:
    """Chạy một image, hỏi `texts` lần lượt và trả phản hồi kèm thông tin container."""
    port = free_port()
    container = Container(
        f"edupulse-amd64-check-{label}", image, args.cpus, args.memory, port, platform
    )
    try:
        container.start()
        base = f"http://127.0.0.1:{port}"
        startup = wait_for_health(base, container, timeout=args.startup_timeout)
        started = time.monotonic()
        analyses = fetch_analyses(base, texts, args.request_timeout)
        elapsed = time.monotonic() - started
        facts = container_facts(container)
    finally:
        container.remove()
    return {
        "image": image,
        "platform": platform,
        "startup_s": round(startup, 1),
        "requests_s": round(elapsed, 1),
        **facts,
        "analyses": analyses,
    }


def build_amd64(args: argparse.Namespace, settings: Settings) -> float:
    """Dựng image amd64 (không tải dữ liệu demo: không cần để kiểm nhãn). Trả số giây."""
    command = [
        "docker",
        "build",
        "--platform=linux/amd64",
        "--build-arg",
        f"BUNDLE_VERSION={args.bundle_version}",
        "--build-arg",
        "EDUPULSE_BUILD_DEMO=0",
        "--build-context",
        f"artifacts={settings.artifacts_dir(args.bundle_version)}",
        "-t",
        args.amd64_image,
        ".",
    ]
    started = time.monotonic()
    if subprocess.run(command, cwd=REPO_ROOT).returncode != 0:  # tiến độ hiện thẳng ra màn hình
        raise CheckError("docker build --platform=linux/amd64 lỗi")
    return round(time.monotonic() - started, 1)


def run(args: argparse.Namespace, settings: Settings) -> dict:
    expected, golden_texts = load_golden(settings.artifacts_dir(args.bundle_version))
    sentences = load_sentences(args.sentences, settings)
    rows = pick_indices(len(sentences), args.n, args.seed)
    texts = golden_texts + [sentences[row] for row in rows]
    build_s = build_amd64(args, settings) if args.build else None
    host = host_info(docker_mode=True)
    sides = {
        "arm64": run_side("arm64", args.arm64_image, "linux/arm64", texts, args),
        "amd64": run_side("amd64", args.amd64_image, "linux/amd64", texts, args),
    }
    result = evaluate(
        expected, rows, sides["arm64"].pop("analyses"), sides["amd64"].pop("analyses")
    )
    verdict = verdict_rows(result)
    return {
        "date": date.today().isoformat(),
        "commit": git_commit(),
        "bundle_version": args.bundle_version,
        "host": host,
        "sample": {"seed": args.seed, "val_available": len(sentences), "val_n": len(rows)},
        "build_s": build_s,
        "limits": {"cpus": args.cpus, "memory": args.memory},
        "containers": sides,
        **result,
        "verdict": verdict,
        "go": is_go(verdict),
    }


# --- Báo cáo ------------------------------------------------------------------------------------


def render_markdown(report: dict) -> str:
    host, sample, sides = report["host"], report["sample"], report["containers"]
    golden, val, gap = report["golden"], report["val"], report["prob_gap"]

    def golden_ok(arch: str) -> int:
        return golden["total"] - len(golden[f"{arch}_mismatches"])

    docker_vm = f"VM {host.get('docker_cpus', '-')} CPU / {host.get('docker_mem_gb', '-')} GB"
    lines = [
        f"# Kiểm tra amd64 so với arm64 ({report['date']})",
        "",
        f"Kết luận: **{'ĐẠT' if report['go'] else 'KHÔNG ĐẠT'}**",
        "",
        f"- Máy: {host['platform']}; Docker {host.get('docker_server', '-')}, {docker_vm}",
        f"- Commit `{report['commit']}`; bundle `{report['bundle_version']}`",
        f"- Giới hạn mỗi container: `--cpus={report['limits']['cpus']:g}"
        f" --memory={report['limits']['memory']}`, không swap",
        f"- {golden['total']} câu golden và {sample['val_n']} câu val (hạt giống {sample['seed']},"
        f" lấy trong {sample['val_available']} câu của tập val)",
        "",
        "## Bảng kết luận",
        "",
        *table(
            ["Chỉ số", "Ngưỡng", "Đo được", "Kết quả"],
            [
                [r["metric"], r["threshold"], r["value"], OUTCOME[r["ok"]]]
                for r in report["verdict"]
            ],
        ),
        "",
        f"- Golden trùng nhãn: {golden_ok('arm64')}/{golden['total']} trên arm64,"
        f" {golden_ok('amd64')}/{golden['total']} trên amd64",
        f"- Trong {val['n']} câu val: trùng nhãn cảm xúc {val['same_sentiment']}, trùng nhãn chủ đề"
        f" {val['same_topic']}, trùng cả hai {val['same_both']} (cần ít nhất {val['needed']})",
        f"- Chênh xác suất lớn nhất giữa hai kiến trúc trên mọi câu: {gap['max']}"
        f" (trung bình {gap['mean']})",
        "",
        "## Container",
        "",
        *table(
            ["", "arm64", "amd64"],
            [
                ["Image", sides["arm64"]["image"], sides["amd64"]["image"]],
                ["Kiến trúc trong container", sides["arm64"]["machine"], sides["amd64"]["machine"]],
                [
                    "Khởi động tới health 200 (s)",
                    sides["arm64"]["startup_s"],
                    sides["amd64"]["startup_s"],
                ],
                [
                    "Hỏi lần lượt mọi câu (s)",
                    sides["arm64"]["requests_s"],
                    sides["amd64"]["requests_s"],
                ],
                [
                    "memory.peak (MB)",
                    sides["arm64"]["memory_peak_mb"],
                    sides["amd64"]["memory_peak_mb"],
                ],
                [
                    "Cờ CPU liên quan int8",
                    " ".join(sides["arm64"]["cpu_flags"]) or "-",
                    " ".join(sides["amd64"]["cpu_flags"]) or "-",
                ],
            ],
        ),
    ]
    if report.get("build_s") is not None:
        lines.append(f"\nDựng image amd64 mất {report['build_s']} s (không tải dữ liệu demo).")
    for title, items in (
        ("Câu golden lệch nhãn trên amd64", golden["amd64_mismatches"]),
        ("Câu golden lệch nhãn trên arm64", golden["arm64_mismatches"]),
    ):
        if items:
            lines += ["", f"## {title}", ""]
            lines += table(
                ["Mã câu", "Kỳ vọng", "Nhận được"],
                [
                    [i["id"], " / ".join(i["expected"]), " / ".join(i["got"] or ["-"])]
                    for i in items
                ],
            )
    if val["differences"]:
        lines += ["", "## Câu val lệch nhãn (không chép nội dung câu)", ""]
        lines += table(
            ["Dòng trong tập val", "arm64 (cảm xúc / chủ đề)", "amd64 (cảm xúc / chủ đề)"],
            [
                [d["val_row"], " / ".join(d["arm64"]), " / ".join(d["amd64"])]
                for d in val["differences"]
            ],
        )
    lines += [
        "",
        "Chạy giả lập nên nhân int8 có thể khác nhân AVX2/VNNI trên VPS x86 thật. Kết quả này chứng"
        " minh wheel, JRE và bộ tách từ của amd64 cho cùng nhãn, không phải số đo hiệu năng.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(settings: Settings) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--bundle-version", default=settings.bundle_version)
    add("--arm64-image", help="mặc định edupulse:<bundle>")
    add("--amd64-image", help="mặc định <image arm64>-amd64")
    add("--build", action="store_true", help="dựng image amd64 trước (giả lập nên chậm)")
    add("--sentences", help="file mỗi dòng một câu thay cho tập val")
    add("--n", type=int, default=SAMPLE_SIZE, help="số câu val")
    add("--seed", type=int, default=SAMPLE_SEED, help="hạt giống chọn câu val")
    add("--cpus", type=float, default=2.0)
    add("--memory", default="1536m")
    add("--startup-timeout", type=float, default=900.0, help="giây chờ health (giả lập chậm)")
    add("--request-timeout", type=float, default=120.0)
    add("--out-dir", type=Path, default=REPO_ROOT / "results" / "perf")
    args = parser.parse_args()
    args.arm64_image = args.arm64_image or f"edupulse:{args.bundle_version}"
    args.amd64_image = args.amd64_image or f"{args.arm64_image}-amd64"
    return args


def main() -> int:
    settings = load_settings()
    args = parse_args(settings)
    try:
        report = run(args, settings)
    except SmokeError as error:  # gồm CheckError
        print(f"Kiểm tra amd64 lỗi: {error}", file=sys.stderr)
        return 3
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"amd64-check-{report['date']}"
    stem.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(report)
    stem.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Đã ghi {stem}.json và .md")
    return 0 if report["go"] else 1


if __name__ == "__main__":
    sys.exit(main())
