"""Smoke test giới hạn tài nguyên cho image EduPulse: go/no-go cho VPS 2 vCPU / 2GB.

Chế độ mặc định chạy container Docker với đúng giới hạn của VPS (`--cpus=2 --memory=1536m`, không
swap) rồi đo: thời gian khởi động, độ trễ 1 câu + giải thích, 1.000 câu end-to-end (bench trong
một container riêng cùng giới hạn), bộ nhớ đỉnh của cgroup, OOM-kill, và kiểm tra nhật ký không lộ
nội dung. Chế độ `--url` chỉ đo độ trễ và hành vi HTTP của một dịch vụ đã chạy (ví dụ VPS thật).

    uv run python scripts/smoke_test.py --bundle-version v0-smoke --image edupulse:smoke
    uv run python scripts/smoke_test.py --url http://vps.example:8000

Ghi `results/perf/smoke-<ngày>-<kiến trúc>.{json,md}`; thoát mã 1 nếu có chỉ số không đạt.
Từ chối chạy khi khoá MPS đang giữ (đang train): lúc ấy số đo không đáng tin và Docker làm swap.

`python -m edupulse.engine.bench --json` in một dòng JSON; script đọc các khoá `total_s`,
`steps_s`, `rss_mb` và `cgroup_memory_peak_mb`.
"""

import argparse
import fcntl
import json
import math
import os
import platform
import random
import re
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from edupulse.config import REPO_ROOT, XAI_MAX_WORDS, Settings, load_settings

# Ngưỡng go khi đo trên M1 với 2 CPU / 1,5GB. VPS x86 chia sẻ chậm hơn nên đọc số M1 với hệ số
# an toàn ×3: 0,6 s → 1,8 s < 2 s; 30 s → 90 s < 2 phút.
GO = {
    "memory_peak_mb": 1200,
    "single_p95_s": 0.6,
    "batch_total_s": 30.0,
    "startup_s": 20.0,
}
MARKERS = ("SMOKE-NOI-DUNG-9f3a", "SMOKE-QUERY-7c1d", "SMOKE-ID-4b2e", "SMOKE-NHOM-1a6c")
LONG_WORDS = 30  # câu 30–40 từ là ca xấu nhất của giải thích bằng bỏ từng từ
OUTCOME = {True: "ĐẠT", False: "KHÔNG ĐẠT", None: "không đo"}


class SmokeError(RuntimeError):
    """Lỗi làm hỏng phép đo (container không lên, bench lỗi, ...), khác với chỉ số không đạt."""


# --- Khoá MPS, Docker, HTTP ---------------------------------------------------------------------


def mps_lock_held(path: Path) -> bool:
    """Tiến trình train giữ `flock(LOCK_EX | LOCK_NB)`; BlockingIOError nghĩa là đang bị giữ."""
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def docker(*args: str, check: bool = True, timeout: float = 900.0) -> str:
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise SmokeError(f"docker {' '.join(args[:2])} lỗi: {result.stderr.strip()[-600:]}")
    return result.stdout


@dataclass
class Reply:
    status: int
    content_type: str
    body: bytes
    seconds: float

    def json(self) -> dict:
        try:
            parsed = json.loads(self.body)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def call(
    base: str, method: str, path: str, payload: dict | None = None, timeout: float = 60.0
) -> Reply:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, headers_in, body = response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        status, headers_in, body = error.code, error.headers, error.read()
    elapsed = time.perf_counter() - started
    return Reply(status, headers_in.get("Content-Type", ""), body, elapsed)


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class Container:
    def __init__(self, name: str, image: str, cpus: float, memory: str, port: int | None = None):
        self.name, self.image, self.cpus, self.memory, self.port = name, image, cpus, memory, port

    def start(self, *command: str, mounts: tuple[str, ...] = ()) -> None:
        docker("rm", "-f", self.name, check=False)
        limits = [
            f"--cpus={self.cpus:g}",
            f"--memory={self.memory}",
            f"--memory-swap={self.memory}",
        ]
        ports = ["-p", f"127.0.0.1:{self.port}:8000"] if self.port else []
        volumes = [arg for mount in mounts for arg in ("-v", mount)]
        docker("run", "-d", "--name", self.name, *limits, *ports, *volumes, self.image, *command)

    def running(self) -> bool:
        state = docker("inspect", "-f", "{{.State.Running}}", self.name, check=False)
        return state.strip() == "true"

    def logs(self) -> tuple[str, str]:
        result = subprocess.run(
            ["docker", "logs", self.name], capture_output=True, text=True, timeout=60
        )
        return result.stdout, result.stderr

    def state(self) -> dict:
        fmt = "{{.State.OOMKilled}} {{.State.ExitCode}} {{.RestartCount}}"
        oom, code, restarts = docker("inspect", "-f", fmt, self.name).split()
        return {"oom_killed": oom == "true", "exit_code": int(code), "restarts": int(restarts)}

    def memory_peak_mb(self) -> float | None:
        out = docker("exec", self.name, "cat", "/sys/fs/cgroup/memory.peak", check=False).strip()
        return round(int(out) / 1048576, 1) if out.isdigit() else None

    def remove(self) -> None:
        docker("rm", "-f", self.name, check=False)


# --- Phép đo ------------------------------------------------------------------------------------


def summarize(values: list[float]) -> dict:
    """p95 theo hạng gần nhất; với 30 mẫu đó là giá trị lớn thứ hai."""
    ordered = sorted(values)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    return {
        "n": len(ordered),
        "p50_s": round(statistics.median(ordered), 3),
        "p95_s": round(p95, 3),
        "max_s": round(ordered[-1], 3),
    }


def post_text(base: str, path: str, text: str) -> float:
    reply = call(base, "POST", path, {"text": text})
    if reply.status != 200:
        raise SmokeError(f"POST {path} trả {reply.status}: {reply.body[:200]!r}")
    return reply.seconds


def measure_latency(base: str, texts: list[str], repeat: int) -> dict:
    """Độ trễ analyze, explain, và cặp gọi song song như giao diện (tính tới khi cả hai xong)."""
    analyze, explain, pair = [], [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i in range(repeat):
            text = texts[i % len(texts)]
            analyze.append(post_text(base, "/api/analyze", text))
            explain.append(post_text(base, "/api/explain", text))
            started = time.perf_counter()
            both = [pool.submit(post_text, base, p, text) for p in ("/api/analyze", "/api/explain")]
            for future in both:
                future.result()
            pair.append(time.perf_counter() - started)
    return {
        "analyze": summarize(analyze),
        "explain": summarize(explain),
        "analyze_explain": summarize(pair),
    }


class Load:
    """Vài luồng gọi /api/analyze liên tục: mô phỏng việc khác đang chiếm CPU và khoá suy luận."""

    def __init__(self, base: str, texts: list[str], workers: int = 2):
        self.base, self.texts, self.workers = base, texts, workers
        self.stop = threading.Event()
        self.done = 0
        self.errors = 0
        self.seconds = 0.0
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []
        self._started = 0.0

    def _run(self, seed: int) -> None:
        rng = random.Random(seed)
        while not self.stop.is_set():
            try:
                post_text(self.base, "/api/analyze", rng.choice(self.texts))
                ok = True
            except (SmokeError, OSError):
                ok = False
            with self._lock:
                if ok:
                    self.done += 1
                else:
                    self.errors += 1

    def __enter__(self) -> "Load":
        self._started = time.perf_counter()
        self._threads = [threading.Thread(target=self._run, args=(i,)) for i in range(self.workers)]
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        for thread in self._threads:
            thread.join()
        self.seconds = time.perf_counter() - self._started

    def summary(self) -> dict:
        rate = self.done / self.seconds if self.seconds else 0.0
        return {
            "workers": self.workers,
            "requests": self.done,
            "errors": self.errors,
            "req_per_s": round(rate, 1),
        }


def check_api(base: str) -> dict[str, bool]:
    """Hành vi HTTP cố định của khung API và bản web."""
    missing = call(base, "GET", "/api/khong-co")
    root = call(base, "GET", "/")
    page = call(base, "GET", "/khong-co/")
    wrong_method = call(base, "POST", "/api/health")
    return {
        "api_unknown_is_json_404": missing.status == 404
        and "json" in missing.content_type
        and missing.json().get("detail", {}).get("code") == "not_found",
        "root_is_html": root.status == 200 and "text/html" in root.content_type,
        "unknown_page_is_html_404": page.status == 404 and "text/html" in page.content_type,
        "wrong_method_is_405": wrong_method.status == 405,
    }


def probe_logged_content(base: str, text: str) -> None:
    """Gửi yêu cầu mang dấu hiệu nhận biết trong thân, query và đường dẫn để soi nhật ký sau đó."""
    body_marker, query_marker, id_marker, group_marker = MARKERS
    call(base, "POST", f"/api/analyze?q={query_marker}", {"text": f"{text} {body_marker}"})
    call(base, "GET", f"/api/batches/{id_marker}/results?group={group_marker}")


def check_logs(logs: str) -> dict[str, bool]:
    return {
        "logs_have_no_request_content": not any(marker in logs for marker in MARKERS),
        "logs_have_route_template": re.search(r"POST /api/analyze 200 \d+ms", logs) is not None,
    }


def run_bench(
    image: str, name: str, cpus: float, memory: str, sentences: Path, n: int, xai: int
) -> dict:
    """1.000 câu end-to-end trong container riêng cùng giới hạn; bộ nhớ đỉnh do bench tự đọc."""
    container = Container(name, image, cpus, memory)
    command = ("python", "-m", "edupulse.engine.bench", "--input", "/tmp/sentences.txt")
    command += ("--n", str(n), "--xai", str(xai), "--json")
    try:
        container.start(*command, mounts=(f"{sentences}:/tmp/sentences.txt:ro",))
        exit_code = int(docker("wait", name, timeout=1800).strip())
        state = container.state()
        stdout, stderr = container.logs()
    finally:
        container.remove()
    lines = [line for line in stdout.splitlines() if line.startswith("{")]
    if exit_code != 0 or not lines:
        return {
            "error": f"bench thoát mã {exit_code}",
            "stderr_tail": stderr.strip()[-600:],
            **state,
        }
    return {**json.loads(lines[-1]), **state}


def wait_for_health(base: str, container: Container, timeout: float = 180.0) -> float:
    started = time.monotonic()
    while time.monotonic() - started < timeout:
        if not container.running():
            stdout, stderr = container.logs()
            raise SmokeError(f"Container dừng trước khi sẵn sàng:\n{(stdout + stderr)[-800:]}")
        try:
            if call(base, "GET", "/api/health", timeout=2).status == 200:
                return time.monotonic() - started
        except OSError:
            pass
        time.sleep(0.2)
    raise SmokeError(f"/api/health chưa trả 200 sau {timeout:.0f} s")


def measure_service(base: str, texts: list[str], repeat: int) -> dict:
    """Phần đo HTTP dùng chung cho Docker và `--url`: độ trễ lúc rảnh, câu dài, và khi có tải."""
    if not texts:
        raise SmokeError(f"Không có câu nào ≤ {XAI_MAX_WORDS} từ để đo")
    long = [t for t in texts if len(t.split()) >= LONG_WORDS]
    first = post_text(base, "/api/analyze", texts[0])  # gồm nạp JVM ở lần đầu
    for text in texts[:3]:
        post_text(base, "/api/explain", text)
    result: dict = {"first_analyze_s": round(first, 3)}
    result["latency_idle"] = measure_latency(base, texts, repeat)
    if len(long) >= 5:
        result["latency_idle_long"] = measure_latency(base, long, repeat)
    with Load(base, texts) as load:
        time.sleep(1.0)
        result["latency_under_load"] = measure_latency(base, texts, repeat)
    result["load"] = load.summary()
    return result


def run_scenario(
    args: argparse.Namespace, label: str, cpus: float, texts: list[str], bench_file: Path
) -> dict:
    """Một lượt đo đầy đủ trong Docker với `cpus` CPU và `args.memory` RAM."""
    base = f"http://127.0.0.1:{args.port or free_port()}"
    port = int(base.rsplit(":", 1)[1])
    server = Container(f"edupulse-smoke-{label}", args.image, cpus, args.memory, port)
    result: dict = {"label": label, "cpus": cpus, "memory": args.memory}
    try:
        server.start()
        result["startup_s"] = round(wait_for_health(base, server), 2)
        result["health"] = call(base, "GET", "/api/health").json()
        result.update(measure_service(base, texts, args.repeat))
        result["checks"] = check_api(base)
        probe_logged_content(base, texts[0])
        bench_name = f"{server.name}-bench"
        result["bench"] = run_bench(
            args.image, bench_name, cpus, args.memory, bench_file, args.n, args.xai
        )
        # Đọc sau bench để các dòng nhật ký của yêu cầu dò đã kịp ghi ra.
        result["checks"].update(check_logs("".join(server.logs())))
        result["server_memory_peak_mb"] = server.memory_peak_mb()
        result["server_state"] = server.state()
    finally:
        server.remove()
    return result


def run_remote(args: argparse.Namespace, texts: list[str]) -> dict:
    base = args.url.rstrip("/")
    result: dict = {"label": "remote", "url": base, "startup_s": None}
    result["health"] = call(base, "GET", "/api/health").json()
    result.update(measure_service(base, texts, args.repeat))
    result["checks"] = check_api(base)
    return result


# --- Dữ liệu vào, dựng image --------------------------------------------------------------------


def read_sentences(args: argparse.Namespace, settings: Settings) -> list[str]:
    """Câu thật để đo: `--sentences FILE` (mỗi dòng một câu) hoặc tập val đã xử lý ở local."""
    if args.sentences:
        lines = Path(args.sentences).read_text(encoding="utf-8").splitlines()
        sentences = [line.strip() for line in lines if line.strip()]
    else:
        parquet = settings.data_dir / "processed" / "validation.parquet"
        if not parquet.is_file():
            raise SmokeError(f"Thiếu {parquet}: chạy `make data` hoặc truyền --sentences FILE")
        import pyarrow.parquet as pq  # chỉ cần ở máy dev, không thuộc group app

        column = pq.read_table(parquet, columns=["sentence"]).column(0)
        sentences = [str(s) for s in column.to_pylist()]
    if not args.url and len(sentences) < args.n:
        raise SmokeError(f"Cần ít nhất {args.n} câu, chỉ có {len(sentences)}")
    return sentences


_SIZE_UNITS = {"B": 1, "kB": 1e3, "MB": 1e6, "GB": 1e9}


def image_layers_mb(image: str) -> int:
    """Tổng dung lượng các lớp chưa nén (MB). `docker image inspect .Size` của kho containerd cộng
    cả bản nén lưu trong kho nên lớn gấp rưỡi con số người dùng tải về hay thấy trên đĩa."""
    total = 0.0
    for size in docker("history", "--format", "{{.Size}}", image).split():
        match = re.fullmatch(r"([\d.]+)([kMG]?B)", size)
        if match is None:
            raise SmokeError(f"Không hiểu dung lượng lớp {size!r} của {image}")
        total += float(match.group(1)) * _SIZE_UNITS[match.group(2)]
    return round(total / 1e6)


def build_image(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    if not args.skip_build:
        command = ["make", "docker-build", f"BUNDLE_VERSION={args.bundle_version}"]
        command.append(f"IMAGE={args.image}")
        built = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
        if built.returncode != 0:
            raise SmokeError(f"make docker-build lỗi:\n{(built.stdout + built.stderr)[-1500:]}")
    if not docker("image", "inspect", "-f", "{{.Id}}", args.image, check=False).strip():
        raise SmokeError(f"Không thấy image {args.image}; bỏ --skip-build hoặc `make docker-build`")
    return {
        "tag": args.image,
        "size_mb": image_layers_mb(args.image),
        "build_s": None if args.skip_build else round(time.monotonic() - started, 1),
    }


# --- Kết luận và báo cáo ------------------------------------------------------------------------


def verdict_rows(gate: dict) -> list[dict]:
    """Bảng go/no-go; `ok` là None khi chỉ số không đo được ở chế độ này."""
    bench = gate.get("bench") or {}
    peaks = [gate.get("server_memory_peak_mb"), bench.get("cgroup_memory_peak_mb")]
    peak = max((p for p in peaks if p is not None), default=None)
    ooms = [gate.get("server_state", {}).get("oom_killed"), bench.get("oom_killed")]
    pair = gate["latency_idle"]["analyze_explain"]["p95_s"]
    checks = gate.get("checks", {})

    def within(value: float | None, limit: float) -> bool | None:
        return None if value is None else value <= limit

    def row(metric: str, threshold: str, value, ok: bool | None) -> dict:
        return {"metric": metric, "threshold": threshold, "value": value, "ok": ok}

    measured_ooms = None not in ooms
    return [
        row(
            "OOM-kill",
            "0",
            sum(ooms) if measured_ooms else None,
            not any(ooms) if measured_ooms else None,
        ),
        row(
            "memory.peak (MB)",
            f"≤ {GO['memory_peak_mb']}",
            peak,
            within(peak, GO["memory_peak_mb"]),
        ),
        row(
            "1 câu + giải thích, p95 (s)",
            f"≤ {GO['single_p95_s']}",
            pair,
            within(pair, GO["single_p95_s"]),
        ),
        row(
            "1.000 câu end-to-end (s)",
            f"≤ {GO['batch_total_s']:g}",
            bench.get("total_s"),
            within(bench.get("total_s"), GO["batch_total_s"]),
        ),
        row(
            "Khởi động tới health 200 (s)",
            f"≤ {GO['startup_s']:g}",
            gate.get("startup_s"),
            within(gate.get("startup_s"), GO["startup_s"]),
        ),
        row(
            "Hành vi API và nhật ký",
            "tất cả đạt",
            f"{sum(checks.values())}/{len(checks)}",
            all(checks.values()),
        ),
    ]


def table(header: list[str], rows: list[list]) -> list[str]:
    cells = lambda row: "| " + " | ".join("-" if c is None else str(c) for c in row) + " |"  # noqa: E731
    return [cells(header), "|" + "---|" * len(header), *(cells(row) for row in rows)]


def latency_rows(gate: dict) -> list[list]:
    titles = {
        "latency_idle": "rảnh",
        "latency_idle_long": "rảnh, câu 30–40 từ",
        "latency_under_load": "đang có tải",
    }
    return [
        [title, name, s["n"], s["p50_s"], s["p95_s"], s["max_s"]]
        for key, title in titles.items()
        for name, s in gate.get(key, {}).items()
    ]


def render_markdown(report: dict) -> str:
    gate, host, image = report["runs"][0], report["host"], report.get("image")
    bundle_version = gate.get("health", {}).get("bundle_version", "-")
    docker_vm = f"VM {host.get('docker_cpus', '-')} CPU / {host.get('docker_mem_gb', '-')} GB"
    load = f"{host['loadavg_start']} → {host['loadavg_end']}"
    lines = [
        f"# Smoke test {report['date']} ({report['arch']})",
        "",
        f"Kết luận: **{'GO' if report['go'] else 'NO-GO'}**",
        "",
        f"- Máy: {host['platform']}; Docker {host.get('docker_server', '-')}, {docker_vm}",
        f"- Tải máy (load average 1/5/15 phút, đầu → cuối): {load}",
        f"- Container khác đang chạy trong Docker: {host.get('other_containers', '-')}",
        f"- Commit `{report['commit']}`; bundle `{bundle_version}`",
    ]
    if image:
        lines.append(
            f"- Image `{image['tag']}`: {image['size_mb']} MB (tổng các lớp, chưa nén);"
            f" dựng mất {image['build_s']} s"
        )
    limits = f"--cpus={gate.get('cpus', '-')} --memory={gate.get('memory', '-')}"
    lines += [f"- Giới hạn đo chính: `{limits}`, không swap", "", "## Bảng go/no-go", ""]
    verdict = [
        [r["metric"], r["threshold"], r["value"], OUTCOME[r["ok"]]] for r in report["verdict"]
    ]
    lines += table(["Chỉ số", "Ngưỡng", "Đo được", "Kết quả"], verdict)

    lines += ["", "## Độ trễ từ máy chủ (giây)", ""]
    lines += table(["Tình huống", "Lệnh", "n", "p50", "p95", "max"], latency_rows(gate))
    if gate.get("load"):
        bg = gate["load"]
        lines += [
            "",
            f"Tải nền: {bg['workers']} luồng gọi /api/analyze liên tục, {bg['req_per_s']} req/s,"
            f" {bg['errors']} lỗi.",
        ]
    lines.append(f"Lần gọi đầu sau khi lên (gồm nạp JVM): {gate.get('first_analyze_s', '-')} s.")

    bench = gate.get("bench")
    if bench and "error" not in bench:
        steps = [[step, seconds] for step, seconds in bench.get("steps_s", {}).items()]
        steps.append(["tổng", bench["total_s"]])
        lines += ["", "## Bench 1.000 câu trong container riêng", ""] + table(
            ["Bước", "Giây"], steps
        )
        lines += [
            "",
            f"RSS cuối {bench.get('rss_mb', '-')} MB;"
            f" cgroup memory.peak {bench.get('cgroup_memory_peak_mb', '-')} MB.",
        ]
    elif bench:
        lines += [
            "",
            "## Bench 1.000 câu",
            "",
            f"Lỗi: {bench['error']}",
            "",
            f"```\n{bench.get('stderr_tail', '')}\n```",
        ]

    lines += ["", "## Kiểm tra hành vi", ""]
    lines += [
        f"- {name}: {'đạt' if ok else 'KHÔNG ĐẠT'}" for name, ok in gate.get("checks", {}).items()
    ]
    for extra in report["runs"][1:]:
        pair = extra["latency_idle"]["analyze_explain"]["p95_s"]
        total = (extra.get("bench") or {}).get("total_s", "-")
        lines += ["", f"## Tham khảo: {extra['cpus']:g} CPU", ""]
        lines += [
            f"- Khởi động {extra.get('startup_s')} s; 1 câu + giải thích p95 {pair} s;"
            f" 1.000 câu {total} s"
        ]
    lines += ["", "Số đo trên M1 đọc với hệ số an toàn ×3 cho VPS x86 chia sẻ; chưa đo x86."]
    return "\n".join(lines) + "\n"


def host_info(docker_mode: bool) -> dict:
    info: dict = {
        "platform": f"{platform.system()} {platform.machine()}",
        "loadavg_start": [round(x, 1) for x in os.getloadavg()],
    }
    if docker_mode:
        fmt = "{{.ServerVersion}} {{.NCPU}} {{.MemTotal}}"
        server, cpus, mem = docker("info", "--format", fmt).split()
        info["docker_server"] = server
        info["docker_cpus"] = int(cpus)
        info["docker_mem_gb"] = round(int(mem) / 2**30, 1)
        info["other_containers"] = len(docker("ps", "-q").split())
    return info


def git_commit() -> str:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    run = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return run.stdout.strip() or "-"


def parse_args(settings: Settings) -> argparse.Namespace:
    arch = {"aarch64": "arm64", "x86_64": "amd64"}.get(platform.machine(), platform.machine())
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add = parser.add_argument
    add("--url", help="đo dịch vụ đã chạy (không dùng Docker, không đo bộ nhớ và bench)")
    add("--bundle-version", default=settings.bundle_version)
    add("--image", default="edupulse:smoke")
    add("--skip-build", action="store_true", help="dùng image sẵn có")
    add("--cpus", type=float, default=2.0)
    add("--memory", default="1536m")
    add("--also-1cpu", action="store_true", help="thêm một lượt tham khảo với 1 CPU")
    add("--port", type=int, default=0, help="cổng máy chủ; 0 = tự chọn cổng trống")
    add("--sentences", help="file mỗi dòng một câu; mặc định lấy tập val đã xử lý")
    add("--n", type=int, default=1000, help="số câu của bench")
    add("--xai", type=int, default=20, help="số câu được giải thích tự động")
    add("--repeat", type=int, default=30, help="số lần lặp mỗi phép đo độ trễ")
    add("--tag", default=arch, help="hậu tố kiến trúc trong tên file báo cáo")
    add("--out-dir", type=Path, default=REPO_ROOT / "results" / "perf")
    return parser.parse_args()


def measure(args: argparse.Namespace, settings: Settings) -> tuple[list[dict], dict | None]:
    texts = read_sentences(args, settings)
    short = [t for t in texts if len(t.split()) <= XAI_MAX_WORDS]
    if args.url:
        return [run_remote(args, short)], None
    image = build_image(args)
    with tempfile.TemporaryDirectory() as workdir:
        os.chmod(workdir, 0o755)  # container chạy uid 1000 phải đọc được thư mục gắn vào
        bench_file = Path(workdir) / "sentences.txt"
        bench_file.write_text(
            "\n".join(t.replace("\n", " ") for t in texts[: args.n]) + "\n", encoding="utf-8"
        )
        bench_file.chmod(0o644)
        runs = [run_scenario(args, "gate", args.cpus, short, bench_file)]
        if args.also_1cpu:
            runs.append(run_scenario(args, "1cpu", 1.0, short, bench_file))
    return runs, image


def main() -> int:
    settings = load_settings()
    args = parse_args(settings)
    try:
        if args.url is None and mps_lock_held(settings.mps_lock_path):
            print(
                f"Từ chối đo: khoá MPS {settings.mps_lock_path} đang giữ (đang train).",
                file=sys.stderr,
            )
            return 2
        host = host_info(docker_mode=args.url is None)
        runs, image = measure(args, settings)
    except SmokeError as error:
        print(f"Smoke test lỗi: {error}", file=sys.stderr)
        return 3

    host["loadavg_end"] = [round(x, 1) for x in os.getloadavg()]
    verdict = verdict_rows(runs[0])
    report = {
        "date": date.today().isoformat(),
        "arch": args.tag,
        "commit": git_commit(),
        "host": host,
        "image": image,
        "runs": runs,
        "verdict": verdict,
        "go": all(row["ok"] is not False for row in verdict),
        "thresholds": GO,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"smoke-{report['date']}-{args.tag}"
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
