"""Smoke test giới hạn tài nguyên cho image EduPulse: go/no-go cho VPS 2 vCPU / 2GB.

Chế độ mặc định chạy container Docker với đúng giới hạn của VPS (`--cpus=2 --memory=1536m`, không
swap) rồi đo: thời gian khởi động, độ trễ của nhãn và của giải thích (hai chỉ số p95 tách biệt),
1.000 câu end-to-end (bench trong một container riêng cùng giới hạn), upload file 5MB chạy xong,
bộ nhớ đỉnh của cgroup, OOM-kill, và kiểm tra nhật ký không lộ nội dung. Chế độ `--url` đo độ
trễ, upload 5MB và hành vi HTTP của một dịch vụ đã chạy (ví dụ VPS thật).

    uv run python scripts/smoke_test.py --bundle-version v0-smoke --image edupulse:smoke
    uv run python scripts/smoke_test.py --url http://vps.example:8000

Ghi `results/perf/smoke-<ngày>-<kiến trúc>.{json,md}`; thoát mã 1 nếu có chỉ số không đạt.
Từ chối chạy khi khoá MPS đang giữ (đang train): lúc ấy số đo không đáng tin và Docker làm swap.

Tiêu chí độ trễ theo PDR 5 (phương án A ở mốc M1: nhãn hiện ngay, giải thích tải sau):
    nhãn      p95 của /api/analyze <= 0,5 s, trên câu bất kỳ độ dài;
    giải thích p95 của /api/explain <= 2 s, chỉ trên câu mà máy chủ giải thích được, tức không quá
              `limits.xai_max_words` từ (đọc từ /api/config, mặc định 25).

`python -m edupulse.engine.bench --json` in một dòng JSON; script đọc các khoá `total_s`,
`steps_s`, `rss_mb` và `cgroup_memory_peak_mb`.
"""

import argparse
import csv
import fcntl
import io
import json
import math
import os
import platform
import random
import re
import secrets
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from edupulse.config import MAX_ROWS, REPO_ROOT, Settings, load_settings

# Ngưỡng go. Độ trễ là con số PDR 5 (đổi ở mốc M1, phương án A): nhãn hiện ngay, giải thích tải
# sau, nên hai p95 được chấm riêng. Giới hạn số từ của giải thích (`xai_max_words`) chốt lại sau khi
# đo trên VPS thật bằng `--url`. Khi đo trên M1 với 2 CPU / 1,5GB, VPS x86 chia sẻ chậm hơn nên đọc
# thời gian 1.000 câu với hệ số an toàn ×3: 30 s → 90 s < 2 phút.
GO = {
    "memory_peak_mb": 1200,
    "label_p95_s": 0.5,
    "explain_p95_s": 2.0,
    "batch_total_s": 30.0,
    "startup_s": 20.0,
}
MARKERS = ("SMOKE-NOI-DUNG-9f3a", "SMOKE-QUERY-7c1d", "SMOKE-ID-4b2e", "SMOKE-NHOM-1a6c")
FILE_MARKER = "SMOKE-TEN-FILE-5d8e"  # nằm trong tên file của lần upload thử
LABEL_SAMPLES = 100  # số câu mẫu của phép đo nhãn
INFO = "tham khảo"  # ngưỡng của dòng chỉ để đọc, không chấm đạt/không đạt
BASELINE_LOADED = re.compile(r"Đã nạp mô hình cơ sở trong ([\d.]+) s")
# Việc nặng của người khác trên cùng máy làm nhiễu số đo; `[x]` để pgrep không tự khớp lệnh của nó.
HEAVY_PATTERN = "[p]ytest|[p]laywright|[v]itest|[n]ext build"
WATCH_PERIOD_S = 2.0  # chu kỳ lấy mẫu việc nặng khác
UPLOAD_MB = 5  # PDR 5: file 5MB phải chạy xong
UPLOAD_TIMEOUT_S = 900.0
LONG_SHARE = 0.8  # câu dài từ ~80% giới hạn giải thích là ca xấu nhất của bỏ từng từ
BATCHED_REFERENCE = 32  # lô của bench tham khảo (nhanh hơn, nhưng kết quả phụ thuộc bạn cùng lô)
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
    base: str,
    method: str,
    path: str,
    payload: dict | None = None,
    timeout: float = 60.0,
    raw: tuple[bytes, str] | None = None,
) -> Reply:
    """Một yêu cầu HTTP. `payload` gửi JSON; `raw` = (thân, Content-Type) gửi nguyên văn."""
    data, content_type = (None, None)
    if raw is not None:
        data, content_type = raw
    elif payload is not None:
        data, content_type = json.dumps(payload).encode(), "application/json"
    headers = {} if content_type is None else {"Content-Type": content_type}
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
    def __init__(
        self,
        name: str,
        image: str,
        cpus: float,
        memory: str,
        port: int | None = None,
        platform: str | None = None,
    ):
        self.name, self.image, self.cpus, self.memory, self.port = name, image, cpus, memory, port
        self.platform = platform  # vd. "linux/amd64" để chạy giả lập trên máy arm64

    def start(self, *command: str, mounts: tuple[str, ...] = ()) -> None:
        docker("rm", "-f", self.name, check=False)
        limits = [
            f"--cpus={self.cpus:g}",
            f"--memory={self.memory}",
            f"--memory-swap={self.memory}",
        ]
        ports = ["-p", f"127.0.0.1:{self.port}:8000"] if self.port else []
        volumes = [arg for mount in mounts for arg in ("-v", mount)]
        emulation = ["--platform", self.platform] if self.platform else []
        options = ["-d", "--name", self.name, *emulation, *limits, *ports, *volumes]
        docker("run", *options, self.image, *command)

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


# --- Giờ từng pha và việc nặng của người khác ---------------------------------------------------


def clock(epoch: float) -> str:
    """Giờ máy HH:MM:SS, để đối chiếu với nhật ký của người khác."""
    return time.strftime("%H:%M:%S", time.localtime(epoch))


def heavy_processes() -> list[str]:
    """Tên tệp thực thi của các tiến trình nặng khác đang chạy trên máy (rỗng: máy rảnh)."""
    try:
        found = subprocess.run(
            ["pgrep", "-f", HEAVY_PATTERN], capture_output=True, text=True, timeout=10
        ).stdout.split()
        if not found:
            return []
        names = subprocess.run(
            ["ps", "-o", "comm=", "-p", ",".join(found)], capture_output=True, text=True, timeout=10
        ).stdout.splitlines()
    except (OSError, subprocess.TimeoutExpired):  # không có pgrep hay ps: coi như không biết
        return []
    return sorted({Path(name.strip()).name for name in names if name.strip()})


class BusyWatch:
    """Lấy mẫu định kỳ xem máy có việc nặng khác (pytest, vitest, next build...) chạy cùng không.

    Máy dev dùng chung nên số đo dễ bị nhiễu; các khoảng bận được ghi lại để biết pha nào cần đo
    lại. Một lần lấy mẫu chỉ phản ánh đúng lúc nó được lấy, nên việc ngắn hơn chu kỳ có thể sót.
    """

    def __init__(
        self, probe: Callable[[], list[str]] = heavy_processes, period: float = WATCH_PERIOD_S
    ):
        self.probe, self.period = probe, period
        self.samples: list[tuple[float, list[str]]] = []  # (giờ epoch, tên các việc nặng)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append((time.time(), self.probe()))
            self._stop.wait(self.period)

    def __enter__(self) -> "BusyWatch":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=30)

    def seen_between(self, start: float, end: float) -> list[str]:
        """Tên các việc nặng thấy trong [start, end], tính cả lần lấy mẫu ngay trước `start`."""
        names: set[str] = set()
        for taken, kinds in list(self.samples):
            if start - self.period <= taken <= end:
                names.update(kinds)
        return sorted(names)

    def busy_windows(self) -> list[dict]:
        """Các khoảng liên tục có việc nặng khác: giờ bắt đầu, giờ kết thúc và tên các việc."""
        windows: list[dict] = []
        last = None
        for taken, kinds in list(self.samples):
            if not kinds:
                continue
            if windows and last is not None and taken - last <= 2 * self.period:
                windows[-1]["end"] = clock(taken)
                windows[-1]["kinds"] = sorted({*windows[-1]["kinds"], *kinds})
            else:
                windows.append({"start": clock(taken), "end": clock(taken), "kinds": sorted(kinds)})
            last = taken
        return windows

    def summary(self) -> dict:
        samples = list(self.samples)
        busy = sum(1 for _, kinds in samples if kinds)
        return {"samples": len(samples), "busy_samples": busy, "windows": self.busy_windows()}


class Timeline:
    """Giờ bắt đầu/kết thúc từng pha đo và việc nặng khác thấy được trong pha đó (`noisy_with`).

    `noisy_with` là None khi không theo dõi, [] khi theo dõi mà máy sạch, còn lại là danh sách tên.
    """

    def __init__(self, watch: BusyWatch | None = None):
        self.watch = watch
        self.phases: list[dict] = []

    @contextmanager
    def phase(self, name: str):
        started = time.time()
        try:
            yield
        finally:  # pha lỗi vẫn được ghi giờ
            ended = time.time()
            seen = None if self.watch is None else self.watch.seen_between(started, ended)
            self.phases.append(
                {
                    "name": name,
                    "start": clock(started),
                    "end": clock(ended),
                    "seconds": round(ended - started, 1),
                    "noisy_with": seen,
                }
            )


# --- Phép đo ------------------------------------------------------------------------------------


def summarize(values: list[float]) -> dict:
    """p95 theo hạng gần nhất; với 30 mẫu đó là giá trị lớn thứ hai. Không có mẫu thì n = 0."""
    if not values:
        return {"n": 0, "p50_s": None, "p95_s": None, "max_s": None}
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


def analyze_once(base: str, text: str) -> tuple[float, bool]:
    """Thời gian của /api/analyze và việc máy chủ có giải thích được câu này không."""
    reply = call(base, "POST", "/api/analyze", {"text": text})
    if reply.status != 200:
        raise SmokeError(f"POST /api/analyze trả {reply.status}: {reply.body[:200]!r}")
    return reply.seconds, reply.json().get("xai_available") is True


def measure_latency(base: str, texts: list[str], repeat: int) -> dict:
    """Độ trễ nhãn và giải thích, đo riêng từng cái (lần lượt, không chồng nhau).

    - `analyze`: /api/analyze trên `LABEL_SAMPLES` câu (rẻ, và đủ nhiều để có cả đuôi câu dài);
    - `explain`: /api/explain trên tối đa `repeat` câu trong số đó mà máy chủ báo giải thích được
      (`xai_available`); `not_explained` đếm các câu mẫu quá dài nên bị bỏ khỏi phép đo này;
    - `analyze_explain`: hai lời gọi song song như giao diện, tính tới khi cả hai xong (tham khảo).
    """
    label_n = max(repeat, min(len(texts), LABEL_SAMPLES))
    analyze: list[float] = []
    explainable: list[str] = []
    for i in range(label_n):
        text = texts[i % len(texts)]
        seconds, can_explain = analyze_once(base, text)
        analyze.append(seconds)
        if can_explain:
            explainable.append(text)
    explain, pair = [], []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for text in explainable[:repeat]:
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
        "not_explained": label_n - len(explainable),
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


def build_upload_csv(
    sentences: list[str], target_bytes: int, rows: int, group_marker: str
) -> bytes:
    """CSV UTF-8 có BOM với đúng `rows` dòng và đúng `target_bytes` byte (nếu phần câu chưa vượt).

    `Góp ý` là câu thật, `Nhóm` mang dấu hiệu nhận biết để soi nhật ký, `Ghi chú` là phần đệm cho đủ
    dung lượng: file 5MB trong 2.000 dòng nghĩa là mỗi dòng mang nhiều dữ liệu ngoài câu góp ý.
    """

    def render(padding: list[str]) -> bytes:
        out = io.StringIO()
        writer = csv.writer(out, lineterminator="\r\n")
        writer.writerow(["Góp ý", "Nhóm", "Ghi chú"])
        for i in range(rows):
            text = sentences[i % len(sentences)].replace("\n", " ")
            writer.writerow([text, f"{group_marker} {i % 4}", padding[i]])
        return b"\xef\xbb\xbf" + out.getvalue().encode("utf-8")

    base = len(render([""] * rows))
    per_row = max(0, (target_bytes - base) // rows)
    filler = ("ghi chu dem " * (per_row // 12 + 1))[:per_row]
    padding = [filler] * rows
    padding[-1] += "x" * max(0, target_bytes - len(render(padding)))
    return render(padding)


def multipart_file(field: str, filename: str, data: bytes) -> tuple[bytes, str]:
    """Thân `multipart/form-data` chứa một file; trả về (thân, giá trị Content-Type)."""
    boundary = "----smoke" + secrets.token_hex(8)
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: text/csv\r\n\r\n"
    ).encode()
    return (
        head + data + f"\r\n--{boundary}--\r\n".encode(),
        f"multipart/form-data; boundary={boundary}",
    )


def poll_until_done(base: str, batch_id: str, timeout: float) -> tuple[dict, list[float]]:
    """Hỏi trạng thái mỗi giây tới khi batch xong hoặc lỗi; trả về (trạng thái, các tiến độ)."""
    deadline = time.monotonic() + timeout
    progress: list[float] = []
    status: dict = {}
    while time.monotonic() < deadline:
        reply = call(base, "GET", f"/api/batches/{batch_id}")
        if reply.status != 200:
            return {"state": f"http_{reply.status}"}, progress
        status = reply.json()
        progress.append(status.get("progress", 0.0))
        if status.get("state") in ("done", "failed"):
            return status, progress
        time.sleep(1.0)
    return {**status, "state": "timeout"}, progress


def check_upload(
    base: str, sentences: list[str], megabytes: int = UPLOAD_MB, timeout: float = UPLOAD_TIMEOUT_S
) -> tuple[dict, str | None]:
    """Đường batch như người dùng thật: tải lên file ~5MB (2.000 dòng), chạy, đọc kết quả, đóng.

    Trả về (kết quả đo, id batch). `problems` liệt kê từng bước sai và `ok` chỉ đúng khi không có
    bước nào sai. Id batch trả riêng để phép kiểm nhật ký chắc rằng nó không bị ghi ra. Mất kết nối
    giữa chừng (vd. container bị OOM-kill) thành một `problem` chứ không làm hỏng cả lượt đo.
    """
    data = build_upload_csv(sentences, megabytes * 1024 * 1024, MAX_ROWS, MARKERS[3])
    problems: list[str] = []
    result: dict = {"bytes": len(data), "rows": MAX_ROWS, "problems": problems, "ok": False}
    batch_id: str | None = None
    try:
        batch_id = _upload_steps(base, data, timeout, result, problems)
    except OSError as error:
        problems.append(f"mất kết nối giữa chừng ({type(error).__name__})")
    result["ok"] = not problems
    return result, batch_id


def _upload_steps(
    base: str, data: bytes, timeout: float, result: dict, problems: list[str]
) -> str | None:
    """Các bước của `check_upload`; trả về id batch (nếu đã tạo được) để dọn và soi nhật ký."""
    rows = result["rows"]
    body = multipart_file("file", f"{FILE_MARKER}.csv", data)
    upload = call(base, "POST", "/api/batches", raw=body, timeout=120)
    result["upload_s"] = round(upload.seconds, 2)
    batch_id = upload.json().get("batch_id") if upload.status == 201 else None
    if not isinstance(batch_id, str):
        problems.append(f"tải lên trả {upload.status}")
        return None
    if upload.json().get("n_rows") != rows:
        problems.append(f"n_rows = {upload.json().get('n_rows')}, cần {rows}")

    started = time.perf_counter()
    run_body = {"text_col": "Góp ý", "group_col": "Nhóm", "review_budget": 10}
    run = call(base, "POST", f"/api/batches/{batch_id}/run", run_body)
    if run.status != 202:
        problems.append(f"chạy trả {run.status}")
        return batch_id
    status, progress = poll_until_done(base, batch_id, timeout)
    result["run_s"] = round(time.perf_counter() - started, 1)
    if status.get("state") != "done":
        problems.append(f"batch kết thúc ở trạng thái {status.get('state')}")
    if progress != sorted(progress):
        problems.append("tiến độ bị giảm")

    if status.get("state") == "done":
        results = call(base, "GET", f"/api/batches/{batch_id}/results", timeout=120)
        summary = results.json().get("summary", {}) if results.status == 200 else {}
        if results.status != 200:
            problems.append(f"kết quả trả {results.status}")
        elif summary.get("n_rows") != rows or len(results.json().get("rows", [])) != rows:
            problems.append("kết quả không đủ số dòng")
        elif summary.get("n_flagged", 0) < math.ceil(0.10 * rows):
            problems.append("số câu bị cờ ít hơn ngân sách 10%")
        result["flagged"] = summary.get("n_flagged")
    closed = call(base, "POST", f"/api/batches/{batch_id}/close")
    gone = call(base, "GET", f"/api/batches/{batch_id}")
    if closed.status != 204 or gone.status != 404:
        problems.append(f"đóng batch: close {closed.status}, đọc lại {gone.status}")
    return batch_id


def measure_compare(base: str, text: str) -> dict:
    """So sánh với mô hình cơ sở NGAY sau khi khởi động: thời gian nạp lạnh và RSS trước/sau.

    Mô hình cơ sở (scikit-learn + skops) chỉ được nạp ở lần `compare` đầu tiên. Phép đo này chạy
    trước mọi phép đo khác nên `memory.peak` cuối lượt đã gồm cả phần nạp đó: cổng RAM chấm khi đã
    bật so sánh. RSS lấy từ /api/health, nên phải đo trong container chỉ có group `app` (trên máy
    dev `import skops.io` còn kéo theo torch và pandas).
    """
    before = call(base, "GET", "/api/health").json().get("rss_mb")
    cold = call(base, "POST", "/api/analyze", {"text": text, "compare": True}, timeout=180)
    warm = call(base, "POST", "/api/analyze", {"text": text, "compare": True})
    after = call(base, "GET", "/api/health").json().get("rss_mb")
    baseline = cold.json().get("baseline") if cold.status == 200 else None
    numbers = isinstance(before, int | float) and isinstance(after, int | float)
    return {
        "ok": isinstance(baseline, dict) and warm.status == 200,
        "cold_s": round(cold.seconds, 3),
        "warm_s": round(warm.seconds, 3),
        "predict_ms": baseline.get("elapsed_ms") if isinstance(baseline, dict) else None,
        "rss_before_mb": before,
        "rss_after_mb": after,
        "rss_delta_mb": round(after - before, 1) if numbers else None,
    }


def baseline_load_seconds(logs: str) -> float | None:
    """Thời gian nạp mô hình cơ sở mà máy chủ tự ghi vào log lúc nạp lười."""
    found = BASELINE_LOADED.search(logs)
    return float(found.group(1)) if found else None


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


def check_logs(logs: str, extra_markers: tuple[str, ...] = ()) -> dict[str, bool]:
    """`extra_markers` là những chuỗi chỉ biết sau khi chạy (vd. id batch của lần upload thử)."""
    forbidden = (*MARKERS, FILE_MARKER, *extra_markers)
    return {
        "logs_have_no_request_content": not any(marker in logs for marker in forbidden),
        "logs_have_route_template": re.search(r"POST /api/analyze 200 \d+ms", logs) is not None,
    }


def run_bench(
    image: str,
    name: str,
    cpus: float,
    memory: str,
    sentences: Path,
    n: int,
    xai: int,
    batch_size: int | None,
) -> dict:
    """1.000 câu end-to-end trong container riêng cùng giới hạn; bộ nhớ đỉnh do bench tự đọc."""
    container = Container(name, image, cpus, memory)
    command = ("python", "-m", "edupulse.engine.bench", "--input", "/tmp/sentences.txt")
    command += ("--n", str(n), "--xai", str(xai), "--json")
    if batch_size is not None:
        command += ("--batch-size", str(batch_size))
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


def server_limits(base: str) -> dict:
    """Giới hạn máy chủ đang áp dụng, đọc từ /api/config để đo đúng thứ đang chạy (VPS có thể đặt
    EDUPULSE_XAI_MAX_WORDS khác máy đo). Không đọc được thì dùng cấu hình của máy đo."""
    try:
        reply = call(base, "GET", "/api/config", timeout=10)
    except OSError:
        reply = None
    limits = reply.json().get("limits") if reply is not None and reply.status == 200 else None
    value = limits.get("xai_max_words") if isinstance(limits, dict) else None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return {"xai_max_words": value, "source": "/api/config"}
    return {"xai_max_words": load_settings().xai_max_words, "source": "cấu hình máy đo"}


def measure_service(
    base: str, texts: list[str], repeat: int, timeline: Timeline | None = None
) -> dict:
    """Phần đo HTTP dùng chung cho Docker và `--url`: độ trễ lúc rảnh, câu dài, và khi có tải.

    `texts` là câu thật với độ dài tự nhiên: nhãn đo trên tất cả, giải thích chỉ trên câu ≤ giới
    hạn của máy chủ. Giờ từng pha được ghi vào `timeline` nếu có.
    """
    timeline = timeline or Timeline()
    if not texts:
        raise SmokeError("Không có câu nào để đo")
    limits = server_limits(base)
    cap = limits["xai_max_words"]
    sample = random.Random(0).sample(texts, min(len(texts), max(repeat, LABEL_SAMPLES)))
    explainable = [t for t in sample if len(t.split()) <= cap]
    if not explainable:
        raise SmokeError(f"Không có câu nào ≤ {cap} từ để đo giải thích")
    near_cap = [t for t in texts if round(cap * LONG_SHARE) <= len(t.split()) <= cap]
    result: dict = {"limits": limits}
    with timeline.phase("Độ trễ khi rảnh: nhãn, giải thích"):
        first = post_text(base, "/api/analyze", sample[0])  # gồm nạp JVM ở lần đầu
        for text in explainable[:3]:
            post_text(base, "/api/explain", text)
        result["first_analyze_s"] = round(first, 3)
        result["latency_idle"] = measure_latency(base, sample, repeat)
    if result["latency_idle"]["explain"]["n"] == 0:
        raise SmokeError(
            f"Máy chủ không giải thích câu nào trong {repeat} câu mẫu (giới hạn {cap})"
        )
    if len(near_cap) >= 5:
        with timeline.phase("Độ trễ khi rảnh: câu dài sát giới hạn giải thích"):
            result["latency_idle_long"] = measure_latency(base, near_cap, repeat)
    with timeline.phase("Độ trễ khi có tải nền"), Load(base, sample) as load:
        time.sleep(1.0)
        result["latency_under_load"] = measure_latency(base, sample, repeat)
    result["load"] = load.summary()
    return result


def run_scenario(
    args: argparse.Namespace,
    label: str,
    cpus: float,
    texts: list[str],
    bench_file: Path,
    watch: BusyWatch | None = None,
) -> dict:
    """Một lượt đo đầy đủ trong Docker với `cpus` CPU và `args.memory` RAM."""
    base = f"http://127.0.0.1:{args.port or free_port()}"
    port = int(base.rsplit(":", 1)[1])
    server = Container(f"edupulse-smoke-{label}", args.image, cpus, args.memory, port)
    timeline = Timeline(watch)
    result: dict = {"label": label, "cpus": cpus, "memory": args.memory}
    try:
        with timeline.phase("Khởi động container tới health 200"):
            server.start()
            result["startup_s"] = round(wait_for_health(base, server), 2)
        result["health"] = call(base, "GET", "/api/health").json()
        with timeline.phase("So sánh với mô hình cơ sở: nạp lạnh, RSS"):
            result["compare"] = measure_compare(base, texts[0])
        result.update(measure_service(base, texts, args.repeat, timeline))
        with timeline.phase("Hành vi API"):
            result["checks"] = check_api(base)
            result["checks"]["compare_with_baseline_works"] = result["compare"]["ok"]
        with timeline.phase(f"Upload {UPLOAD_MB}MB chạy xong"):
            result["upload"], batch_id = check_upload(base, texts)
        with timeline.phase("Yêu cầu dò nội dung nhật ký"):
            probe_logged_content(base, texts[0])
        bench_name = f"{server.name}-bench"
        with timeline.phase(f"Bench {args.n} câu trong container riêng"):
            result["bench"] = run_bench(
                args.image,
                bench_name,
                cpus,
                args.memory,
                bench_file,
                args.n,
                args.xai,
                args.bench_batch_size,
            )
        if args.also_batched and args.bench_batch_size != BATCHED_REFERENCE:
            with timeline.phase(f"Bench lô ONNX {BATCHED_REFERENCE} (tham khảo)"):
                result["bench_batched"] = run_bench(
                    args.image,
                    f"{bench_name}-batched",
                    cpus,
                    args.memory,
                    bench_file,
                    args.n,
                    args.xai,
                    BATCHED_REFERENCE,
                )
        # Đọc sau bench để các dòng nhật ký của yêu cầu dò đã kịp ghi ra.
        logs = "".join(server.logs())
        extra_markers = () if batch_id is None else (batch_id,)
        result["checks"].update(check_logs(logs, extra_markers))
        result["compare"]["load_s"] = baseline_load_seconds(logs)
        result["server_memory_peak_mb"] = server.memory_peak_mb()
        result["server_state"] = server.state()
    finally:
        server.remove()
    result["phases"] = timeline.phases
    return result


def run_remote(args: argparse.Namespace, texts: list[str], watch: BusyWatch | None = None) -> dict:
    base = args.url.rstrip("/")
    timeline = Timeline(watch)
    result: dict = {"label": "remote", "url": base, "startup_s": None}
    result["health"] = call(base, "GET", "/api/health").json()
    with timeline.phase("So sánh với mô hình cơ sở: nạp lạnh, RSS"):
        result["compare"] = measure_compare(base, texts[0])
    result.update(measure_service(base, texts, args.repeat, timeline))
    with timeline.phase("Hành vi API"):
        result["checks"] = check_api(base)
        result["checks"]["compare_with_baseline_works"] = result["compare"]["ok"]
    with timeline.phase(f"Upload {UPLOAD_MB}MB chạy xong"):
        result["upload"], _ = check_upload(base, texts)
    result["phases"] = timeline.phases
    return result


# --- Dữ liệu vào, dựng image --------------------------------------------------------------------


def load_sentences(sentences_file: str | None, settings: Settings) -> list[str]:
    """Câu thật: `FILE` (mỗi dòng một câu) hoặc tập val đã xử lý ở local."""
    if sentences_file:
        lines = Path(sentences_file).read_text(encoding="utf-8").splitlines()
        return [line.strip() for line in lines if line.strip()]
    parquet = settings.data_dir / "processed" / "validation.parquet"
    if not parquet.is_file():
        raise SmokeError(f"Thiếu {parquet}: chạy `make data` hoặc truyền --sentences FILE")
    import pyarrow.parquet as pq  # chỉ cần ở máy dev, không thuộc group app

    column = pq.read_table(parquet, columns=["sentence"]).column(0)
    return [str(s) for s in column.to_pylist()]


def read_sentences(args: argparse.Namespace, settings: Settings) -> list[str]:
    """Câu thật để đo: `--sentences FILE` hoặc tập val, đủ cho bench `--n` câu."""
    sentences = load_sentences(args.sentences, settings)
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


def image_content_mb(image: str) -> int | None:
    """Dung lượng nén (phần phải tải về) lấy từ cột CONTENT SIZE của `docker image ls`; None nếu
    bản Docker không có cột đó hoặc không đọc được."""
    rows = docker("image", "ls", image, check=False).splitlines()
    if len(rows) < 2:
        return None
    fields = re.split(r"\s{2,}", rows[1].strip())
    match = re.fullmatch(r"([\d.]+)([kMG]?B)", fields[3]) if len(fields) >= 4 else None
    return None if match is None else round(float(match[1]) * _SIZE_UNITS[match[2]] / 1e6)


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
        "compressed_mb": image_content_mb(args.image),
        "build_s": None if args.skip_build else round(time.monotonic() - started, 1),
    }


# --- Kết luận và báo cáo ------------------------------------------------------------------------


def p95_of(gate: dict, situation: str, command: str) -> float | None:
    """p95 đã đo của `command` ở `situation` (vd. "latency_idle", "analyze"); None nếu chưa đo."""
    summary = gate.get(situation, {}).get(command)
    return summary.get("p95_s") if isinstance(summary, dict) else None


def describe_upload(upload: dict | None) -> str | None:
    if not upload:
        return None
    text = f"{upload['bytes'] / 1048576:.1f} MB, {upload['rows']} dòng"
    return text if "run_s" not in upload else f"{text}, chạy {upload['run_s']:g} s"


def verdict_rows(gate: dict) -> list[dict]:
    """Bảng go/no-go; `ok` là None khi chỉ số không đo được ở chế độ này.

    Nhãn và giải thích là hai dòng riêng với hai p95 riêng (PDR 5): nhãn đo trên mọi câu, giải
    thích chỉ trên câu ≤ `limits.xai_max_words` của máy chủ.
    """
    bench = gate.get("bench") or {}
    peaks = [gate.get("server_memory_peak_mb"), bench.get("cgroup_memory_peak_mb")]
    peak = max((p for p in peaks if p is not None), default=None)
    ooms = [gate.get("server_state", {}).get("oom_killed"), bench.get("oom_killed")]
    label = p95_of(gate, "latency_idle", "analyze")
    explain = p95_of(gate, "latency_idle", "explain")
    cap = gate.get("limits", {}).get("xai_max_words", "?")
    upload = gate.get("upload")
    compare = gate.get("compare") or {}
    rss_after = compare.get("rss_after_mb")
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
            "memory.peak (MB), đã gồm so sánh với mô hình cơ sở",
            f"≤ {GO['memory_peak_mb']}",
            peak,
            within(peak, GO["memory_peak_mb"]),
        ),
        row(
            "RSS tiến trình sau khi nạp mô hình cơ sở (MB)",
            f"≤ {GO['memory_peak_mb']}",
            rss_after,
            within(rss_after, GO["memory_peak_mb"]),
        ),
        row(
            "So sánh mô hình: nạp lạnh lần đầu (s)",
            INFO,
            compare.get("cold_s"),
            None,
        ),
        row(
            "Nhãn 1 câu: p95 /api/analyze (s)",
            f"≤ {GO['label_p95_s']:g}",
            label,
            within(label, GO["label_p95_s"]),
        ),
        row(
            f"Giải thích: p95 /api/explain, câu ≤ {cap} từ (s)",
            f"≤ {GO['explain_p95_s']:g}",
            explain,
            within(explain, GO["explain_p95_s"]),
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
            f"Upload {UPLOAD_MB}MB chạy xong",
            "không lỗi",
            describe_upload(upload),
            None if not upload else bool(upload.get("ok")),
        ),
        row(
            "Hành vi API và nhật ký",
            "tất cả đạt",
            f"{sum(checks.values())}/{len(checks)}",
            all(checks.values()),
        ),
    ]


def is_go(verdict: list[dict]) -> bool:
    """GO khi không dòng nào KHÔNG ĐẠT; chỉ số không đo được (ok = None) không làm hỏng kết luận."""
    return all(row["ok"] is not False for row in verdict)


def table(header: list[str], rows: list[list]) -> list[str]:
    cells = lambda row: "| " + " | ".join("-" if c is None else str(c) for c in row) + " |"  # noqa: E731
    return [cells(header), "|" + "---|" * len(header), *(cells(row) for row in rows)]


COMMAND_TITLES = {
    "analyze": "nhãn (/api/analyze)",
    "explain": "giải thích (/api/explain)",
    "analyze_explain": "cặp song song như giao diện (tham khảo)",
}


def latency_rows(gate: dict) -> list[list]:
    titles = {
        "latency_idle": "rảnh",
        "latency_idle_long": "rảnh, câu dài sát giới hạn giải thích",
        "latency_under_load": "đang có tải",
    }
    return [
        [title, COMMAND_TITLES.get(name, name), s["n"], s["p50_s"], s["p95_s"], s["max_s"]]
        for key, title in titles.items()
        for name, s in gate.get(key, {}).items()
        if isinstance(s, dict)  # bỏ `not_explained` (một số nguyên)
    ]


def latency_summary(gate: dict) -> list[str]:
    """Hai p95 chấm riêng, nêu thẳng ngưỡng, số mẫu và vì sao số mẫu giải thích ít hơn."""
    idle = gate.get("latency_idle")
    if not idle:
        return []
    label, explain = idle["analyze"], idle["explain"]
    limits = gate.get("limits", {})
    cap, source = limits.get("xai_max_words", "?"), limits.get("source", "-")
    return [
        "",
        "Hai chỉ số độ trễ chấm riêng (PDR 5: nhãn hiện ngay, giải thích tải sau), đo lúc máy rảnh,"
        " từng lời gọi một:",
        "",
        f"- **Nhãn**, p95 của /api/analyze: **{label['p95_s']} s**"
        f" (ngưỡng ≤ {GO['label_p95_s']:g}), trên {label['n']} câu thật, mọi độ dài.",
        f"- **Giải thích**, p95 của /api/explain: **{explain['p95_s']} s**"
        f" (ngưỡng ≤ {GO['explain_p95_s']:g}), trên {explain['n']} câu không quá {cap} từ"
        f" (giới hạn đọc từ {source}); {idle.get('not_explained', 0)}/{label['n']} câu mẫu dài hơn"
        " giới hạn nên không được giải thích và không tính vào số này.",
    ]


def compare_section(gate: dict) -> list[str]:
    """Nút "So sánh với mô hình cơ sở": nạp lạnh lần đầu và bộ nhớ sau khi nạp."""
    compare = gate.get("compare")
    if not compare:
        return []
    lines = ["", "## So sánh với mô hình cơ sở (nạp lười ở lần đầu)", ""]
    loaded = compare.get("load_s")
    lines.append(
        f"- Lần `compare` đầu sau khi khởi động (nạp lạnh): {compare['cold_s']} s;"
        f" lần sau {compare['warm_s']} s; mô hình cơ sở tự báo {compare.get('predict_ms', '-')} ms"
        " cho phần dự đoán"
    )
    if loaded is not None:
        lines.append(f"- Nạp scikit-learn và hai mô hình skops: {loaded} s theo log của máy chủ")
    if compare.get("rss_delta_mb") is not None:
        lines.append(
            f"- RSS tiến trình: {compare['rss_before_mb']} MB → {compare['rss_after_mb']} MB"
            f" (+{compare['rss_delta_mb']} MB)"
        )
    peak = gate.get("server_memory_peak_mb")
    if peak is not None:
        lines.append(
            f"- memory.peak của container cuối lượt đo: {peak} MB, tính cả phần nạp mô hình cơ sở"
            f" (ngưỡng ≤ {GO['memory_peak_mb']})"
        )
    if not compare["ok"]:
        lines.append("- **Lỗi**: so sánh với mô hình cơ sở không chạy được")
    return lines


def watch_text(watch: dict | None) -> str:
    """Một câu về việc nặng của người khác trong lúc đo (kết quả của `BusyWatch.summary`)."""
    if not watch:
        return "chưa theo dõi việc nặng khác (pytest, playwright, vitest, next build)"
    if not watch["windows"]:
        return f"không thấy việc nặng khác chạy cùng ({watch['samples']} lần lấy mẫu)"
    spans = "; ".join(
        f"{w['start']}–{w['end']} ({', '.join(w['kinds'])})" for w in watch["windows"]
    )
    return (
        f"**có việc nặng khác chạy cùng** lúc {spans}"
        f" ({watch['busy_samples']}/{watch['samples']} lần lấy mẫu)"
    )


def noisy_phases(run: dict) -> list[str]:
    """Tên các pha của một lượt đo có việc nặng khác chạy cùng lúc."""
    return [p["name"] for p in run.get("phases", []) if p.get("noisy_with")]


def noise_cell(noisy_with: list[str] | None) -> str:
    if noisy_with is None:
        return "-"
    return f"CÓ: {', '.join(noisy_with)}" if noisy_with else "không"


def phase_section(report: dict) -> list[str]:
    """Giờ bắt đầu/kết thúc từng pha của mỗi lượt đo, để về sau đối chiếu với việc khác trên máy."""
    tables: list[str] = []
    for run in report["runs"]:
        if run.get("phases"):
            rows = [
                [p["name"], p["start"], p["end"], p["seconds"], noise_cell(p.get("noisy_with"))]
                for p in run["phases"]
            ]
            header = ["Pha", "Bắt đầu", "Kết thúc", "Giây", "Nhiễu"]
            tables += ["", f"Lượt `{run.get('label', '-')}`:", "", *table(header, rows)]
    if not tables:
        return []
    note = (
        "Giờ máy (HH:MM:SS). Cột Nhiễu cho biết pytest, playwright, vitest hay next build có chạy"
        f" cùng lúc không (lấy mẫu mỗi {WATCH_PERIOD_S:g} s); dấu - là không theo dõi."
    )
    return ["", "## Giờ từng pha đo", "", note, *tables]


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
            f"- Image `{image['tag']}`: {image['size_mb']} MB tổng các lớp chưa nén,"
            f" {image.get('compressed_mb') or '-'} MB nén (phần tải về);"
            f" dựng mất {image['build_s']} s"
        )
    limits = f"--cpus={gate.get('cpus', '-')} --memory={gate.get('memory', '-')}"
    lines.append(f"- Giới hạn đo chính: `{limits}`, không swap")
    lines.append(
        f"- Giờ đo {host.get('started', '-')} → {host.get('ended', '-')}; "
        + watch_text(host.get("heavy_watch"))
    )
    noisy = noisy_phases(gate)
    if noisy:
        lines.append(
            f"- **Cảnh báo nhiễu**: pha {', '.join(noisy)} của lượt đo chính trùng việc nặng khác;"
            " số đo của chúng nên được đo lại (kết luận GO/NO-GO vẫn tính theo số đã đo)"
        )
    lines += ["", "## Bảng go/no-go", ""]
    verdict = [
        [
            r["metric"],
            r["threshold"],
            r["value"],
            INFO if r["threshold"] == INFO else OUTCOME[r["ok"]],
        ]
        for r in report["verdict"]
    ]
    lines += table(["Chỉ số", "Ngưỡng", "Đo được", "Kết quả"], verdict)
    lines += latency_summary(gate)

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
        batch = bench.get("batch_size")
        title = "## Bench 1.000 câu trong container riêng" + (
            "" if batch is None else f" (lô ONNX {batch})"
        )
        lines += ["", title, ""] + table(["Bước", "Giây"], steps)
        lines += [
            "",
            f"RSS cuối {bench.get('rss_mb', '-')} MB;"
            f" cgroup memory.peak {bench.get('cgroup_memory_peak_mb', '-')} MB.",
        ]
        throttle = bench.get("cgroup_cpu_throttle")
        if throttle:
            lines.append(
                f"Giới hạn CPU (CFS): bị cắt {throttle['throttled']}/{throttle['periods']} chu kỳ,"
                f" tổng {throttle['throttled_s']} s."
            )
    elif bench:
        lines += [
            "",
            "## Bench 1.000 câu",
            "",
            f"Lỗi: {bench['error']}",
            "",
            f"```\n{bench.get('stderr_tail', '')}\n```",
        ]

    batched = gate.get("bench_batched")
    if batched and "error" not in batched:
        ratio = bench["total_s"] / batched["total_s"] if bench and "total_s" in bench else None
        lines += [
            "",
            f"## Tham khảo: bench theo lô ONNX {BATCHED_REFERENCE} (kết quả phụ thuộc bạn cùng lô)",
            "",
            f"- 1.000 câu end-to-end {batched['total_s']} s"
            + ("" if ratio is None else f" (mặc định chậm hơn {ratio:.2f} lần)"),
            f"- predict {batched['steps_s']['predict']} s;"
            f" explain {batched['steps_s']['explain']} s;"
            f" cgroup memory.peak {batched.get('cgroup_memory_peak_mb', '-')} MB",
        ]
    elif batched:
        lines += ["", "## Tham khảo: bench theo lô 32", "", f"Lỗi: {batched['error']}"]

    lines += compare_section(gate)

    upload = gate.get("upload")
    if upload:
        lines += ["", f"## Upload {UPLOAD_MB}MB (đường batch như người dùng thật)", ""]
        lines.append(
            f"- File {upload['bytes'] / 1048576:.1f} MB, {upload['rows']} dòng:"
            f" tải lên {upload.get('upload_s', '-')} s, chạy {upload.get('run_s', '-')} s,"
            f" {upload.get('flagged', '-')} câu bị cờ"
        )
        lines.append(f"- Kết quả: {'đạt' if upload['ok'] else 'KHÔNG ĐẠT'}")
        lines += [f"- Lỗi: {problem}" for problem in upload.get("problems", [])]

    lines += ["", "## Kiểm tra hành vi", ""]
    lines += [
        f"- {name}: {'đạt' if ok else 'KHÔNG ĐẠT'}" for name, ok in gate.get("checks", {}).items()
    ]
    for extra in report["runs"][1:]:
        idle = extra["latency_idle"]
        total = (extra.get("bench") or {}).get("total_s", "-")
        lines += ["", f"## Tham khảo: {extra['cpus']:g} CPU", ""]
        lines += [
            f"- Khởi động {extra.get('startup_s')} s; nhãn p95 {idle['analyze']['p95_s']} s;"
            f" giải thích p95 {idle['explain']['p95_s']} s; 1.000 câu {total} s"
        ]
    lines += phase_section(report)
    lines += [
        "",
        "Số đo trên M1 (Docker trong VM Mac); chưa đo x86. Thời gian 1.000 câu đọc với hệ số an"
        " toàn ×3 cho VPS x86 chia sẻ. Giới hạn số từ của giải thích (XAI_MAX_WORDS) chốt lại"
        " sau khi đo trên VPS thật bằng `--url`.",
    ]
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
    add("--also-batched", action="store_true", help="thêm bench tham khảo chạy theo lô 32")
    add("--port", type=int, default=0, help="cổng máy chủ; 0 = tự chọn cổng trống")
    add("--sentences", help="file mỗi dòng một câu; mặc định lấy tập val đã xử lý")
    add("--n", type=int, default=1000, help="số câu của bench")
    add("--xai", type=int, default=20, help="số câu được giải thích tự động")
    add("--bench-batch-size", type=int, help="lô ONNX của bench; 1 = từng câu một (tham khảo)")
    add("--repeat", type=int, default=30, help="số lần lặp mỗi phép đo độ trễ")
    add("--tag", default=arch, help="hậu tố kiến trúc trong tên file báo cáo")
    add("--out-dir", type=Path, default=REPO_ROOT / "results" / "perf")
    return parser.parse_args()


def measure(
    args: argparse.Namespace, settings: Settings, watch: BusyWatch | None = None
) -> tuple[list[dict], dict | None]:
    texts = read_sentences(args, settings)
    if args.url:
        return [run_remote(args, texts, watch)], None
    image = build_image(args)
    with tempfile.TemporaryDirectory() as workdir:
        os.chmod(workdir, 0o755)  # container chạy uid 1000 phải đọc được thư mục gắn vào
        bench_file = Path(workdir) / "sentences.txt"
        bench_file.write_text(
            "\n".join(t.replace("\n", " ") for t in texts[: args.n]) + "\n", encoding="utf-8"
        )
        bench_file.chmod(0o644)
        runs = [run_scenario(args, "gate", args.cpus, texts, bench_file, watch)]
        if args.also_1cpu:
            runs.append(run_scenario(args, "1cpu", 1.0, texts, bench_file, watch))
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
        host["started"] = clock(time.time())
        with BusyWatch() as watch:
            runs, image = measure(args, settings, watch)
    except SmokeError as error:
        print(f"Smoke test lỗi: {error}", file=sys.stderr)
        return 3

    host["ended"] = clock(time.time())
    host["heavy_watch"] = watch.summary()
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
        "go": is_go(verdict),
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
