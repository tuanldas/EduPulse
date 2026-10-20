"""Kiểm câu gốc UIT-VSFC và mẫu bí mật trong lịch sử git trước khi đẩy lên repo public.

Repo public chỉ chứa câu viết lại, không chứa câu gốc của bộ dữ liệu. Một dòng bị coi là câu gốc
khi có ít nhất `N` token liên tiếp trùng một câu trong `$EDUPULSE_HOME/data/raw`: không phân biệt
hoa thường, `_` của tách từ tính như khoảng trắng, `\\uXXXX` được giải mã, văn bản chuẩn hoá NFC.
Cùng lượt quét tìm token Hugging Face, GitHub, OpenAI, AWS và khoá riêng.

    python -m ml.check_raw_sentences                     # mọi commit của mọi ref
    python -m ml.check_raw_sentences --head              # chỉ cây HEAD
    python -m ml.check_raw_sentences X --not --remotes   # đối số chuyển cho `git rev-list`

Mã thoát: 0 sạch, 1 có câu gốc hoặc bí mật, 2 không có dữ liệu gốc để so. Chỉ in đường dẫn và
loại mẫu, không in đoạn trùng hay giá trị bí mật.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from ml.common import raw_dir

N = 8
MAX_BLOB_BYTES = 5_000_000
BINARY_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".onnx", ".skops", ".parquet",
     ".zip", ".pdf", ".jar"}
)
TOKEN = re.compile(r"[^\W_]+")
UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
SECRETS = {
    "token Hugging Face": re.compile(r"\bhf_[A-Za-z0-9]{30,}"),
    "token GitHub": re.compile(r"\b(?:gh[opsu]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})"),
    "khoá OpenAI": re.compile(r"\bsk-[A-Za-z0-9_-]{30,}"),
    "khoá AWS": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "khoá riêng": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
}

Grams = set[tuple[str, ...]]


def tokens(text: str) -> list[str]:
    text = UNICODE_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)
    return [token.lower() for token in TOKEN.findall(unicodedata.normalize("NFC", text))]


def load_grams(raw: Path) -> Grams:
    """Mọi chuỗi `N` token liên tiếp của các câu trong tệp parquet ở `raw`."""
    import pyarrow.parquet as pq

    grams: Grams = set()
    for path in sorted(raw.glob("*.parquet")):
        for sentence in pq.read_table(path, columns=["sentence"]).column("sentence").to_pylist():
            words = tokens(sentence)
            grams.update(tuple(words[i : i + N]) for i in range(len(words) - N + 1))
    return grams


def has_raw(line: str, grams: Grams) -> bool:
    words = tokens(line)
    return any(tuple(words[i : i + N]) in grams for i in range(len(words) - N + 1))


def git(repo: Path, *args: str, stdin: str | None = None) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, input=stdin, capture_output=True, text=True, check=True
    ).stdout


def text_blobs(repo: Path, revs: list[str], head: bool) -> list[tuple[str, str]]:
    """(sha, đường dẫn) của các blob cần quét, mỗi sha một lần, bỏ tệp nhị phân và tệp quá lớn."""
    if head:
        tree = git(repo, "ls-tree", "-r", "-z", "HEAD").split("\0")
        entries = [entry.split("\t", 1) for entry in tree if entry]
        listing = [(meta.split()[2], path) for meta, path in entries]
    else:
        lines = git(repo, "rev-list", "--objects", *revs).splitlines()
        listing = [tuple(line.split(" ", 1)) for line in lines if " " in line]
    seen: set[str] = set()
    picked = []
    for sha, path in listing:
        if sha not in seen and Path(path).suffix.lower() not in BINARY_SUFFIXES:
            seen.add(sha)
            picked.append((sha, path))
    if not picked:
        return []
    checked = git(repo, "cat-file", "--batch-check", stdin="".join(f"{sha}\n" for sha, _ in picked))
    kinds = {parts[0]: (parts[1], int(parts[2])) for parts in map(str.split, checked.splitlines())
             if len(parts) == 3}
    return [(sha, path) for sha, path in picked
            if kinds.get(sha, ("", 0))[0] == "blob" and kinds[sha][1] <= MAX_BLOB_BYTES]


def read_blobs(repo: Path, shas: list[str]) -> Iterator[bytes]:
    with subprocess.Popen(
        ["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as proc:
        assert proc.stdin and proc.stdout
        for sha in shas:
            proc.stdin.write(f"{sha}\n".encode())
            proc.stdin.flush()
            size = int(proc.stdout.readline().split()[2])
            yield proc.stdout.read(size)
            proc.stdout.read(1)
        proc.stdin.close()


def scan(repo: Path, grams: Grams, revs: list[str], head: bool):
    """Số dòng câu gốc theo tệp, loại bí mật theo tệp, số dòng câu gốc trong commit message."""
    raw_hits: Counter[str] = Counter()
    secret_hits: dict[str, set[str]] = defaultdict(set)
    blobs = text_blobs(repo, revs, head)
    for (_, path), data in zip(blobs, read_blobs(repo, [sha for sha, _ in blobs]), strict=True):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        secret_hits[path].update(name for name, rx in SECRETS.items() if rx.search(text))
        raw_hits[path] += sum(has_raw(line, grams) for line in text.splitlines())
    message_hits = 0
    if not head:
        messages = git(repo, "log", "--format=%B", *revs)
        message_hits = sum(has_raw(line, grams) for line in messages.splitlines())
        secret_hits["(commit message)"].update(
            name for name, rx in SECRETS.items() if rx.search(messages)
        )
    return +raw_hits, {path: names for path, names in secret_hits.items() if names}, message_hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--head", action="store_true", help="chỉ quét cây HEAD, bỏ lịch sử")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="mặc định: thư mục hiện hành")
    parser.add_argument("--raw-dir", type=Path, help="mặc định: $EDUPULSE_HOME/data/raw")
    args, revs = parser.parse_known_args(argv)
    raw = args.raw_dir or raw_dir()
    if not any(raw.glob("*.parquet")):
        print(f"không có parquet gốc ở {raw}: chạy `make data` rồi kiểm lại", file=sys.stderr)
        return 2
    raw_hits, secret_hits, message_hits = scan(args.repo, load_grams(raw), revs or ["--all"],
                                               args.head)
    for path, count in sorted(raw_hits.items()):
        print(f"câu gốc: {path} ({count} dòng)")
    for path, names in sorted(secret_hits.items()):
        print(f"bí mật: {path} ({', '.join(sorted(names))})")
    if message_hits:
        print(f"câu gốc trong commit message: {message_hits} dòng")
    if raw_hits or secret_hits or message_hits:
        return 1
    print("sạch: không có câu gốc UIT-VSFC hay bí mật")
    return 0


if __name__ == "__main__":
    sys.exit(main())
