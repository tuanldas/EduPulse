"""Quét tệp xem có câu nào của UIT-VSFC lọt vào không (khớp n-gram từ, mặc định n = 8).

    python -m ml.audit_sentences results              # thư mục hoặc tệp, nhiều đường dẫn
    python -m ml.audit_sentences results docs --show  # in cả đoạn khớp

Giấy phép UIT-VSFC chỉ cho học thuật, phi thương mại, nên repo công khai không được chứa câu gốc.
Khớp theo n-gram từ (bỏ dấu câu, không phân biệt hoa thường, chuẩn hoá NFC) để bắt cả câu bị cắt
ngắn hay viết lại dấu câu chứ không chỉ khớp nguyên câu. Cụm ngắn hơn n từ (câu ngắn thông dụng như
"em rất hài lòng") không bị tính.

Mã thoát: 0 sạch, 1 có tệp trùng, 2 không đọc được dữ liệu để đối chiếu.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path

from ml.common import SPLITS, load_processed

LOG_PREFIX = "audit_sentences"
DEFAULT_NGRAM = 8
TEXT_SUFFIXES = frozenset(
    {".md", ".csv", ".json", ".txt", ".html", ".ts", ".tsx", ".js", ".py", ".yml", ".yaml", ".toml"}
)
SKIPPED_DIRS = frozenset({".git", "node_modules", "__pycache__", ".venv", ".next", "out"})
_WORD = re.compile(r"\w+")


def words(text: str) -> list[str]:
    """Từ viết thường (NFC) của `text`, bỏ dấu câu."""
    return _WORD.findall(unicodedata.normalize("NFC", text).lower())


def grams_of(tokens: list[str], n: int) -> Iterator[str]:
    for start in range(len(tokens) - n + 1):
        yield " ".join(tokens[start : start + n])


def corpus_grams(n: int = DEFAULT_NGRAM) -> set[str]:
    """Mọi n-gram từ của các câu UIT-VSFC (cả ba split; chỉ tìm chuỗi, không tính chỉ số nào)."""
    found: set[str] = set()
    for split in SPLITS:
        for sentence in load_processed(split, allow_test=True)["sentence"]:
            found.update(grams_of(words(str(sentence)), n))
    return found


def scan_text(text: str, grams: set[str], n: int = DEFAULT_NGRAM) -> list[tuple[int, str]]:
    """(số dòng từ 1, n-gram đầu tiên khớp) của mỗi dòng có n-gram trùng với dữ liệu."""
    hits = []
    for number, line in enumerate(text.splitlines(), start=1):
        for gram in grams_of(words(line), n):
            if gram in grams:
                hits.append((number, gram))
                break
    return hits


def iter_files(paths: Iterable[Path]) -> Iterator[Path]:
    """Các tệp văn bản dưới `paths` (tệp, hoặc thư mục duyệt đệ quy bỏ thư mục sinh ra)."""
    for path in paths:
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                relative = child.relative_to(path).parts
                if child.is_file() and child.suffix in TEXT_SUFFIXES:
                    if not SKIPPED_DIRS.intersection(relative):
                        yield child


def audit(paths: Iterable[Path], grams: set[str], n: int = DEFAULT_NGRAM) -> dict[Path, list]:
    """{tệp: [(dòng, n-gram)]} chỉ gồm tệp có trùng."""
    report = {}
    for file in iter_files(paths):
        hits = scan_text(file.read_text(encoding="utf-8", errors="ignore"), grams, n)
        if hits:
            report[file] = hits
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="tệp hoặc thư mục cần quét")
    parser.add_argument("--ngram", type=int, default=DEFAULT_NGRAM, help="số từ liên tiếp")
    parser.add_argument("--show", action="store_true", help="in đoạn khớp của từng dòng")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.ngram < 2:
        print(f"{LOG_PREFIX}: --ngram phải từ 2 trở lên", file=sys.stderr)
        return 2
    try:
        grams = corpus_grams(args.ngram)
    except (OSError, RuntimeError) as error:
        print(f"{LOG_PREFIX}: không đọc được dữ liệu để đối chiếu: {error}", file=sys.stderr)
        return 2
    report = audit(args.paths, grams, args.ngram)
    for file, hits in report.items():
        for number, gram in hits:
            print(f"{file}:{number}" + (f"  …{gram}…" if args.show else ""))
    lines = sum(len(hits) for hits in report.values())
    print(f"{LOG_PREFIX}: {len(report)} tệp, {lines} dòng có {args.ngram}-gram trùng UIT-VSFC")
    return 1 if report else 0


if __name__ == "__main__":
    sys.exit(main())
