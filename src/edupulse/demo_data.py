"""Dữ liệu demo tính lúc BUILD image: file mẫu, câu mẫu và kết quả dự đoán của tập test.

Repo và bundle không chứa câu gốc của UIT-VSFC (bộ dữ liệu học thuật, phi thương mại). Câu gốc chỉ
được tải về máy build rồi biến thành bốn tệp trong thư mục demo:

    sample.csv         200 câu val phân tầng, cột `Góp ý` và `Nhóm (minh hoạ)` (Nhóm A-E ngẫu nhiên)
    ui_samples.json    câu mẫu cho trang Phân tích câu: 6 câu val ngắn + 1 bản bỏ dấu
    demo_results.json  dự đoán của `Analyzer` trên 3.166 câu test, dùng cho dashboard demo
    demo_manifest.json ok / lý do / revision / bundle / kiến trúc máy build (ghi cuối cùng)

Không tải được dữ liệu (không có mạng, Hugging Face lỗi) thì ghi `ok: false` kèm lý do và thoát 0:
build image KHÔNG hỏng, API báo `demo_available = false` và giao diện ẩn phần demo. Lỗi khác (bundle
hỏng, mô hình lỗi) vẫn làm lệnh thất bại.

    python -m edupulse.demo_data                         # tải từ HF, ghi vào EDUPULSE_DEMO_DIR
    python -m edupulse.demo_data --source-dir data/raw   # dùng parquet có sẵn, không cần mạng

Phần nạp lúc chạy (`load_demo`) không cần pyarrow hay huggingface_hub: chỉ phần dựng mới import
chúng, và chỉ khi được gọi.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import platform
import random
import re
import sys
import tempfile
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from edupulse.bundle_schema import DemoManifest, DemoResults
from edupulse.config import Settings, load_settings
from edupulse.labels import SENTIMENT, TOPIC

if TYPE_CHECKING:
    from edupulse.engine.service import Analyzer

LOG = logging.getLogger("demo_data")

DATASET_REPO = "uitnlp/vietnamese_students_feedback"
# Commit của nhánh `refs/convert/parquet` (cùng giá trị với ml/download_data.py).
DATASET_REVISION = "2d76906157232f87e9883fa37ebfc91ffdee1f83"
REMOTE_PATHS = {
    "validation": "default/validation/0000.parquet",
    "test": "default/test/0000.parquet",
}
# sha256 của hai tệp trên (khớp data/processed/MANIFEST.json `raw_sha256`); lệch nghĩa là tải hỏng.
EXPECTED_SHA256 = {
    "validation": "68b0a7c97ced26c8c971e0a996719a345e7897870e8d3ea0d319cdce16b90514",
    "test": "74a223204066690a2d6b6c3dac85178e8acc43b0a7f2b11e25e52d5837911e78",
}
EXPECTED_ROWS = {"validation": 1583, "test": 3166}

SAMPLE_CSV = "sample.csv"
UI_SAMPLES = "ui_samples.json"
DEMO_RESULTS = "demo_results.json"
DEMO_MANIFEST = "demo_manifest.json"

TEXT_COLUMN = "Góp ý"
GROUP_COLUMN = "Nhóm (minh hoạ)"  # chứa "minh hoạ": API gắn cờ `illustrative_groups`
GROUP_NAMES = ("Nhóm A", "Nhóm B", "Nhóm C", "Nhóm D", "Nhóm E")
SAMPLE_SIZE = 200
SEED = 42
MIN_PER_CELL = 2  # mỗi ô (cảm xúc × chủ đề) có trong val ít nhất chừng này câu trong file mẫu
SAMPLE_WORDS = (3, 50)  # độ dài (số từ cách nhau bằng dấu cách) của câu được chọn vào file mẫu
UI_SAMPLE_WORDS = (6, 14)
# Câu mẫu của trang 1: (chủ đề, cảm xúc), theo thứ tự chip trong wireframe.
UI_SAMPLE_TARGETS = (
    ("lecturer", "positive"),
    ("lecturer", "neutral"),
    ("training_program", "positive"),
    ("facility", "negative"),
    ("others", "neutral"),
    ("lecturer", "negative"),
)
SKIPPED_REASON = "Bỏ qua dựng dữ liệu demo (EDUPULSE_BUILD_DEMO=0)."

_NAME_TOKEN = re.compile(r"\bwzjwz\d+\b")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;!?)\]])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[])\s+")
# Ô CSV bắt đầu bằng ký tự này bị Excel hiểu là công thức, nên bỏ chúng ở đầu câu.
_FORMULA_LEAD = "=+-@\t\r"
# Câu mẫu gọn: một câu kết thúc bằng dấu chấm, không số, dấu hai chấm, ngoặc, gạch.
_CLEAN_SENTENCE = re.compile(r"^[^\W\d_][^0-9:()/\-]*\.$")
_MAX_REASON_CHARS = 300


class DemoBuildError(RuntimeError):
    """Lỗi thật khi dựng demo (khác với việc không tải được dữ liệu)."""


class SourceUnavailable(RuntimeError):
    """Không lấy được dữ liệu nguồn; build vẫn tiếp tục với `demo_available = false`."""


# --- Định dạng tệp (dùng chung lúc build và lúc chạy) ------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class UiSampleItem(_Strict):
    text: str
    no_diacritics: bool


class UiSamplesFile(_Strict):
    schema_version: Literal[1] = 1
    samples: list[UiSampleItem]


class DemoRowData(_Strict):
    """Một câu test đã dự đoán. Nhãn gốc của bộ dữ liệu KHÔNG được lưu."""

    display_text: str  # đã ẩn thông tin cá nhân, viết hoa chữ đầu
    model_text: str  # đầu ra của `prepare()`, để đếm n-gram đúng như lúc đếm nền
    sentiment: Literal["negative", "neutral", "positive"]
    topic: Literal["lecturer", "training_program", "facility", "others"]
    sentiment_confidence: float
    topic_confidence: float
    n_words: int


@dataclass(frozen=True)
class DemoData:
    """Dữ liệu demo đã nạp. `available = false` thì phần còn lại rỗng và `reason` nêu lý do."""

    available: bool
    reason: str | None
    sample_csv: bytes
    ui_samples: tuple[UiSampleItem, ...]
    rows: tuple[DemoRowData, ...]
    manifest: DemoManifest | None


def _unavailable(reason: str, manifest: DemoManifest | None = None) -> DemoData:
    return DemoData(False, reason, b"", (), (), manifest)


def _check_sample_header(data: bytes) -> None:
    """File mẫu phải là CSV UTF-8 có đúng hai cột `Góp ý` và `Nhóm (minh hoạ)`."""
    header = next(csv.reader(io.StringIO(data.decode("utf-8-sig"))), [])
    if header != [TEXT_COLUMN, GROUP_COLUMN]:
        raise ValueError("tiêu đề file mẫu không đúng")


def load_demo(demo_dir: Path, bundle_version: str) -> DemoData:
    """Nạp dữ liệu demo; mọi vấn đề (thiếu tệp, tệp hỏng, lệch bundle) thành `available = false`.

    Không bao giờ ném lỗi: thiếu demo không được làm API chết. Lý do chỉ nêu loại vấn đề, không
    chép nội dung tệp.
    """
    manifest_path = demo_dir / DEMO_MANIFEST
    if not manifest_path.is_file():
        return _unavailable(
            "Chưa có dữ liệu demo: build image khi có mạng hoặc chạy `make demo-data`."
        )
    try:
        manifest = DemoManifest.model_validate_json(manifest_path.read_bytes())
    except (OSError, ValidationError):
        return _unavailable(f"{DEMO_MANIFEST} không đọc được hoặc sai định dạng.")
    if not manifest.ok:
        return _unavailable(manifest.reason or "Dữ liệu demo không khả dụng.", manifest)
    if manifest.bundle_version != bundle_version:
        return _unavailable(
            f"Dữ liệu demo tính bằng bundle {manifest.bundle_version}, đang chạy {bundle_version}: "
            "dựng lại dữ liệu demo.",
            manifest,
        )
    try:
        sample_csv = (demo_dir / SAMPLE_CSV).read_bytes()
        _check_sample_header(sample_csv)
        samples = UiSamplesFile.model_validate_json((demo_dir / UI_SAMPLES).read_bytes())
        results = DemoResults.model_validate_json((demo_dir / DEMO_RESULTS).read_bytes())
        rows = tuple(DemoRowData.model_validate(row.model_dump()) for row in results.rows)
    except (OSError, ValueError) as exc:  # ValidationError và lỗi giải mã đều là ValueError
        return _unavailable(f"Tệp dữ liệu demo thiếu hoặc hỏng ({type(exc).__name__}).", manifest)
    if results.bundle_version != bundle_version or len(rows) != manifest.n_rows:
        return _unavailable("Dữ liệu demo không khớp manifest: dựng lại.", manifest)
    return DemoData(True, None, sample_csv, tuple(samples.samples), rows, manifest)


# --- Chọn câu -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceRow:
    sentence: str
    sentiment: str  # nhãn chữ ("negative"...)
    topic: str


def beautify(text: str) -> str:
    """Làm đẹp nhẹ câu gốc để hiển thị: gọn khoảng trắng, dấu câu sát chữ, viết hoa chữ đầu.

    Giữ nguyên chữ nghĩa và lỗi chính tả gốc (design-guidelines §8). Tên người bị thay trong dữ
    liệu gốc (`wzjwz<k>`) hiện thành [TÊN].
    """
    text = _NAME_TOKEN.sub("[TÊN]", text)
    text = " ".join(text.split()).lstrip(_FORMULA_LEAD + " ")
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return text[:1].upper() + text[1:]


def strip_diacritics(text: str) -> str:
    """Bỏ dấu tiếng Việt: "Thầy giảng dạy" -> "Thay giang day"."""
    decomposed = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _n_words(text: str) -> int:
    return len(text.split())


def _eligible(row: SourceRow, words: tuple[int, int]) -> bool:
    """Câu không có tên bị che (`wzjwz<k>`) và độ dài vừa phải."""
    low, high = words
    return _NAME_TOKEN.search(row.sentence) is None and low <= _n_words(row.sentence) <= high


def _allocate(sizes: dict[tuple[str, str], int], total: int, minimum: int) -> dict:
    """Chia `total` suất cho các ô theo tỉ lệ (phần dư lớn nhất), mỗi ô có ít nhất `minimum`."""
    population = sum(sizes.values())
    if total > population:
        raise DemoBuildError(f"chỉ có {population} câu hợp lệ, cần {total}")
    floor = {cell: min(size, minimum) for cell, size in sizes.items()}
    if sum(floor.values()) > total:
        raise DemoBuildError(f"{len(sizes)} ô × {minimum} câu vượt {total}")
    exact = {cell: total * size / population for cell, size in sizes.items()}
    quota = {cell: max(floor[cell], int(exact[cell])) for cell in sizes}
    # Thêm theo phần dư lớn nhất, bớt từ ô đang vượt mức tỉ lệ nhiều nhất, tới khi đủ `total`.
    while sum(quota.values()) < total:
        cell = max(
            (c for c in sizes if quota[c] < sizes[c]),
            key=lambda c: (exact[c] - quota[c], sizes[c], c),
        )
        quota[cell] += 1
    while sum(quota.values()) > total:
        cell = max(
            (c for c in sizes if quota[c] > floor[c]), key=lambda c: (quota[c] - exact[c], c)
        )
        quota[cell] -= 1
    return quota


def pick_sample(rows: Sequence[SourceRow], size: int = SAMPLE_SIZE, seed: int = SEED) -> list:
    """`size` câu phân tầng theo (cảm xúc × chủ đề), không trùng chữ, thứ tự xáo trộn theo seed."""
    rng = random.Random(seed)
    seen: set[str] = set()
    cells: dict[tuple[str, str], list[SourceRow]] = {}
    for row in rows:
        key = " ".join(row.sentence.lower().split())
        if not _eligible(row, SAMPLE_WORDS) or key in seen:
            continue
        seen.add(key)
        cells.setdefault((row.sentiment, row.topic), []).append(row)
    quota = _allocate({cell: len(items) for cell, items in cells.items()}, size, MIN_PER_CELL)
    chosen: list[SourceRow] = []
    for cell in sorted(cells):
        items = cells[cell]
        rng.shuffle(items)
        chosen.extend(items[: quota[cell]])
    rng.shuffle(chosen)
    return chosen


def assign_groups(count: int, seed: int = SEED) -> list[str]:
    """Nhóm A-E gán ngẫu nhiên nhưng cố định theo seed. Đây là dữ liệu MINH HOẠ, không có thật."""
    rng = random.Random(seed)
    return [rng.choice(GROUP_NAMES) for _ in range(count)]


def sample_csv_bytes(rows: Sequence[SourceRow], seed: int = SEED) -> bytes:
    """CSV UTF-8 có BOM (mở thẳng bằng Excel) gồm cột `Góp ý` và `Nhóm (minh hoạ)`."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow([TEXT_COLUMN, GROUP_COLUMN])
    for row, group in zip(rows, assign_groups(len(rows), seed), strict=True):
        writer.writerow([beautify(row.sentence), group])
    return out.getvalue().encode("utf-8-sig")


def pick_ui_samples(rows: Sequence[SourceRow], seed: int = SEED) -> list[UiSampleItem]:
    """Mỗi cặp (chủ đề, cảm xúc) trong `UI_SAMPLE_TARGETS` một câu ngắn, rồi thêm bản bỏ dấu của
    câu đầu. Chọn theo seed, không phụ thuộc mô hình."""
    rng = random.Random(seed + 1)
    pool = [row for row in rows if _eligible(row, UI_SAMPLE_WORDS)]
    rng.shuffle(pool)
    samples: list[UiSampleItem] = []
    used: set[str] = set()
    for topic, sentiment in UI_SAMPLE_TARGETS:
        candidates = [
            r
            for r in pool
            if (r.topic, r.sentiment) == (topic, sentiment) and r.sentence not in used
        ]
        # Ưu tiên câu gọn; thiếu thì lấy câu bất kỳ đúng chủ đề × cảm xúc.
        clean = [r for r in candidates if _CLEAN_SENTENCE.match(beautify(r.sentence))]
        chosen = (clean or candidates or [None])[0]
        if chosen is None:
            raise DemoBuildError(f"không có câu ngắn cho chủ đề {topic} × cảm xúc {sentiment}")
        used.add(chosen.sentence)
        samples.append(UiSampleItem(text=beautify(chosen.sentence), no_diacritics=False))
    samples.append(UiSampleItem(text=strip_diacritics(samples[0].text), no_diacritics=True))
    return samples


# --- Nguồn dữ liệu ------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_source(path: Path, split: str) -> list[SourceRow]:
    """Đọc một tệp parquet của UIT-VSFC, kiểm cột, số dòng và miền nhãn."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if tuple(table.column_names) != ("sentence", "sentiment", "topic"):
        raise DemoBuildError(f"{path.name}: cột {table.column_names} không như mong đợi")
    if table.num_rows != EXPECTED_ROWS[split]:
        raise DemoBuildError(f"{split}: {table.num_rows} dòng, cần {EXPECTED_ROWS[split]}")
    rows = []
    for sentence, sentiment, topic in zip(
        table.column("sentence").to_pylist(),
        table.column("sentiment").to_pylist(),
        table.column("topic").to_pylist(),
        strict=True,
    ):
        if not (isinstance(sentence, str) and 0 <= sentiment < len(SENTIMENT)):
            raise DemoBuildError(f"{split}: dòng không hợp lệ")
        if not 0 <= topic < len(TOPIC):
            raise DemoBuildError(f"{split}: nhãn chủ đề ngoài miền")
        rows.append(SourceRow(sentence, SENTIMENT[sentiment], TOPIC[topic]))
    return rows


def fetch_from_hub(revision: str) -> dict[str, Path]:
    """Tải hai tệp parquet ở đúng commit đã ghim. Mọi lỗi tải thành `SourceUnavailable`."""
    try:
        from huggingface_hub import hf_hub_download

        paths = {
            split: Path(
                hf_hub_download(DATASET_REPO, remote, repo_type="dataset", revision=revision)
            )
            for split, remote in REMOTE_PATHS.items()
        }
    except Exception as exc:  # mạng, DNS, 404, hết dung lượng... đều không được làm hỏng build
        raise SourceUnavailable(
            f"Không tải được UIT-VSFC từ Hugging Face ({type(exc).__name__})."
        ) from exc
    _check_hashes(paths)
    return paths


def local_sources(source_dir: Path) -> dict[str, Path]:
    paths = {split: source_dir / f"{split}.parquet" for split in REMOTE_PATHS}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise SourceUnavailable(f"Thiếu {', '.join(missing)} trong {source_dir}.")
    _check_hashes(paths)
    return paths


def find_sources(
    source_dir: Path | None, settings: Settings, revision: str = DATASET_REVISION
) -> dict[str, Path]:
    """Nguồn dữ liệu theo thứ tự: thư mục chỉ định, bản `make data` đã tải (nếu khớp sha256), HF.

    Thư mục chỉ định mà sai là lỗi của người chạy (`DemoBuildError`); không tìm được nguồn nào
    khác thì `SourceUnavailable` để build tiếp tục không có demo.
    """
    if source_dir is not None:
        try:
            return local_sources(source_dir)
        except SourceUnavailable as exc:
            raise DemoBuildError(str(exc)) from exc
    if revision == DATASET_REVISION:
        try:
            paths = local_sources(settings.data_dir / "raw")
        except SourceUnavailable:
            pass
        else:
            LOG.info("dùng dữ liệu đã tải ở %s", settings.data_dir / "raw")
            return paths
    return fetch_from_hub(revision)


def _check_hashes(paths: dict[str, Path]) -> None:
    for split, path in paths.items():
        if _sha256(path) != EXPECTED_SHA256[split]:
            raise SourceUnavailable(f"{path.name} không khớp sha256 đã ghim ({split}).")


# --- Dựng ---------------------------------------------------------------------------------------


def predict_test(analyzer: Analyzer, rows: Sequence[SourceRow]) -> list[DemoRowData]:
    """Chạy `Analyzer` trên tập test (mỗi câu một lượt, đúng như `/api/analyze`)."""
    prepared = analyzer.prepare_many([row.sentence for row in rows])
    kept = [p for p in prepared if p.n_words > 0]
    if len(kept) != len(prepared):
        LOG.warning("bỏ %d câu test không còn nội dung sau tiền xử lý", len(prepared) - len(kept))
    analyses = analyzer.infer(kept)
    return [
        DemoRowData(
            display_text=beautify(a.prepared.display_text),
            model_text=a.prepared.model_text,
            sentiment=a.sentiment.label,
            topic=a.topic.label,
            sentiment_confidence=round(a.sentiment.confidence, 4),
            topic_confidence=round(a.topic.confidence, 4),
            n_words=a.prepared.n_words,
        )
        for a in analyses
    ]


def _write_atomic(path: Path, data: bytes) -> None:
    """Ghi qua tệp tạm cùng thư mục rồi đổi tên, để không bao giờ để lại tệp ghi dở."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _json_bytes(payload: BaseModel) -> bytes:
    text = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False, indent=1)
    return (text + "\n").encode("utf-8")


def write_manifest(out_dir: Path, manifest: DemoManifest) -> None:
    _write_atomic(out_dir / DEMO_MANIFEST, _json_bytes(manifest))


def build_demo(
    out_dir: Path,
    validation: Sequence[SourceRow],
    test: Sequence[SourceRow],
    analyzer: Analyzer,
    *,
    revision: str,
    seed: int = SEED,
    sample_size: int = SAMPLE_SIZE,
) -> DemoManifest:
    """Dựng cả bốn tệp. Manifest ghi cuối cùng nên thư mục chỉ "có demo" khi mọi tệp đã xong."""
    sample = pick_sample(validation, sample_size, seed)
    ui_samples = pick_ui_samples(validation, seed)
    LOG.info("dự đoán %d câu test", len(test))
    rows = predict_test(analyzer, test)

    bundle_version = analyzer.bundle.manifest.bundle_version
    results = DemoResults(bundle_version=bundle_version, rows=[r.model_dump() for r in rows])
    _write_atomic(out_dir / SAMPLE_CSV, sample_csv_bytes(sample, seed))
    _write_atomic(out_dir / UI_SAMPLES, _json_bytes(UiSamplesFile(samples=ui_samples)))
    _write_atomic(out_dir / DEMO_RESULTS, _json_bytes(results))
    manifest = DemoManifest(
        ok=True,
        dataset_revision=revision,
        bundle_version=bundle_version,
        arch=platform.machine(),
        n_rows=len(rows),
    )
    write_manifest(out_dir, manifest)
    return manifest


def _short(reason: str) -> str:
    return reason if len(reason) <= _MAX_REASON_CHARS else reason[: _MAX_REASON_CHARS - 1] + "…"


def parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="thư mục ghi (mặc định EDUPULSE_DEMO_DIR)")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="thư mục có validation.parquet và test.parquet; bỏ trống thì dùng data/raw nếu có, "
        "không thì tải từ Hugging Face",
    )
    parser.add_argument(
        "--revision", default=DATASET_REVISION, help="commit của bộ dữ liệu trên HF"
    )
    parser.add_argument("--bundle-dir", type=Path, help="bundle mô hình (mặc định theo cấu hình)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument(
        "--skip",
        metavar="LÝ_DO",
        nargs="?",
        const=SKIPPED_REASON,
        help="không dựng, chỉ ghi manifest `ok: false` kèm lý do (build với EDUPULSE_BUILD_DEMO=0)",
    )
    return parser.parse_args(None if argv is None else list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings: Settings = load_settings()
    out_dir: Path = args.out or settings.demo_dir

    if args.skip is not None:
        write_manifest(out_dir, DemoManifest(ok=False, reason=_short(args.skip)))
        LOG.info("%s", args.skip)
        return 0
    try:
        sources = find_sources(args.source_dir, settings, args.revision)
    except SourceUnavailable as exc:
        write_manifest(out_dir, DemoManifest(ok=False, reason=_short(str(exc))))
        LOG.warning("%s Demo bị tắt; build vẫn tiếp tục.", exc)
        return 0

    from edupulse.engine.service import Analyzer

    if args.bundle_dir:
        settings = replace(settings, bundle_dir=args.bundle_dir.resolve())
    manifest = build_demo(
        out_dir,
        read_source(sources["validation"], "validation"),
        read_source(sources["test"], "test"),
        Analyzer.load(settings),
        revision=args.revision,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    LOG.info("xong: %d câu demo, bundle %s, %s", manifest.n_rows, manifest.bundle_version, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
