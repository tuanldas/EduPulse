"""Tải UIT-VSFC từ nhánh parquet đã ghim commit vào `$EDUPULSE_HOME/data/raw`.

Các bước: liệt kê cây tệp ở đúng commit và kiểm đúng 3 tệp, tải, đối chiếu sha256 với bản LFS trên
Hub, kiểm số dòng và cột, rồi ghi `MANIFEST.json`. Chạy lại là không tải thêm khi tệp đã khớp.
Dữ liệu là bộ học thuật phi thương mại: chỉ nằm ở máy, không commit.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path

from ml.common import SPLITS, raw_dir, read_json, sha256_file, write_json

LOG = logging.getLogger("download_data")

DATASET_REPO = "uitnlp/vietnamese_students_feedback"
# Commit của nhánh `refs/convert/parquet`. Ghim SHA vì Hub có thể tạo lại nhánh convert.
DATASET_REVISION = "2d76906157232f87e9883fa37ebfc91ffdee1f83"
# Tên tệp THẬT trên Hub là `0000.parquet` (`0.parquet` là 404).
REMOTE_PATHS = {split: f"default/{split}/0000.parquet" for split in SPLITS}
EXPECTED_ROWS = {"train": 11426, "validation": 1583, "test": 3166}
EXPECTED_COLUMNS = ("sentence", "sentiment", "topic")
NUM_CLASSES = {"sentiment": 3, "topic": 4}


def list_remote_files(revision: str = DATASET_REVISION) -> dict[str, dict]:
    """Cây tệp ở commit đã ghim; báo lỗi nếu không đúng 3 tệp parquet mong đợi."""
    from huggingface_hub import HfApi

    entries = HfApi().list_repo_tree(
        DATASET_REPO, path_in_repo="default", repo_type="dataset", revision=revision, recursive=True
    )
    found = {item.path: item for item in entries if getattr(item, "size", None) is not None}
    expected = set(REMOTE_PATHS.values())
    if set(found) != expected:
        raise RuntimeError(
            f"cây tệp ở {revision[:8]} không như mong đợi: có {sorted(found)}, "
            f"cần {sorted(expected)}"
        )
    files: dict[str, dict] = {}
    for split, path in REMOTE_PATHS.items():
        item = found[path]
        lfs = getattr(item, "lfs", None)
        if lfs is None or not getattr(lfs, "sha256", None):
            raise RuntimeError(f"{path}: Hub không trả sha256 LFS để đối chiếu")
        files[split] = {"remote_path": path, "bytes": int(item.size), "sha256": lfs.sha256}
    return files


def validate_parquet(path: Path, split: str) -> int:
    """Kiểm cột, số dòng, kiểu và miền nhãn; trả về số dòng."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = tuple(table.column_names)
    if columns != EXPECTED_COLUMNS:
        raise RuntimeError(f"{path.name}: cột {columns}, cần {EXPECTED_COLUMNS}")
    if table.num_rows != EXPECTED_ROWS[split]:
        raise RuntimeError(f"{split}: {table.num_rows} dòng, cần {EXPECTED_ROWS[split]}")
    for name in EXPECTED_COLUMNS:
        if table.column(name).null_count:
            raise RuntimeError(f"{split}.{name}: có giá trị rỗng")
    for task, n_classes in NUM_CLASSES.items():
        values = set(table.column(task).to_pylist())
        if not values <= set(range(n_classes)):
            raise RuntimeError(f"{split}.{task}: nhãn ngoài 0..{n_classes - 1}: {sorted(values)}")
    return table.num_rows


def _fetch(split: str, remote: dict, target: Path) -> None:
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        DATASET_REPO, remote["remote_path"], repo_type="dataset", revision=DATASET_REVISION
    )
    # Sao qua tệp tạm cùng thư mục rồi đổi tên, để lần chạy lỗi giữa chừng không để lại tệp dở.
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with open(fd, "wb") as out, open(cached, "rb") as src:
            shutil.copyfileobj(src, out)
        if sha256_file(tmp) != remote["sha256"]:
            raise RuntimeError(f"{split}: sha256 tải về không khớp bản LFS trên Hub")
        tmp.replace(target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def download(force: bool = False) -> dict:
    out = raw_dir()
    out.mkdir(parents=True, exist_ok=True)
    remote_files = list_remote_files()
    manifest_files: dict[str, dict] = {}
    for split in SPLITS:
        remote = remote_files[split]
        target = out / f"{split}.parquet"
        if force or not target.is_file() or sha256_file(target) != remote["sha256"]:
            LOG.info("tải %s (%d byte)", remote["remote_path"], remote["bytes"])
            _fetch(split, remote, target)
        else:
            LOG.info("giữ %s (đã khớp sha256)", target.name)
        rows = validate_parquet(target, split)
        manifest_files[split] = {
            "file": target.name,
            "remote_path": remote["remote_path"],
            "bytes": target.stat().st_size,
            "sha256": remote["sha256"],
            "rows": rows,
        }
    manifest = {
        "dataset": DATASET_REPO,
        "revision": DATASET_REVISION,
        "columns": list(EXPECTED_COLUMNS),
        "files": manifest_files,
    }
    write_json(out / "MANIFEST.json", manifest)
    return manifest


def load_raw_manifest() -> dict:
    """MANIFEST dữ liệu thô; kiểm sha256 từng tệp còn khớp trước khi dùng."""
    path = raw_dir() / "MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(f"chưa có {path} — chạy `make data`")
    manifest = read_json(path)
    if manifest.get("revision") != DATASET_REVISION:
        raise RuntimeError(
            f"MANIFEST thô ghim {manifest.get('revision')}, mã đang ghim {DATASET_REVISION}"
        )
    for info in manifest["files"].values():
        if sha256_file(raw_dir() / info["file"]) != info["sha256"]:
            raise RuntimeError(f"{info['file']} đã thay đổi so với MANIFEST — chạy `make data`")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="tải lại dù tệp đã khớp")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = download(force=args.force)
    for split, info in manifest["files"].items():
        print(
            f"{split:<10} {info['rows']:>6} dòng  {info['bytes']:>8} byte  "
            f"sha256={info['sha256'][:12]}…"
        )
    print(f"revision {manifest['revision'][:12]}… -> {raw_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
