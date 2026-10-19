"""Chứng minh hai bản chuẩn hoá cho model_text giống hệt trên mọi câu UIT-VSFC.

    python -m ml.normalizer_invariance --old-processed DIR [--from 1]

Dùng khi đổi luật chuẩn hoá (vd. ẩn thêm tên người) mà muốn giữ nguyên mô hình đã huấn luyện, số
liệu đã chấm test và freeze. Quy trình: chép `data/processed` hiện có (bản cũ) sang DIR; tăng
`NORMALIZER_VERSION`; chạy `python -m ml.preprocess` dựng lại `data/processed`; chạy lệnh này.

Lệnh so từng cột của từng split giữa bản cũ và bản mới (cùng revision dữ liệu, mỗi tệp đúng sha256
trong MANIFEST của nó). Chỉ khi KHÔNG câu nào khác mới ghi tệp bằng chứng
`results/metrics/normalizer_v<N>_invariance.json`, kèm revision dữ liệu và sha256 của dữ liệu cũ,
mới; `ml.common.normalizer_compatible` đọc tệp đó để cho phép dùng run, cổng int8 và baseline dựng
bằng bản cũ. Dữ liệu đổi (revision hay sha256) thì bằng chứng tự hết hiệu lực.

Mã thoát: 0 đã ghi, 1 có câu khác hoặc dữ liệu không hợp lệ.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from edupulse.config import REPO_ROOT
from edupulse.text import NORMALIZER_VERSION
from edupulse.text.pii import mask
from ml.common import SPLITS, processed_dir, read_json, sha256_file, write_json
from ml.download_data import DATASET_REPO

LOG_PREFIX = "normalizer_invariance"
SCHEMA_VERSION = 1


class InvarianceError(RuntimeError):
    """Hai bản dữ liệu đã xử lý không thể so sánh, hoặc không giống hệt nhau."""


@dataclass(frozen=True)
class Processed:
    manifest: dict[str, Any]
    manifest_sha256: str
    frames: dict[str, pd.DataFrame]


def read_processed(directory: Path) -> Processed:
    """Dữ liệu đã xử lý trong `directory`, sau khi kiểm sha256 từng tệp theo MANIFEST của nó."""
    manifest_path = directory / "MANIFEST.json"
    if not manifest_path.is_file():
        raise InvarianceError(f"thiếu {manifest_path}")
    manifest = read_json(manifest_path)
    frames = {}
    for split in SPLITS:
        info = manifest["files"][split]
        path = directory / info["file"]
        if not path.is_file() or sha256_file(path) != info["sha256"]:
            raise InvarianceError(f"{path} thiếu hoặc không khớp sha256 trong MANIFEST của nó")
        frames[split] = pd.read_parquet(path)
    return Processed(manifest, sha256_file(manifest_path), frames)


def has_capital(sentence: str) -> bool:
    return any(char.isupper() for char in sentence)


def compare(
    old: Processed, new: Processed, *, old_version: str, new_version: str
) -> dict[str, Any]:
    """Bằng chứng (dict) rằng hai bản cho dữ liệu giống hệt; ném InvarianceError nếu không."""
    for name, data, version in (("cũ", old, old_version), ("mới", new, new_version)):
        found = data.manifest.get("normalizer_version")
        if found != version:
            raise InvarianceError(
                f"dữ liệu {name} dựng bằng bản chuẩn hoá {found!r}, cần {version!r}"
            )
    for key in ("dataset_revision", "raw_sha256"):
        if old.manifest[key] != new.manifest[key]:
            raise InvarianceError(f"{key} của hai bản dữ liệu khác nhau: không so sánh được")

    totals = {"sentences": 0, "different_model_text": 0, "different_columns": 0}
    capitals = pii_hits = 0
    for split in SPLITS:
        before, after = old.frames[split], new.frames[split]
        if list(before.columns) != list(after.columns) or len(before) != len(after):
            raise InvarianceError(f"{split}: cột hoặc số dòng khác nhau giữa hai bản")
        totals["sentences"] += len(after)
        totals["different_model_text"] += int((before["text_rdr"] != after["text_rdr"]).sum())
        totals["different_columns"] += int((before != after).any(axis=1).sum())
        sentences = after["sentence"].astype(str).tolist()
        capitals += sum(has_capital(sentence) for sentence in sentences)
        pii_hits += sum(bool(mask(sentence).types) for sentence in sentences)
    if totals["different_model_text"] or totals["different_columns"]:
        raise InvarianceError(
            f"{totals['different_model_text']} câu có model_text khác, "
            f"{totals['different_columns']} dòng khác ở ít nhất một cột "
            f"(trên {totals['sentences']} câu): bản mới KHÔNG bất biến với dữ liệu"
        )

    def side(data: Processed) -> dict[str, Any]:
        return {
            "normalizer_version": data.manifest["normalizer_version"],
            "manifest_sha256": data.manifest_sha256,
            "files": {split: info["sha256"] for split, info in data.manifest["files"].items()},
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "from": old_version,
        "to": new_version,
        "identical": True,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dataset": {
            "name": DATASET_REPO,
            "revision": new.manifest["dataset_revision"],
            "rows": {split: info["rows"] for split, info in new.manifest["files"].items()},
        },
        "processed": {"from": side(old), "to": side(new)},
        "checks": {
            "sentences": totals["sentences"],
            "columns_compared": list(new.frames[SPLITS[0]].columns),
            "parquet_files_byte_identical": side(old)["files"] == side(new)["files"],
            "rows_with_different_model_text": totals["different_model_text"],
            "rows_with_different_columns": totals["different_columns"],
            "sentences_with_capital_letters": capitals,
            "sentences_with_pii_under_new_rules": pii_hits,
        },
        "why": (
            "Dữ liệu UIT-VSFC toàn chữ thường và không có thông tin cá nhân: luật tên cần chữ hoa "
            "nên không bao giờ chạy, các mẫu mới của bản mới không khớp câu nào. Mọi cột của dữ "
            "liệu đã xử lý, kể cả model_text, giống hệt giữa hai bản, nên mô hình đã huấn luyện, "
            "cổng int8, baseline và số liệu chấm test của bản cũ áp dụng nguyên vẹn cho bản mới."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--old-processed", type=Path, required=True, help="bản sao data/processed của bản cũ"
    )
    parser.add_argument("--from", dest="old_version", default="1", help="bản chuẩn hoá cũ")
    parser.add_argument("--out", type=Path, help="tệp bằng chứng (mặc định results/metrics/...)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    name = f"normalizer_v{NORMALIZER_VERSION}_invariance.json"
    out = args.out or REPO_ROOT / "results" / "metrics" / name
    try:
        evidence = compare(
            read_processed(args.old_processed),
            read_processed(processed_dir()),
            old_version=args.old_version,
            new_version=NORMALIZER_VERSION,
        )
    except (InvarianceError, KeyError, OSError) as error:
        print(f"{LOG_PREFIX}: LỖI: {error}", file=sys.stderr)
        return 1
    write_json(out, evidence)
    checks = evidence["checks"]
    print(
        f"{LOG_PREFIX}: bản {args.old_version} và {NORMALIZER_VERSION} giống hệt trên "
        f"{checks['sentences']} câu (chữ hoa {checks['sentences_with_capital_letters']}, "
        f"PII theo luật mới {checks['sentences_with_pii_under_new_rules']}) -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
