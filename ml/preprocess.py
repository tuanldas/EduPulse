"""Tiền xử lý UIT-VSFC bằng đúng pipeline chuẩn hoá dùng khi phục vụ, rồi ghi bộ nhớ đệm.

Đầu ra (dưới `$EDUPULSE_HOME`):
- `data/processed/{train,validation,test}.parquet`: sentence, sentiment, topic, text_rdr, n_words,
  no_diacritics;
- `data/processed/MANIFEST.json`: normalizer_version, revision dữ liệu, sha256 từng tệp, thống kê;
- `outputs/data/insight_background.json`: đếm n-gram nền của tập train cho insight.

Tập test chỉ được chuẩn hoá và tách từ ở đây, không dùng để chọn gì cả.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from edupulse.bundle_schema import InsightBackground
from edupulse.labels import SENTIMENT, TOPIC
from edupulse.text import NORMALIZER_VERSION, count_ngrams, prepare_many
from edupulse.text.normalize import normalize_tone
from ml.common import SPLITS, processed_dir, raw_dir, settings, sha256_file, write_json
from ml.download_data import load_raw_manifest

LOG = logging.getLogger("preprocess")

INSIGHT_ALPHA0 = 1000.0  # độ mạnh prior Dirichlet của log-odds có trọng số
INSIGHT_MIN_COUNT = 2  # n-gram chỉ xuất hiện 1 lần bị bỏ: gọn và không giữ cụm hiếm gần nguyên văn
_SPECIAL = re.compile(r"\b(?:colon\w+|doubledot|vdotv|dotdotdot|\w*dot\w*|wzjwz\d+)\b")


def _write_parquet(frame: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def process_split(split: str, raw_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    sentences = raw["sentence"].astype(str).tolist()
    # Dữ liệu huấn luyện không được phụ thuộc EDUPULSE_NER: NER chỉ chạy trên chữ hoa mà dữ liệu
    # gốc toàn chữ thường.
    prepared = prepare_many(sentences, use_ner=False)
    return pd.DataFrame(
        {
            "sentence": sentences,
            "sentiment": raw["sentiment"].astype("int64"),
            "topic": raw["topic"].astype("int64"),
            "text_rdr": [p.model_text for p in prepared],
            "n_words": [p.n_words for p in prepared],
            "no_diacritics": [p.no_diacritics for p in prepared],
            "pii_hits": [len(p.pii_types) for p in prepared],
        }
    )


def split_stats(frame: pd.DataFrame) -> dict[str, Any]:
    """Thống kê đếm (không chứa câu) để đối chiếu với nghiên cứu và phát hiện bất thường."""
    return {
        "rows": int(len(frame)),
        "tone_old_style": int((frame["sentence"].map(normalize_tone) != frame["sentence"]).sum()),
        "with_special_tokens": int(frame["text_rdr"].str.contains(_SPECIAL).sum()),
        "no_diacritics": int(frame["no_diacritics"].sum()),
        "pii_hits": int((frame["pii_hits"] > 0).sum()),
        "empty_after_prepare": int((frame["n_words"] == 0).sum()),
        "n_words_mean": round(float(frame["n_words"].mean()), 3),
        "n_words_p99": float(frame["n_words"].quantile(0.99)),
    }


def insight_background(train: pd.DataFrame) -> InsightBackground:
    """Đếm nền của tập train: n-gram, tỉ lệ cảm xúc, tỉ trọng và mức nền tiêu cực theo chủ đề."""
    counts = count_ngrams(train["text_rdr"])
    n = len(train)
    negative = SENTIMENT.index("negative")
    sentiment_share = {
        name: float((train["sentiment"] == i).mean()) for i, name in enumerate(SENTIMENT)
    }
    topic_share = {name: float((train["topic"] == i).mean()) for i, name in enumerate(TOPIC)}
    negative_rate = {
        name: float((train.loc[train["topic"] == i, "sentiment"] == negative).mean())
        for i, name in enumerate(TOPIC)
    }
    return InsightBackground(
        normalizer_version=NORMALIZER_VERSION,
        alpha0=INSIGHT_ALPHA0,
        min_count=INSIGHT_MIN_COUNT,
        n_sentences=n,
        total_ngrams=int(sum(counts.values())),
        ngram_counts={g: c for g, c in sorted(counts.items()) if c >= INSIGHT_MIN_COUNT},
        sentiment_share=sentiment_share,
        topic_share=topic_share,
        negative_rate_by_topic=negative_rate,
    )


def run() -> dict[str, Any]:
    raw_manifest = load_raw_manifest()
    out_dir = processed_dir()
    files: dict[str, Any] = {}
    stats: dict[str, Any] = {}
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        info = raw_manifest["files"][split]
        frame = process_split(split, raw_dir() / info["file"])
        stats[split] = split_stats(frame)
        LOG.info("%s: %s", split, stats[split])
        if stats[split]["pii_hits"] or stats[split]["empty_after_prepare"]:
            raise RuntimeError(f"{split}: dữ liệu gốc bất ngờ có PII hoặc câu rỗng: {stats[split]}")
        target = out_dir / f"{split}.parquet"
        _write_parquet(frame.drop(columns=["pii_hits"]), target)
        files[split] = {"file": target.name, "sha256": sha256_file(target), "rows": len(frame)}
        frames[split] = frame

    background = insight_background(frames["train"])
    background_path = settings().outputs_dir / "data" / "insight_background.json"
    write_json(background_path, background.model_dump(mode="json"))
    LOG.info(
        "n-gram nền: %d (tổng %d) -> %s",
        len(background.ngram_counts),
        background.total_ngrams,
        background_path,
    )

    manifest = {
        "normalizer_version": NORMALIZER_VERSION,
        "dataset_revision": raw_manifest["revision"],
        "raw_sha256": {s: raw_manifest["files"][s]["sha256"] for s in SPLITS},
        "files": files,
        "stats": stats,
    }
    # Ghi MANIFEST sau cùng: có nó nghĩa là cả ba tệp đã hoàn tất.
    write_json(out_dir / "MANIFEST.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = run()
    print(f"normalizer_version={manifest['normalizer_version']} -> {processed_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
