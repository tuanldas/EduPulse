"""Phân tích khám phá UIT-VSFC cho báo cáo: hình, bảng đếm và tóm tắt, ghi vào `results/`.

    python -m ml.eda        # make eda

Đầu vào là dữ liệu đã tiền xử lý. Chỉ thống kê mô tả (phân bố nhãn, độ dài, token đặc biệt, trùng
lặp); không tính chỉ số mô hình nào trên test. Mọi bảng chỉ chứa số đếm hoặc từ/cụm từ ngắn
(không có câu gốc), nhãn và chú thích bằng tiếng Việt có dấu.
"""

from __future__ import annotations

import argparse
import collections
import logging
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from edupulse.config import REPO_ROOT  # noqa: E402
from edupulse.labels import LABELS, LABELS_VI, SENTIMENT, TASKS, TOPIC  # noqa: E402
from edupulse.text import NORMALIZER_VERSION, count_ngrams  # noqa: E402
from edupulse.text.normalize import TONE_MAP  # noqa: E402
from ml.common import SPLITS, load_processed, processed_manifest, write_json  # noqa: E402
from ml.phobert_model import BASE_REVISION, load_tokenizer  # noqa: E402

LOG = logging.getLogger("eda")
RESULTS = REPO_ROOT / "results"
SPLIT_VI = {"train": "Train", "validation": "Validation", "test": "Test"}

# Màu ngữ nghĩa của docs/design-guidelines.md mục 2.2 (fill biểu đồ, dùng chung hai theme).
COLORS = {
    "negative": "#C94827",
    "neutral": "#8B909A",
    "positive": "#1B9A6E",
    "lecturer": "#0167BB",
    "training_program": "#944A76",
    "facility": "#B88818",
    "others": "#66837A",
}
INK, MUTED, GRID, BACKGROUND = "#14171E", "#5B6170", "#DCDAD2", "#FAFAF8"

# Số ghi trong thẻ dataset (đồng thuận giữa người gán nhãn); không tự đo được.
INTER_ANNOTATOR_AGREEMENT = {"sentiment": 0.91, "topic": 0.71}
INSIGHT_ALPHA0 = 1000.0
LOGODDS_TOP = 15
LENGTH_UNITS = {
    "syllables": "Âm tiết (tách theo dấu cách)",
    "words": "Từ sau RDRSegmenter",
    "tokens": "Token PhoBERT (gồm <s> </s>)",
}
# Số trong nghiên cứu (r1, r3) để đối chiếu; sai khác vượt dung sai thì phải điều tra.
EXPECTED = {
    "train_sentiment_counts": [5325, 458, 5643],
    "train_topic_counts": [8166, 2201, 497, 562],
    "old_style_tone_sentences": 284,
    "phobert_tokens_p95": 30,
    "negative_rate_by_topic": {
        "lecturer": 0.356,
        "training_program": 0.780,
        "facility": 0.954,
        "others": 0.402,
    },
}


def vn(value: float, digits: int = 1) -> str:
    """Số kiểu Việt: dấu phẩy thập phân, dấu chấm hàng nghìn."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def setup_style() -> None:
    families = {f.name for f in font_manager.fontManager.ttflist}
    preferred = "Be Vietnam Pro" if "Be Vietnam Pro" in families else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": preferred,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> Path:
    path = RESULTS / "figures" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_table(frame: pd.DataFrame, name: str) -> Path:
    path = RESULTS / "tables" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


def load_all() -> pd.DataFrame:
    # allow_test: chỉ để thống kê mô tả (nhãn, độ dài, trùng lặp), không tính chỉ số mô hình.
    frames = [load_processed(split, allow_test=True).assign(split=split) for split in SPLITS]
    return pd.concat(frames, ignore_index=True)


# --- Phân bố nhãn -------------------------------------------------------------------------------


def label_distribution(data: pd.DataFrame) -> dict[str, Any]:
    """Nhãn theo tập: thanh nhóm (train/validation/test) cho từng lớp, ghi % ở đầu thanh."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"wspace": 0.32})
    alphas = {"train": 1.0, "validation": 0.62, "test": 0.38}
    bar = 0.27
    summary: dict[str, Any] = {}
    for ax, task in zip(axes, TASKS, strict=True):
        names = LABELS[task]
        summary[task] = {}
        for split in SPLITS:
            counts = data.loc[data["split"] == split, task].value_counts()
            counts = counts.reindex(range(len(names)), fill_value=0)
            shares = counts / counts.sum()
            summary[task][split] = {
                "counts": counts.tolist(),
                "shares": [round(float(s), 4) for s in shares],
            }
        for index, name in enumerate(names):
            for offset, split in enumerate(SPLITS):
                share = summary[task][split]["shares"][index]
                y = index + (offset - 1) * bar
                ax.barh(y, share, height=bar * 0.92, color=COLORS[name], alpha=alphas[split])
                ax.text(share + 0.01, y, vn(share * 100, 1) + "%", va="center", fontsize=8.5)
        ax.set_yticks(range(len(names)), [LABELS_VI[task][n] for n in names])
        ax.invert_yaxis()
        ax.set_xlim(0, 0.9)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        ax.set_title("Cảm xúc" if task == "sentiment" else "Chủ đề", loc="left", fontweight="bold")
    sizes = {s: int((data["split"] == s).sum()) for s in SPLITS}
    handles = [plt.Rectangle((0, 0), 1, 1, color=MUTED, alpha=alphas[s]) for s in SPLITS]
    labels = [f"{SPLIT_VI[s]} (n = {vn(sizes[s], 0)})" for s in SPLITS]
    fig.legend(
        handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle(
        "Phân bố nhãn theo tập (tỉ lệ % trong từng tập)",
        x=0.01,
        y=1.03,
        ha="left",
        fontweight="bold",
    )
    save_figure(fig, "eda_label_dist.png")
    return summary


def topic_sentiment(data: pd.DataFrame) -> dict[str, Any]:
    """Cảm xúc theo chủ đề; mức nền tiêu cực và tỉ trọng chủ đề lấy từ tập train."""
    rows = []
    for split in SPLITS:
        part = data[data["split"] == split]
        for t, topic in enumerate(TOPIC):
            for s, sentiment in enumerate(SENTIMENT):
                rows.append(
                    {
                        "split": split,
                        "chu_de": LABELS_VI["topic"][topic],
                        "cam_xuc": LABELS_VI["sentiment"][sentiment],
                        "so_cau": int(((part["topic"] == t) & (part["sentiment"] == s)).sum()),
                    }
                )
    long = pd.DataFrame(rows)
    train = data[data["split"] == "train"]
    base = []
    for t, topic in enumerate(TOPIC):
        cell = train[train["topic"] == t]
        base.append(
            {
                "chu_de": LABELS_VI["topic"][topic],
                "so_cau_train": int(len(cell)),
                "ti_trong_chu_de": round(len(cell) / len(train), 4),
                "muc_nen_tieu_cuc": round(float((cell["sentiment"] == 0).mean()), 4),
                "muc_nen_trung_tinh": round(float((cell["sentiment"] == 1).mean()), 4),
                "muc_nen_tich_cuc": round(float((cell["sentiment"] == 2).mean()), 4),
            }
        )
    base_frame = pd.DataFrame(base)
    save_table(long, "eda_topic_sentiment.csv")
    save_table(base_frame, "eda_topic_base_rates.csv")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for row, t in enumerate(reversed(range(len(TOPIC)))):
        cell = train[train["topic"] == t]
        shares = cell["sentiment"].value_counts(normalize=True).reindex(range(3), fill_value=0.0)
        left = 0.0
        for s, sentiment in enumerate(SENTIMENT):
            share = float(shares.iloc[s])
            ax.barh(row, share, left=left, color=COLORS[sentiment], height=0.62)
            if share >= 0.06:
                ax.text(
                    left + share / 2,
                    row,
                    vn(share * 100, 1) + "%",
                    ha="center",
                    va="center",
                    color="white" if sentiment == "negative" else INK,
                    fontsize=9,
                )
            left += share
        ax.text(1.01, row, f"n = {vn(len(cell), 0)}", va="center", color=MUTED, fontsize=9)
    ax.set_yticks(
        range(len(TOPIC)), [LABELS_VI["topic"][TOPIC[t]] for t in reversed(range(len(TOPIC)))]
    )
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[n]) for n in SENTIMENT]
    ax.legend(
        handles,
        [LABELS_VI["sentiment"][n] for n in SENTIMENT],
        ncol=3,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
    )
    ax.set_title(
        "Cảm xúc trong từng chủ đề (tập train): mức nền tiêu cực khác nhau rất xa",
        loc="left",
        fontweight="bold",
    )
    save_figure(fig, "eda_topic_sentiment.png")
    return {
        row["chu_de"]: {k: v for k, v in row.items() if k != "chu_de"}
        for row in base_frame.to_dict("records")
    }


# --- Độ dài -------------------------------------------------------------------------------------


def length_columns(data: pd.DataFrame) -> pd.DataFrame:
    tokenizer = load_tokenizer(max_len=100_000)  # không cắt: cần độ dài thật
    ids = tokenizer.encode_batch(data["text_rdr"].tolist())
    return data.assign(
        syllables=data["sentence"].str.split().str.len(),
        words=data["n_words"],
        tokens=[len(e.ids) for e in ids],
    )


def length_report(data: pd.DataFrame) -> dict[str, Any]:
    data = length_columns(data)
    rows = []
    for unit in LENGTH_UNITS:
        for split in ("all", *SPLITS):
            values = data[unit] if split == "all" else data.loc[data["split"] == split, unit]
            rows.append(
                {
                    "don_vi": LENGTH_UNITS[unit],
                    "tap": split,
                    "trung_binh": round(float(values.mean()), 2),
                    "p50": float(values.quantile(0.50)),
                    "p90": float(values.quantile(0.90)),
                    "p95": float(values.quantile(0.95)),
                    "p99": float(values.quantile(0.99)),
                    "p99_9": float(values.quantile(0.999)),
                    "toi_da": int(values.max()),
                    "ti_le_tren_64": round(float((values > 64).mean()), 5),
                    "ti_le_tren_128": round(float((values > 128).mean()), 5),
                }
            )
    table = pd.DataFrame(rows)
    save_table(table, "eda_length_percentiles.csv")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"wspace": 0.22})
    for ax, unit in zip(axes, LENGTH_UNITS, strict=True):
        values = data[unit]
        top = float(values.quantile(0.995))
        shown = values[values <= top]
        ax.hist(shown, bins=np.arange(0, top + 2) - 0.5, color="#3A4BD0", alpha=0.85)
        ceiling = ax.get_ylim()[1]
        for q, style, height in ((0.95, "--", 0.93), (0.99, ":", 0.80)):
            point = float(values.quantile(q))
            ax.axvline(point, color=INK, linestyle=style, linewidth=1)
            ax.text(
                point - 0.6,
                ceiling * height,
                f"p{int(q * 100)} = {vn(point, 0)}",
                ha="right",
                fontsize=8.5,
            )
        ax.set_title(LENGTH_UNITS[unit], loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel(
            f"Độ dài (câu dài hơn {vn(top, 0)}: {vn((values > top).mean() * 100, 1)}%, ẩn)"
        )
        ax.set_ylabel("Số câu" if unit == "syllables" else "")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    fig.suptitle("Phân bố độ dài câu (cả ba tập)", x=0.01, y=1.04, ha="left", fontweight="bold")
    save_figure(fig, "eda_length.png")
    return {
        unit: {
            k: float(data[unit].quantile(q))
            for k, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
        }
        | {"mean": round(float(data[unit].mean()), 2), "max": int(data[unit].max())}
        for unit in LENGTH_UNITS
    }


# --- Token đặc biệt, dấu thanh, trùng lặp ----------------------------------------------------

_EMOTICON = re.compile(r"\bcolon\w+\b")
_SPECIAL = {
    "Tên người (wzjwz<N>)": re.compile(r"\bwzjwz\d+\b"),
    "Emoticon (colon...)": _EMOTICON,
    "Dấu hai chấm (doubledot)": re.compile(r"\bdoubledot\b|\d+doubledot\d+"),
    "Viết tắt v.v (vdotv)": re.compile(r"\bvdotv\b"),
    "Dấu ba chấm (dotdotdot)": re.compile(r"\bdotdotdot\b"),
    "Số/tên miền chứa dấu chấm (…dot…)": re.compile(
        r"\b(?!\w*doubledot)(?!(?:vdotv|dotdotdot)\b)(?:\w*\wdot\w+|dot(?:net|com|vn))\b"
    ),
}


def special_tokens(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split in (*SPLITS, "all"):
        part = data if split == "all" else data[data["split"] == split]
        texts = part["sentence"]
        for name, pattern in _SPECIAL.items():
            hits = texts.str.count(pattern)
            rows.append(
                {
                    "tap": split,
                    "loai": name,
                    "so_cau": int((hits > 0).sum()),
                    "so_lan": int(hits.sum()),
                    "ti_le_cau": round(float((hits > 0).mean()), 5),
                }
            )
    emoticons = collections.Counter(
        token for sentence in data["sentence"] for token in _EMOTICON.findall(sentence)
    )
    for token, count in emoticons.most_common():
        rows.append(
            {"tap": "all", "loai": f"  {token}", "so_cau": count, "so_lan": count, "ti_le_cau": 0}
        )
    table = pd.DataFrame(rows)
    save_table(table, "eda_special_tokens.csv")
    return table


def old_style_tone(data: pd.DataFrame) -> dict[str, Any]:
    """Dấu thanh kiểu cũ (hòa, khóa, thủy...) mà vocab PhoBERT-v2 không có."""
    sentences = data["sentence"]
    rows = []
    for old, new in TONE_MAP.items():
        counts = {
            split: int(sentences[data["split"] == split].str.count(old).sum()) for split in SPLITS
        }
        rows.append({"kieu_cu": old, "kieu_moi": new, **counts, "tong": sum(counts.values())})
    table = pd.DataFrame(rows).sort_values("tong", ascending=False)
    words = collections.Counter(
        word
        for sentence in sentences
        for word in sentence.split()
        if any(o in word for o in TONE_MAP)
    )
    save_table(table, "eda_tone_old_style.csv")
    affected = sentences.map(lambda s: any(o in s for o in TONE_MAP))
    summary = {
        "sentences_affected": int(affected.sum()),
        "share": round(float(affected.mean()), 5),
        "by_split": {split: int(affected[data["split"] == split].sum()) for split in SPLITS},
        "top_words": dict(words.most_common(8)),
    }
    return summary


def duplicates(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sets = {split: set(data.loc[data["split"] == split, "sentence"]) for split in SPLITS}
    for split in SPLITS:
        part = data[data["split"] == split]
        conflict = part.groupby("sentence")[["sentiment", "topic"]].nunique()
        rows.append(
            {
                "phep_do": f"trùng trong {split}",
                "so_cau": int(part["sentence"].duplicated().sum()),
                "nhan_mau_thuan": int(
                    ((conflict["sentiment"] > 1) | (conflict["topic"] > 1)).sum()
                ),
            }
        )
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        rows.append(
            {"phep_do": f"giao {a} ∩ {b}", "so_cau": len(sets[a] & sets[b]), "nhan_mau_thuan": 0}
        )
    table = pd.DataFrame(rows)
    save_table(table, "eda_duplicates.csv")
    return table


# --- Từ khoá nổi bật theo ô chủ đề × cảm xúc -------------------------------------------------


def log_odds(
    cell: collections.Counter,
    rest: collections.Counter,
    prior: collections.Counter,
    alpha0: float = INSIGHT_ALPHA0,
) -> pd.DataFrame:
    """Log-odds có trọng số với prior Dirichlet thông tin (Monroe, Colaresi, Quinn 2008)."""
    total_prior = sum(prior.values())
    n_cell, n_rest = sum(cell.values()), sum(rest.values())
    rows = []
    for gram, background in prior.items():
        alpha = alpha0 * background / total_prior
        y1, y2 = cell.get(gram, 0), rest.get(gram, 0)
        delta = np.log((y1 + alpha) / (n_cell + alpha0 - y1 - alpha)) - np.log(
            (y2 + alpha) / (n_rest + alpha0 - y2 - alpha)
        )
        z = delta / np.sqrt(1 / (y1 + alpha) + 1 / (y2 + alpha))
        rows.append(
            {"ngram": gram, "so_lan_trong_o": y1, "so_lan_ngoai_o": y2, "z": round(float(z), 3)}
        )
    return pd.DataFrame(rows).sort_values("z", ascending=False)


def top_keywords(data: pd.DataFrame) -> pd.DataFrame:
    train = data[data["split"] == "train"]
    prior = collections.Counter(
        {g: c for g, c in count_ngrams(train["text_rdr"]).items() if c >= 2}
    )
    cells: list[tuple[str, pd.Series]] = []
    for t, topic in enumerate(TOPIC):
        label = f"{LABELS_VI['topic'][topic]} × Tiêu cực"
        cells.append((label, (train["topic"] == t) & (train["sentiment"] == 0)))
    cells.append(("Mọi chủ đề × Tích cực", train["sentiment"] == 2))
    frames = []
    for label, mask in cells:
        cell = count_ngrams(train.loc[mask, "text_rdr"])
        rest = count_ngrams(train.loc[~mask, "text_rdr"])
        top = log_odds(cell, rest, prior).head(LOGODDS_TOP).assign(o=label, n_cau=int(mask.sum()))
        frames.append(top)
    table = pd.concat(frames, ignore_index=True)[
        ["o", "n_cau", "ngram", "so_lan_trong_o", "so_lan_ngoai_o", "z"]
    ]
    table = table.rename(columns={"o": "o_chu_de_cam_xuc"})
    save_table(table, "eda_top_logodds.csv")
    return table


# --- Đối chiếu nghiên cứu ---------------------------------------------------------------------


def check(name: str, observed: Any, expected: Any, tolerance: float = 0.0) -> dict[str, Any]:
    if isinstance(expected, list):
        ok = observed == expected
    else:
        ok = abs(float(observed) - float(expected)) <= tolerance
    return {
        "check": name,
        "observed": observed,
        "expected": expected,
        "tolerance": tolerance,
        "ok": bool(ok),
    }


def verify_against_research(summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = summary["labels"]
    checks = [
        check(
            "train: số câu cảm xúc",
            labels["sentiment"]["train"]["counts"],
            EXPECTED["train_sentiment_counts"],
        ),
        check(
            "train: số câu chủ đề",
            labels["topic"]["train"]["counts"],
            EXPECTED["train_topic_counts"],
        ),
        check(
            "câu có dấu thanh kiểu cũ",
            summary["old_style_tone"]["sentences_affected"],
            EXPECTED["old_style_tone_sentences"],
        ),
        check(
            "p95 token PhoBERT",
            summary["length"]["tokens"]["p95"],
            EXPECTED["phobert_tokens_p95"],
            1,
        ),
    ]
    for topic, expected in EXPECTED["negative_rate_by_topic"].items():
        observed = summary["topic_base_rates"][LABELS_VI["topic"][topic]]["muc_nen_tieu_cuc"]
        checks.append(
            check(f"mức nền tiêu cực: {LABELS_VI['topic'][topic]}", observed, expected, 0.001)
        )
    return checks


def run() -> dict[str, Any]:
    setup_style()
    manifest = processed_manifest()
    data = load_all()
    LOG.info("%d câu từ %d tập", len(data), len(SPLITS))
    summary: dict[str, Any] = {
        "normalizer_version": NORMALIZER_VERSION,
        "dataset_revision": manifest["dataset_revision"],
        "phobert_revision": BASE_REVISION,
        "rows": {split: int((data["split"] == split).sum()) for split in SPLITS},
        "inter_annotator_agreement": INTER_ANNOTATOR_AGREEMENT,
        "labels": label_distribution(data),
        "topic_base_rates": topic_sentiment(data),
        "length": length_report(data),
        "old_style_tone": old_style_tone(data),
        "no_diacritics_sentences": int(data["no_diacritics"].sum()),
    }
    specials = special_tokens(data)
    summary["special_tokens"] = {
        row["loai"]: {"so_cau": row["so_cau"], "so_lan": row["so_lan"]}
        for _, row in specials[
            (specials["tap"] == "all") & (~specials["loai"].str.startswith(" "))
        ].iterrows()
    }
    dup = duplicates(data)
    summary["duplicates"] = {
        row["phep_do"]: {"so_cau": int(row["so_cau"]), "nhan_mau_thuan": int(row["nhan_mau_thuan"])}
        for _, row in dup.iterrows()
    }
    top_keywords(data)
    summary["checks"] = verify_against_research(summary)
    write_json(RESULTS / "metrics" / "data_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    summary = run()
    for item in summary["checks"]:
        print(
            f"{'OK  ' if item['ok'] else 'LỆCH'} {item['check']}: {item['observed']} "
            f"(nghiên cứu: {item['expected']})"
        )
    return 0 if all(item["ok"] for item in summary["checks"]) else 1


if __name__ == "__main__":
    sys.exit(main())
