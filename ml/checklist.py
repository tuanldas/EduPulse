"""Bộ kiểm thử hành vi kiểu CheckList (Ribeiro et al., ACL 2020) trên câu validation.

    python -m ml.checklist                # make report (run 'best')
    python -m ml.checklist --limit 50     # chạy thử trên 50 câu đầu, KHÔNG phải số liệu chính thức

Mỗi biến thể được sinh từ câu GỐC (cột `sentence`), đi qua `prepare_many(..., use_ner=False)` như
lúc phục vụ, mã hoá bằng `ml.export_onnx.encode` rồi chạy ONNX int8 từng câu một (batch 1) với
nhiệt độ T trong `calibration.json` của run. Dự đoán của câu gốc cũng chạy lại cùng đường, nên mọi
phép so sánh đều cùng điều kiện (không dùng logits val lưu sẵn).

Ba loại kiểm thử:
- bất biến (invariance): nhãn cảm xúc VÀ chủ đề đều không đổi sau một biến đổi vô hại;
- có hướng (directional): xác suất phải đổi đúng chiều sau một biến đổi có nghĩa;
- bền vững (robustness): đo mức suy giảm khi đầu vào xấu đi (bỏ dấu).

`n` của mỗi dòng là số câu đủ điều kiện. Câu mà biến đổi không đổi gì (viết hoa câu không có chữ)
hoặc không áp dụng được (câu không có "thầy"/"cô") bị loại khỏi `n`. Dòng có nhiều biến thể thì một
câu đạt khi đạt ở MỌI biến thể áp dụng được cho nó; ghi chú đếm riêng từng biến thể.

Đầu ra (chỉ số đếm và tỉ lệ, không chứa câu gốc): `results/tables/checklist.md` và
`results/metrics/checklist.json` (các dòng theo schema `ChecklistRow`).
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from edupulse import config as edupulse_config
from edupulse.bundle_schema import Calibration, ChecklistRow
from edupulse.labels import LABELS, TASKS
from edupulse.scoring import softmax_t
from edupulse.text import prepare_many
from edupulse.text.pii import NAME_TOKEN_PREFIX
from ml.calibrate import output_path as calibration_output_path
from ml.common import load_processed, read_json, resolve_run, settings, write_json
from ml.plotting import vn
from ml.predictors import deploy_onnx_path

SEED = 0
# Dung sai khi so xác suất của câu gốc với biến thể: chấp nhiễu của int8 động.
DIRECTIONAL_TOLERANCE = 0.01
# Câu gốc chỉ được xét ở phép thêm "không" khi được dự đoán tích cực với xác suất tối thiểu này.
POSITIVE_MIN_PROB = 0.5
MAX_NAME_ID = 327  # chỉ số lớn nhất của tên ẩn `wzjwz<k>` trong UIT-VSFC
NEGATION = "không"
SAD_EMOTICON = ":("
TEACHER_TITLES = ("thầy", "cô")
# Thông tin cá nhân chèn vào giữa câu; pipeline phải nhận ra và xoá trước khi tới mô hình.
PII_SNIPPETS: dict[str, str] = {
    "SĐT": "0912345678",
    "email": "ten.ho@example.com",
    "MSSV": "mssv 18520001",
}

Transform = Callable[[str], str]

_TOKEN = re.compile(r"\S+")
_NAME_TOKEN = re.compile(rf"\b{NAME_TOKEN_PREFIX}(\d+)\b")
_POSITIVE_ID = LABELS["sentiment"].index("positive")
_NO_ELIGIBLE = "không có câu đủ điều kiện"
KIND_VI = {"invariance": "Bất biến", "directional": "Có hướng", "robustness": "Bền vững"}


# --- Dự đoán ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Prediction:
    """Dự đoán cho một danh sách câu: xác suất hai head (sau nhiệt độ) và cờ "không dấu"."""

    probs: Mapping[str, np.ndarray]  # nhiệm vụ -> (n, số lớp)
    no_diacritics: np.ndarray  # (n,) bool: hệ thống gắn cờ câu nghi gõ không dấu

    def __post_init__(self) -> None:
        if set(self.probs) != set(TASKS):
            raise ValueError(
                f"probs phải có đúng các nhiệm vụ {list(TASKS)}, nhận {list(self.probs)}"
            )
        n = len(self.no_diacritics)
        for task in TASKS:
            shape = (n, len(LABELS[task]))
            if self.probs[task].shape != shape:
                raise ValueError(
                    f"probs[{task!r}] có kích thước {self.probs[task].shape}, cần {shape}"
                )

    def __len__(self) -> int:
        return len(self.no_diacritics)

    def labels(self, task: str) -> np.ndarray:
        """Id nhãn dự đoán của từng câu."""
        return self.probs[task].argmax(axis=1)

    def prob(self, task: str, label: str) -> np.ndarray:
        """Xác suất của một nhãn (theo tên) ở từng câu."""
        return self.probs[task][:, LABELS[task].index(label)]


# Nhận danh sách câu GỐC (chưa tiền xử lý), trả dự đoán cùng thứ tự.
Predict = Callable[[Sequence[str]], Prediction]


def _predict(predict: Predict, texts: Sequence[str]) -> Prediction:
    result = predict(texts)
    if len(result) != len(texts):
        raise ValueError(f"bộ dự đoán trả {len(result)} dòng cho {len(texts)} câu")
    return result


def build_predictor(onnx_path: Path, temperatures: Mapping[str, float], threads: int) -> Predict:
    """Bộ dự đoán thật: tiền xử lý như lúc phục vụ, ONNX int8 từng câu một, softmax theo T."""
    from ml.export_onnx import encode, make_session, run_logits  # nạp muộn: kéo theo torch, onnx

    session = make_session(onnx_path, threads)

    def predict(texts: Sequence[str]) -> Prediction:
        prepared = prepare_many(texts, use_ner=False)
        logits = run_logits(session, encode([p.model_text for p in prepared]), batch_size=1)
        return Prediction(
            probs={task: softmax_t(logits[task], temperatures[task]) for task in TASKS},
            no_diacritics=np.array([p.no_diacritics for p in prepared], dtype=bool),
        )

    return predict


# --- Biến đổi chuỗi ---------------------------------------------------------------------------
# Mỗi hàm nhận một câu, trả một chuỗi (câu rỗng hay lạ cũng không ném lỗi). Biến đổi không áp dụng
# được thì trả lại nguyên câu; người gọi coi "không đổi" là không đủ điều kiện.


def to_upper(sentence: str) -> str:
    """Viết HOA toàn câu."""
    return sentence.upper()


def to_title_case(sentence: str) -> str:
    """Viết Hoa Chữ Đầu Mỗi Từ (từ = cụm ký tự không khoảng trắng); giữ nguyên khoảng trắng."""
    return _TOKEN.sub(lambda match: match.group()[0].upper() + match.group()[1:], sentence)


def append_suffix(sentence: str, suffix: str) -> str:
    """Nối `suffix` (đã gồm dấu cách đứng trước, vd. " .", " !!") vào cuối câu."""
    return sentence + suffix


def double_spaces(sentence: str) -> str:
    """Nhân đôi mọi dấu cách."""
    return sentence.replace(" ", "  ")


def pad_spaces(sentence: str) -> str:
    """Thêm một dấu cách ở đầu và ở cuối."""
    return f" {sentence} "


def swap_teacher_title(sentence: str) -> str:
    """Đổi "thầy" <-> "cô" khi câu có đúng một trong hai token (tách bằng khoảng trắng).

    Câu có cả hai token, hoặc không có token nào, được trả lại nguyên vẹn. Mọi lần xuất hiện của
    token đang có đều được đổi.
    """
    tokens = set(sentence.split())
    present = [title for title in TEACHER_TITLES if title in tokens]
    if len(present) != 1:
        return sentence
    source = present[0]
    target = TEACHER_TITLES[1 - TEACHER_TITLES.index(source)]
    return _TOKEN.sub(lambda match: target if match.group() == source else match.group(), sentence)


def _swap_from(source: str, sentence: str) -> str:
    """`swap_teacher_title` chỉ khi token đang có là `source` (để đếm riêng theo chiều đổi)."""
    return swap_teacher_title(sentence) if source in sentence.split() else sentence


def reindex_names(sentence: str, rng: random.Random) -> str:
    """Thay chỉ số của tên ẩn `wzjwz<k>` bằng chỉ số khác trong 1..`MAX_NAME_ID`.

    Cùng một tên (cùng k) thì giữ cùng chỉ số mới, tên khác nhau thì chỉ số mới khác nhau, và không
    chỉ số mới nào trùng chỉ số cũ trong câu, để cấu trúc "thầy A, thầy B" được bảo toàn.
    """
    old_ids = list(dict.fromkeys(int(k) for k in _NAME_TOKEN.findall(sentence)))
    if not old_ids:
        return sentence
    pool = [k for k in range(1, MAX_NAME_ID + 1) if k not in old_ids]
    if len(pool) < len(old_ids):  # chỉ khi câu có gần 327 tên khác nhau; không có trong dữ liệu
        return sentence
    new_ids = dict(zip(old_ids, rng.sample(pool, len(old_ids)), strict=True))
    return _NAME_TOKEN.sub(
        lambda match: f"{NAME_TOKEN_PREFIX}{new_ids[int(match.group(1))]}", sentence
    )


def insert_middle(sentence: str, snippet: str) -> str:
    """Chèn `snippet` vào giữa câu, sau nửa số token (chia nguyên); khoảng trắng được gộp lại."""
    tokens = sentence.split()
    middle = len(tokens) // 2
    return " ".join([*tokens[:middle], snippet, *tokens[middle:]])


def add_negation(sentence: str) -> str:
    """Đặt "không " ngay trước câu."""
    return f"{NEGATION} {sentence}"


def add_sad_emoticon(sentence: str) -> str:
    """Thêm " :(" vào cuối câu."""
    return f"{sentence} {SAD_EMOTICON}"


def strip_diacritics(sentence: str) -> str:
    """Bỏ dấu thanh và dấu mũ/móc/trăng (NFD rồi bỏ dấu kết hợp); đ/Đ thành d/D."""
    folded = unicodedata.normalize("NFD", sentence.replace("đ", "d").replace("Đ", "D"))
    kept = "".join(char for char in folded if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", kept)


PUNCTUATION_VARIANTS: dict[str, Transform] = {
    "thêm ' .'": partial(append_suffix, suffix=" ."),
    "thêm ' !!'": partial(append_suffix, suffix=" !!"),
    "gấp đôi dấu cách": double_spaces,
    "đệm dấu cách đầu/cuối": pad_spaces,
}
SWAP_VARIANTS: dict[str, Transform] = {
    "thầy→cô": partial(_swap_from, "thầy"),
    "cô→thầy": partial(_swap_from, "cô"),
}


# --- Tính từng dòng ---------------------------------------------------------------------------


def _variant_texts(sentences: Sequence[str], transform: Transform) -> tuple[np.ndarray, list[str]]:
    """Chỉ số và văn bản biến thể của các câu mà phép biến đổi thật sự đổi nội dung.

    Câu rỗng hoặc chỉ có khoảng trắng không bao giờ đủ điều kiện.
    """
    index: list[int] = []
    texts: list[str] = []
    for i, sentence in enumerate(sentences):
        variant = transform(sentence)
        if sentence.strip() and variant != sentence:
            index.append(i)
            texts.append(variant)
    return np.asarray(index, dtype=np.intp), texts


def _count(n_failed: int, applicable: int, n: int) -> str:
    """Số hỏng của một biến thể; thêm `/số áp dụng` khi biến thể chỉ áp dụng cho một phần câu."""
    return vn(n_failed, 0) if applicable == n else f"{vn(n_failed, 0)}/{vn(applicable, 0)}"


def invariance_row(
    name: str,
    sentences: Sequence[str],
    base: Prediction,
    predict: Predict,
    variants: Mapping[str, Transform],
) -> ChecklistRow:
    """Bất biến: nhãn của CẢ HAI tác vụ giữ nguyên so với câu gốc ở mọi biến thể áp dụng được."""
    eligible = np.zeros(len(sentences), dtype=bool)
    failed = np.zeros(len(sentences), dtype=bool)
    per_variant: dict[str, tuple[int, int]] = {}
    for label, transform in variants.items():
        index, texts = _variant_texts(sentences, transform)
        per_variant[label] = (0, len(index))
        if not len(index):
            continue
        new = _predict(predict, texts)
        unchanged = np.ones(len(index), dtype=bool)
        for task in TASKS:
            unchanged &= new.labels(task) == base.labels(task)[index]
        eligible[index] = True
        failed[index[~unchanged]] = True
        per_variant[label] = (int((~unchanged).sum()), len(index))
    n, n_failed = int(eligible.sum()), int(failed.sum())
    rule = "Đạt khi nhãn cảm xúc và chủ đề đều không đổi"
    if n == 0:
        return ChecklistRow(
            name=name, kind="invariance", n=0, pass_rate=None, note=f"{rule}; {_NO_ELIGIBLE}."
        )
    note = f"{rule}; hỏng {vn(n_failed, 0)}/{vn(n, 0)} câu"
    if len(per_variant) > 1:
        parts = [f"{label} {_count(f, total, n)}" for label, (f, total) in per_variant.items()]
        note += f" ({', '.join(parts)})"
    return ChecklistRow(
        name=name, kind="invariance", n=n, pass_rate=(n - n_failed) / n, note=f"{note}."
    )


def _directional_row(
    name: str,
    sentences: Sequence[str],
    base: Prediction,
    predict: Predict,
    *,
    transform: Transform,
    eligible: np.ndarray,
    label: str,
    passes: Callable[[np.ndarray, np.ndarray], np.ndarray],
    rule: str,
) -> ChecklistRow:
    """Có hướng: so xác suất cảm xúc của `label` giữa câu gốc và biến thể ở các câu đủ điều kiện."""
    nonblank = np.fromiter((bool(s.strip()) for s in sentences), dtype=bool, count=len(sentences))
    index = np.flatnonzero(eligible & nonblank)
    if not len(index):
        return ChecklistRow(
            name=name, kind="directional", n=0, pass_rate=None, note=f"{rule}; {_NO_ELIGIBLE}."
        )
    new = _predict(predict, [transform(sentences[i]) for i in index])
    ok = passes(base.prob("sentiment", label)[index], new.prob("sentiment", label))
    n, n_failed = len(index), int((~ok).sum())
    return ChecklistRow(
        name=name,
        kind="directional",
        n=n,
        pass_rate=(n - n_failed) / n,
        note=f"{rule}; hỏng {vn(n_failed, 0)}/{vn(n, 0)} câu.",
    )


def negation_row(sentences: Sequence[str], base: Prediction, predict: Predict) -> ChecklistRow:
    """Thêm "không" trước câu: P(tích cực) phải giảm. Chỉ xét câu gốc được dự đoán tích cực."""
    positive = base.prob("sentiment", "positive")
    return _directional_row(
        'Thêm "không" phủ định',
        sentences,
        base,
        predict,
        transform=add_negation,
        eligible=(base.labels("sentiment") == _POSITIVE_ID) & (positive >= POSITIVE_MIN_PROB),
        label="positive",
        passes=lambda before, after: after < before,
        rule=(
            f"Xét câu gốc dự đoán tích cực (P ≥ {vn(POSITIVE_MIN_PROB, 1)}). "
            "Đạt khi P(tích cực) giảm"
        ),
    )


def sad_emoticon_row(
    sentences: Sequence[str],
    base: Prediction,
    predict: Predict,
    tolerance: float = DIRECTIONAL_TOLERANCE,
) -> ChecklistRow:
    """Thêm ":(" cuối câu: P(tiêu cực) không được giảm quá `tolerance`. Xét mọi câu."""
    return _directional_row(
        'Thêm ":(" cuối câu',
        sentences,
        base,
        predict,
        transform=add_sad_emoticon,
        eligible=np.ones(len(sentences), dtype=bool),
        label="negative",
        passes=lambda before, after: after >= before - tolerance,
        rule=f"Đạt khi P(tiêu cực) không giảm quá {vn(tolerance, 2)}",
    )


def _drop_text(points: float) -> str:
    return f"giảm {vn(points, 1)} điểm" if points >= 0 else f"tăng {vn(-points, 1)} điểm"


def no_diacritics_row(
    sentences: Sequence[str],
    targets: Mapping[str, Sequence[int]],
    base: Prediction,
    predict: Predict,
) -> ChecklistRow:
    """Bỏ dấu toàn câu: tỉ lệ câu bị hệ thống gắn cờ "không dấu" và mức giảm accuracy hai tác vụ.

    `pass_rate` là tỉ lệ câu bị gắn cờ (hệ thống luôn gắn cờ những câu này để người xem xét).
    Accuracy của câu gốc và của câu bỏ dấu tính trên cùng tập câu đủ điều kiện, so với nhãn thật.
    """
    name = "Bỏ dấu toàn câu"
    index, texts = _variant_texts(sentences, strip_diacritics)
    n = len(index)
    if n == 0:
        return ChecklistRow(
            name=name, kind="robustness", n=0, pass_rate=None, note=f"{_NO_ELIGIBLE}."
        )
    new = _predict(predict, texts)
    flagged = int(new.no_diacritics.sum())
    parts = []
    for task, task_vi in (("sentiment", "cảm xúc"), ("topic", "chủ đề")):
        truth = np.asarray(targets[task])[index]
        before = float((base.labels(task)[index] == truth).mean()) * 100
        after = float((new.labels(task) == truth).mean()) * 100
        parts.append(f"{task_vi} {_drop_text(before - after)} ({vn(before, 1)}% → {vn(after, 1)}%)")
    note = (
        f"Đạt khi hệ thống gắn cờ không dấu (lọt cờ {vn(n - flagged, 0)}/{vn(n, 0)} câu). "
        f"Accuracy: {', '.join(parts)}."
    )
    return ChecklistRow(name=name, kind="robustness", n=n, pass_rate=flagged / n, note=note)


def compute_rows(
    sentences: Sequence[str],
    targets: Mapping[str, Sequence[int]],
    predict: Predict,
    log: Callable[[str], None] | None = None,
) -> list[ChecklistRow]:
    """Mọi dòng CheckList cho `sentences` (câu gốc), nhãn thật `targets`, bộ dự đoán `predict`."""
    for task in TASKS:
        if len(targets[task]) != len(sentences):
            raise ValueError(f"nhãn {task} có {len(targets[task])} dòng, cần {len(sentences)}")
    say = log or (lambda _message: None)
    base = _predict(predict, sentences)
    rng = random.Random(SEED)
    names_variants: dict[str, Transform] = {"": partial(reindex_names, rng=rng)}
    pii_variants: dict[str, Transform] = {
        label: partial(insert_middle, snippet=snippet) for label, snippet in PII_SNIPPETS.items()
    }
    builders: tuple[Callable[[], ChecklistRow], ...] = (
        lambda: invariance_row("Viết HOA toàn câu", sentences, base, predict, {"": to_upper}),
        lambda: invariance_row(
            "Viết Hoa Chữ Đầu Mỗi Từ", sentences, base, predict, {"": to_title_case}
        ),
        lambda: invariance_row(
            "Dấu câu và khoảng trắng thừa", sentences, base, predict, PUNCTUATION_VARIANTS
        ),
        lambda: invariance_row("Đổi thầy ↔ cô", sentences, base, predict, SWAP_VARIANTS),
        lambda: invariance_row(
            "Đổi chỉ số tên ẩn (wzjwz + số)", sentences, base, predict, names_variants
        ),
        lambda: invariance_row("Chèn thông tin cá nhân", sentences, base, predict, pii_variants),
        lambda: negation_row(sentences, base, predict),
        lambda: sad_emoticon_row(sentences, base, predict),
        lambda: no_diacritics_row(sentences, targets, base, predict),
    )
    rows = []
    for build in builders:
        started = time.perf_counter()
        row = build()
        rows.append(row)
        rate = "–" if row.pass_rate is None else f"{vn(row.pass_rate * 100, 1)}%"
        say(f"  {row.name:<34} n={row.n:>5}  đạt {rate:>7}  ({time.perf_counter() - started:.0f}s)")
    return rows


# --- Ghi kết quả ------------------------------------------------------------------------------


def default_results_dir() -> Path:
    return edupulse_config.REPO_ROOT / "results"


def result_paths(results_dir: Path) -> tuple[Path, Path]:
    """(bảng Markdown, JSON) trong `results_dir`."""
    return results_dir / "tables" / "checklist.md", results_dir / "metrics" / "checklist.json"


def build_payload(rows: Sequence[ChecklistRow], *, run: str, n_sentences: int) -> dict[str, Any]:
    """Nội dung `checklist.json`; mỗi dòng được kiểm lại bằng schema `ChecklistRow`."""
    return {
        "split": "validation",
        "run": run,
        "n_sentences": n_sentences,
        "seed": SEED,
        "rows": [
            ChecklistRow.model_validate(row.model_dump()).model_dump(mode="json") for row in rows
        ],
    }


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def render_markdown(
    rows: Sequence[ChecklistRow],
    *,
    run: str,
    n_sentences: int,
    temperatures: Mapping[str, float],
    partial_run: bool = False,
) -> str:
    """Bảng tiếng Việt: kiểm thử, loại, n, tỉ lệ đạt, ghi chú; đầu bảng nêu điều kiện chạy."""
    lines = [
        "# CheckList: kiểm thử hành vi của mô hình trên validation",
        "",
        f"Validation n = {vn(n_sentences, 0)} câu · run `{run}` · mô hình ONNX int8, từng câu một "
        f"(batch 1) · seed {SEED} · nhiệt độ T: cảm xúc {vn(temperatures['sentiment'], 2)}, "
        f"chủ đề {vn(temperatures['topic'], 2)}.",
        "",
    ]
    if partial_run:
        lines += ["**Chạy thử trên một phần validation, không phải số liệu chính thức.**", ""]
    lines += ["| Kiểm thử | Loại | n | Tỉ lệ đạt | Ghi chú |", "|---|---|---:|---:|---|"]
    for row in rows:
        rate = "–" if row.pass_rate is None else f"{vn(row.pass_rate * 100, 1)}%"
        lines.append(
            f"| {_cell(row.name)} | {KIND_VI[row.kind]} | {vn(row.n, 0)} | {rate} "
            f"| {_cell(row.note)} |"
        )
    lines += [
        "",
        "Bất biến: nhãn cảm xúc và chủ đề cùng không đổi sau một biến đổi vô hại. Có hướng: xác "
        "suất phải đổi đúng chiều. Bền vững: đo mức suy giảm khi đầu vào xấu đi. `n` là số câu đủ "
        "điều kiện; mỗi biến thể sinh từ câu gốc rồi qua đúng đường tiền xử lý và suy luận lúc "
        "phục vụ.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(
    rows: Sequence[ChecklistRow],
    *,
    run: str,
    n_sentences: int,
    temperatures: Mapping[str, float],
    results_dir: Path,
    partial_run: bool = False,
) -> tuple[Path, Path]:
    """Ghi bảng Markdown và JSON vào `results_dir`; chỉ có số đếm và tỉ lệ, không có câu gốc."""
    table_path, metrics_path = result_paths(results_dir)
    write_json(metrics_path, build_payload(rows, run=run, n_sentences=n_sentences))
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        render_markdown(
            rows,
            run=run,
            n_sentences=n_sentences,
            temperatures=temperatures,
            partial_run=partial_run,
        ),
        encoding="utf-8",
    )
    return table_path, metrics_path


# --- Chạy thật --------------------------------------------------------------------------------


def load_deployment(run_dir: Path) -> tuple[Path, Calibration]:
    """ONNX int8 và calibration.json của run; thiếu thì báo lỗi kèm lệnh cần chạy trước."""
    onnx_path = deploy_onnx_path(run_dir.name)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"chưa có {onnx_path} — chạy `make export` trước")
    calibration_path = calibration_output_path(run_dir)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"chưa có {calibration_path} — chạy `make calibrate` trước")
    return onnx_path, Calibration.model_validate(read_json(calibration_path))


def run(
    run_spec: str = "best",
    *,
    limit: int | None = None,
    results_dir: Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Chạy CheckList trên validation bằng mô hình triển khai; ghi kết quả, trả nội dung JSON."""
    if limit is not None and limit < 1:
        raise ValueError(f"limit phải >= 1, nhận {limit}")
    run_dir = resolve_run(run_spec)
    onnx_path, calibration = load_deployment(run_dir)
    frame = load_processed("validation")  # chỉ validation: tập test không bao giờ được mở ở đây
    total = len(frame)
    if limit is not None:
        frame = frame.head(limit)
    sentences = frame["sentence"].astype(str).tolist()
    targets = {task: frame[task].to_numpy() for task in TASKS}
    temperatures = {
        "sentiment": calibration.sentiment.temperature,
        "topic": calibration.topic.temperature,
    }
    log(
        f"CheckList: {len(sentences)}/{total} câu validation · run {run_dir.name} · "
        f"T cảm xúc {temperatures['sentiment']:.3f}, chủ đề {temperatures['topic']:.3f}"
    )
    predict = build_predictor(onnx_path, temperatures, settings().ort_threads)
    rows = compute_rows(sentences, targets, predict, log=log)
    write_outputs(
        rows,
        run=run_dir.name,
        n_sentences=len(sentences),
        temperatures=temperatures,
        results_dir=default_results_dir() if results_dir is None else results_dir,
        partial_run=len(sentences) < total,
    )
    return build_payload(rows, run=run_dir.name, n_sentences=len(sentences))


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"cần số nguyên, nhận {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"cần số nguyên >= 1, nhận {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="best", help="'best', tên run hoặc thư mục checkpoint")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="chỉ chạy N câu validation đầu tiên để thử nhanh (không phải số liệu chính thức)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.run, limit=args.limit)
    except FileNotFoundError as error:
        print(f"checklist: {error}", file=sys.stderr)
        return 2
    table_path, metrics_path = result_paths(default_results_dir())
    print(table_path.read_text(encoding="utf-8"))
    print(f"-> {table_path}\n-> {metrics_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
