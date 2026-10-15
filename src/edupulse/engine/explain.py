"""Giải thích dự đoán bằng cách bỏ từng từ (occlusion).

Với câu N từ (đơn vị là token của `model_text`: từ ghép RDR và dấu câu), dựng N biến thể bỏ đúng
một từ, rồi chạy N+1 câu (cùng câu gốc) qua mô hình trong một lượt chia lô nhỏ. Mô hình đa nhiệm
nên cả hai head được giải thích cùng lúc. Với mỗi head, mức ảnh hưởng của một từ là

    Δ = p(đủ câu) − p(bỏ từ đó)     (xác suất của lớp mà đủ câu được dự đoán)

Δ dương: từ ủng hộ nhãn; Δ âm: từ đi ngược nhãn. Xác suất dùng nhiệt độ đã hiệu chỉnh nên khớp
với độ tin cậy hiển thị.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from edupulse.bundle_schema import Calibration, HeadCalibration
from edupulse.engine.onnx_model import OnnxModel
from edupulse.labels import LABELS
from edupulse.scoring import softmax_t

EXPLAIN_BATCH = 16  # lô nhỏ: thêm <40MB RAM và nhường khoá suy luận cho yêu cầu khác
MIN_EFFECT = 0.01  # dưới 1 điểm phần trăm thì chưa đáng gọi là ủng hộ hay đi ngược
TOP_SUPPORT = 3
TOP_OPPOSE = 1
TOO_LONG = "xai_too_long"

# Token chữ mà bước chuẩn hoá dùng thay cho ký hiệu (xem text/normalize.py), đổi ngược để hiển thị.
_SYMBOL_TOKENS = {
    "colonsmilesmile": ":))",
    "colonsmile": ":)",
    "colonsad": ":(",
    "colonbigsmile": ":D",
    "colonlove": "<3",
    "coloncontemn": ":v",
    "colonsurprise": ":o",
    "colonhihi": ":3",
    "colonsmallsmile": ":>",
    "doubledot": ":",
    "vdotv": "v.v",
    "dotdotdot": "...",
}
_NAME_TOKEN = re.compile(r"wzjwz\d+")
# 3dot0 -> 3.0, uitdotedudotvn -> uit.edu.vn
_DOTTED_TOKEN = re.compile(
    r"\d+(?:dot\d+)+|(?:[a-z0-9]+(?:dot[a-z0-9]+)*)?dot(?:com|net|org|edu|vn|io)"
)


class ExplanationUnavailable(ValueError):
    """Câu quá dài để giải thích bằng bỏ từng từ (tốn N+1 lượt chạy)."""

    code = TOO_LONG

    def __init__(self, n_words: int, max_words: int) -> None:
        super().__init__(f"câu có {n_words} từ, giải thích chỉ hỗ trợ tối đa {max_words} từ")
        self.n_words = n_words
        self.max_words = max_words


@dataclass(frozen=True, slots=True)
class TokenEffect:
    text: str  # dạng hiển thị: "giảng viên", "[TÊN]", ":)"
    delta: float  # p(đủ câu) − p(bỏ từ này)
    p_without: float  # xác suất của lớp dự đoán khi bỏ từ này


@dataclass(frozen=True, slots=True)
class HeadExplanation:
    label: str  # lớp dự đoán của câu đầy đủ
    probability: float  # xác suất của lớp đó, đã hiệu chỉnh
    tokens: tuple[TokenEffect, ...]  # mọi từ của câu, theo thứ tự
    support: tuple[int, ...]  # chỉ số các từ ủng hộ mạnh nhất, mạnh trước
    oppose: tuple[int, ...]  # chỉ số các từ đi ngược mạnh nhất


@dataclass(frozen=True, slots=True)
class Explanation:
    n_forwards: int  # số câu đã chạy qua mô hình: N + 1
    sentiment: HeadExplanation
    topic: HeadExplanation


def display_token(token: str) -> str:
    """Token của mô hình -> chữ cho người đọc: `_` thành dấu cách, tên và ký hiệu được khôi phục."""
    if token in _SYMBOL_TOKENS:
        return _SYMBOL_TOKENS[token]
    if _NAME_TOKEN.fullmatch(token):
        return "[TÊN]"
    if _DOTTED_TOKEN.fullmatch(token):
        return token.replace("dot", ".")
    return token.replace("_", " ")


def _top(deltas: np.ndarray, *, descending: bool, limit: int) -> tuple[int, ...]:
    """Chỉ số các từ có |Δ| đủ lớn theo chiều đã chọn; bằng nhau thì từ đứng trước được ưu tiên."""
    sign = 1.0 if descending else -1.0
    ranked = sorted(
        (i for i, d in enumerate(deltas) if sign * d >= MIN_EFFECT),
        key=lambda i: (-sign * deltas[i], i),
    )
    return tuple(ranked[:limit])


def _explain_head(
    task: str, logits: np.ndarray, calibration: HeadCalibration, shown: Sequence[str]
) -> HeadExplanation:
    """`logits` có N+1 hàng: câu đầy đủ rồi N biến thể bỏ một từ."""
    probs = softmax_t(logits, calibration.temperature)
    predicted = int(np.argmax(probs[0]))
    full = float(probs[0, predicted])
    without = probs[1:, predicted]
    deltas = full - without
    effects = tuple(
        TokenEffect(text=text, delta=float(d), p_without=float(w))
        for text, d, w in zip(shown, deltas, without, strict=True)
    )
    return HeadExplanation(
        label=LABELS[task][predicted],
        probability=full,
        tokens=effects,
        support=_top(deltas, descending=True, limit=TOP_SUPPORT),
        oppose=_top(deltas, descending=False, limit=TOP_OPPOSE),
    )


def explain_tokens(
    model: OnnxModel, calibration: Calibration, tokens: Sequence[str]
) -> Explanation:
    """Giải thích cho câu đã tách từ (`model_text.split()`); gọi sau khi kiểm giới hạn độ dài."""
    variants = [" ".join(tokens)]
    variants += [" ".join((*tokens[:i], *tokens[i + 1 :])) for i in range(len(tokens))]
    logits = model.predict(variants, batch_size=EXPLAIN_BATCH)
    shown = [display_token(token) for token in tokens]
    return Explanation(
        n_forwards=len(variants),
        sentiment=_explain_head("sentiment", logits.sentiment, calibration.sentiment, shown),
        topic=_explain_head("topic", logits.topic, calibration.topic, shown),
    )
