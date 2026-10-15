"""Giải thích dự đoán bằng cách bỏ từng từ (occlusion).

Với câu có N từ (đơn vị là token của `model_text`: từ ghép RDR, tên, emoticon và dấu câu), dựng
một biến thể bỏ đúng một từ cho mỗi từ không phải dấu câu, rồi chạy các biến thể cùng câu gốc qua
mô hình, mỗi câu một lượt. Dấu câu không mang nội dung để giải thích nên không bị che: Δ = 0 sẵn,
bớt khoảng một phần tám số lượt chạy. Mô hình đa nhiệm nên cả hai head được giải thích cùng lúc.
Với mỗi head, mức ảnh hưởng của một từ là

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

# Dưới ngưỡng này thì chưa đáng gọi là ủng hộ hay đi ngược. Int8 động làm Δ lệch ngẫu nhiên cỡ 1
# điểm phần trăm (p95 ≈ 3,5 điểm, đo so với FP32 trên PhoBERT-base), nên 5 điểm để không đưa
# nhiễu lượng tử hoá ra thành "từ quan trọng".
MIN_EFFECT = 0.05
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
_PUNCTUATION = re.compile(r"[^\w]+")  # token chỉ gồm dấu câu và ký hiệu
_PUNCTUATION_TOKENS = frozenset({"doubledot", "dotdotdot"})  # ":" và "..." sau chuẩn hoá
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
    n_forwards: (
        int  # số câu đã chạy qua mô hình: câu gốc + một biến thể cho mỗi từ không phải dấu câu
    )
    sentiment: HeadExplanation
    topic: HeadExplanation


def is_punctuation(token: str) -> bool:
    """Dấu câu (kể cả ":" và "..." đã đổi thành token chữ). Emoticon và tên thì không phải."""
    return token in _PUNCTUATION_TOKENS or _PUNCTUATION.fullmatch(token) is not None


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
    task: str,
    logits: np.ndarray,
    calibration: HeadCalibration,
    shown: Sequence[str],
    occluded: Sequence[int] | None = None,
) -> HeadExplanation:
    """`logits` có hàng đầu là câu đầy đủ, các hàng sau là biến thể bỏ từ `occluded[k]`.

    Từ không nằm trong `occluded` (dấu câu) không được chạy lại: xác suất không đổi, Δ = 0.
    `occluded=None` nghĩa là mọi từ đều bị che lần lượt.
    """
    probs = softmax_t(logits, calibration.temperature)
    predicted = int(np.argmax(probs[0]))
    full = float(probs[0, predicted])
    without = np.full(len(shown), full)
    positions = range(len(shown)) if occluded is None else occluded
    without[list(positions)] = probs[1:, predicted]
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
    """Giải thích cho câu đã tách từ (`model_text.split()`); gọi sau khi kiểm giới hạn độ dài.

    Mỗi câu chạy riêng (lô 1): int8 động làm kết quả phụ thuộc bạn cùng lô, nên chia lô sẽ khiến
    câu đầy đủ có thể ra nhãn khác với `/api/analyze` và Δ nhiễu theo cách chia. Lô 1 cho câu đầy đủ
    đúng bằng kết quả của `analyze`, và Δ của một từ không phụ thuộc các từ khác bị che hay không.
    Chia lô cũng không nhanh hơn đáng kể ở mô hình này (đo ~12%).
    """
    occluded = [i for i, token in enumerate(tokens) if not is_punctuation(token)]
    variants = [" ".join(tokens)]
    variants += [" ".join((*tokens[:i], *tokens[i + 1 :])) for i in occluded]
    logits = model.predict(variants, batch_size=1)
    shown = [display_token(token) for token in tokens]
    return Explanation(
        n_forwards=len(variants),
        sentiment=_explain_head(
            "sentiment", logits.sentiment, calibration.sentiment, shown, occluded
        ),
        topic=_explain_head("topic", logits.topic, calibration.topic, shown, occluded),
    )
