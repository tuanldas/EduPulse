"""Điểm vào suy luận cho API và job nền: chuẩn hoá → ONNX → hiệu chỉnh → cờ → giải thích.

`Analyzer` nạp một lần lúc khởi động rồi dùng chung giữa các luồng (khoá nằm ở từng lô ONNX và ở
bộ tách từ). Các bước tách riêng (`prepare_many`, `infer`, `flag`) để job nền báo tiến độ và để
`bench` đo từng bước; `analyze` / `analyze_many` là đường ghép sẵn.

Một câu và cả lô dùng luật cờ khác nhau (xem `flags`): `analyze` so với ngưỡng ngân sách mặc định,
`analyze_many` xếp hạng cả lô theo ngân sách người dùng chọn.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from edupulse.bundle_schema import Calibration
from edupulse.config import DEFAULT_BUDGET, XAI_MAX_WORDS, Settings
from edupulse.engine.bundle import Bundle, load_bundle
from edupulse.engine.explain import Explanation, ExplanationUnavailable, explain_tokens
from edupulse.engine.flags import (
    NOT_FLAGGED,
    Flag,
    batch_needs_review,
    build_flag,
    single_needs_review,
    weakest_task,
)
from edupulse.engine.onnx_model import OnnxModel
from edupulse.labels import LABELS
from edupulse.scoring import confidence, flag_score, quantile_rank, softmax_t
from edupulse.text import Prepared, prepare, prepare_many

# Câu có chữ hoa, để lần khởi động nạp luôn JVM và (khi bật) NER thay vì đợi yêu cầu đầu tiên.
WARMUP_SENTENCE = "Xin chào Thầy Nguyễn Văn An"


class NoContent(ValueError):
    """Câu không còn từ nào để phân tích sau khi ẩn thông tin cá nhân và bỏ ký hiệu.

    Ví dụ câu chỉ có số điện thoại, email hay emoji. Mô hình vẫn trả nhãn cho chuỗi rỗng nhưng
    đó là nhãn vô nghĩa, nên engine từ chối thay vì trả kết quả trông như thật. Job nền lọc trước
    bằng `Prepared.n_words` (các dòng này không có nhãn) rồi mới gọi `infer`.
    """

    code = "no_content"

    def __init__(self, rows: Sequence[int]) -> None:
        super().__init__(f"{len(rows)} câu không còn nội dung để phân tích")
        self.rows = tuple(rows)


@dataclass(frozen=True, slots=True)
class HeadResult:
    label: str
    confidence: float  # xác suất của lớp dự đoán, đã hiệu chỉnh nhiệt độ
    probs: tuple[float, ...]  # theo thứ tự nhãn trong `labels.py`
    quantile: float  # phân vị của độ tin cậy trong phân phối trên val (0..1)


@dataclass(frozen=True, slots=True)
class Analysis:
    prepared: Prepared
    sentiment: HeadResult
    topic: HeadResult
    score: float  # điểm cờ = min(phân vị cảm xúc, phân vị chủ đề); thấp là ít chắc chắn
    flag: Flag = NOT_FLAGGED


def _head_results(
    task: str, logits: np.ndarray, temperature: float, cdf: Sequence[float]
) -> list[HeadResult]:
    probs = softmax_t(logits, temperature)
    quantiles = quantile_rank(confidence(probs), cdf)
    labels = LABELS[task]
    return [
        HeadResult(
            label=labels[int(np.argmax(row))],
            confidence=float(row.max()),
            probs=tuple(float(p) for p in row),
            quantile=float(q),
        )
        for row, q in zip(probs, quantiles, strict=True)
    ]


class Analyzer:
    def __init__(self, bundle: Bundle, model: OnnxModel, *, use_ner: bool = False) -> None:
        self.bundle = bundle
        self._model = model
        self._use_ner = use_ner

    @classmethod
    def load(cls, settings: Settings) -> "Analyzer":
        """Nạp bundle, mô hình và bộ tách từ rồi chạy thử; lỗi thì dừng với thông báo rõ."""
        bundle = load_bundle(settings.bundle_dir)
        analyzer = cls(bundle, OnnxModel(bundle, settings.ort_threads), use_ner=settings.ner)
        analyzer.warmup()
        return analyzer

    def warmup(self) -> None:
        prepare(WARMUP_SENTENCE, use_ner=self._use_ner)
        self._model.warmup()

    @property
    def calibration(self) -> Calibration:
        return self.bundle.calibration

    @property
    def use_ner(self) -> bool:
        return self._use_ner

    # --- Các bước ---------------------------------------------------------------------------

    def prepare_many(self, texts: Sequence[str]) -> list[Prepared]:
        return prepare_many(texts, use_ner=self._use_ner)

    def infer(
        self,
        prepared: Sequence[Prepared],
        progress: Callable[[int, int], None] | None = None,
        batch_size: int | None = None,
    ) -> list[Analysis]:
        """Dự đoán và điểm cờ từng câu, chưa quyết định cờ; `progress(xong, tổng)` sau mỗi lô.

        Int8 động làm kết quả một câu phụ thuộc nhẹ vào bạn cùng lô. `batch_size=1` cho đúng kết
        quả của `analyze` và của lúc hiệu chỉnh (chậm hơn vài lần); mặc định dùng lô của mô hình.
        Ném `NoContent` (kèm chỉ số các dòng) nếu có câu không còn từ nào để phân tích.
        """
        empty = [i for i, p in enumerate(prepared) if p.n_words == 0]
        if empty:
            raise NoContent(empty)
        total = len(prepared)
        logits = self._model.predict(
            [p.model_text for p in prepared],
            batch_size=batch_size,
            progress=None if progress is None else lambda done: progress(done, total),
        )
        calibration = self.bundle.calibration
        sentiment = _head_results(
            "sentiment",
            logits.sentiment,
            calibration.sentiment.temperature,
            calibration.sentiment.cdf,
        )
        topic = _head_results(
            "topic", logits.topic, calibration.topic.temperature, calibration.topic.cdf
        )
        scores = flag_score([s.quantile for s in sentiment], [t.quantile for t in topic])
        return [
            Analysis(prepared=p, sentiment=s, topic=t, score=float(score))
            for p, s, t, score in zip(prepared, sentiment, topic, scores, strict=True)
        ]

    @staticmethod
    def _flagged(analysis: Analysis, flagged: bool) -> Analysis:
        task = weakest_task(analysis.sentiment.quantile, analysis.topic.quantile)
        head = analysis.sentiment if task == "sentiment" else analysis.topic
        flag = build_flag(
            flagged=flagged,
            no_diacritics=analysis.prepared.no_diacritics,
            task=task,
            probs=head.probs,
        )
        return replace(analysis, flag=flag)

    def flag(
        self, analyses: Sequence[Analysis], budget_pct: float = DEFAULT_BUDGET
    ) -> list[Analysis]:
        """Cờ cả lô: ⌈B·n/100⌉ câu điểm thấp nhất, câu gõ không dấu được cờ trước."""
        if not analyses:
            return []
        flagged = batch_needs_review(
            [a.score for a in analyses], [a.prepared.no_diacritics for a in analyses], budget_pct
        )
        return [self._flagged(a, bool(f)) for a, f in zip(analyses, flagged, strict=True)]

    # --- Đường ghép sẵn -----------------------------------------------------------------------

    def analyze(self, text: str) -> Analysis:
        """Một câu; cờ theo ngưỡng của ngân sách mặc định."""
        (analysis,) = self.infer(self.prepare_many([text]))
        tau = self.bundle.calibration.tau[DEFAULT_BUDGET]
        flagged = single_needs_review(analysis.score, analysis.prepared.no_diacritics, tau)
        return self._flagged(analysis, flagged)

    def analyze_many(
        self,
        texts: Sequence[str],
        budget_pct: float = DEFAULT_BUDGET,
        progress: Callable[[int, int], None] | None = None,
        batch_size: int | None = None,
    ) -> list[Analysis]:
        prepared = self.prepare_many(texts)
        return self.flag(self.infer(prepared, progress, batch_size), budget_pct)

    def explain_prepared(self, prepared: Prepared) -> Explanation:
        if prepared.n_words == 0:
            raise NoContent([0])
        if prepared.n_words > XAI_MAX_WORDS:
            raise ExplanationUnavailable(prepared.n_words, XAI_MAX_WORDS)
        return explain_tokens(self._model, self.bundle.calibration, prepared.model_text.split())

    def explain(self, text: str) -> Explanation:
        """Giải thích bằng bỏ từng từ. Ném `ExplanationUnavailable` nếu câu dài hơn giới hạn,
        `NoContent` nếu câu không còn từ nào."""
        return self.explain_prepared(prepare(text, use_ner=self._use_ner))
