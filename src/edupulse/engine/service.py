"""Điểm vào suy luận cho API và job nền: chuẩn hoá → ONNX → hiệu chỉnh → cờ → giải thích.

`Analyzer` nạp một lần lúc khởi động rồi dùng chung giữa các luồng (khoá nằm ở từng lô ONNX và ở
bộ tách từ). Các bước tách riêng (`prepare_many`, `infer`, `flag`) để `bench` đo từng bước;
`analyze` / `analyze_many` là đường ghép sẵn cho một câu / một lô nhỏ, còn `run_batch` là đường
của job nền: cùng các bước đó kèm mẫu kiểm tra, giải thích tự động, tiến độ và huỷ giữa chừng.

Một câu và cả lô dùng luật cờ khác nhau (xem `flags`): `analyze` so với ngưỡng ngân sách mặc định,
`analyze_many` xếp hạng cả lô theo ngân sách người dùng chọn.
"""

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from edupulse.bundle_schema import Calibration
from edupulse.config import AUDIT_RATE, AUTO_XAI_TOP, DEFAULT_BUDGET, XAI_MAX_WORDS, Settings
from edupulse.engine.baseline import Baseline
from edupulse.engine.bundle import Bundle, load_bundle
from edupulse.engine.explain import Explanation, ExplanationUnavailable, explain_tokens
from edupulse.engine.flags import (
    NOT_FLAGGED,
    Flag,
    audit_sample,
    batch_needs_review,
    build_flag,
    review_order,
    single_needs_review,
    weakest_task,
)
from edupulse.engine.onnx_model import OnnxModel
from edupulse.labels import LABELS
from edupulse.scoring import confidence, flag_score, quantile_rank, softmax_t
from edupulse.text import Prepared, prepare, prepare_many

# Câu có chữ hoa, để lần khởi động nạp luôn JVM và (khi bật) NER thay vì đợi yêu cầu đầu tiên.
WARMUP_SENTENCE = "Xin chào Thầy Nguyễn Văn An"
# Số câu mỗi lần gọi tiền xử lý trong job nền: đủ nhỏ để báo tiến độ và nhận lệnh huỷ kịp thời.
PREPARE_CHUNK = 50


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


class BatchCancelled(Exception):
    """Job nền bị huỷ giữa chừng (batch đã đóng hoặc hết hạn)."""


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


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """Kết quả của `run_batch`; mọi danh sách khớp thứ tự câu đầu vào."""

    prepared: list[Prepared]
    analyses: list[Analysis | None]  # None: câu không còn nội dung nên không có nhãn
    audit: NDArray[np.bool_]  # mẫu kiểm tra (câu không cờ được chọn ngẫu nhiên)
    review_order: list[int]  # chỉ số câu cần duyệt (cờ + kiểm tra), độ tin cậy tăng dần
    explanations: dict[int, Explanation]  # giải thích tính sẵn, khoá theo chỉ số câu
    steps_s: dict[str, float]  # thời gian từng bước, giây

    @property
    def n_flagged(self) -> int:
        return sum(a is not None and a.flag.needs_review for a in self.analyses)

    @property
    def n_skipped(self) -> int:
        return sum(a is None for a in self.analyses)


# Tên bước báo tiến độ: `progress(bước, xong, tổng)`.
ProgressCallback = Callable[[str, int, int], None]


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
    def __init__(
        self,
        bundle: Bundle,
        model: OnnxModel,
        *,
        use_ner: bool = False,
        xai_max_words: int = XAI_MAX_WORDS,
    ) -> None:
        self.bundle = bundle
        self._model = model
        self._use_ner = use_ner
        self.xai_max_words = xai_max_words  # câu dài hơn thì không giải thích
        self.baseline = Baseline(bundle.root / "baseline")  # nạp lười khi có người so sánh

    @classmethod
    def load(cls, settings: Settings) -> "Analyzer":
        """Nạp bundle, mô hình và bộ tách từ rồi chạy thử; lỗi thì dừng với thông báo rõ."""
        bundle = load_bundle(settings.bundle_dir)
        analyzer = cls(
            bundle,
            OnnxModel(bundle, settings.ort_threads),
            use_ner=settings.ner,
            xai_max_words=settings.xai_max_words,
        )
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

        Int8 động làm kết quả một câu phụ thuộc vào bạn cùng lô, nên mặc định chạy đúng cỡ lô mà
        bundle đã dùng khi hiệu chỉnh (`logits_batch_size`, hiện là 1: từng câu một): nhãn và cờ
        của một câu khi đó không đổi giữa `analyze` và batch job. `batch_size` lớn hơn nhanh hơn
        nhưng kết quả lệch nhẹ theo bạn cùng lô.
        Ném `NoContent` (kèm chỉ số các dòng) nếu có câu không còn từ nào để phân tích.
        """
        empty = [i for i, p in enumerate(prepared) if p.n_words == 0]
        if empty:
            raise NoContent(empty)
        total = len(prepared)
        logits = self._model.predict(
            [p.model_text for p in prepared],
            batch_size=batch_size or self.bundle.calibration.logits_batch_size,
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

    def run_batch(
        self,
        texts: Sequence[str],
        budget_pct: float = DEFAULT_BUDGET,
        *,
        seed: int,
        explain_top: int = AUTO_XAI_TOP,
        batch_size: int | None = None,
        progress: ProgressCallback | None = None,
        cancelled: Callable[[], bool] | None = None,
        on_prepared: Callable[[list[Prepared]], None] | None = None,
    ) -> BatchOutcome:
        """Đường của job nền: tiền xử lý → dự đoán → cờ theo ngân sách → mẫu kiểm tra → giải thích
        tự động cho `explain_top` câu cờ có điểm thấp nhất (chỉ câu không dài quá giới hạn).

        Câu không còn nội dung không có nhãn và không tính vào ngân sách. `seed` quyết định mẫu kiểm
        tra, nên cùng batch luôn cho cùng kết quả. `cancelled()` được hỏi giữa các đợt xử lý; trả
        True thì ném `BatchCancelled`. `on_prepared` nhận kết quả tiền xử lý ngay khi xong bước đó
        (để báo số câu có thông tin bị ẩn sớm). Ném `NoContent` nếu không câu nào còn nội dung.
        """

        def report(step: str, done: int, total: int) -> None:
            if cancelled is not None and cancelled():
                raise BatchCancelled
            if progress is not None:
                progress(step, done, total)

        steps: dict[str, float] = {}
        total = len(texts)

        started = time.perf_counter()
        prepared: list[Prepared] = []
        report("prepare", 0, total)
        for start in range(0, total, PREPARE_CHUNK):
            prepared.extend(self.prepare_many(texts[start : start + PREPARE_CHUNK]))
            report("prepare", len(prepared), total)
        steps["prepare"] = time.perf_counter() - started
        if on_prepared is not None:
            on_prepared(prepared)

        kept = [i for i, p in enumerate(prepared) if p.n_words > 0]
        if not kept:
            raise NoContent(range(total))

        started = time.perf_counter()
        analyses = self.infer(
            [prepared[i] for i in kept],
            progress=lambda done, count: report("predict", done, count),
            batch_size=batch_size,
        )
        steps["predict"] = time.perf_counter() - started

        started = time.perf_counter()
        report("flag", 0, 1)
        analyses = self.flag(analyses, budget_pct)
        flagged = np.array([a.flag.needs_review for a in analyses])
        audit = audit_sample(flagged, AUDIT_RATE, seed)
        order = review_order([a.score for a in analyses], flagged | audit)
        steps["flags"] = time.perf_counter() - started
        report("flag", 1, 1)

        started = time.perf_counter()
        explainable = sorted(
            (
                pos
                for pos in np.flatnonzero(flagged)
                if analyses[pos].prepared.n_words <= self.xai_max_words
            ),
            key=lambda pos: (analyses[pos].score, pos),
        )[:explain_top]
        explanations: dict[int, Explanation] = {}
        report("explain", 0, len(explainable))
        for done, pos in enumerate(explainable, start=1):
            explanations[kept[pos]] = self.explain_prepared(analyses[pos].prepared)
            report("explain", done, len(explainable))
        steps["explain"] = time.perf_counter() - started

        full: list[Analysis | None] = [None] * total
        audit_full = np.zeros(total, dtype=bool)
        for pos, index in enumerate(kept):
            full[index] = analyses[pos]
            audit_full[index] = audit[pos]
        return BatchOutcome(
            prepared=prepared,
            analyses=full,
            audit=audit_full,
            review_order=[kept[pos] for pos in order],
            explanations=explanations,
            steps_s=steps,
        )

    def explain_prepared(self, prepared: Prepared) -> Explanation:
        if prepared.n_words == 0:
            raise NoContent([0])
        if prepared.n_words > self.xai_max_words:
            raise ExplanationUnavailable(prepared.n_words, self.xai_max_words)
        return explain_tokens(self._model, self.bundle.calibration, prepared.model_text.split())

    def explain(self, text: str) -> Explanation:
        """Giải thích bằng bỏ từng từ. Ném `ExplanationUnavailable` nếu câu dài hơn giới hạn,
        `NoContent` nếu câu không còn từ nào."""
        return self.explain_prepared(prepare(text, use_ner=self._use_ner))
