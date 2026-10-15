"""Phân tích và giải thích một câu. Giao diện gọi hai endpoint song song để kết quả hiện trước.

Endpoint khai báo bằng `def` để FastAPI chạy trong threadpool: suy luận tốn CPU, không được chặn
vòng lặp sự kiện (và `/api/health` vẫn trả lời khi mô hình đang bận).
"""

from fastapi import APIRouter

from edupulse.api import schemas
from edupulse.api.deps import AnalyzerDep
from edupulse.api.errors import ApiError, not_implemented
from edupulse.config import XAI_MAX_WORDS
from edupulse.engine import explain as xai
from edupulse.engine import flags
from edupulse.engine.service import Analysis, HeadResult, NoContent
from edupulse.labels import LABELS

router = APIRouter(
    prefix="/api",
    tags=["analyze"],
    responses={422: {"model": schemas.ErrorResponse}, 503: {"model": schemas.ErrorResponse}},
)

_DIGITS = 4


def _p(value: float) -> float:
    return round(value, _DIGITS)


def _probs(task: str, head: HeadResult) -> dict[str, float]:
    return {label: _p(p) for label, p in zip(LABELS[task], head.probs, strict=True)}


def _reason(reason: flags.FlagReason) -> schemas.FlagReason:
    return schemas.FlagReason(
        code=reason.code,
        task=reason.task,
        candidates=[
            schemas.Candidate(label=c.label, probability=_p(c.probability))
            for c in reason.candidates
        ],
        message=reason.message,
    )


def to_analyze_response(analysis: Analysis) -> schemas.AnalyzeResponse:
    prepared = analysis.prepared
    return schemas.AnalyzeResponse(
        display_text=prepared.display_text,
        pii_types=list(prepared.pii_types),
        no_diacritics=prepared.no_diacritics,
        n_words=prepared.n_words,
        xai_available=prepared.n_words <= XAI_MAX_WORDS,
        sentiment=schemas.SentimentPrediction(
            label=analysis.sentiment.label,
            confidence=_p(analysis.sentiment.confidence),
            probs=_probs("sentiment", analysis.sentiment),
        ),
        topic=schemas.TopicPrediction(
            label=analysis.topic.label,
            confidence=_p(analysis.topic.confidence),
            probs=_probs("topic", analysis.topic),
        ),
        flag=schemas.Flag(
            needs_review=analysis.flag.needs_review,
            reasons=[_reason(r) for r in analysis.flag.reasons],
        ),
        baseline=None,
    )


def _head_explanation(head: xai.HeadExplanation) -> schemas.HeadExplanation:
    return schemas.HeadExplanation(
        label=head.label,
        probability=_p(head.probability),
        tokens=[
            schemas.TokenEffect(text=t.text, delta=_p(t.delta), p_without=_p(t.p_without))
            for t in head.tokens
        ],
        support=list(head.support),
        oppose=list(head.oppose),
    )


def to_explain_response(explanation: xai.Explanation) -> schemas.ExplainResponse:
    return schemas.ExplainResponse(
        n_forwards=explanation.n_forwards,
        sentiment=_head_explanation(explanation.sentiment),
        topic=_head_explanation(explanation.topic),
    )


NO_CONTENT_MESSAGE = (
    "Câu chỉ gồm thông tin cá nhân hoặc ký hiệu nên không có nội dung để phân tích."
)


@router.post("/analyze")
def analyze(body: schemas.AnalyzeRequest, analyzer: AnalyzerDep) -> schemas.AnalyzeResponse:
    if body.compare:
        raise not_implemented()
    try:
        analysis = analyzer.analyze(body.text)
    except NoContent as exc:
        raise ApiError(422, exc.code, NO_CONTENT_MESSAGE) from exc
    return to_analyze_response(analysis)


@router.post("/explain")
def explain(body: schemas.TextRequest, analyzer: AnalyzerDep) -> schemas.ExplainResponse:
    try:
        explanation = analyzer.explain(body.text)
    except NoContent as exc:
        raise ApiError(422, exc.code, NO_CONTENT_MESSAGE) from exc
    except xai.ExplanationUnavailable as exc:
        raise ApiError(
            422,
            exc.code,
            f"Câu có {exc.n_words} từ, giải thích chỉ hỗ trợ tối đa {exc.max_words} từ.",
            params={"n_words": exc.n_words, "max_words": exc.max_words},
        ) from exc
    return to_explain_response(explanation)
