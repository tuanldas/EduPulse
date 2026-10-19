"""Phân tích và giải thích một câu. Giao diện gọi hai endpoint song song để kết quả hiện trước.

Endpoint khai báo bằng `def` để FastAPI chạy trong threadpool: suy luận tốn CPU, không được chặn
vòng lặp sự kiện (và `/api/health` vẫn trả lời khi mô hình đang bận).
"""

from fastapi import APIRouter

from edupulse.api import schemas
from edupulse.api.deps import AnalyzerDep
from edupulse.api.errors import ApiError
from edupulse.engine import explain as xai
from edupulse.engine import flags
from edupulse.engine.baseline import NAME as BASELINE_NAME
from edupulse.engine.baseline import BaselineUnavailable
from edupulse.engine.explain import display_token
from edupulse.engine.service import Analysis, Analyzer, HeadResult, NoContent
from edupulse.labels import LABELS

router = APIRouter(
    prefix="/api",
    tags=["analyze"],
    responses={422: {"model": schemas.ErrorResponse}, 503: {"model": schemas.ErrorResponse}},
)

_DIGITS = 4


def prob(value: float) -> float:
    return round(value, _DIGITS)


def _probs(task: str, head: HeadResult) -> dict[str, float]:
    return {label: prob(p) for label, p in zip(LABELS[task], head.probs, strict=True)}


def to_reason(reason: flags.FlagReason) -> schemas.FlagReason:
    return schemas.FlagReason(
        code=reason.code,
        task=reason.task,
        candidates=[
            schemas.Candidate(label=c.label, probability=prob(c.probability))
            for c in reason.candidates
        ],
        message=reason.message,
    )


def to_analyze_response(analysis: Analysis, xai_max_words: int) -> schemas.AnalyzeResponse:
    prepared = analysis.prepared
    return schemas.AnalyzeResponse(
        display_text=prepared.display_text,
        pii_types=list(prepared.pii_types),
        no_diacritics=prepared.no_diacritics,
        n_words=prepared.n_words,
        xai_available=prepared.n_words <= xai_max_words,
        sentiment=schemas.SentimentPrediction(
            label=analysis.sentiment.label,
            confidence=prob(analysis.sentiment.confidence),
            probs=_probs("sentiment", analysis.sentiment),
        ),
        topic=schemas.TopicPrediction(
            label=analysis.topic.label,
            confidence=prob(analysis.topic.confidence),
            probs=_probs("topic", analysis.topic),
        ),
        flag=schemas.Flag(
            needs_review=analysis.flag.needs_review,
            reasons=[to_reason(r) for r in analysis.flag.reasons],
        ),
        baseline=None,
    )


def _head_explanation(head: xai.HeadExplanation) -> schemas.HeadExplanation:
    return schemas.HeadExplanation(
        label=head.label,
        probability=prob(head.probability),
        tokens=[
            schemas.TokenEffect(text=t.text, delta=prob(t.delta), p_without=prob(t.p_without))
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


def compare_with_baseline(analyzer: Analyzer, analysis: Analysis) -> schemas.BaselineComparison:
    """Kết quả của mô hình cơ sở trên cùng câu, kèm việc hai mô hình có đồng ý không."""
    try:
        result = analyzer.baseline.predict(analysis.prepared.model_text)
    except BaselineUnavailable as exc:
        raise ApiError(503, exc.code, "Bundle không có mô hình cơ sở để so sánh.") from exc
    return schemas.BaselineComparison(
        name=BASELINE_NAME,
        sentiment=schemas.BaselineSentiment(
            label=result.sentiment.label,
            confidence=prob(result.sentiment.confidence),
            top_terms=[display_token(t) for t in result.sentiment.top_terms],
        ),
        topic=schemas.BaselineTopic(
            label=result.topic.label,
            confidence=prob(result.topic.confidence),
            top_terms=[display_token(t) for t in result.topic.top_terms],
        ),
        agree_sentiment=result.sentiment.label == analysis.sentiment.label,
        agree_topic=result.topic.label == analysis.topic.label,
        elapsed_ms=round(result.elapsed_ms, 2),
    )


NO_CONTENT_MESSAGE = (
    "Câu chỉ gồm thông tin cá nhân hoặc ký hiệu nên không có nội dung để phân tích."
)


@router.post("/analyze")
def analyze(body: schemas.AnalyzeRequest, analyzer: AnalyzerDep) -> schemas.AnalyzeResponse:
    try:
        analysis = analyzer.analyze(body.text)
    except NoContent as exc:
        raise ApiError(422, exc.code, NO_CONTENT_MESSAGE) from exc
    response = to_analyze_response(analysis, analyzer.xai_max_words)
    if body.compare:
        response.baseline = compare_with_baseline(analyzer, analysis)
    return response


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
