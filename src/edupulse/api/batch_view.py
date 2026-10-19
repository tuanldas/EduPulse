"""Chuyển dữ liệu batch trong RAM (`jobs`) sang các schema của API."""

import math
from collections import Counter

from edupulse.api import schemas
from edupulse.api.jobs import Batch, BatchResult, RowData, Snapshot
from edupulse.api.limits import PREVIEW_CELL_CHARS, PREVIEW_ROWS
from edupulse.api.parsing import ParsedTable, suggest_group_column, suggest_text_column
from edupulse.api.routes.analyze import prob, to_reason

# Thứ tự loại thông tin cá nhân khi liệt kê (khớp thứ tự khai báo trong `schemas.PiiType`).
PII_ORDER = ("name", "phone", "email", "url", "student_id", "national_id")
KNOWN_ENCODINGS = ("utf-8", "utf-8-sig", "cp1258")


def upload_response(batch: Batch, table: ParsedTable) -> schemas.BatchUpload:
    text_col = suggest_text_column(table)
    encoding = table.encoding if table.encoding in KNOWN_ENCODINGS else None
    return schemas.BatchUpload(
        batch_id=batch.id,
        filename=batch.filename,
        n_rows=len(table.rows),
        columns=table.columns,
        preview=[[cell[:PREVIEW_CELL_CHARS] for cell in row] for row in table.rows[:PREVIEW_ROWS]],
        suggested_text_col=text_col,
        suggested_group_col=suggest_group_column(table, text_col),
        encoding=encoding,
        warnings=[
            schemas.BatchWarning(code=w.code, message=w.message, count=w.count)
            for w in table.warnings
        ],
        illustrative_groups=batch.illustrative_groups,
    )


def status_response(snapshot: Snapshot) -> schemas.BatchStatus:
    return schemas.BatchStatus(
        state=snapshot.state,
        step=snapshot.step,
        done=snapshot.done,
        total=snapshot.total,
        progress=round(snapshot.progress, 4),
        eta_s=None if snapshot.eta_s is None else round(snapshot.eta_s, 1),
        n_rows=snapshot.n_rows,
        pii_rows=snapshot.pii_rows,
        error_code=snapshot.error_code,
        illustrative_groups=snapshot.illustrative_groups,
    )


def budget_target(budget: int, n_analyzed: int) -> int:
    """⌈ngân sách · số câu / 100⌉, làm tròn 9 chữ số như `scoring` để không lệch một câu."""
    return math.ceil(round(budget * n_analyzed / 100, 9))


def _result_row(row: RowData, result: BatchResult, xai_max_words: int) -> schemas.ResultRow:
    analysis = row.analysis
    assert analysis is not None
    flagged = analysis.flag.needs_review
    source = "flag" if flagged else ("audit" if row.audit else "none")
    return schemas.ResultRow(
        index=row.index,
        source_row=row.source_row,
        text=row.prepared.display_text,
        n_words=row.prepared.n_words,
        no_diacritics=row.prepared.no_diacritics,
        pii_types=list(row.prepared.pii_types),
        sentiment=analysis.sentiment.label,
        sentiment_confidence=prob(analysis.sentiment.confidence),
        topic=analysis.topic.label,
        topic_confidence=prob(analysis.topic.confidence),
        needs_review=flagged,
        reasons=[to_reason(r) for r in analysis.flag.reasons],
        source=source,
        in_review=source != "none",
        group=row.group,
        has_explanation=row.index in result.explanations,
        explainable=row.prepared.n_words <= xai_max_words,
    )


def _summary(batch: Batch, result: BatchResult) -> schemas.ResultsSummary:
    analyzed = [r for r in result.rows if r.analysis is not None]
    flagged = sum(r.analysis.flag.needs_review for r in analyzed)
    pii = Counter(t for r in result.rows for t in set(r.prepared.pii_types))
    return schemas.ResultsSummary(
        n_rows=len(result.rows),
        n_analyzed=len(analyzed),
        n_skipped=len(result.rows) - len(analyzed),
        n_flagged=flagged,
        flagged_share=round(flagged / len(analyzed), 4) if analyzed else 0.0,
        n_audit=sum(r.audit for r in analyzed),
        n_review=len(result.review_order),
        budget=result.budget,
        budget_target=budget_target(result.budget, len(analyzed)),
        n_no_diacritics=sum(r.prepared.no_diacritics for r in analyzed),
        pii_rows=sum(bool(r.prepared.pii_types) for r in result.rows),
        pii_counts=[schemas.PiiCount(type=t, rows=pii[t]) for t in PII_ORDER if pii[t]],
        n_truncated=sum(r.truncated for r in result.rows),
        n_explained=len(result.explanations),
        n_groups=None
        if batch.group_col is None
        else len({r.group for r in analyzed if r.group is not None}),
        elapsed_s=round(result.elapsed_s, 2),
        illustrative_groups=batch.illustrative_groups,
    )


def results_response(
    batch: Batch, result: BatchResult, xai_max_words: int
) -> schemas.ResultsResponse:
    return schemas.ResultsResponse(
        summary=_summary(batch, result),
        rows=[_result_row(r, result, xai_max_words) for r in result.rows if r.analysis is not None],
        review_order=list(result.review_order),
        skipped=[
            schemas.SkippedRow(index=r.index, source_row=r.source_row, text=r.prepared.display_text)
            for r in result.rows
            if r.analysis is None
        ],
    )
