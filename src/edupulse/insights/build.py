"""Dựng dữ liệu dashboard & insight từ các dòng đã dự đoán.

Mọi số liệu tính trên NHÃN DỰ ĐOÁN. So với mức nền của tập train (`InsightBackground`) thay vì một
ngưỡng cố định, vì tỉ lệ tiêu cực nền của từng chủ đề rất khác nhau (cơ sở vật chất ~95%, giảng
viên ~36%). Quy tắc cảnh báo:

- Tỉ lệ tiêu cực của chủ đề: cần n >= 20 câu VÀ cận dưới Wilson 95% > mức nền + 10 điểm.
- Chủ đề có mức nền đã sát 100% (cơ sở vật chất): xét tỉ trọng của chủ đề, cận dưới Wilson 95% > tỉ
  trọng nền.
- Có cột nhóm và chủ đề chưa vượt ở mức chung: xét từng nhóm với cùng luật.
- Nêu tối đa 3 cảnh báo, nặng nhất trước.
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from edupulse.bundle_schema import InsightBackground
from edupulse.insights import models as m
from edupulse.insights.rows import InsightRow
from edupulse.insights.rules import LECTURER_NOTE, TOPIC_PHRASE, recommend
from edupulse.insights.stats import document_frequency, weighted_log_odds, wilson_interval
from edupulse.labels import SENTIMENT, TOPIC
from edupulse.text.ngrams import ngrams

MIN_N = 20  # số câu tối thiểu để nêu cảnh báo / tô màu một ô heatmap
ALERT_MARGIN = 0.10  # cận dưới Wilson phải vượt mức nền ngần này
MIN_TOPIC_ROWS = 10  # cảnh báo tỉ trọng cần ít nhất ngần này câu thuộc chủ đề
MAX_ALERTS = 3
HIGHLIGHT_MIN_N = 20  # nhóm nổi bật trong cảnh báo chủ đề phải có ít nhất ngần này câu
ALL_GROUPS_MIN_N = 5  # nhóm tính vào "cao ở cả N nhóm" phải có ít nhất ngần này câu

MIN_KEYWORD_DF = 3  # từ khoá phải có trong ít nhất ngần này câu tiêu cực
MIN_KEYWORD_Z = 1.0
MAX_KEYWORDS = 8
KEYWORD_POOL = 20  # số từ dùng để chấm điểm và gắn nhãn câu tiêu biểu
COVERED_BY_BIGRAM = 0.6  # đơn từ bị bỏ nếu một cụm hai từ chứa nó chiếm ngần này số câu của nó

QUOTE_POOL = 50
QUOTE_MIN_CONFIDENCE = 0.7
QUOTE_WORDS = (6, 30)
NEAR_DUPLICATE = 0.8  # Jaccard của tập từ

KEYWORD_TOPICS = ("lecturer", "training_program", "facility")  # "Khác" quá phân tán để có từ khoá
TOPIC_TITLE = {
    "lecturer": "Giảng dạy",
    "training_program": "Chương trình",
    "facility": "Cơ sở vật chất",
    "others": "Chủ đề Khác",
}

_UPPER = "A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
# Từ xưng hô rồi đến một từ Viết hoa: tên người mà bước ẩn thông tin cá nhân còn sót.
_NAME_AFTER_ADDRESS = re.compile(
    rf"(?i:(?<!\w)(?:thầy giáo|cô giáo|giảng viên|giáo viên|thầy|cô|anh|chị|bạn|ông|bà|gv))\s+"
    rf"[{_UPPER}][^\W\d_]*"
)


@dataclass(frozen=True, slots=True)
class _Scoped:
    """Một dòng cùng tập n-gram của nó, tính một lần cho mỗi lần dựng."""

    row: InsightRow
    grams: tuple[str, ...]
    words: frozenset[str]


# --- Định dạng số theo tiếng Việt ----------------------------------------------------------------


def percent(value: float) -> str:
    """93% / 4,4%: một chữ số thập phân khi dưới 10%, dấu phẩy thập phân."""
    if value < 0.1:
        return f"{value * 100:.1f}".replace(".", ",") + "%"
    return f"{round(value * 100)}%"


def times(ratio: float) -> str:
    if 1.8 <= ratio <= 2.2:
        return "gấp đôi"
    return f"gấp {ratio:.1f} lần".replace(".", ",")


def display_term(term: str) -> str:
    return term.replace("_", " ")


# --- Chỉ số --------------------------------------------------------------------------------------


def _shares(rows: Sequence[InsightRow]) -> m.SentimentShares:
    n = len(rows)
    counts = Counter(r.sentiment for r in rows)
    values = {label: counts[label] / n if n else 0.0 for label in SENTIMENT}
    return m.SentimentShares(**values)


def _baseline_shares(background: InsightBackground) -> m.SentimentShares:
    return m.SentimentShares(**{label: background.sentiment_share[label] for label in SENTIMENT})


def _kpis(
    rows: Sequence[InsightRow],
    background: InsightBackground,
    *,
    has_groups: bool,
    has_flags: bool,
) -> m.Kpis:
    shares, baseline = _shares(rows), _baseline_shares(background)
    flagged = sum(bool(r.flagged) for r in rows)
    return m.Kpis(
        n_rows=len(rows),
        n_groups=len({r.group for r in rows if r.group is not None}) if has_groups else None,
        shares=shares,
        baseline_shares=baseline,
        delta_points=m.SentimentShares(
            **{
                label: round((getattr(shares, label) - getattr(baseline, label)) * 100, 1)
                for label in SENTIMENT
            }
        ),
        n_flagged=flagged if has_flags else None,
        flagged_share=(flagged / len(rows) if rows else 0.0) if has_flags else None,
    )


def _topic_stats(
    rows: Sequence[InsightRow], background: InsightBackground
) -> dict[str, m.TopicStat]:
    stats = {}
    for topic in TOPIC:
        members = [r for r in rows if r.topic == topic]
        negative = sum(r.sentiment == "negative" for r in members)
        stats[topic] = m.TopicStat(
            topic=topic,
            n=len(members),
            share=len(members) / len(rows) if rows else 0.0,
            baseline_share=background.topic_share[topic],
            sentiment_shares=_shares(members),
            negative_n=negative,
            baseline_negative_rate=background.negative_rate_by_topic[topic],
            negative_ci_low=wilson_interval(negative, len(members))[0] if members else None,
        )
    return stats


# --- Từ khoá và câu tiêu biểu --------------------------------------------------------------------


def _keywords_for(
    cell: Sequence[_Scoped], everything: Counter[str], background: InsightBackground
) -> tuple[list[m.Keyword], dict[str, float]]:
    """Từ khoá nổi bật của một ô (chủ đề × tiêu cực) và bảng z của nhóm từ dùng để chấm câu.

    Trả về (từ khoá hiển thị, {n-gram: z} của cả `KEYWORD_POOL` từ đầu).
    """
    if not cell:
        return [], {}
    inside: Counter[str] = Counter()
    for item in cell:
        inside.update(item.grams)
    outside = everything - inside
    z = weighted_log_odds(
        inside,
        outside,
        background.ngram_counts,
        prior_total=background.total_ngrams,
        alpha0=background.alpha0,
    )
    frequency = document_frequency(item.grams for item in cell)
    candidates = {
        term: score
        for term, score in z.items()
        if score >= MIN_KEYWORD_Z and frequency[term] >= MIN_KEYWORD_DF
    }
    # Đơn từ chỉ là một phần của cụm hai từ nói rõ hơn thì nhường chỗ cho cụm đó.
    bigrams = [t for t in candidates if " " in t]
    covered = {
        part
        for bigram in bigrams
        for part in bigram.split(" ")
        if part in candidates and frequency[bigram] >= COVERED_BY_BIGRAM * frequency[part]
    }
    ranked = sorted((t for t in candidates if t not in covered), key=lambda t: (-candidates[t], t))[
        :KEYWORD_POOL
    ]
    keywords = [
        m.Keyword(term=display_term(t), n=frequency[t], score=round(candidates[t], 2))
        for t in ranked[:MAX_KEYWORDS]
    ]
    return keywords, {t: candidates[t] for t in ranked}


def _suspects_a_name(text: str) -> bool:
    return _NAME_AFTER_ADDRESS.search(text) is not None


def _quotes_for(candidates: Sequence[_Scoped], pool: dict[str, float]) -> list[m.Quote]:
    low, high = QUOTE_WORDS
    usable = [
        item
        for item in candidates
        if low <= item.row.n_words <= high
        and min(item.row.sentiment_confidence, item.row.topic_confidence) >= QUOTE_MIN_CONFIDENCE
        and not _suspects_a_name(item.row.display_text)
    ]

    def matched(item: _Scoped) -> list[str]:
        present = set(item.grams)
        return sorted((t for t in pool if t in present), key=lambda t: (-pool[t], t))

    scored = []
    for item in usable:
        terms = matched(item)
        score = sum(pool[t] for t in terms) / max(item.row.n_words, 1) ** 0.5
        scored.append((score, item.row.sentiment_confidence, item, terms))
    scored.sort(key=lambda s: (-s[0], -s[1], s[2].row.display_text))
    chosen: list[tuple[_Scoped, list[str]]] = []
    for _, _, item, terms in scored:
        if any(_jaccard(item.words, kept.words) > NEAR_DUPLICATE for kept, _ in chosen):
            continue
        chosen.append((item, terms))
        if len(chosen) == QUOTE_POOL:
            break
    return [
        m.Quote(
            text=item.row.display_text,
            sentiment=item.row.sentiment,
            confidence=round(item.row.sentiment_confidence, 4),
            group=item.row.group,
            matched=[display_term(t) for t in terms],
        )
        for item, terms in chosen
    ]


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = len(a | b)
    return len(a & b) / union if union else 1.0


# --- Cảnh báo ------------------------------------------------------------------------------------


def _evidence(k: int, n: int, baseline: float) -> m.AlertEvidence:
    rate = k / n
    return m.AlertEvidence(
        n=n,
        k=k,
        rate=round(rate, 4),
        ci_low=round(wilson_interval(k, n)[0], 4),
        baseline=round(baseline, 4),
        delta_points=round((rate - baseline) * 100, 1),
    )


def _group_rates(rows: Sequence[InsightRow], topic: str) -> dict[str, tuple[int, int]]:
    """{nhóm: (số câu tiêu cực, số câu)} của một chủ đề."""
    totals: Counter[str] = Counter()
    negatives: Counter[str] = Counter()
    for row in rows:
        if row.topic == topic and row.group is not None:
            totals[row.group] += 1
            negatives[row.group] += row.sentiment == "negative"
    return {g: (negatives[g], totals[g]) for g in totals}


def _rate_summary(topic: str, group: str | None, evidence: m.AlertEvidence) -> str:
    where = f" ở nhóm {group}" if group else ""
    return (
        f"{percent(evidence.rate)} góp ý về {TOPIC_PHRASE[topic]}{where} là tiêu cực "
        f"({evidence.k}/{evidence.n} câu), cao hơn mức nền {percent(evidence.baseline)} "
        f"khoảng {round(evidence.delta_points)} điểm."
    )


def _highlight(
    rows: Sequence[InsightRow], topic: str, baseline: float
) -> tuple[m.GroupHighlight | None, str]:
    """Nhóm có tỉ lệ tiêu cực cao nhất của chủ đề và câu mô tả "cao ở cả N nhóm" (nếu đúng)."""
    rates = _group_rates(rows, topic)
    if len(rates) < 2:
        return None, ""
    sizeable = {g: v for g, v in rates.items() if v[1] >= ALL_GROUPS_MIN_N}
    note = ""
    if len(sizeable) >= 2 and all(k / n > baseline for k, n in sizeable.values()):
        note = f" Cao ở cả {len(sizeable)} nhóm"
    eligible = {g: v for g, v in rates.items() if v[1] >= HIGHLIGHT_MIN_N}
    if not eligible:
        return None, (note + ".") if note else ""
    group, (k, n) = max(eligible.items(), key=lambda item: (item[1][0] / item[1][1], item[0]))
    highlight = m.GroupHighlight(group=group, k=k, n=n, rate=round(k / n, 4))
    return (
        highlight,
        f"{note}{', nhiều nhất' if note else ' Nhiều nhất'} ở nhóm {group} ({k}/{n} câu).",
    )


def _alerts(
    rows: Sequence[InsightRow],
    stats: dict[str, m.TopicStat],
    keywords: dict[str, list[m.Keyword]],
    has_groups: bool,
) -> list[m.Alert]:
    found: list[tuple[float, m.Alert]] = []
    for topic in TOPIC:
        stat = stats[topic]
        baseline = stat.baseline_negative_rate
        terms = [k.term for k in keywords.get(topic, [])[:3]]
        note = LECTURER_NOTE if topic == "lecturer" else None
        if baseline + ALERT_MARGIN >= 1.0:
            # Mức nền đã sát 100%: tỉ lệ tiêu cực không phân biệt được gì, xét tỉ trọng chủ đề.
            alert = _share_alert(topic, stat, len(rows), terms, note)
            if alert is not None:
                found.append((alert.evidence.ci_low - alert.evidence.baseline, alert))
            continue
        if stat.n >= MIN_N and (stat.negative_ci_low or 0.0) > baseline + ALERT_MARGIN:
            evidence = _evidence(stat.negative_n, stat.n, baseline)
            highlight, extra = _highlight(rows, topic, baseline) if has_groups else (None, "")
            alert = m.Alert(
                kind="negative_rate",
                topic=topic,
                group=None,
                title=f"{TOPIC_TITLE[topic]}: tỉ lệ tiêu cực cao hơn mức nền",
                summary=_rate_summary(topic, None, evidence) + extra,
                evidence=evidence,
                keywords=terms,
                note=note,
                highlight=highlight,
            )
            found.append((evidence.ci_low - baseline, alert))
        elif has_groups:
            for group, (k, n) in sorted(_group_rates(rows, topic).items()):
                if n < MIN_N or wilson_interval(k, n)[0] <= baseline + ALERT_MARGIN:
                    continue
                evidence = _evidence(k, n, baseline)
                alert = m.Alert(
                    kind="negative_rate",
                    topic=topic,
                    group=group,
                    title=(f"{TOPIC_TITLE[topic]} ở nhóm {group}: tỉ lệ tiêu cực cao hơn mức nền"),
                    summary=_rate_summary(topic, group, evidence),
                    evidence=evidence,
                    keywords=terms,
                    note=note,
                    highlight=None,
                )
                found.append((evidence.ci_low - baseline, alert))
    found.sort(key=lambda item: (-item[0], item[1].topic, item[1].group or ""))
    return [alert for _, alert in found[:MAX_ALERTS]]


def _share_alert(
    topic: str, stat: m.TopicStat, n_rows: int, terms: list[str], note: str | None
) -> m.Alert | None:
    if n_rows < MIN_N or stat.n < MIN_TOPIC_ROWS:
        return None
    evidence = _evidence(stat.n, n_rows, stat.baseline_share)
    if evidence.ci_low <= stat.baseline_share:
        return None
    negative = f" {percent(stat.negative_n / stat.n)} trong số đó là tiêu cực." if stat.n else ""
    return m.Alert(
        kind="topic_share",
        topic=topic,
        group=None,
        title=f"{TOPIC_TITLE[topic]} được nhắc nhiều hơn thường lệ",
        summary=(
            f"{percent(evidence.rate)} góp ý nói về {TOPIC_TITLE[topic]} "
            f"({evidence.k}/{evidence.n} câu), {times(evidence.rate / stat.baseline_share)} "
            f"mức thường gặp {percent(stat.baseline_share)}.{negative}"
        ),
        evidence=evidence,
        keywords=terms,
        note=note,
        highlight=None,
    )


# --- Heatmap -------------------------------------------------------------------------------------


def _heatmap(rows: Sequence[InsightRow], background: InsightBackground) -> m.Heatmap:
    sizes = Counter(r.group for r in rows if r.group is not None)
    groups = sorted(sizes, key=lambda g: (-sizes[g], g))
    baseline = [background.negative_rate_by_topic[t] for t in TOPIC]
    cells = []
    for group in groups:
        line = []
        for topic, base in zip(TOPIC, baseline, strict=True):
            members = [r for r in rows if r.group == group and r.topic == topic]
            n = len(members)
            rate = sum(r.sentiment == "negative" for r in members) / n if n else None
            line.append(
                m.HeatmapCell(
                    n=n,
                    negative_rate=None if rate is None else round(rate, 4),
                    delta_points=None if rate is None else round((rate - base) * 100, 1),
                    shaded=n >= MIN_N,
                )
            )
        cells.append(line)
    return m.Heatmap(
        groups=groups,
        topics=list(TOPIC),
        baseline_negative_rate=[round(b, 4) for b in baseline],
        min_n=MIN_N,
        cells=cells,
    )


# --- Điểm vào ------------------------------------------------------------------------------------


def build_insights(
    rows: Sequence[InsightRow],
    background: InsightBackground,
    *,
    source: m.InsightSource,
    illustrative_groups: bool,
    groups: Sequence[str] | None = None,
) -> m.InsightsResponse:
    """Dashboard cho `rows`, lọc theo `groups` nếu tập có cột nhóm (tên lạ bị bỏ qua)."""
    sizes = Counter(r.group for r in rows if r.group is not None)
    has_groups = bool(sizes)
    has_flags = any(r.flagged is not None for r in rows)
    options = (
        [m.GroupOption(name=g, n=sizes[g]) for g in sorted(sizes, key=lambda g: (-sizes[g], g))]
        if has_groups
        else None
    )
    requested = set(groups or ())
    wanted = [o.name for o in options if o.name in requested] if options else []
    scope = [r for r in rows if r.group in wanted] if wanted else list(rows)
    selected = wanted if wanted else None

    stats = _topic_stats(scope, background)
    items = [
        _Scoped(r, tuple(ngrams(r.model_text)), frozenset(r.model_text.split())) for r in scope
    ]
    everything: Counter[str] = Counter()
    for item in items:
        everything.update(item.grams)

    keyword_blocks: list[m.TopicKeywords] = []
    keywords_by_topic: dict[str, list[m.Keyword]] = {}
    pools: dict[str, dict[str, float]] = {}
    negatives: dict[str, list[InsightRow]] = {}
    for topic in TOPIC:
        negatives[topic] = [r for r in scope if r.topic == topic and r.sentiment == "negative"]
    for topic in KEYWORD_TOPICS:
        cell = [i for i in items if i.row.topic == topic and i.row.sentiment == "negative"]
        keywords, pool = _keywords_for(cell, everything, background)
        keyword_blocks.append(m.TopicKeywords(topic=topic, n_negative=len(cell), keywords=keywords))
        keywords_by_topic[topic] = keywords
        pools[topic] = pool

    quote_blocks = []
    for topic in TOPIC:
        members = [
            i
            for i in items
            if i.row.topic == topic and (topic == "others" or i.row.sentiment == "negative")
        ]
        quote_blocks.append(
            m.TopicQuotes(topic=topic, quotes=_quotes_for(members, pools.get(topic, {})))
        )

    return m.InsightsResponse(
        source=source,
        illustrative_groups=illustrative_groups,
        groups=options,
        selected_groups=selected,
        kpis=_kpis(scope, background, has_groups=has_groups, has_flags=has_flags),
        topics=[stats[t] for t in TOPIC],
        alerts=_alerts(scope, stats, keywords_by_topic, has_groups),
        heatmap=_heatmap(scope, background) if has_groups else None,
        keywords=keyword_blocks,
        quotes=quote_blocks,
        recommendations=recommend(negatives, len({r.group for r in scope if r.group is not None})),
    )
