"""Dashboard & insight: số liệu kiểm tay trên tập nhỏ, cảnh báo, heatmap, từ khoá, câu tiêu biểu."""

import random

import pytest

from edupulse.bundle_schema import InsightBackground
from edupulse.config import load_settings
from edupulse.insights import build as insights
from edupulse.insights.models import InsightsResponse
from edupulse.insights.rows import InsightRow
from edupulse.insights.stats import wilson_lower
from edupulse.labels import SENTIMENT, TOPIC

BASELINE_NEGATIVE = {
    "lecturer": 0.3562331618907666,
    "training_program": 0.7796456156292594,
    "facility": 0.9537223340040242,
    "others": 0.40213523131672596,
}
TOPIC_SHARE = {
    "lecturer": 0.7146858043059688,
    "training_program": 0.1926308419394364,
    "facility": 0.04349728688955015,
    "others": 0.049186066865044634,
}
SENTIMENT_SHARE = {
    "negative": 0.46604235953089446,
    "neutral": 0.040084018904253456,
    "positive": 0.4938736215648521,
}


def make_background(ngram_counts: dict[str, int] | None = None) -> InsightBackground:
    return InsightBackground(
        normalizer_version="1",
        alpha0=1000,
        min_count=2,
        n_sentences=11426,
        total_ngrams=199244,
        ngram_counts=ngram_counts
        or {"phòng": 700, "máy_chiếu": 60, "phòng máy": 40, "thầy": 4000, "dạy": 3000},
        sentiment_share=SENTIMENT_SHARE,
        topic_share=TOPIC_SHARE,
        negative_rate_by_topic=BASELINE_NEGATIVE,
    )


BACKGROUND = make_background()


def row(
    text: str = "câu mẫu số một về điều gì đó",
    topic: str = "lecturer",
    sentiment: str = "negative",
    *,
    model_text: str | None = None,
    confidence: float = 0.9,
    topic_confidence: float | None = None,
    group: str | None = None,
    flagged: bool | None = None,
    n_words: int | None = None,
) -> InsightRow:
    return InsightRow(
        display_text=text,
        model_text=model_text if model_text is not None else text.lower(),
        sentiment=sentiment,
        sentiment_confidence=confidence,
        topic=topic,
        topic_confidence=confidence if topic_confidence is None else topic_confidence,
        n_words=len(text.split()) if n_words is None else n_words,
        group=group,
        flagged=flagged,
    )


def many(
    count: int, template: str = "câu số {i} nói về điều gì đó hay", **kwargs
) -> list[InsightRow]:
    return [row(template.format(i=i), **kwargs) for i in range(count)]


def build(rows, **kwargs) -> InsightsResponse:
    kwargs.setdefault("source", "batch")
    kwargs.setdefault("illustrative_groups", False)
    return insights.build_insights(rows, BACKGROUND, **kwargs)


def topic_rows(topic: str, negative: int, positive: int = 0, **kwargs) -> list[InsightRow]:
    return [
        *many(
            negative,
            f"{topic} tiêu cực {{i}} lời góp ý dài",
            topic=topic,
            sentiment="negative",
            **kwargs,
        ),
        *many(
            positive,
            f"{topic} tích cực {{i}} lời góp ý dài",
            topic=topic,
            sentiment="positive",
            **kwargs,
        ),
    ]


# --- Chỉ số tổng quan, kiểm tay -------------------------------------------------------------------


@pytest.fixture(scope="module")
def hundred() -> list[InsightRow]:
    """100 câu: giảng viên 50 (20 tiêu cực, 5 trung tính, 25 tích cực), chương trình 30 (24 tiêu
    cực, 6 tích cực), cơ sở vật chất 10 (10 tiêu cực), khác 10 (4 tiêu cực, 6 tích cực)."""
    data = topic_rows("lecturer", 20, 25) + many(
        5, "lecturer trung tính {i} lời góp ý dài", sentiment="neutral"
    )
    data += topic_rows("training_program", 24, 6)
    data += topic_rows("facility", 10)
    data += topic_rows("others", 4, 6)
    assert len(data) == 100
    return data


def test_kpis_match_the_hand_computed_shares(hundred):
    kpis = build(hundred).kpis
    assert kpis.n_rows == 100 and kpis.n_groups is None
    assert (kpis.shares.negative, kpis.shares.neutral, kpis.shares.positive) == (0.58, 0.05, 0.37)
    assert kpis.baseline_shares.negative == pytest.approx(SENTIMENT_SHARE["negative"])
    assert kpis.delta_points.negative == pytest.approx(11.4)  # (0,58 - 0,466) × 100
    assert kpis.delta_points.neutral == pytest.approx(1.0)
    assert kpis.delta_points.positive == pytest.approx(-12.4)
    assert (kpis.n_flagged, kpis.flagged_share) == (None, None)


def test_topic_stats_match_the_hand_computed_numbers(hundred):
    topics = {t.topic: t for t in build(hundred).topics}
    assert list(topics) == list(TOPIC)
    lecturer = topics["lecturer"]
    assert (lecturer.n, lecturer.negative_n, lecturer.share) == (50, 20, 0.5)
    assert lecturer.sentiment_shares.negative == 0.4
    assert lecturer.sentiment_shares.neutral == 0.1
    assert lecturer.baseline_negative_rate == pytest.approx(BASELINE_NEGATIVE["lecturer"])
    assert lecturer.baseline_share == pytest.approx(TOPIC_SHARE["lecturer"])
    assert lecturer.negative_ci_low == pytest.approx(wilson_lower(20, 50))
    assert topics["facility"].negative_ci_low == pytest.approx(wilson_lower(10, 10))


def test_an_empty_topic_has_no_interval_and_zero_shares():
    topics = {t.topic: t for t in build(topic_rows("lecturer", 5)).topics}
    assert topics["facility"].n == 0 and topics["facility"].negative_ci_low is None
    assert topics["facility"].sentiment_shares.negative == 0.0


def test_flag_counts_come_from_the_rows_when_the_source_has_them():
    flagged = [row(f"câu {i} có đủ chữ", flagged=i < 3) for i in range(10)]
    kpis = build(flagged).kpis
    assert (kpis.n_flagged, kpis.flagged_share) == (3, 0.3)
    assert build(flagged, groups=None).kpis.n_flagged == 3


def test_empty_input_gives_a_valid_empty_dashboard():
    out = build([])
    assert out.kpis.n_rows == 0 and out.kpis.shares.negative == 0.0
    assert [t.n for t in out.topics] == [0, 0, 0, 0]
    assert out.alerts == [] and out.heatmap is None and out.groups is None
    assert all(not b.keywords for b in out.keywords) and all(not b.quotes for b in out.quotes)
    assert out.recommendations == []


def test_the_response_survives_a_json_round_trip(hundred):
    out = build(hundred, source="demo")
    assert InsightsResponse.model_validate_json(out.model_dump_json()) == out
    assert out.source == "demo"


def test_the_same_rows_in_any_order_give_the_same_numbers(hundred):
    shuffled = list(hundred)
    random.Random(3).shuffle(shuffled)
    a, b = build(hundred), build(shuffled)
    assert (a.kpis, a.topics, a.alerts) == (b.kpis, b.topics, b.alerts)
    assert a.keywords == b.keywords and a.recommendations == b.recommendations
    assert build(hundred) == a


# --- Cảnh báo -------------------------------------------------------------------------------------


def test_no_alert_when_everything_is_around_the_baseline():
    rows = topic_rows("lecturer", 36, 64) + topic_rows("training_program", 78, 22)
    assert build(rows).alerts == []


def test_negative_rate_alert_needs_the_wilson_lower_bound_above_baseline_plus_ten_points():
    # 96/100 chương trình: cận dưới Wilson 0,90 > 0,78 + 0,10; 88/100 thì 0,80 < 0,88.
    (alert,) = build(topic_rows("training_program", 96, 4)).alerts
    assert (alert.kind, alert.topic, alert.group) == ("negative_rate", "training_program", None)
    assert alert.evidence.n == 100 and alert.evidence.k == 96
    assert alert.evidence.ci_low == pytest.approx(wilson_lower(96, 100), abs=1e-4)
    assert alert.evidence.baseline == pytest.approx(BASELINE_NEGATIVE["training_program"], abs=1e-4)
    assert alert.evidence.delta_points == pytest.approx(96 - 77.96, abs=0.1)
    assert alert.title == "Chương trình: tỉ lệ tiêu cực cao hơn mức nền"
    assert alert.summary == (
        "96% góp ý về Chương trình là tiêu cực (96/100 câu), cao hơn mức nền 78% khoảng 18 điểm."
    )
    assert build(topic_rows("training_program", 88, 12)).alerts == []


def test_the_point_estimate_alone_is_not_enough():
    # 60% > 36% + 10 điểm nhưng chỉ 10 câu: cận dưới Wilson thấp và n < 20.
    assert build(topic_rows("lecturer", 6, 4)).alerts == []
    # n = 25, 15 tiêu cực (60%): cận dưới 0,41 < 0,456.
    assert build(topic_rows("lecturer", 15, 10)).alerts == []


def test_at_least_twenty_sentences_are_needed():
    assert build(topic_rows("lecturer", 19)).alerts == []  # 19/19 nhưng n = 19
    assert len(build(topic_rows("lecturer", 20)).alerts) == 1


def test_lecturer_alerts_carry_the_aggregate_note_and_speak_of_teaching():
    (alert,) = build(topic_rows("lecturer", 60, 40)).alerts
    assert alert.note == "Tổng hợp theo nhóm, không gắn với cá nhân."
    assert "giảng dạy" in alert.summary and "Giảng viên" not in alert.title


def test_facility_is_judged_by_its_share_because_its_baseline_is_already_near_100_percent():
    rows = topic_rows("facility", 20) + topic_rows("lecturer", 90, 90)  # 20/200 = 10% so với 4,3%
    alerts = build(rows).alerts
    share = [a for a in alerts if a.topic == "facility"]
    assert len(share) == 1 and share[0].kind == "topic_share"
    assert share[0].evidence.n == 200 and share[0].evidence.k == 20
    assert share[0].evidence.baseline == pytest.approx(TOPIC_SHARE["facility"], abs=1e-4)
    assert share[0].title == "Cơ sở vật chất được nhắc nhiều hơn thường lệ"
    assert share[0].summary == (
        "10% góp ý nói về Cơ sở vật chất (20/200 câu), gấp 2,3 lần mức thường gặp 4,3%. "
        "100% trong số đó là tiêu cực."
    )


def test_a_facility_share_close_to_the_baseline_raises_nothing():
    rows = topic_rows("facility", 9) + topic_rows("lecturer", 91, 100)  # 4,5% so với 4,3%
    assert [a for a in build(rows).alerts if a.topic == "facility"] == []
    rows = topic_rows("facility", 9) + topic_rows("lecturer", 21, 70)  # đủ cao nhưng chỉ 9 câu
    assert [a for a in build(rows).alerts if a.topic == "facility"] == []


def test_times_wording():
    assert insights.times(2.0) == "gấp đôi"
    assert insights.times(2.3) == "gấp 2,3 lần"
    assert insights.times(5.0) == "gấp 5,0 lần"


def test_percent_wording():
    assert insights.percent(0.93) == "93%"
    assert insights.percent(0.044) == "4,4%"
    assert insights.percent(1.0) == "100%"
    assert insights.percent(0.0) == "0,0%"


def test_at_most_three_alerts_most_severe_first():
    rows = (
        topic_rows("lecturer", 60, 40)  # cận dưới ~0,50 vs 0,36: vượt ~0,15
        + topic_rows("training_program", 96, 4)  # vượt ~0,12
        + topic_rows("others", 40)  # cận dưới ~0,91 vs 0,40: vượt ~0,51
        + topic_rows("facility", 40)  # tỉ trọng
    )
    alerts = build(rows).alerts
    assert len(alerts) == 3
    margins = [a.evidence.ci_low - a.evidence.baseline for a in alerts]
    assert margins == sorted(margins, reverse=True)
    assert alerts[0].topic == "others"


def test_group_alert_when_only_one_group_is_high_and_the_overall_rate_is_not():
    rows = [
        *topic_rows("lecturer", 20, 0, group="A"),  # 20/20 tiêu cực
        *topic_rows("lecturer", 20, 80, group="B"),  # 20%
    ]
    out = build(rows)
    (alert,) = out.alerts
    assert (alert.kind, alert.group, alert.topic) == ("negative_rate", "A", "lecturer")
    assert alert.title == "Giảng dạy ở nhóm A: tỉ lệ tiêu cực cao hơn mức nền"
    assert "ở nhóm A" in alert.summary and alert.evidence.n == 20
    assert alert.note == "Tổng hợp theo nhóm, không gắn với cá nhân."


def test_topic_alert_names_the_standout_group_and_says_when_all_groups_are_high():
    rows = [*topic_rows("lecturer", 45, 5, group="A"), *topic_rows("lecturer", 30, 20, group="B")]
    (alert,) = build(rows).alerts
    assert alert.group is None
    assert alert.highlight is not None and alert.highlight.group == "A"
    assert (alert.highlight.k, alert.highlight.n) == (45, 50)
    assert "Cao ở cả 2 nhóm, nhiều nhất ở nhóm A (45/50 câu)." in alert.summary


def test_a_group_below_baseline_stops_the_all_groups_claim():
    # Chung 110/200 = 55% vẫn vượt nền, nhưng nhóm B chỉ 20% nên không được nói "cao ở cả hai nhóm".
    rows = [*topic_rows("lecturer", 90, 10, group="A"), *topic_rows("lecturer", 20, 80, group="B")]
    (alert,) = build(rows).alerts
    assert alert.group is None
    assert "Cao ở cả" not in alert.summary and "Nhiều nhất ở nhóm A (90/100 câu)." in alert.summary


# --- Nhóm và heatmap ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grouped() -> list[InsightRow]:
    return [
        *topic_rows("lecturer", 10, 15, group="Nhóm A"),
        *topic_rows("training_program", 6, 2, group="Nhóm A"),
        *topic_rows("lecturer", 5, 10, group="Nhóm B"),
        *topic_rows("facility", 3, 0, group="Nhóm B"),
    ]


def test_groups_are_listed_with_their_sizes_largest_first(grouped):
    out = build(grouped)
    assert [(g.name, g.n) for g in out.groups] == [("Nhóm A", 33), ("Nhóm B", 18)]
    assert out.selected_groups is None and out.kpis.n_groups == 2 and out.kpis.n_rows == 51


def test_filtering_by_group_narrows_every_number_but_keeps_the_options(grouped):
    out = build(grouped, groups=["Nhóm B"])
    assert out.selected_groups == ["Nhóm B"]
    assert out.kpis.n_rows == 18 and out.kpis.n_groups == 1
    assert [(g.name, g.n) for g in out.groups] == [("Nhóm A", 33), ("Nhóm B", 18)]
    assert [t.n for t in out.topics] == [15, 0, 3, 0]
    assert out.heatmap.groups == ["Nhóm B"]


def test_several_groups_unknown_names_and_duplicates(grouped):
    out = build(grouped, groups=["Nhóm A", "Nhóm B", "Nhóm A"])
    assert out.selected_groups == ["Nhóm A", "Nhóm B"] and out.kpis.n_rows == 51
    mixed = build(grouped, groups=["Nhóm B", "Không có"])
    assert mixed.selected_groups == ["Nhóm B"] and mixed.kpis.n_rows == 18


def test_a_filter_matching_no_group_is_ignored(grouped):
    out = build(grouped, groups=["Không có"])
    assert out.selected_groups is None and out.kpis.n_rows == 51
    assert build(grouped, groups=[]).kpis.n_rows == 51


def test_a_filter_is_ignored_when_the_file_has_no_groups(hundred):
    out = build(hundred, groups=["Nhóm A"])
    assert out.selected_groups is None and out.groups is None and out.heatmap is None
    assert out.kpis.n_rows == 100 and out.kpis.n_groups is None


def test_heatmap_cells_show_rate_delta_and_whether_to_shade(grouped):
    heatmap = build(grouped).heatmap
    assert heatmap.groups == ["Nhóm A", "Nhóm B"] and heatmap.topics == list(TOPIC)
    assert heatmap.min_n == 20
    assert heatmap.baseline_negative_rate == pytest.approx(
        [round(BASELINE_NEGATIVE[t], 4) for t in TOPIC]
    )
    cell = heatmap.cells[0][0]  # Nhóm A × giảng viên: 25 câu, 10 tiêu cực
    assert (cell.n, cell.negative_rate, cell.shaded) == (25, 0.4, True)
    assert cell.delta_points == pytest.approx((0.4 - BASELINE_NEGATIVE["lecturer"]) * 100, abs=0.1)
    small = heatmap.cells[0][1]  # Nhóm A × chương trình: 8 câu
    assert (small.n, small.shaded) == (8, False) and small.negative_rate == 0.75
    empty = heatmap.cells[0][2]  # Nhóm A × cơ sở vật chất: không có câu
    assert (empty.n, empty.negative_rate, empty.delta_points, empty.shaded) == (
        0,
        None,
        None,
        False,
    )
    assert len(heatmap.cells) == 2 and all(len(line) == 4 for line in heatmap.cells)


# --- Từ khoá --------------------------------------------------------------------------------------


def facility_story() -> list[InsightRow]:
    """30 câu tiêu cực về máy chiếu, 12 về phòng máy, rải rác vài từ hiếm; nền là câu khác."""
    rows = many(30, "máy chiếu mờ số {i} quá", topic="facility", model_text="máy_chiếu mờ rất")
    rows += many(12, "phòng máy nóng số {i} quá", topic="facility", model_text="phòng máy nóng rất")
    rows += many(2, "quạt hỏng số {i} quá", topic="facility", model_text="quạt hỏng rất")  # df = 2
    rows += many(
        200,
        "thầy dạy rất hay số {i}",
        topic="lecturer",
        sentiment="positive",
        model_text="thầy dạy rất hay",
    )
    return rows


def keywords_of(out: InsightsResponse, topic: str):
    return next(b for b in out.keywords if b.topic == topic)


def test_keywords_come_from_negative_sentences_of_the_topic_and_show_document_counts():
    block = keywords_of(build(facility_story()), "facility")
    terms = {k.term: k for k in block.keywords}
    assert block.n_negative == 44
    # `_` hiện thành dấu cách; n = số câu có từ. "mờ" luôn đi sau "máy chiếu" nên cụm hai từ
    # nói rõ hơn và thay cho cả hai đơn từ.
    assert terms["máy chiếu mờ"].n == 30 and "máy chiếu" not in terms and "mờ" not in terms
    assert terms["phòng máy"].n == 12
    assert "quạt" not in terms  # chỉ 2 câu < MIN_KEYWORD_DF
    scores = [k.score for k in block.keywords]
    assert scores == sorted(scores, reverse=True) and all(
        s >= insights.MIN_KEYWORD_Z for s in scores
    )


def test_common_words_that_everything_uses_are_not_keywords():
    block = keywords_of(build(facility_story()), "facility")
    assert "rất" not in {k.term for k in block.keywords}


def test_a_single_word_inside_a_better_phrase_gives_way_to_the_phrase():
    # "phòng" chỉ xuất hiện trong "phòng máy": giữ cụm, bỏ đơn từ.
    terms = {k.term for k in keywords_of(build(facility_story()), "facility").keywords}
    assert "phòng máy" in terms and "phòng" not in terms
    # Nhưng "phòng" xuất hiện nhiều nơi khác ngoài cụm thì được giữ.
    spread = []
    for second in ("học", "thi", "lab", "ồn"):
        spread += many(
            10, f"phòng {second} số {{i}}", topic="facility", model_text=f"phòng {second} rộng"
        )
    terms = {k.term for k in keywords_of(build(facility_story() + spread), "facility").keywords}
    assert "phòng" in terms


def test_keywords_are_limited_to_eight_and_only_for_three_topics():
    rows = []
    for i in range(20):
        rows += many(5, f"câu {i}" + " số {i}", topic="facility", model_text=f"từ{i}a từ{i}b")
    rows += many(100, "nền {i}", topic="lecturer", sentiment="positive", model_text="chung chung")
    out = build(rows)
    assert len(keywords_of(out, "facility").keywords) == insights.MAX_KEYWORDS
    assert [b.topic for b in out.keywords] == ["lecturer", "training_program", "facility"]


def test_a_topic_without_negative_sentences_has_no_keywords():
    block = keywords_of(build(facility_story()), "training_program")
    assert block.n_negative == 0 and block.keywords == []


# --- Câu tiêu biểu -------------------------------------------------------------------------------


def quotes_of(out: InsightsResponse, topic: str):
    return next(b for b in out.quotes if b.topic == topic).quotes


def test_quotes_are_negative_sentences_of_the_topic_ranked_by_keyword_strength():
    out = build(facility_story())
    quotes = quotes_of(out, "facility")
    assert quotes and all(q.sentiment == "negative" for q in quotes)
    assert quotes[0].matched  # câu xếp đầu chứa từ khoá
    assert all(
        set(q.matched) <= {k.term for k in keywords_of(out, "facility").keywords} | set(q.matched)
        for q in quotes
    )
    assert quotes_of(out, "lecturer") == []  # câu về giảng viên đều tích cực


def test_quotes_need_enough_confidence_in_both_heads_and_a_reasonable_length():
    base = dict(topic="facility", model_text="máy_chiếu mờ rất")
    good = row("máy chiếu mờ không nhìn rõ chữ", **base)
    low_sentiment = row("máy chiếu rất mờ nên khó nhìn", confidence=0.6, **base)
    low_topic = row("máy chiếu mờ khó nhìn thấy gì", topic_confidence=0.5, **base)
    short = row("máy chiếu mờ", **base)
    long = row(" ".join(["mờ"] * 40), **base)
    others = many(60, "nền {i} khác hẳn nhau", topic="lecturer", sentiment="positive")
    texts = [
        q.text
        for q in quotes_of(
            build([good, low_sentiment, low_topic, short, long, *others]), "facility"
        )
    ]
    assert texts == [good.display_text]


def test_near_duplicate_quotes_are_dropped():
    base = dict(topic="facility", model_text="máy_chiếu mờ")
    a = row("máy chiếu mờ nên không nhìn rõ chữ trên bảng", **base)
    b = row("máy chiếu mờ nên không nhìn rõ chữ trên bảng!", **base)  # gần như trùng
    c = row(
        "phòng học nóng quạt kêu to ảnh hưởng buổi học",
        topic="facility",
        model_text="phòng nóng quạt",
    )
    texts = [
        q.text
        for q in quotes_of(
            build([a, b, c, *many(50, "nền {i} khác", topic="lecturer", sentiment="positive")]),
            "facility",
        )
    ]
    assert len(texts) == 2 and c.display_text in texts


def test_sentences_that_still_carry_a_name_after_an_address_word_are_not_quoted():
    named = row(
        "Thầy Hùng giảng quá nhanh nên em khó theo kịp bài", "lecturer", model_text="a b c d"
    )
    masked = row(
        "Thầy [TÊN] giảng quá nhanh nên em khó theo kịp bài", "lecturer", model_text="e f g h"
    )
    plain = row("Cô giảng quá nhanh nên em khó theo kịp bài học", "lecturer", model_text="i j k l")
    texts = [
        q.text
        for q in quotes_of(
            build([named, masked, plain, *many(40, "nền {i} khác", topic="facility")]), "lecturer"
        )
    ]
    assert named.display_text not in texts
    assert masked.display_text in texts and plain.display_text in texts


@pytest.mark.parametrize(
    ("text", "suspect"),
    [
        ("Thầy Hùng dạy nhanh quá", True),
        ("em thấy cô Lan giảng hay", True),
        ("Giảng viên Nguyễn Văn An rất nhiệt tình", True),
        ("bạn Hà hay đi trễ", True),
        ("Thầy dạy rất nhanh", False),
        ("Thầy [TÊN] dạy rất nhanh", False),
        ("cô giáo rất nhiệt tình", False),
        ("Anh ấy giảng hay", False),
    ],
)
def test_name_suspicion(text: str, suspect: bool):
    assert insights._suspects_a_name(text) is suspect


def test_the_quote_pool_is_capped_at_fifty():
    rows = [
        row(
            f"máy chiếu mờ nên em thấy rất khó nhìn chữ số {i} {'x' * i}",
            topic="facility",
            model_text=f"máy_chiếu mờ khác{i} lạ{i}",
        )
        for i in range(80)
    ]
    assert len(quotes_of(build(rows), "facility")) == insights.QUOTE_POOL


def test_quotes_of_the_other_topic_include_every_sentiment_and_rank_by_confidence():
    rows = [
        row("em không có ý kiến gì thêm về môn này", "others", "neutral", confidence=0.95),
        row("mọi thứ đều rất ổn với em cả học kỳ", "others", "positive", confidence=0.8),
        row("em thấy hơi chán vì không có gì mới lạ", "others", "negative", confidence=0.9),
    ]
    quotes = quotes_of(build(rows), "others")
    assert [q.sentiment for q in quotes] == ["neutral", "negative", "positive"]
    assert all(q.matched == [] for q in quotes)


def test_quotes_carry_their_group_and_a_rounded_confidence():
    rows = many(
        40,
        "máy chiếu mờ số {i} khó nhìn",
        topic="facility",
        confidence=0.91234567,
        group="Nhóm A",
        model_text="máy_chiếu mờ",
    )
    rows += many(
        40, "thầy dạy tốt số {i} lắm", topic="lecturer", sentiment="positive", group="Nhóm B"
    )
    first = quotes_of(build(rows), "facility")[0]
    assert first.group == "Nhóm A" and first.confidence == 0.9123


# --- Khuyến nghị và dữ liệu thật -----------------------------------------------------------------


def test_recommendations_are_built_from_the_negative_sentences_in_scope():
    rows = many(12, "Máy chiếu mờ số {i} quá", topic="facility", group="Nhóm A")
    rows += many(12, "Quạt hỏng số {i} quá", topic="facility", group="Nhóm B")
    out = build(rows, groups=["Nhóm A"])
    assert [r.id for r in out.recommendations] == ["facility_projector"]
    assert out.recommendations[0].n == 12


def test_the_illustrative_flag_and_source_are_passed_through(hundred):
    out = build(hundred, source="demo", illustrative_groups=True)
    assert out.source == "demo" and out.illustrative_groups is True


# --- Dữ liệu thật ---------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_background() -> InsightBackground:
    from edupulse.engine.bundle import load_bundle, load_insight_background

    return load_insight_background(load_bundle(load_settings().bundle_dir, verify_files=False))


@pytest.mark.data
@pytest.mark.model
def test_keywords_on_the_real_training_set_have_the_right_sign(real_background):
    from ml.common import load_processed

    train = load_processed("train")
    rows = [
        InsightRow(
            display_text=text,
            model_text=text_rdr,
            sentiment=SENTIMENT[sentiment],
            sentiment_confidence=1.0,
            topic=TOPIC[topic],
            topic_confidence=1.0,
            n_words=len(text_rdr.split()),
            group=None,
            flagged=None,
        )
        for text, text_rdr, sentiment, topic in zip(
            train["sentence"], train["text_rdr"], train["sentiment"], train["topic"], strict=True
        )
    ]
    out = insights.build_insights(rows, real_background, source="batch", illustrative_groups=False)
    facility = {k.term for k in keywords_of(out, "facility").keywords}
    assert {"phòng", "máy chiếu"} <= facility or {"phòng", "phòng học"} <= facility
    lecturer = {k.term for k in keywords_of(out, "lecturer").keywords}
    assert "không" in lecturer
    program = {k.term for k in keywords_of(out, "training_program").keywords}
    assert {"thực hành", "lý thuyết"} & program
    # Dữ liệu train chính là mức nền: không có gì vượt nền.
    assert out.alerts == []
    assert out.kpis.shares.negative == pytest.approx(SENTIMENT_SHARE["negative"], abs=1e-3)


@pytest.mark.model
def test_the_demo_predictions_give_a_complete_dashboard(real_background):
    from edupulse.demo_data import load_demo

    settings = load_settings()
    demo = load_demo(settings.demo_dir, real_background and settings.bundle_version)
    if not demo.available:
        pytest.skip("không có dữ liệu demo — chạy `make demo-data`")
    rows = [
        InsightRow(
            display_text=r.display_text,
            model_text=r.model_text,
            sentiment=r.sentiment,
            sentiment_confidence=r.sentiment_confidence,
            topic=r.topic,
            topic_confidence=r.topic_confidence,
            n_words=r.n_words,
            group=None,
            flagged=None,
        )
        for r in demo.rows
    ]
    out = insights.build_insights(rows, real_background, source="demo", illustrative_groups=False)
    assert out.kpis.n_rows == len(rows) == 3166
    assert [t.n for t in out.topics] and sum(t.n for t in out.topics) == 3166
    assert all(quotes_of(out, t) for t in TOPIC)
    assert out.recommendations
    for block in out.keywords:
        assert block.keywords, block.topic
    assert all("[TÊN]" in q.text or "wzjwz" not in q.text for b in out.quotes for q in b.quotes)
