"""Luật khuyến nghị: khớp nguyên từ, ngưỡng, căn cứ và ghi chú."""

import re
from pathlib import Path

import pytest

from edupulse.config import load_settings
from edupulse.insights.rows import InsightRow
from edupulse.insights.rules import (
    LECTURER_NOTE,
    MAX_RECOMMENDATIONS,
    MIN_RULE_ROWS,
    RULES,
    Rule,
    match_rule,
    recommend,
)
from edupulse.labels import SENTIMENT, TOPIC

BY_ID = {rule.id: rule for rule in RULES}


def row(text: str, topic: str = "facility", group: str | None = None) -> InsightRow:
    return InsightRow(
        display_text=text,
        model_text=text.lower(),
        sentiment="negative",
        sentiment_confidence=0.9,
        topic=topic,
        topic_confidence=0.9,
        n_words=len(text.split()),
        group=group,
        flagged=None,
    )


def rows(texts: list[str], topic: str = "facility", group: str | None = None):
    return [row(t, topic, group) for t in texts]


# --- Hình dạng của bộ luật ----------------------------------------------------------------------


def test_there_are_about_a_dozen_rules_with_unique_ids():
    assert 10 <= len(RULES) <= 14
    assert len({r.id for r in RULES}) == len(RULES)


def test_every_rule_names_a_known_topic_an_action_and_terms():
    for rule in RULES:
        assert rule.topic in TOPIC and rule.topic != "others"
        assert rule.action and rule.terms and len(set(rule.terms)) == len(rule.terms)


def test_only_lecturer_rules_carry_the_group_level_note():
    for rule in RULES:
        assert (rule.note == LECTURER_NOTE) == (rule.topic == "lecturer"), rule.id
    assert "không gắn với cá nhân" in LECTURER_NOTE


# --- Khớp -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rule_id", "text", "expected"),
    [
        ("program_exam", "Em thấy bài thi cuối kỳ quá khó.", True),
        ("program_exam", "Thi vào buổi tối rất bất tiện.", True),
        ("program_exam", "Cần nâng cấp thiết bị trong lớp.", False),  # "thi" không khớp "thiết"
        ("program_exam", "Môn này thích hợp với sinh viên.", False),  # cũng không khớp "thích"
        ("facility_projector", "MÁY CHIẾU quá MỜ.", True),  # không phân biệt hoa/thường
        ("facility_projector", "Loa trong phòng rè.", True),
        ("facility_projector", "Ánh sáng làm chói mắt.", False),
        ("facility_ventilation", "Phòng thiếu điều hoà nên rất nóng.", True),
        ("facility_ventilation", "Phòng thiếu điều hòa.", True),
        ("facility_network", "Wi-fi yếu, mạng chập chờn.", True),
        ("facility_network", "Em đến phòng mạng máy tính.", True),
        ("lecturer_pace", "Thầy giảng nhanh quá, khó hiểu.", True),
        ("lecturer_pace", "Thầy dạy rất tận tình.", False),
        ("lecturer_absence", "Thầy hay nghỉ mà không thông báo.", True),
        ("lecturer_absence", "Bài tập tuần này khá nhiều.", False),
        ("program_materials", "Giáo trình đã cũ, cần cập nhật.", True),
    ],
)
def test_rules_match_whole_words_only(rule_id: str, text: str, expected: bool):
    assert bool(BY_ID[rule_id].pattern.search(text)) is expected


def test_longest_alternative_wins_so_terms_are_reported_whole():
    found = BY_ID["facility_projector"].pattern.search("Máy chiếu mờ.")
    assert found is not None and found.group(0) == "Máy chiếu"


def test_match_counts_sentences_not_occurrences():
    texts = ["Máy chiếu mờ, mờ lắm, máy chiếu cũ.", "Máy chiếu hỏng.", "Phòng sạch sẽ."]
    match = match_rule(BY_ID["facility_projector"], rows(texts))
    assert match is not None and match.rows == 2
    assert dict(match.terms) == {"máy chiếu": 2, "mờ": 1}


def test_terms_are_ranked_by_how_many_sentences_use_them():
    texts = ["Máy chiếu mờ.", "Máy chiếu hỏng.", "Máy chiếu cũ.", "Loa rè."]
    match = match_rule(BY_ID["facility_projector"], rows(texts))
    assert [t for t, _ in match.terms][:2] == ["máy chiếu", "loa"]


def test_no_match_returns_none():
    assert match_rule(BY_ID["facility_projector"], rows(["Phòng sạch sẽ."])) is None
    assert match_rule(BY_ID["facility_projector"], []) is None


def test_the_busiest_group_is_reported_when_the_file_has_groups():
    data = rows(["Máy chiếu mờ."] * 5, group="A") + rows(["Máy chiếu hỏng."], group="B")
    match = match_rule(BY_ID["facility_projector"], data)
    assert match.top_group == ("A", 5)


# --- Khuyến nghị ----------------------------------------------------------------------------------


def projector_rows(n: int, group: str | None = None) -> list[InsightRow]:
    return rows(["Máy chiếu quá mờ, nhìn không rõ."] * n, group=group)


def test_evidence_states_the_count_topic_and_matched_terms():
    data = rows(["Máy chiếu mờ."] * 15 + ["Loa rè, micro hỏng."] * 7)
    (rec,) = recommend({"facility": data}, n_groups=0)
    assert rec.id == "facility_projector" and rec.n == 22
    assert rec.action == "Bảo trì thiết bị trình chiếu và âm thanh ở phòng học"
    assert (
        rec.evidence == "Căn cứ: 22 câu tiêu cực về Cơ sở vật chất nhắc “máy chiếu”, “mờ”, “loa”."
    )
    assert rec.terms == ["máy chiếu", "mờ", "loa"]
    assert rec.note is None and rec.topic == "facility"


def test_every_recommendation_has_a_basis_line():
    negatives = {
        "facility": projector_rows(10),
        "training_program": rows(["Quá nhiều bài tập và đồ án."] * 10, "training_program"),
        "lecturer": rows(["Thầy giảng nhanh, khó hiểu."] * 10, "lecturer"),
    }
    recs = recommend(negatives, n_groups=0)
    assert len(recs) == 3
    assert all(r.evidence.startswith("Căn cứ: ") and str(r.n) in r.evidence for r in recs)


def test_lecturer_recommendations_speak_of_teaching_and_never_of_a_person():
    (rec,) = recommend({"lecturer": rows(["Thầy giảng nhanh quá."] * 6, "lecturer")}, n_groups=0)
    assert "về giảng dạy" in rec.evidence and "Giảng viên" not in rec.evidence
    assert rec.note == LECTURER_NOTE


def test_a_rule_needs_at_least_three_matching_sentences():
    assert recommend({"facility": projector_rows(MIN_RULE_ROWS - 1)}, n_groups=0) == []
    assert len(recommend({"facility": projector_rows(MIN_RULE_ROWS)}, n_groups=0)) == 1


def test_a_rule_needs_five_percent_of_the_topic_negatives():
    filler = rows(["Phòng học bình thường."] * 100)
    assert recommend({"facility": [*projector_rows(4), *filler]}, n_groups=0) == []  # 4/104 < 5%
    assert len(recommend({"facility": [*projector_rows(6), *filler]}, n_groups=0)) == 1  # 6/106


def test_a_rule_only_sees_sentences_of_its_own_topic():
    mixed = {"training_program": projector_rows(10)}  # nói về máy chiếu nhưng thuộc chủ đề khác
    assert recommend(mixed, n_groups=0) == []


def test_recommendations_are_ordered_by_evidence_and_capped():
    negatives = {
        "facility": [
            *projector_rows(5),
            *rows(["Quạt hỏng, phòng nóng."] * 9),
            *rows(["Wifi yếu."] * 7),
            *rows(["Máy tính cấu hình thấp."] * 3),
            *rows(["Phòng chật, ghế gãy."] * 4),
        ],
        "training_program": rows(["Đề thi khó."] * 8 + ["Giáo trình cũ."] * 6, "training_program"),
    }
    recs = recommend(negatives, n_groups=0)
    counts = [r.n for r in recs]
    assert counts == sorted(counts, reverse=True)
    assert len(recs) <= MAX_RECOMMENDATIONS
    assert recs[0].id == "facility_ventilation"
    assert len(recommend(negatives, n_groups=0, limit=2)) == 2


def test_ties_keep_the_order_of_the_rule_table():
    negatives = {"facility": [*projector_rows(5), *rows(["Quạt hỏng."] * 5)]}
    ids = [r.id for r in recommend(negatives, n_groups=0)]
    assert ids == ["facility_projector", "facility_ventilation"]


def test_the_busiest_group_is_named_only_when_it_stands_out():
    stands_out = projector_rows(8, "Nhóm A") + projector_rows(2, "Nhóm B")
    (rec,) = recommend({"facility": stands_out}, n_groups=2)
    assert rec.evidence.endswith("Nhiều nhất ở Nhóm A (8 câu).")
    assert "nhóm Nhóm" not in rec.evidence  # tên nhóm giữ nguyên, không thêm chữ "nhóm"
    even = projector_rows(4, "A") + projector_rows(4, "B") + projector_rows(4, "C")
    (rec,) = recommend({"facility": even}, n_groups=3)
    assert "Nhiều nhất" not in rec.evidence  # 4/12 < 40%
    (rec,) = recommend({"facility": stands_out}, n_groups=1)
    assert "Nhiều nhất" not in rec.evidence  # chỉ một nhóm: không có gì để so
    few = projector_rows(4, "A") + projector_rows(1, "B")
    (rec,) = recommend({"facility": few}, n_groups=2)
    assert "Nhiều nhất" not in rec.evidence  # nhóm A chỉ 4 câu < 5


def test_no_rows_no_recommendations():
    assert recommend({}, n_groups=0) == []
    assert recommend({"facility": []}, n_groups=0) == []


# --- Trên câu thật (cần `make data`) --------------------------------------------------------------


@pytest.fixture(scope="module")
def train_negatives() -> dict[str, list[str]]:
    path = Path(load_settings().data_dir / "raw" / "train.parquet")
    if not path.is_file():
        pytest.skip("chưa có data/raw — chạy `make data`")
    pq = pytest.importorskip("pyarrow.parquet")

    table = pq.read_table(path)
    out: dict[str, list[str]] = {t: [] for t in TOPIC}
    for sentence, sentiment, topic in zip(
        table.column("sentence").to_pylist(),
        table.column("sentiment").to_pylist(),
        table.column("topic").to_pylist(),
        strict=True,
    ):
        if SENTIMENT[sentiment] == "negative":
            out[TOPIC[topic]].append(sentence)
    return out


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.id)
def test_every_rule_fires_on_real_negative_sentences_of_its_topic(rule: Rule, train_negatives):
    sentences = train_negatives[rule.topic]
    hits = sum(1 for s in sentences if rule.pattern.search(s))
    assert hits >= 15, f"{rule.id}: chỉ {hits}/{len(sentences)} câu khớp"


def test_real_data_gives_recommendations_for_every_topic(train_negatives):
    negatives = {
        topic: [row(s, topic) for s in sentences]
        for topic, sentences in train_negatives.items()
        if topic != "others"
    }
    recs = recommend(negatives, n_groups=0, limit=20)
    assert {r.topic for r in recs} == {"facility", "training_program", "lecturer"}
    assert all(
        re.fullmatch(r"Căn cứ: \d+ câu tiêu cực về .+ nhắc “.+”\.", r.evidence) for r in recs
    )
