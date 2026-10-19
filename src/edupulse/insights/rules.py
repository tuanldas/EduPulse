"""Khuyến nghị theo luật: chủ đề + từ khoá trong câu tiêu cực -> hành động gợi ý.

Mỗi luật gồm một chủ đề, danh sách từ/cụm từ và một hành động. Luật khớp bằng biểu thức chính quy
trên câu đã ẩn thông tin cá nhân (không cần tách từ, không phân biệt hoa/thường, khớp nguyên từ nên
"thi" không khớp "thiết bị"). Một luật chỉ được nêu khi đủ số câu tiêu cực khớp và luôn kèm căn cứ
("Căn cứ: n câu…"); khuyến nghị chỉ để gợi ý, người đọc phải đọc câu gốc trước khi quyết định.

Luật về giảng dạy mang ghi chú "Tổng hợp theo nhóm, không gắn với cá nhân" (design-guidelines §8).
"""

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from edupulse.insights.models import Recommendation, TopicLabel
from edupulse.insights.rows import InsightRow

MIN_RULE_ROWS = 3  # số câu tiêu cực khớp tối thiểu để nêu một khuyến nghị
MIN_RULE_SHARE = 0.05  # và tối thiểu bằng ngần này số câu tiêu cực của chủ đề
MAX_RECOMMENDATIONS = 6
MAX_EVIDENCE_TERMS = 3
GROUP_MENTION_SHARE = 0.4  # nêu nhóm "nhiều nhất" khi nhóm đó chiếm ít nhất ngần này số câu khớp
GROUP_MENTION_MIN_ROWS = 5

TOPIC_PHRASE: dict[str, str] = {
    "facility": "Cơ sở vật chất",
    "training_program": "Chương trình",
    "lecturer": "giảng dạy",
    "others": "chủ đề Khác",
}
LECTURER_NOTE = "Tổng hợp theo nhóm, không gắn với cá nhân."


def _pattern(terms: Sequence[str]) -> re.Pattern[str]:
    # Dài trước để "máy chiếu" thắng "máy"; (?<!\w)/(?!\w) = ranh giới từ hiểu cả chữ có dấu.
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", re.IGNORECASE)


@dataclass(frozen=True)
class Rule:
    id: str
    topic: TopicLabel
    action: str
    terms: tuple[str, ...]
    note: str | None = None
    pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "pattern", _pattern(self.terms))


RULES: tuple[Rule, ...] = (
    # Cơ sở vật chất
    Rule(
        "facility_projector",
        "facility",
        "Bảo trì thiết bị trình chiếu và âm thanh ở phòng học",
        ("máy chiếu", "mờ", "loa", "micro", "âm thanh"),
    ),
    Rule(
        "facility_ventilation",
        "facility",
        "Kiểm tra thông gió và làm mát phòng học",
        ("quạt", "nóng", "điều hòa", "điều hoà", "máy lạnh"),
    ),
    Rule(
        "facility_computers",
        "facility",
        "Lập kế hoạch nâng cấp phòng máy và thiết bị thực hành",
        ("máy tính", "thiết bị", "cấu hình", "phòng máy", "lab"),
    ),
    Rule(
        "facility_network",
        "facility",
        "Kiểm tra hạ tầng mạng ở khu học tập",
        ("mạng", "wifi", "wi-fi", "internet"),
    ),
    Rule(
        "facility_space",
        "facility",
        "Rà soát không gian và bàn ghế phòng học",
        ("chật", "bàn ghế", "ghế", "phòng nhỏ", "phòng bé", "ồn"),
    ),
    # Chương trình
    Rule(
        "program_practice",
        "training_program",
        "Rà soát tỉ lệ lý thuyết – thực hành và tiến độ các môn",
        ("thực hành", "lý thuyết"),
    ),
    Rule(
        "program_exam",
        "training_program",
        "Xem lại hình thức và đề thi",
        ("đề thi", "cuối kỳ", "kiểm tra", "hình thức thi", "thi"),
    ),
    Rule(
        "program_materials",
        "training_program",
        "Cập nhật giáo trình, tài liệu và đề cương môn học",
        ("giáo trình", "tài liệu", "cập nhật", "đề cương", "nội dung"),
    ),
    Rule(
        "program_workload",
        "training_program",
        "Cân đối khối lượng bài tập, đồ án và thời gian học",
        ("thời gian", "đồ án", "bài tập", "quá tải", "deadline"),
    ),
    # Giảng dạy
    Rule(
        "lecturer_pace",
        "lecturer",
        "Trao đổi với bộ môn về nhịp giảng và cách giải thích",
        ("nhanh", "khó hiểu", "chậm"),
        LECTURER_NOTE,
    ),
    Rule(
        "lecturer_absence",
        "lecturer",
        "Nhắc lại quy trình báo nghỉ và dạy bù",
        ("nghỉ", "trễ", "thông báo", "bù"),
        LECTURER_NOTE,
    ),
    Rule(
        "lecturer_engagement",
        "lecturer",
        "Tăng tương tác và ví dụ trong bài giảng",
        ("slide", "buồn ngủ", "ví dụ", "tương tác"),
        LECTURER_NOTE,
    ),
)


@dataclass(frozen=True, slots=True)
class RuleMatch:
    rule: Rule
    rows: int  # số câu tiêu cực khớp
    terms: list[tuple[str, int]]  # (từ, số câu có từ đó), nhiều nhất trước
    top_group: tuple[str, int] | None  # nhóm có nhiều câu khớp nhất, nếu tập có nhóm


def match_rule(rule: Rule, rows: Sequence[InsightRow]) -> RuleMatch | None:
    """Số câu trong `rows` (câu tiêu cực của đúng chủ đề) khớp luật; None nếu không câu nào."""
    matched_rows = 0
    terms: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    for row in rows:
        found = {m.group(0).lower() for m in rule.pattern.finditer(row.display_text)}
        if not found:
            continue
        matched_rows += 1
        terms.update(found)
        if row.group is not None:
            groups[row.group] += 1
    if matched_rows == 0:
        return None
    ranked = sorted(terms.items(), key=lambda item: (-item[1], item[0]))
    top_group = max(groups.items(), key=lambda item: (item[1], item[0])) if groups else None
    return RuleMatch(rule, matched_rows, ranked, top_group)


def _quote(term: str) -> str:
    return f"“{term}”"


def _evidence(match: RuleMatch, n_groups: int) -> str:
    topic = TOPIC_PHRASE[match.rule.topic]
    mentioned = ", ".join(_quote(t) for t, _ in match.terms[:MAX_EVIDENCE_TERMS])
    text = f"Căn cứ: {match.rows} câu tiêu cực về {topic} nhắc {mentioned}."
    if match.top_group is not None and n_groups >= 2:
        group, count = match.top_group
        if count >= GROUP_MENTION_MIN_ROWS and count >= GROUP_MENTION_SHARE * match.rows:
            text += f" Nhiều nhất ở {group} ({count} câu)."
    return text


def recommend(
    negatives: dict[str, Sequence[InsightRow]], n_groups: int, limit: int = MAX_RECOMMENDATIONS
) -> list[Recommendation]:
    """Khuyến nghị từ các câu tiêu cực của từng chủ đề (`negatives[chủ đề]`), nhiều căn cứ nhất
    trước. `n_groups` là số nhóm trong tập (0 hoặc 1: không nêu nhóm)."""
    matches: list[tuple[int, RuleMatch]] = []
    for position, rule in enumerate(RULES):
        topic_rows = negatives.get(rule.topic, ())
        match = match_rule(rule, topic_rows)
        if match is None:
            continue
        if match.rows < MIN_RULE_ROWS or match.rows < MIN_RULE_SHARE * len(topic_rows):
            continue
        matches.append((position, match))
    matches.sort(key=lambda item: (-item[1].rows, item[0]))
    return [
        Recommendation(
            id=match.rule.id,
            topic=match.rule.topic,
            action=match.rule.action,
            evidence=_evidence(match, n_groups),
            n=match.rows,
            terms=[t for t, _ in match.terms[:MAX_EVIDENCE_TERMS]],
            note=match.rule.note,
        )
        for _, match in matches[:limit]
    ]
