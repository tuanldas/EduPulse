"""Một dòng đầu vào của insight: đủ để tính mọi thứ ở dashboard mà không cần mô hình."""

from dataclasses import dataclass

from edupulse.insights.models import SentimentLabel, TopicLabel


@dataclass(frozen=True, slots=True)
class InsightRow:
    display_text: str  # câu đã ẩn thông tin cá nhân
    model_text: str  # đầu ra của `prepare()`: đếm n-gram trên chuỗi này, như lúc đếm nền
    sentiment: SentimentLabel
    sentiment_confidence: float
    topic: TopicLabel
    topic_confidence: float
    n_words: int
    group: str | None
    flagged: bool | None  # None: nguồn không có bước gắn cờ (dữ liệu demo)
