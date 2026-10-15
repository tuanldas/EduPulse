"""Thứ tự nhãn UIT-VSFC và tên hiển thị tiếng Việt.

Vị trí trong tuple chính là id gốc (0..n-1) của bộ dữ liệu. Huấn luyện, bundle và API đều dùng
chung các hằng này để chỉ số lớp không bao giờ lệch nhau.
"""

from typing import Final

SENTIMENT: Final = ("negative", "neutral", "positive")
TOPIC: Final = ("lecturer", "training_program", "facility", "others")

TASKS: Final = ("sentiment", "topic")
LABELS: Final = {"sentiment": SENTIMENT, "topic": TOPIC}

# Thuật ngữ cố định của giao diện (docs/design-guidelines.md, mục "Thuật ngữ").
SENTIMENT_VI: Final = {"negative": "Tiêu cực", "neutral": "Trung tính", "positive": "Tích cực"}
TOPIC_VI: Final = {
    "lecturer": "Giảng viên",
    "training_program": "Chương trình",
    "facility": "Cơ sở vật chất",
    "others": "Khác",
}
LABELS_VI: Final = {"sentiment": SENTIMENT_VI, "topic": TOPIC_VI}
