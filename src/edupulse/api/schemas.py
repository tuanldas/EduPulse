"""Mô hình dữ liệu của API: lỗi, sức khoẻ, phân tích một câu và giải thích.

Giá trị phân loại (`Literal`) khoá bằng tiếng Anh; giao diện ánh xạ sang tiếng Việt. Thứ tự và tên
nhãn lấy từ `edupulse.labels` (có test canh các `Literal` dưới đây khớp với nó). Xác suất làm tròn
4 chữ số thập phân.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, StringConstraints

from edupulse.bundle_schema import Stage
from edupulse.config import MAX_CHARS

SentimentLabel = Literal["negative", "neutral", "positive"]
TopicLabel = Literal["lecturer", "training_program", "facility", "others"]
Task = Literal["sentiment", "topic"]
PiiType = Literal["name", "phone", "email", "url", "student_id", "national_id"]
ReasonCode = Literal["low_confidence", "no_diacritics"]

# Bỏ khoảng trắng hai đầu rồi mới đo độ dài, nên câu toàn dấu cách bị từ chối như câu rỗng.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_CHARS)]


# --- Lỗi và sức khoẻ ----------------------------------------------------------------------------


class FieldError(BaseModel):
    """Một trường không hợp lệ; không kèm giá trị gửi lên để không lặp lại nội dung người dùng."""

    loc: list[str | int]
    type: str


class ErrorBody(BaseModel):
    code: str  # khoá ổn định để giao diện dịch
    message: str  # chỉ để người phát triển đọc
    fields: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    detail: ErrorBody


class Health(BaseModel):
    status: Literal["ok"]
    bundle_version: str
    model_version: str | None = None  # None khi app chạy mà chưa nạp mô hình (chỉ trong test)
    stage: Stage | None = None
    ner_enabled: bool
    rss_mb: float  # bộ nhớ thường trú của tiến trình


# --- Phân tích một câu --------------------------------------------------------------------------


class TextRequest(BaseModel):
    text: Text


class Candidate(BaseModel):
    label: str
    probability: float


class FlagReason(BaseModel):
    code: ReasonCode
    task: Task | None = None  # head không chắc (chỉ có với low_confidence)
    candidates: list[Candidate] = []  # các lớp cao nhất của head đó
    message: str  # tiếng Việt, dùng cho giao diện và cột "Lý do" của file xuất


class Flag(BaseModel):
    needs_review: bool
    reasons: list[FlagReason]


class SentimentPrediction(BaseModel):
    label: SentimentLabel
    confidence: float  # xác suất của nhãn, đã hiệu chỉnh nhiệt độ
    probs: dict[str, float]  # nhãn -> xác suất, theo thứ tự nhãn trong labels.py


class TopicPrediction(BaseModel):
    label: TopicLabel
    confidence: float
    probs: dict[str, float]


class AnalyzeResponse(BaseModel):
    display_text: str  # câu đã ẩn thông tin cá nhân, giữ hoa/thường
    pii_types: list[PiiType]
    no_diacritics: bool
    n_words: int
    xai_available: bool  # false khi câu dài hơn giới hạn của giải thích
    sentiment: SentimentPrediction
    topic: TopicPrediction
    flag: Flag


# --- Giải thích ---------------------------------------------------------------------------------


class TokenEffect(BaseModel):
    text: str  # dạng hiển thị: "giảng viên", "[TÊN]", ":)"
    delta: float  # p(đủ câu) − p(bỏ từ này); dương là ủng hộ nhãn
    p_without: float


class HeadExplanation(BaseModel):
    label: str  # lớp dự đoán của câu đầy đủ
    probability: float
    tokens: list[TokenEffect]  # mọi từ của câu, theo thứ tự
    support: list[int]  # chỉ số các từ ủng hộ mạnh nhất, mạnh trước
    oppose: list[int]  # chỉ số các từ đi ngược mạnh nhất


class ExplainResponse(BaseModel):
    n_forwards: int  # số câu đã chạy qua mô hình (số từ + 1)
    sentiment: HeadExplanation
    topic: HeadExplanation
