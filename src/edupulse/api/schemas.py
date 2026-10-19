"""Hợp đồng dữ liệu của API: lỗi, cấu hình, phân tích một câu, giải thích, batch, xuất file.

Giá trị phân loại (`Literal`) khoá bằng tiếng Anh; giao diện ánh xạ sang tiếng Việt. Thứ tự và tên
nhãn lấy từ `edupulse.labels` (có test canh các `Literal` dưới đây khớp với nó). Xác suất làm tròn
4 chữ số thập phân. Mô hình trả về kế thừa `ApiModel` để mọi trường luôn có trong JSON (kiểu
TypeScript sinh từ OpenAPI không có `?:` thừa). Dữ liệu của dashboard nằm ở `insights.models`.

Mọi thay đổi ở đây là thay đổi hợp đồng với giao diện: chạy `make gen-api` và commit
`apps/web/openapi.json` cùng `apps/web/src/lib/api/schema.d.ts` (test snapshot sẽ báo nếu quên).
"""

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
)
from pydantic_core import PydanticCustomError

from edupulse.api.limits import MAX_GROUP_CHARS, MAX_GROUPS
from edupulse.bundle_schema import BudgetEfficiency, Stage
from edupulse.config import DEFAULT_BUDGET, MAX_CHARS, MAX_ROWS
from edupulse.insights.models import (
    ApiModel,
    InsightsResponse,
    SentimentLabel,
    TopicLabel,
)

__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "ApiModel",
    "AppConfig",
    "BaselineComparison",
    "BatchStatus",
    "BatchUpload",
    "ErrorResponse",
    "ExplainResponse",
    "ExportRequest",
    "Health",
    "InsightsRequest",
    "InsightsResponse",
    "ResultsResponse",
    "RunRequest",
    "SentimentLabel",
    "TextRequest",
    "TopicLabel",
]

Task = Literal["sentiment", "topic"]
PiiType = Literal["name", "phone", "email", "url", "student_id", "national_id"]
ReasonCode = Literal["low_confidence", "no_diacritics"]
# Ngân sách duyệt (% số câu). Test canh khớp `config.BUDGETS`.
ReviewBudget = Literal[5, 10, 20, 30]

_TEXT_EMPTY = "text_empty"
_TEXT_TOO_LONG = "text_too_long"


def _check_text(value: str) -> str:
    """Mã lỗi riêng cho câu rỗng và câu quá dài, để giao diện báo đúng thay vì lỗi chung."""
    if not value:
        raise PydanticCustomError(_TEXT_EMPTY, "Câu không được để trống.")
    if len(value) > MAX_CHARS:
        raise PydanticCustomError(
            _TEXT_TOO_LONG, "Câu dài hơn {max_chars} ký tự.", {"max_chars": MAX_CHARS}
        )
    return value


# Bỏ khoảng trắng hai đầu rồi mới đo độ dài, nên câu toàn dấu cách bị từ chối như câu rỗng.
# `json_schema_extra` giữ lại giới hạn trong OpenAPI vì kiểm tra nằm ở validator tuỳ biến.
Text = Annotated[
    str,
    StringConstraints(strip_whitespace=True),
    AfterValidator(_check_text),
    Field(json_schema_extra={"minLength": 1, "maxLength": MAX_CHARS}),
]


class StrictRequest(BaseModel):
    """Thân yêu cầu: trường lạ bị từ chối để lỗi gõ nhầm tên trường không bị bỏ qua âm thầm."""

    model_config = ConfigDict(extra="forbid")


# --- Lỗi và sức khoẻ ----------------------------------------------------------------------------


class FieldError(BaseModel):
    """Một trường không hợp lệ; không kèm giá trị gửi lên để không lặp lại nội dung người dùng."""

    loc: list[str | int]
    type: str


class ErrorBody(BaseModel):
    code: str  # khoá ổn định để giao diện dịch
    message: str  # chỉ để người phát triển đọc
    fields: list[FieldError] | None = None  # chỉ có với lỗi `validation_error`
    # Số nguyên đi kèm mã lỗi để giao diện tự ghép câu, vd. `too_many_rows` có n_rows, max_rows.
    params: dict[str, int] | None = None


class ErrorResponse(BaseModel):
    detail: ErrorBody


class Health(ApiModel):
    status: Literal["ok"]
    bundle_version: str
    model_version: str | None  # None khi app chạy mà chưa nạp mô hình (chỉ trong test)
    stage: Stage | None
    ner_enabled: bool
    rss_mb: float  # bộ nhớ thường trú của tiến trình


# --- Cấu hình cho giao diện ---------------------------------------------------------------------


class ModelInfo(ApiModel):
    model_version: str
    bundle_version: str
    stage: Stage


class Limits(ApiModel):
    max_chars: int  # ký tự tối đa của một câu
    max_upload_mb: int
    max_rows: int  # dòng tối đa của một file
    max_groups: int  # số giá trị khác nhau tối đa của cột nhóm
    xai_max_words: int  # câu dài hơn số từ này không được giải thích
    auto_xai_top: int  # số câu cờ thấp nhất được giải thích sẵn khi chạy batch


class BudgetOption(ApiModel):
    budget: ReviewBudget
    default: bool
    efficiency: BudgetEfficiency  # ước tính trên tập validation


class UiSample(ApiModel):
    text: str
    no_diacritics: bool  # câu mẫu gõ không dấu (để minh hoạ cảnh báo)


class AppConfig(ApiModel):
    model: ModelInfo
    ner_enabled: bool  # tên người được nhận diện bằng NER (nếu không, chỉ theo luật xưng hô)
    limits: Limits
    budgets: list[BudgetOption]
    default_budget: ReviewBudget
    heartbeat_s: int  # chu kỳ gửi heartbeat khi tab đang mở
    ui_samples: list[UiSample]  # rỗng khi `demo_available` = false
    demo_available: bool  # dữ liệu demo, file mẫu và câu mẫu đã được dựng lúc build image
    demo_reason: str | None  # vì sao không có demo (khi `demo_available` = false)
    source_url: str  # liên kết mã nguồn (AGPL-3.0 §13); "" nếu chưa cấu hình
    model_url: str  # liên kết tới model trên Hugging Face; "" nếu chưa cấu hình (ẩn nút)


# --- Phân tích một câu --------------------------------------------------------------------------


class TextRequest(BaseModel):
    text: Text


class AnalyzeRequest(BaseModel):
    text: Text
    compare: bool = False  # kèm kết quả của mô hình cơ sở (TF-IDF + LinearSVC)


class Candidate(ApiModel):
    label: str
    probability: float


class FlagReason(ApiModel):
    code: ReasonCode
    task: Task | None  # head không chắc (chỉ có với low_confidence)
    candidates: list[Candidate]  # các lớp cao nhất của head đó
    message: str  # tiếng Việt, dùng cho giao diện và cột "Lý do" của file xuất


class Flag(ApiModel):
    needs_review: bool
    reasons: list[FlagReason]


class SentimentPrediction(ApiModel):
    label: SentimentLabel
    confidence: float  # xác suất của nhãn, đã hiệu chỉnh nhiệt độ
    probs: dict[str, float]  # nhãn -> xác suất, theo thứ tự nhãn trong labels.py


class TopicPrediction(ApiModel):
    label: TopicLabel
    confidence: float
    probs: dict[str, float]


class BaselineSentiment(ApiModel):
    label: SentimentLabel
    confidence: float
    top_terms: list[str]  # từ/cụm đóng góp nhiều nhất cho nhãn dự đoán (hệ số × TF-IDF)


class BaselineTopic(ApiModel):
    label: TopicLabel
    confidence: float
    top_terms: list[str]


class BaselineComparison(ApiModel):
    """Kết quả của mô hình cơ sở trên cùng câu, để so với PhoBERT."""

    name: str  # "TF-IDF + LinearSVC"
    sentiment: BaselineSentiment
    topic: BaselineTopic
    agree_sentiment: bool  # cùng nhãn cảm xúc với mô hình chính
    agree_topic: bool
    elapsed_ms: float


class AnalyzeResponse(ApiModel):
    display_text: str  # câu đã ẩn thông tin cá nhân, giữ hoa/thường
    pii_types: list[PiiType]
    no_diacritics: bool
    n_words: int
    # false khi câu dài hơn `Limits.xai_max_words`: /api/explain sẽ trả 422 `xai_too_long`.
    xai_available: bool
    sentiment: SentimentPrediction
    topic: TopicPrediction
    flag: Flag
    baseline: BaselineComparison | None  # chỉ có khi yêu cầu `compare` = true


# --- Giải thích ---------------------------------------------------------------------------------


class TokenEffect(ApiModel):
    text: str  # dạng hiển thị: "giảng viên", "[TÊN]", ":)"
    delta: float  # p(đủ câu) − p(bỏ từ này); dương là ủng hộ nhãn
    p_without: float


class HeadExplanation(ApiModel):
    label: str  # lớp dự đoán của câu đầy đủ
    probability: float
    tokens: list[TokenEffect]  # mọi từ của câu, theo thứ tự
    support: list[int]  # chỉ số các từ ủng hộ mạnh nhất, mạnh trước
    oppose: list[int]  # chỉ số các từ đi ngược mạnh nhất


class ExplainResponse(ApiModel):
    n_forwards: int  # số câu đã chạy qua mô hình (số từ + 1)
    sentiment: HeadExplanation
    topic: HeadExplanation


# --- Batch: tải file ----------------------------------------------------------------------------

BatchWarningCode = Literal[
    "encoding_fallback",  # file không phải UTF-8, đã đọc bằng cp1258 (có thể sai dấu)
    "blank_rows_skipped",  # có dòng trống bị bỏ
    "other_sheets_ignored",  # XLSX có nhiều sheet, chỉ đọc sheet đầu
    "columns_renamed",  # tiêu đề cột trống hoặc trùng đã được đặt lại tên
]
FileEncoding = Literal["utf-8", "utf-8-sig", "cp1258"]


class BatchWarning(ApiModel):
    code: BatchWarningCode
    message: str
    count: int | None  # số dòng/cột liên quan, nếu có


class BatchUpload(ApiModel):
    """Kết quả tải file hoặc tạo batch từ file mẫu; chưa có phân tích nào được chạy."""

    batch_id: str
    filename: str
    n_rows: int  # số dòng dữ liệu (không tính tiêu đề và dòng trống)
    columns: list[str]
    preview: list[list[str]]  # vài dòng đầu, mỗi dòng cùng thứ tự `columns`
    suggested_text_col: str | None  # None: không có cột nào trông giống văn bản
    suggested_group_col: str | None
    encoding: FileEncoding | None  # None với XLSX
    warnings: list[BatchWarning]
    illustrative_groups: bool  # cột nhóm là dữ liệu minh hoạ (file mẫu)


# --- Batch: chạy và theo dõi --------------------------------------------------------------------

BatchState = Literal["uploaded", "queued", "running", "done", "failed"]
# Ẩn thông tin cá nhân và tách từ chạy chung một bước vì dùng chung một đường tiền xử lý.
BatchStep = Literal["prepare", "predict", "flag", "explain"]


class RunRequest(StrictRequest):
    text_col: Annotated[str, Field(min_length=1, max_length=200)]
    group_col: Annotated[str, Field(min_length=1, max_length=200)] | None = None
    review_budget: ReviewBudget = DEFAULT_BUDGET


class BatchStatus(ApiModel):
    state: BatchState
    step: BatchStep | None  # bước đang chạy; None khi chưa chạy hoặc đã xong
    done: int  # tiến độ trong bước hiện tại (số câu)
    total: int
    progress: float  # 0..1 cả quá trình, không bao giờ giảm
    eta_s: float | None  # thời gian còn lại ước tính; None khi chưa đo được thông lượng
    n_rows: int
    pii_rows: int | None  # số câu có thông tin bị ẩn; có sau bước `prepare`
    error_code: str | None  # có khi state = failed
    illustrative_groups: bool


# --- Batch: kết quả -----------------------------------------------------------------------------

# Gắn cờ / mẫu kiểm tra ngẫu nhiên / không nằm trong hàng đợi duyệt.
RowSource = Literal["flag", "audit", "none"]


class ResultRow(ApiModel):
    index: int  # vị trí dòng trong batch (từ 0); dùng cho đường dẫn và `overrides`
    source_row: int  # số thứ tự dòng dữ liệu trong file gốc (từ 1, không tính tiêu đề)
    text: str  # câu đã ẩn thông tin cá nhân
    n_words: int
    no_diacritics: bool
    pii_types: list[PiiType]
    sentiment: SentimentLabel
    sentiment_confidence: float
    topic: TopicLabel
    topic_confidence: float
    needs_review: bool
    reasons: list[FlagReason]
    source: RowSource
    in_review: bool  # nằm trong danh sách duyệt (cờ hoặc mẫu kiểm tra)
    group: str | None
    has_explanation: bool  # giải thích đã tính sẵn; chưa có thì gọi endpoint giải thích
    explainable: bool  # false khi câu dài hơn `Limits.xai_max_words`


class SkippedRow(ApiModel):
    """Dòng không còn từ nào để phân tích (chỉ có thông tin cá nhân hoặc ký hiệu)."""

    index: int
    source_row: int
    text: str


class PiiCount(ApiModel):
    type: PiiType
    rows: int  # số câu có loại thông tin này


class ResultsSummary(ApiModel):
    n_rows: int  # số dòng trong batch
    n_analyzed: int
    n_skipped: int
    n_flagged: int
    flagged_share: float
    n_audit: int  # số câu kiểm tra ngẫu nhiên (từ phần không cờ)
    n_review: int  # cờ + kiểm tra = độ dài `review_order`
    budget: ReviewBudget
    budget_target: int  # ⌈ngân sách · số câu⌉; `n_flagged` lớn hơn khi câu không dấu vượt ngân sách
    n_no_diacritics: int
    pii_rows: int
    pii_counts: list[PiiCount]
    n_truncated: int  # số câu dài hơn giới hạn ký tự và bị cắt
    n_explained: int  # số câu đã có giải thích sẵn
    n_groups: int | None  # None khi không chọn cột nhóm
    elapsed_s: float
    illustrative_groups: bool


class ResultsResponse(ApiModel):
    summary: ResultsSummary
    rows: list[ResultRow]  # theo thứ tự trong file; bỏ các dòng ở `skipped`
    review_order: list[int]  # `index` của các dòng cần duyệt, độ tin cậy tăng dần, cố định
    skipped: list[SkippedRow]


# --- Batch: insight và xuất file ----------------------------------------------------------------

GroupName = Annotated[str, Field(min_length=1, max_length=MAX_GROUP_CHARS)]


class InsightsRequest(StrictRequest):
    # Nhóm được lọc nằm trong thân yêu cầu, không đưa vào query string (nhật ký không ghi giá trị).
    groups: Annotated[list[GroupName], Field(max_length=MAX_GROUPS)] | None = None


ExportKind = Literal["all", "corrected"]


class LabelOverride(StrictRequest):
    """Nhãn do người duyệt sửa; trường để trống là giữ nhãn của mô hình."""

    index: Annotated[int, Field(ge=0)]
    sentiment: SentimentLabel | None = None
    topic: TopicLabel | None = None


class ExportRequest(StrictRequest):
    kind: ExportKind = "all"
    overrides: Annotated[list[LabelOverride], Field(max_length=MAX_ROWS)] = []
