"""Hình dạng dữ liệu của dashboard & insight (kết quả của `POST /api/batches/{id}/insights`).

Mọi số liệu tính trên NHÃN DỰ ĐOÁN, không phản ánh nhãn người duyệt đã sửa ở giao diện. Tỉ lệ là số
trong [0, 1]; `delta_points` là chênh lệch tính bằng điểm phần trăm (đã nhân 100). Chuỗi hiển thị
(`title`, `summary`, `action`, `evidence`) viết sẵn bằng tiếng Việt theo giọng văn của
docs/design-guidelines.md: chủ ngữ là dữ liệu, luôn kèm căn cứ, không nêu tên cá nhân.

Các `Literal` nhãn và lớp nền `ApiModel` đặt ở đây để cả `insights` lẫn `api` dùng chung mà không
import vòng.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

SentimentLabel = Literal["negative", "neutral", "positive"]
TopicLabel = Literal["lecturer", "training_program", "facility", "others"]


class ApiModel(BaseModel):
    """Lớp nền cho dữ liệu trả về: mọi trường luôn có mặt trong JSON, kể cả trường có mặc định.

    Pydantic mặc định chỉ đánh dấu `required` cho trường không có giá trị mặc định, nên kiểu
    TypeScript sinh từ OpenAPI sẽ có `field?:` dù máy chủ luôn gửi trường đó.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


# --- Chỉ số tổng quan ---------------------------------------------------------------------------


class SentimentShares(ApiModel):
    negative: float
    neutral: float
    positive: float


class GroupOption(ApiModel):
    name: str
    n: int  # số câu của nhóm trong cả tập (trước khi lọc)


class Kpis(ApiModel):
    n_rows: int
    n_groups: int | None  # số nhóm đang xét; None khi tập không có cột nhóm
    shares: SentimentShares  # tỉ lệ cảm xúc dự đoán
    baseline_shares: SentimentShares  # tỉ lệ cảm xúc toàn tập train (mức nền)
    delta_points: SentimentShares  # (tỉ lệ − nền) × 100
    n_flagged: int | None  # số câu được gắn cờ; None với dữ liệu demo (không có bước gắn cờ)
    flagged_share: float | None


class TopicStat(ApiModel):
    topic: TopicLabel
    n: int
    share: float  # tỉ trọng của chủ đề trong tập đang xét
    baseline_share: float  # tỉ trọng của chủ đề trong train
    sentiment_shares: SentimentShares  # cơ cấu cảm xúc trong chủ đề (đều 0 nếu n = 0)
    negative_n: int
    baseline_negative_rate: float  # mức nền tiêu cực của chủ đề (train)
    negative_ci_low: float | None  # cận dưới Wilson 95% của tỉ lệ tiêu cực; None nếu n = 0


# --- Cảnh báo -----------------------------------------------------------------------------------

AlertKind = Literal["negative_rate", "topic_share"]


class AlertEvidence(ApiModel):
    """Căn cứ của một cảnh báo. `negative_rate`: k câu tiêu cực trong n câu của chủ đề (hoặc của
    chủ đề × nhóm). `topic_share`: k câu thuộc chủ đề trong n câu của cả tập."""

    n: int
    k: int
    rate: float  # k / n
    ci_low: float  # cận dưới Wilson 95% của rate
    baseline: float  # mức nền tương ứng
    delta_points: float  # (rate − nền) × 100


class GroupHighlight(ApiModel):
    """Nhóm có tỉ lệ tiêu cực cao nhất trong một cảnh báo theo chủ đề."""

    group: str
    k: int
    n: int
    rate: float


class Alert(ApiModel):
    kind: AlertKind
    topic: TopicLabel
    group: str | None  # nhóm mà cảnh báo áp dụng; None = cả tập
    title: str
    summary: str  # một câu có số, vd. "93% góp ý về Chương trình là tiêu cực (182/195 câu)…"
    evidence: AlertEvidence
    keywords: list[str]  # từ khoá nổi bật của chủ đề, đã đổi `_` thành dấu cách
    note: str | None  # vd. "Tổng hợp theo nhóm, không gắn với cá nhân." với chủ đề Giảng viên
    highlight: GroupHighlight | None


# --- Heatmap nhóm × chủ đề ----------------------------------------------------------------------


class HeatmapCell(ApiModel):
    n: int
    negative_rate: float | None  # None khi n = 0
    delta_points: float | None  # (tỉ lệ − nền của chủ đề) × 100; None khi n = 0
    shaded: bool  # false khi n < `Heatmap.min_n`: giao diện không tô màu và không nêu chênh lệch


class Heatmap(ApiModel):
    groups: list[str]  # hàng
    topics: list[TopicLabel]  # cột, theo thứ tự nhãn
    baseline_negative_rate: list[float]  # mức nền tiêu cực, cùng thứ tự `topics`
    min_n: int
    cells: list[list[HeatmapCell]]  # cells[hàng][cột]


# --- Từ khoá, câu tiêu biểu, khuyến nghị --------------------------------------------------------


class Keyword(ApiModel):
    term: str  # vd. "máy chiếu"
    n: int  # số câu tiêu cực của chủ đề có chứa từ này
    score: float  # z của log-odds có trọng số; lớn là đặc trưng hơn


class TopicKeywords(ApiModel):
    topic: TopicLabel  # chỉ có chủ đề giảng viên / chương trình / cơ sở vật chất
    n_negative: int  # số câu tiêu cực của chủ đề trong tập đang xét
    keywords: list[Keyword]


class Quote(ApiModel):
    text: str  # câu đã ẩn thông tin cá nhân
    sentiment: SentimentLabel
    confidence: float  # độ tin cậy của nhãn cảm xúc
    group: str | None
    matched: list[str]  # từ khoá của chủ đề có trong câu (cùng dạng với `Keyword.term`)


class TopicQuotes(ApiModel):
    topic: TopicLabel
    quotes: list[Quote]  # tối đa ~50 câu, tốt nhất trước; giao diện lọc theo `matched`


class Recommendation(ApiModel):
    id: str  # khoá ổn định của luật, vd. "facility_projector"
    topic: TopicLabel
    action: str
    evidence: str  # "Căn cứ: 22 câu tiêu cực về Cơ sở vật chất nhắc “máy chiếu”, “mờ”."
    n: int  # số câu tiêu cực khớp luật
    terms: list[str]  # từ khoá của luật xuất hiện trong các câu khớp, nhiều nhất trước
    note: str | None


# --- Kết quả ------------------------------------------------------------------------------------

InsightSource = Literal["demo", "batch"]


class InsightsResponse(ApiModel):
    source: InsightSource
    illustrative_groups: bool  # nhóm là dữ liệu minh hoạ (file mẫu), giao diện phải ghi rõ
    groups: list[GroupOption] | None  # các nhóm có trong tập; None khi không có cột nhóm
    selected_groups: list[str] | None  # nhóm đang lọc; None = tất cả
    kpis: Kpis
    topics: list[TopicStat]  # đủ 4 chủ đề, theo thứ tự nhãn
    alerts: list[Alert]  # tối đa 3, nghiêm trọng nhất trước
    heatmap: Heatmap | None  # chỉ khi tập có cột nhóm
    keywords: list[TopicKeywords]
    quotes: list[TopicQuotes]
    recommendations: list[Recommendation]
