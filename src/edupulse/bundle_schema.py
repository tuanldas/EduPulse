"""Hợp đồng dữ liệu giữa `ml/` (đóng gói) và `api/` (nạp bundle): schema Pydantic cho mọi tệp JSON.

Một bundle là thư mục `artifacts/<bundle_version>/`:

    manifest.json            Manifest  (phiên bản, sha256 từng tệp, nhãn, normalizer_version...)
    model.int8.onnx          mô hình đa nhiệm đã lượng tử hoá int8
    tokenizer.json           tokenizer PhoBERT-base-v2 (thư viện `tokenizers`)
    calibration.json         Calibration  (nhiệt độ, phân vị độ tin cậy, ngưỡng ngân sách duyệt)
    model_card.json          ModelCard
    insight_background.json  InsightBackground  (đếm n-gram nền và mức nền của tập train)
    golden_predictions.json  GoldenPredictions  (dự đoán của 20 câu tự viết, để kiểm hồi quy)
    baseline/*.skops         baseline TF-IDF + LinearSVC (nạp lười, cho nút so sánh)

Bundle KHÔNG chứa câu gốc của UIT-VSFC. Mọi schema cấm trường lạ (`extra="forbid"`) để bên ghi và
bên đọc không lệch nhau âm thầm; đổi cấu trúc thì tăng `SCHEMA_VERSION`.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from edupulse.config import BUDGETS, DEFAULT_BUDGET
from edupulse.labels import LABELS, TASKS
from edupulse.scoring import CDF_POINTS

SCHEMA_VERSION = 1

# Kiểu skops phải tin khi nạp `baseline/*.skops`. Danh sách tường minh thay cho `trusted=True`:
# tệp chứa thêm kiểu lạ thì skops từ chối nạp. Cập nhật khi nâng scikit-learn/skops
# (ml/train_baseline.save_model báo kiểu thiếu).
BASELINE_TRUSTED_TYPES: tuple[str, ...] = (
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._TemperatureScaling",
)

Stage = Literal["smoke", "release"]
Split = Literal["val", "test"]

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

Probability = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Manifest ---------------------------------------------------------------------------------


class DatasetInfo(_Strict):
    name: str
    revision: str
    splits: dict[str, int]
    license_note: str


class BaseModelInfo(_Strict):
    name: str
    revision: str
    license: str


class OnnxSpec(_Strict):
    """Giao diện ONNX. Tokenizer.json có nhiều mục hơn bảng embedding gốc của PhoBERT (66.119 so với
    64.001): ONNX được nới bảng tới `embedding_rows` hàng, hàng thừa là bản sao `<unk>`, nên mọi id
    của tokenizer.json đều chạy được và hành xử như `<unk>` mà phía phục vụ không cần đổi id."""

    input_names: tuple[str, str] = ("input_ids", "attention_mask")
    output_names: tuple[str, str] = ("sentiment_logits", "topic_logits")
    pad_id: int = 1
    quantization: str
    base_vocab_size: int = Field(default=64001, gt=0)  # số hàng embedding của checkpoint gốc
    unk_id: int = Field(default=3, ge=0)
    embedding_rows: int = Field(default=64001, gt=0)  # số hàng embedding của ONNX này

    @model_validator(mode="after")
    def _embedding_covers_vocabulary(self) -> OnnxSpec:
        if self.embedding_rows < self.base_vocab_size:
            raise ValueError("embedding_rows không được nhỏ hơn base_vocab_size")
        if not self.unk_id < self.base_vocab_size:
            raise ValueError("unk_id phải nằm trong bảng embedding gốc")
        return self


class Manifest(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    bundle_version: str
    stage: Stage
    model_version: str
    created_at: datetime
    git_commit: str
    normalizer_version: str
    max_len: int = Field(gt=0)
    labels: dict[str, list[str]]
    onnx: OnnxSpec
    files: dict[str, str]  # tên tệp (tương đối trong bundle) -> sha256 hex; không gồm manifest.json
    dataset: DatasetInfo
    base_model: BaseModelInfo

    @field_validator("labels")
    @classmethod
    def _labels_match_code(cls, value: dict[str, list[str]]) -> dict[str, list[str]]:
        expected = {task: list(LABELS[task]) for task in TASKS}
        if value != expected:
            raise ValueError(f"nhãn/thứ tự nhãn lệch mã nguồn: bundle {value}, mã {expected}")
        return value

    @field_validator("files")
    @classmethod
    def _files_are_safe(cls, value: dict[str, str]) -> dict[str, str]:
        for name, digest in value.items():
            parts = name.split("/")
            if not all(_SAFE_PART.fullmatch(part) and part not in {".", ".."} for part in parts):
                raise ValueError(f"tên tệp không an toàn: {name!r}")
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"{name}: sha256 phải là 64 chữ số hex thường")
        return value


# --- Calibration ------------------------------------------------------------------------------


class HeadCalibration(_Strict):
    temperature: float = Field(gt=0)
    # Độ tin cậy (xác suất lớn nhất sau nhiệt độ) tại phân vị 0%, 1%, ..., 100% của val.
    cdf: list[float] = Field(min_length=CDF_POINTS, max_length=CDF_POINTS)
    ece_before: float | None = None
    ece_after: float | None = None

    @field_validator("cdf")
    @classmethod
    def _cdf_is_monotone(cls, value: list[float]) -> list[float]:
        if any(b < a for a, b in zip(value, value[1:], strict=False)):
            raise ValueError("cdf phải không giảm")
        if not all(0.0 <= x <= 1.0 for x in value):
            raise ValueError("cdf phải nằm trong [0, 1]")
        return value


class BudgetEfficiency(_Strict):
    """Hiệu quả của ngân sách duyệt, tính trên val bằng `scoring.rank_batch`."""

    flagged_share: Probability
    caught_sentiment_errors: Probability
    caught_topic_errors: Probability
    accept_accuracy_both: Probability
    all_accuracy_both: Probability


class Calibration(_Strict):
    """Nhiệt độ, bảng phân vị độ tin cậy và ngưỡng cờ, tính từ logits val của bản int8.

    Int8 động phụ thuộc nhẹ vào lô nên logits val được chạy từng câu một (`logits_batch_size` = 1),
    cùng điều kiện với phân tích một câu: `tau` và `cdf` là tham chiếu tất định cho đường đó.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    split: Literal["val"] = "val"
    logits_batch_size: Literal[1] = 1
    n: int = Field(gt=0)
    sentiment: HeadCalibration
    topic: HeadCalibration
    tau: dict[int, float]  # ngân sách % -> ngưỡng điểm cờ (cờ khi điểm < tau)
    efficiency: dict[int, BudgetEfficiency]

    @model_validator(mode="after")
    def _budgets_consistent(self) -> Calibration:
        if set(self.tau) != set(self.efficiency):
            raise ValueError("tau và efficiency phải cùng tập ngân sách")
        if not {DEFAULT_BUDGET, *BUDGETS} <= set(self.tau):
            raise ValueError(f"thiếu ngân sách trong {sorted(BUDGETS)}")
        return self


# --- Model card -------------------------------------------------------------------------------


class ClassMetrics(_Strict):
    label: str
    precision: Probability
    recall: Probability
    f1: Probability
    support: int = Field(ge=0)


class ReliabilityBin(_Strict):
    lower: Probability
    upper: Probability
    n: int = Field(ge=0)
    confidence: Probability | None  # trung bình độ tin cậy trong bin; None nếu bin rỗng
    accuracy: Probability | None  # tỉ lệ đúng trong bin; None nếu bin rỗng


class TaskMetrics(_Strict):
    accuracy: Probability
    macro_f1: Probability
    weighted_f1: Probability
    ece: float = Field(ge=0.0, le=1.0)
    nll: float | None = None
    per_class: list[ClassMetrics]
    confusion: list[list[Probability]]  # chuẩn hoá theo hàng (hàng = nhãn thật)
    reliability: list[ReliabilityBin]


class SplitMetrics(_Strict):
    split: Split
    n: int = Field(gt=0)
    sentiment: TaskMetrics
    topic: TaskMetrics


class BaselineRow(_Strict):
    name: str
    deployed: bool = False
    sentiment_accuracy: Probability
    sentiment_macro_f1: Probability
    topic_accuracy: Probability
    topic_macro_f1: Probability
    sentiment_ece: float | None = None


class SeedStat(_Strict):
    mean: float
    std: float
    n: int = Field(gt=0)


class ArchitectureRow(_Strict):
    """Đa nhiệm so với đơn nhiệm, trung bình theo seed."""

    name: str
    seeds: list[int]
    sentiment_macro_f1: SeedStat
    topic_macro_f1: SeedStat


class BudgetRow(_Strict):
    budget: int
    default: bool = False
    efficiency: BudgetEfficiency


class ChecklistRow(_Strict):
    name: str
    kind: Literal["invariance", "directional", "robustness"]
    n: int = Field(ge=0)
    pass_rate: Probability | None = None
    note: str = ""


class ModelCardNotes(_Strict):
    x86: str  # vd. "Chưa đo trên x86"
    demo: str | None = None


class ModelCard(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    model_version: str
    bundle_version: str
    stage: Stage
    metrics: SplitMetrics
    baselines: list[BaselineRow] = []
    architectures: list[ArchitectureRow] = []
    review_budget: list[BudgetRow] = []
    checklist: list[ChecklistRow] = []
    intended_use: str
    out_of_scope: str
    limitations: list[str]
    notes: ModelCardNotes

    @model_validator(mode="after")
    def _stage_matches_content(self) -> ModelCard:
        if self.stage == "smoke" and self.metrics.split != "val":
            raise ValueError("bản smoke chỉ có số trên val")
        if self.stage == "release":
            if self.metrics.split != "test":
                raise ValueError("bản release phải có số trên test")
            defaults = [row.budget for row in self.review_budget if row.default]
            if defaults != [DEFAULT_BUDGET]:
                raise ValueError(
                    f"bản release phải đánh dấu đúng mức {DEFAULT_BUDGET}% là mặc định"
                )
        return self


# --- Golden predictions -----------------------------------------------------------------------


class GoldenItem(_Strict):
    id: str
    model_text: str  # đầu ra của `prepare()` cho câu này, để kiểm cả tiền xử lý
    sentiment: str
    topic: str
    p_sentiment: list[Probability] = Field(
        min_length=len(LABELS["sentiment"]), max_length=len(LABELS["sentiment"])
    )
    p_topic: list[Probability] = Field(
        min_length=len(LABELS["topic"]), max_length=len(LABELS["topic"])
    )


class GoldenPredictions(_Strict):
    """Xác suất thô (T = 1) của mô hình int8 trên các câu tự viết; câu nằm ở tests/fixtures.

    Int8 động tính thang kích hoạt theo cả tensor, nên xác suất phụ thuộc nhẹ vào các câu cùng lô:
    mỗi câu được chạy riêng một lần (batch 1, không đệm) và test phải chạy cùng điều kiện đó.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    normalizer_version: str
    batch_size: Literal[1] = 1
    items: list[GoldenItem]


# --- Insight ----------------------------------------------------------------------------------


class InsightBackground(_Strict):
    """Đếm nền của tập train cho prior log-odds và mức nền của cảnh báo. Không chứa câu gốc."""

    schema_version: Literal[1] = SCHEMA_VERSION
    normalizer_version: str
    alpha0: float = Field(gt=0)
    min_count: int = Field(ge=1)
    n_sentences: int = Field(gt=0)
    total_ngrams: int = Field(gt=0)  # tổng số lần xuất hiện của MỌI n-gram, kể cả đã bị loại
    ngram_counts: dict[str, int]  # chỉ các n-gram có count >= min_count
    sentiment_share: dict[str, Probability]  # tỉ lệ cảm xúc toàn train
    topic_share: dict[str, Probability]  # tỉ trọng chủ đề toàn train
    negative_rate_by_topic: dict[str, Probability]  # mức nền tiêu cực của từng chủ đề

    @model_validator(mode="after")
    def _keys_are_labels(self) -> InsightBackground:
        checks = (
            (self.sentiment_share, LABELS["sentiment"]),
            (self.topic_share, LABELS["topic"]),
            (self.negative_rate_by_topic, LABELS["topic"]),
        )
        for mapping, labels in checks:
            if set(mapping) != set(labels):
                raise ValueError(f"khoá {sorted(mapping)} không khớp nhãn {sorted(labels)}")
        if any(count < self.min_count for count in self.ngram_counts.values()):
            raise ValueError("ngram_counts chứa n-gram dưới min_count")
        return self


# --- Dữ liệu demo (tính lúc build image; phía API dùng) ---------------------------------------


class DemoManifest(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    ok: bool
    reason: str | None = None  # vì sao không có demo khi ok = false
    dataset_revision: str | None = None
    bundle_version: str | None = None
    arch: str | None = None
    n_rows: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _explain_failure(self) -> DemoManifest:
        if not self.ok and not self.reason:
            raise ValueError("demo không khả dụng thì phải có reason")
        return self


class DemoRow(BaseModel):
    """Một dòng dự đoán của batch demo; cho phép thêm trường để khớp kết quả của `Analyzer`."""

    model_config = ConfigDict(extra="allow")

    display_text: str
    sentiment: str
    topic: str


class DemoResults(_Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    bundle_version: str
    rows: list[DemoRow]
