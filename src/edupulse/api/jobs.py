"""Kho batch trong RAM và worker nền chạy phân tích.

- Id batch là chuỗi ngẫu nhiên 128 bit (`secrets.token_urlsafe(16)`); `demo` là batch chỉ đọc dựng
  từ dữ liệu demo, không hết hạn và không tính vào giới hạn.
- Tối đa `MAX_JOBS` batch; hàng đợi chạy (đang chạy + chờ) tối đa `MAX_QUEUE`; vượt thì 429 `busy`.
  Chỉ một worker (một luồng) chạy mô hình, các yêu cầu `/api/analyze` chen vào giữa từng câu.
- Vòng đời: giao diện gửi heartbeat khi tab còn mở và `close` khi đóng tab. TTL tính từ lần truy cập
  cuối chỉ là lưới an toàn, nên duyệt lâu không mất batch còn đóng tab thì xoá ngay.
- Dữ liệu thô của file chỉ giữ tới khi phân tích xong (hoặc batch bị xoá); sau đó chỉ còn câu đã ẩn
  thông tin cá nhân, nhãn và nhóm. Không có gì ghi ra đĩa.
- Log chỉ có số lượng và loại lỗi: không nội dung câu, tên file, giá trị cột/nhóm hay id batch.
"""

from __future__ import annotations

import logging
import queue
import secrets
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import cast

from edupulse.api.errors import ApiError
from edupulse.api.limits import (
    DEMO_BATCH_ID,
    MAX_GROUP_CHARS,
    MAX_GROUPS,
    MAX_PARALLEL_PARSES,
    PARSE_WAIT_S,
)
from edupulse.api.parsing import ParsedTable, is_illustrative
from edupulse.api.schemas import BatchState, BatchStep
from edupulse.config import AUTO_XAI_TOP, BUDGETS, MAX_CHARS, MAX_JOBS, MAX_QUEUE
from edupulse.demo_data import DemoRowData
from edupulse.engine.explain import Explanation
from edupulse.engine.flags import seed_from
from edupulse.engine.service import Analysis, Analyzer, BatchCancelled, NoContent
from edupulse.insights.models import InsightsResponse
from edupulse.insights.rows import InsightRow
from edupulse.text import Prepared

logger = logging.getLogger("edupulse.jobs")

EMPTY_GROUP = "(trống)"
# Phần của thanh tiến độ tổng dành cho từng bước (tổng bằng 1).
STEP_WEIGHTS: dict[str, float] = {"prepare": 0.10, "predict": 0.75, "flag": 0.02, "explain": 0.13}
STEP_ORDER = tuple(STEP_WEIGHTS)
EXPLAIN_SECONDS_GUESS = 0.2  # ước tính một câu giải thích tự động, dùng tới khi đo được thật
MIN_RATE_SAMPLE_S = 0.3  # chưa đủ thời gian đo thì chưa dám ước tính thời gian còn lại


@dataclass(slots=True)
class RowData:
    """Một dòng của batch sau khi phân tích (hoặc bị bỏ qua vì không còn nội dung)."""

    index: int  # vị trí trong batch, từ 0
    source_row: int  # số thứ tự dòng dữ liệu trong file gốc, từ 1
    prepared: Prepared
    analysis: Analysis | None  # None: không có nhãn
    group: str | None
    truncated: bool  # câu gốc dài hơn MAX_CHARS và đã bị cắt
    audit: bool  # thuộc mẫu kiểm tra


@dataclass(slots=True)
class BatchResult:
    rows: list[RowData]
    review_order: list[int]
    explanations: dict[int, Explanation]
    elapsed_s: float
    budget: int


@dataclass(slots=True)
class Batch:
    id: str
    filename: str
    kind: str  # "upload" | "sample" | "demo"
    n_rows: int
    illustrative_groups: bool
    created: float
    last_access: float
    table: ParsedTable | None
    state: BatchState = "uploaded"
    # Tham số lần chạy.
    text_col: str | None = None
    group_col: str | None = None
    budget: int | None = None
    # Tiến độ.
    step: BatchStep | None = None
    done: int = 0
    total: int = 0
    progress: float = 0.0
    eta_s: float | None = None
    pii_rows: int | None = None
    error_code: str | None = None
    step_started: float = 0.0
    run_started: float = 0.0
    # Kết quả.
    result: BatchResult | None = None
    insight_rows: list[InsightRow] | None = None
    insights_cache: dict[tuple[str, ...], InsightsResponse] = field(default_factory=dict)
    # Điều khiển.
    cancel: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(
        default_factory=threading.RLock
    )  # tiến độ và bộ nhớ đệm giải thích

    @property
    def readonly(self) -> bool:
        return self.kind == "demo"


@dataclass(frozen=True, slots=True)
class Snapshot:
    """Ảnh chụp nhất quán của tiến độ, lấy dưới khoá của batch."""

    state: BatchState
    step: BatchStep | None
    done: int
    total: int
    progress: float
    eta_s: float | None
    n_rows: int
    pii_rows: int | None
    error_code: str | None
    illustrative_groups: bool


def _busy(message: str) -> ApiError:
    return ApiError(429, "busy", message, headers={"Retry-After": "5"})


def _not_found() -> ApiError:
    return ApiError(404, "batch_not_found", "Không có batch này hoặc batch đã hết hạn.")


class JobStore:
    def __init__(
        self,
        analyzer: Analyzer,
        *,
        ttl_s: float,
        max_batches: int = MAX_JOBS,
        max_queue: int = MAX_QUEUE,
        parse_slots: int = MAX_PARALLEL_PARSES,
        parse_wait_s: float = PARSE_WAIT_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.analyzer = analyzer
        self.ttl_s = ttl_s
        self.max_batches = max_batches
        self.max_queue = max_queue
        self.parse_wait_s = parse_wait_s
        self._parse_slots = threading.BoundedSemaphore(parse_slots)
        self._clock = clock
        self._lock = threading.RLock()
        self._batches: dict[str, Batch] = {}
        self._demo: Batch | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    # --- Vòng đời của store ---------------------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._work, name="edupulse-batch", daemon=True)
        self._thread.start()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Dừng worker: huỷ batch đang chạy và đợi luồng thoát (tối đa `timeout` giây)."""
        with self._lock:
            for batch in self._batches.values():
                batch.cancel.set()
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        self.clear()

    def clear(self) -> None:
        """Xoá mọi batch (trừ `demo`) và huỷ việc đang chạy; dùng khi tắt máy chủ và trong test."""
        with self._lock:
            batches = list(self._batches.values())
            self._batches.clear()
        for batch in batches:
            self._discard(batch)

    # --- Truy cập --------------------------------------------------------------------------------

    def set_demo(self, rows: list[DemoRowData]) -> None:
        """Batch `demo`: dự đoán đã tính lúc build, chỉ đọc, không có cột nhóm."""
        insight_rows = [
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
            for r in rows
        ]
        now = self._clock()
        self._demo = Batch(
            id=DEMO_BATCH_ID,
            filename="",
            kind="demo",
            n_rows=len(rows),
            illustrative_groups=False,
            created=now,
            last_access=now,
            table=None,
            state="done",
            progress=1.0,
            done=len(rows),
            total=len(rows),
            insight_rows=insight_rows,
        )

    def get(self, batch_id: str) -> Batch:
        """Batch còn sống (và gia hạn nó); `demo` luôn có nếu đã nạp. Không có thì 404."""
        if batch_id == DEMO_BATCH_ID and self._demo is not None:
            return self._demo
        with self._lock:
            self._purge_locked()
            batch = self._batches.get(batch_id)
            if batch is None:
                raise _not_found()
            batch.last_access = self._clock()
            return batch

    def remove(self, batch_id: str) -> None:
        """Xoá ngay (đóng tab hoặc DELETE). `demo` không bao giờ bị xoá."""
        if batch_id == DEMO_BATCH_ID and self._demo is not None:
            return
        with self._lock:
            batch = self._batches.pop(batch_id, None)
        if batch is None:
            raise _not_found()
        self._discard(batch)

    def count(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._batches)

    def ensure_capacity(self) -> None:
        """429 nếu kho đã đủ batch. Gọi TRƯỚC khi đọc và parse file để không tốn công vô ích."""
        with self._lock:
            self._purge_locked()
            if len(self._batches) >= self.max_batches:
                raise _busy("Máy chủ đang giữ tối đa số batch, hãy thử lại sau ít phút.")

    @contextmanager
    def parsing(self) -> Iterator[None]:
        """Một trong vài chỗ đọc + parse file; hết chỗ quá lâu thì 429.

        Upload song song mà không có trần thì RAM và CPU nhân lên theo số yêu cầu (6 file 9MB cùng
        lúc đo được +286MB), nên mỗi lúc chỉ vài file được mở ra.
        """
        if not self._parse_slots.acquire(timeout=self.parse_wait_s):
            raise _busy("Máy chủ đang đọc nhiều file cùng lúc, hãy thử lại sau ít giây.")
        try:
            yield
        finally:
            self._parse_slots.release()

    def _purge_locked(self) -> None:
        now = self._clock()
        expired = [b for b in self._batches.values() if now - b.last_access > self.ttl_s]
        for batch in expired:
            del self._batches[batch.id]
            self._discard(batch)
        if expired:
            logger.info("Đã xoá %d batch hết hạn", len(expired))

    @staticmethod
    def _discard(batch: Batch) -> None:
        batch.cancel.set()
        with batch.lock:
            batch.table = None
            batch.result = None
            batch.insight_rows = None

    # --- Tạo và chạy ------------------------------------------------------------------------------

    def create(
        self, filename: str, table: ParsedTable, *, kind: str = "upload", illustrative: bool = False
    ) -> Batch:
        with self._lock:
            self._purge_locked()
            if (
                len(self._batches) >= self.max_batches
            ):  # chỗ có thể đã bị lấy từ lúc ensure_capacity
                raise _busy("Máy chủ đang giữ tối đa số batch, hãy thử lại sau ít phút.")
            now = self._clock()
            batch = Batch(
                id=secrets.token_urlsafe(16),
                filename=filename,
                kind=kind,
                n_rows=len(table.rows),
                illustrative_groups=illustrative,
                created=now,
                last_access=now,
                table=table,
            )
            self._batches[batch.id] = batch
            return batch

    def run(self, batch: Batch, *, text_col: str, group_col: str | None, budget: int) -> Snapshot:
        """Kiểm tra tham số rồi xếp batch vào hàng đợi chạy.

        Khoá luôn lấy theo thứ tự store -> batch (như `_purge_locked`) để không bao giờ bế tắc.
        """
        if budget not in BUDGETS:
            raise ApiError(422, "invalid_budget", "Ngân sách duyệt không hợp lệ.")
        with self._lock:
            self._purge_locked()
            if not batch.readonly and self._batches.get(batch.id) is not batch:
                raise _not_found()
            with batch.lock:
                if batch.readonly or batch.state not in ("uploaded", "failed"):
                    code = "batch_readonly" if batch.readonly else "already_running"
                    raise ApiError(409, code, "Batch này đã chạy hoặc không chạy lại được.")
                table = batch.table
                if table is None:
                    raise _not_found()
                _check_columns(table, text_col, group_col)
                active = sum(b.state in ("queued", "running") for b in self._batches.values())
                if active >= self.max_queue:
                    raise _busy("Hàng đợi phân tích đang đầy, hãy thử lại sau ít phút.")
                batch.text_col, batch.group_col, batch.budget = text_col, group_col, budget
                batch.illustrative_groups = is_illustrative(batch.kind, group_col, table.columns)
                batch.state = "queued"
                batch.step, batch.done, batch.total = None, 0, batch.n_rows
                batch.progress, batch.eta_s = 0.0, None
                batch.pii_rows, batch.error_code = None, None
            self._queue.put(batch.id)
        return self.snapshot(batch)

    @staticmethod
    def insight_rows(batch: Batch) -> list[InsightRow]:
        """Các dòng có nhãn của batch ở dạng đầu vào của insight (tính một lần rồi giữ lại).

        Insight tính trên NHÃN DỰ ĐOÁN, không phản ánh nhãn người duyệt đã sửa ở giao diện.
        """
        with batch.lock:
            if batch.insight_rows is None:
                result = batch.result
                if result is None:
                    raise ApiError(409, "not_done", "Batch chưa phân tích xong.")
                batch.insight_rows = [
                    InsightRow(
                        display_text=row.prepared.display_text,
                        model_text=row.prepared.model_text,
                        sentiment=row.analysis.sentiment.label,
                        sentiment_confidence=row.analysis.sentiment.confidence,
                        topic=row.analysis.topic.label,
                        topic_confidence=row.analysis.topic.confidence,
                        n_words=row.prepared.n_words,
                        group=row.group,
                        flagged=row.analysis.flag.needs_review,
                    )
                    for row in result.rows
                    if row.analysis is not None
                ]
            return batch.insight_rows

    def snapshot(self, batch: Batch) -> Snapshot:
        with batch.lock:
            return Snapshot(
                state=batch.state,
                step=batch.step,
                done=batch.done,
                total=batch.total,
                progress=batch.progress,
                eta_s=batch.eta_s,
                n_rows=batch.n_rows,
                pii_rows=batch.pii_rows,
                error_code=batch.error_code,
                illustrative_groups=batch.illustrative_groups,
            )

    # --- Worker ----------------------------------------------------------------------------------

    def _work(self) -> None:
        while True:
            batch_id = self._queue.get()
            if batch_id is None:
                return
            with self._lock:
                batch = self._batches.get(batch_id)
            if batch is None or batch.cancel.is_set():
                continue
            try:
                self._execute(batch)
            except Exception as exc:  # worker không được chết vì một batch hỏng
                logger.error("Job lỗi không xử lý: %s", type(exc).__name__)
                self._fail(batch, "internal_error")

    def _fail(self, batch: Batch, code: str) -> None:
        with batch.lock:
            batch.state = "failed"
            batch.error_code = code
            batch.step, batch.eta_s = None, None

    def _execute(self, batch: Batch) -> None:
        with batch.lock:
            table, text_col, group_col, budget = (
                batch.table,
                batch.text_col,
                batch.group_col,
                batch.budget,
            )
            if table is None or text_col is None or budget is None or batch.cancel.is_set():
                return
            batch.state = "running"
            batch.run_started = self._clock()
        text_index = table.columns.index(text_col)
        group_index = table.columns.index(group_col) if group_col else None
        texts = [row[text_index][:MAX_CHARS] for row in table.rows]
        truncated = [len(row[text_index]) > MAX_CHARS for row in table.rows]
        groups = (
            [_group_value(row[group_index]) for row in table.rows]
            if group_index is not None
            else None
        )

        def progress(step: str, done: int, total: int) -> None:
            self._on_progress(batch, step, done, total)

        def prepared(items: list[Prepared]) -> None:
            with batch.lock:
                batch.pii_rows = sum(bool(p.pii_types) for p in items)

        try:
            outcome = self.analyzer.run_batch(
                texts,
                budget,
                seed=seed_from(batch.id),
                progress=progress,
                cancelled=batch.cancel.is_set,
                on_prepared=prepared,
            )
        except BatchCancelled:
            return
        except NoContent:
            self._fail(batch, "no_content")
            return
        except Exception as exc:
            logger.error("Job phân tích lỗi: %s", type(exc).__name__)
            self._fail(batch, "internal_error")
            return

        rows = [
            RowData(
                index=i,
                source_row=table.source_rows[i],
                prepared=outcome.prepared[i],
                analysis=outcome.analyses[i],
                group=None if groups is None else groups[i],
                truncated=truncated[i],
                audit=bool(outcome.audit[i]),
            )
            for i in range(len(texts))
        ]
        result = BatchResult(
            rows=rows,
            review_order=outcome.review_order,
            explanations=dict(outcome.explanations),
            elapsed_s=self._clock() - batch.run_started,
            budget=budget,
        )
        with batch.lock:
            if batch.cancel.is_set():
                return
            batch.result = result
            batch.table = None  # dữ liệu thô không còn cần: chỉ giữ câu đã ẩn thông tin cá nhân
            batch.state = "done"
            batch.step, batch.eta_s = None, None
            batch.done = batch.total = len(rows)
            batch.progress = 1.0
        logger.info(
            "Batch xong: %d dòng, %d cờ, %.1f s", len(rows), outcome.n_flagged, result.elapsed_s
        )

    def _on_progress(self, batch: Batch, step: str, done: int, total: int) -> None:
        now = self._clock()
        with batch.lock:
            if batch.step != step:
                batch.step = cast(BatchStep, step)  # tên bước do engine phát ra
                batch.step_started = now
            batch.done, batch.total = done, total
            within = 1.0 if total == 0 else done / total
            before = sum(STEP_WEIGHTS[s] for s in STEP_ORDER[: STEP_ORDER.index(step)])
            batch.progress = max(batch.progress, min(1.0, before + STEP_WEIGHTS[step] * within))
            batch.eta_s = _estimate_eta(step, done, total, now - batch.step_started)


def _estimate_eta(step: str, done: int, total: int, elapsed: float) -> float | None:
    """Thời gian còn lại (giây) theo thông lượng đã đo của bước hiện tại; None khi chưa đo được."""
    explain_guess = AUTO_XAI_TOP * EXPLAIN_SECONDS_GUESS
    if step == "predict":
        if done == 0 or elapsed < MIN_RATE_SAMPLE_S:
            return None
        return (total - done) * elapsed / done + explain_guess
    if step == "flag":
        return explain_guess
    if step == "explain":
        if done == 0:
            return max(total, 0) * EXPLAIN_SECONDS_GUESS
        return (total - done) * elapsed / done
    return None


def _group_value(cell: str) -> str:
    return cell[:MAX_GROUP_CHARS] if cell else EMPTY_GROUP


def _check_columns(table: ParsedTable, text_col: str, group_col: str | None) -> None:
    """Cột đã chọn phải có thật, cột văn bản phải có chữ, cột nhóm không quá nhiều giá trị."""
    if text_col not in table.columns:
        raise ApiError(422, "invalid_column", "Cột văn bản không có trong file.")
    if group_col is not None and (group_col not in table.columns or group_col == text_col):
        raise ApiError(422, "invalid_column", "Cột nhóm không hợp lệ.")
    text_index = table.columns.index(text_col)
    if not any(row[text_index] for row in table.rows):
        raise ApiError(422, "no_text_column", "Cột văn bản không có câu nào để phân tích.")
    if group_col is not None:
        group_index = table.columns.index(group_col)
        distinct = {_group_value(row[group_index]) for row in table.rows}
        if len(distinct) > MAX_GROUPS:
            raise ApiError(
                422,
                "too_many_groups",
                f"Cột nhóm có {len(distinct)} giá trị, tối đa {MAX_GROUPS}.",
                params={"n_groups": len(distinct), "max_groups": MAX_GROUPS},
            )


def group_counts(result: BatchResult) -> Counter[str]:
    """Số câu có nhãn của từng nhóm (rỗng khi batch không chọn cột nhóm)."""
    return Counter(r.group for r in result.rows if r.group is not None and r.analysis is not None)
