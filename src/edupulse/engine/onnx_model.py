"""Suy luận ONNX Runtime int8 cho PhoBERT đa nhiệm (cảm xúc + chủ đề trong một lượt chạy).

- Tokenizer là `tokenizers` đọc `tokenizer.json` của bundle (không cần transformers): cắt ở
  `max_len` của manifest, tự pad bằng numpy với `pad_id` của manifest.
- Câu được sắp theo độ dài rồi chia lô để bớt token pad; kết quả trả về theo thứ tự đầu vào.
- Khoá suy luận giữ theo từng lô nhỏ chứ không theo cả lời gọi, nên một yêu cầu lẻ chỉ phải chờ
  tối đa một lô khi có batch đang chạy.
"""

import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from edupulse.engine.bundle import Bundle, BundleError
from edupulse.labels import SENTIMENT, TOPIC

# Mỗi lần gọi chỉ vài chục câu: song song hoá của tokenizers tốn hơn lợi và tranh CPU với ORT.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_BATCH_SIZE = 32
WARMUP_TEXT = "xin chào"


@dataclass(frozen=True)
class Logits:
    """Đầu ra chưa softmax, hàng i ứng với câu thứ i của đầu vào."""

    sentiment: np.ndarray  # (n, 3) float32
    topic: np.ndarray  # (n, 4) float32


class OnnxModel:
    def __init__(
        self,
        bundle: Bundle,
        threads: int,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        mem_arena: bool = True,
    ) -> None:
        if threads < 1 or batch_size < 1:
            raise ValueError("threads và batch_size phải >= 1")
        spec = bundle.manifest.onnx
        self._input_names = spec.input_names
        self._output_names = spec.output_names
        self._pad_id = spec.pad_id
        self._batch_size = batch_size

        tokenizer = Tokenizer.from_file(str(bundle.tokenizer_path))
        tokenizer.enable_truncation(max_length=bundle.manifest.max_len)
        tokenizer.no_padding()
        self._tokenizer = tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = mem_arena
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.log_severity_level = 3  # chỉ lỗi
        try:
            self._session = ort.InferenceSession(
                str(bundle.model_path), options, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # ORT báo lỗi nạp bằng nhiều kiểu ngoại lệ riêng
            raise BundleError(
                f"Không nạp được {bundle.model_path.name}: {type(exc).__name__}: {exc}"
            ) from exc
        self._check_signature()
        self._lock = threading.Lock()

    def _check_signature(self) -> None:
        inputs = {i.name for i in self._session.get_inputs()}
        outputs = {o.name for o in self._session.get_outputs()}
        if not set(self._input_names) <= inputs or not set(self._output_names) <= outputs:
            raise BundleError(
                f"ONNX có đầu vào {sorted(inputs)}, đầu ra {sorted(outputs)}; manifest khai "
                f"{list(self._input_names)} và {list(self._output_names)}"
            )

    def _pad(self, rows: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        width = max(len(row) for row in rows)
        ids = np.full((len(rows), width), self._pad_id, dtype=np.int64)
        mask = np.zeros((len(rows), width), dtype=np.int64)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = row
            mask[i, : len(row)] = 1
        return ids, mask

    def _run(self, ids: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feed = {self._input_names[0]: ids, self._input_names[1]: mask}
        with self._lock:
            sentiment, topic = self._session.run(list(self._output_names), feed)
        return sentiment, topic

    def predict(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        progress: Callable[[int], None] | None = None,
    ) -> Logits:
        """Logits của từng câu. `progress(số_câu_đã_xong)` được gọi sau mỗi lô."""
        count = len(texts)
        sentiment = np.empty((count, len(SENTIMENT)), dtype=np.float32)
        topic = np.empty((count, len(TOPIC)), dtype=np.float32)
        if count == 0:
            return Logits(sentiment, topic)

        size = batch_size or self._batch_size
        encoded = [e.ids for e in self._tokenizer.encode_batch(list(texts))]
        order = np.argsort([len(ids) for ids in encoded], kind="stable")
        for start in range(0, count, size):
            index = order[start : start + size]
            ids, mask = self._pad([encoded[i] for i in index])
            batch_sentiment, batch_topic = self._run(ids, mask)
            sentiment[index] = batch_sentiment
            topic[index] = batch_topic
            if progress is not None:
                progress(min(start + size, count))
        return Logits(sentiment, topic)

    def warmup(self) -> None:
        """Một lượt chạy thử để ORT cấp bộ nhớ trước, yêu cầu đầu tiên của người dùng khỏi chậm."""
        self.predict([WARMUP_TEXT])
