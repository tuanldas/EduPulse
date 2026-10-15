"""Tiền xử lý văn bản tiếng Việt dùng chung cho huấn luyện và phục vụ.

Dùng `prepare` / `prepare_many` làm điểm vào; `ngrams` cho đếm nền và insight. Import gói này không
khởi động JVM: bộ tách từ chỉ nạp ở lần gọi `prepare` đầu tiên.
"""

from edupulse.text.ngrams import count_ngrams, ngrams
from edupulse.text.pipeline import NORMALIZER_VERSION, Prepared, prepare, prepare_many

__all__ = [
    "NORMALIZER_VERSION",
    "Prepared",
    "count_ngrams",
    "ngrams",
    "prepare",
    "prepare_many",
]
