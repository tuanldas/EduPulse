"""PhoBERT-base-v2 nhiều đầu ra: một bộ mã hoá + đầu tuyến tính cho mỗi tác vụ.

Một lớp cho cả hai kiểu: `{"sentiment": 3, "topic": 4}` là đa nhiệm (MTL), một đầu duy nhất là đơn
nhiệm (STL). Cùng kiến trúc đầu nên chỉ có một đường mã.

BẪY: lớp con `PreTrainedModel` phải đặt `base_model_prefix = "roberta"` VÀ giữ bộ mã hoá ở thuộc
tính `self.roberta`. Tên khác thì toàn bộ bộ mã hoá bị khởi tạo ngẫu nhiên mà không báo lỗi (chỉ
có bảng "LOAD REPORT" liệt kê các khoá MISSING). `tests/ml/test_phobert_model.py` canh điều này.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PreTrainedModel, RobertaConfig, RobertaModel

from edupulse.labels import LABELS, TASKS

if TYPE_CHECKING:
    from tokenizers import Tokenizer

BASE_MODEL = "vinai/phobert-base-v2"
# Ghim commit: repo tác giả còn đổi (PR safetensors chưa gộp). Trùng revision trong manifest bundle.
BASE_REVISION = "86cd7fd4c148980922ac11a2cf5e257f2ba639e1"
MAX_LEN = 128
PAD_ID = 1  # <pad> của PhoBERT


def head_sizes(heads: str) -> dict[str, int]:
    """'mtl' -> cả hai tác vụ; 'sentiment' hay 'topic' -> đơn nhiệm."""
    if heads == "mtl":
        tasks = TASKS
    elif heads in TASKS:
        tasks = (heads,)
    else:
        raise ValueError(f"heads phải là 'mtl' hoặc một trong {TASKS}, nhận {heads!r}")
    return {task: len(LABELS[task]) for task in tasks}


class MultiHeadConfig(RobertaConfig):
    """Cấu hình RoBERTa kèm kích thước từng đầu; lưu cùng `config.json` để nạp lại đúng đầu."""

    model_type = "roberta"  # giữ nguyên: checkpoint gốc cũng khai "roberta"

    def __init__(
        self, task_heads: dict[str, int] | None = None, head_dropout: float = 0.1, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.task_heads = dict(task_heads or {})
        self.head_dropout = head_dropout


class MultiHeadPhoBERT(PreTrainedModel):
    config_class = MultiHeadConfig
    base_model_prefix = "roberta"  # BẪY: phải khớp tiền tố khoá trọng số PhoBERT

    def __init__(self, config: MultiHeadConfig) -> None:
        super().__init__(config)
        if not config.task_heads:
            raise ValueError("config.task_heads rỗng: cần ít nhất một đầu ra")
        self.roberta = RobertaModel(config, add_pooling_layer=False)  # BẪY: tên thuộc tính
        self.dropout = nn.Dropout(config.head_dropout)
        self.heads = nn.ModuleDict(
            {name: nn.Linear(config.hidden_size, size) for name, size in config.task_heads.items()}
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        # Chỉ đầu mới cần khởi tạo; bộ mã hoá lấy trọng số từ checkpoint.
        if isinstance(module, nn.Linear) and module in set(self.heads.values()):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            nn.init.zeros_(module.bias)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Logits của từng đầu, tính từ trạng thái ẩn của token `<s>`."""
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled = self.dropout(hidden[:, 0])
        return {name: head(pooled) for name, head in self.heads.items()}


def load_pretrained(heads: str = "mtl") -> MultiHeadPhoBERT:
    """Bộ mã hoá PhoBERT-base-v2 (ghim commit) cộng các đầu khởi tạo mới."""
    return MultiHeadPhoBERT.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, task_heads=head_sizes(heads)
    )


def tokenizer_path() -> Path:
    """`tokenizer.json` của PhoBERT-base-v2 đúng commit đã ghim (tải một lần, dùng cache Hub)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(BASE_MODEL, "tokenizer.json", revision=BASE_REVISION))


def load_tokenizer(max_len: int = MAX_LEN) -> Tokenizer:
    """Tokenizer dùng chung cho train, export và serve: cắt ở `max_len`, không đệm sẵn.

    Cho id giống hệt `PhobertTokenizer` (đã kiểm trên val) nhưng nhanh hơn nhiều và không cần
    transformers lúc chạy.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path()))
    tokenizer.enable_truncation(max_length=max_len)
    return tokenizer
