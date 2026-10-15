"""PhoBERT-base-v2 nhiều đầu ra: một bộ mã hoá + đầu tuyến tính cho mỗi tác vụ.

Một lớp cho cả hai kiểu: `{"sentiment": 3, "topic": 4}` là đa nhiệm (MTL), một đầu duy nhất là đơn
nhiệm (STL). Cùng kiến trúc đầu nên chỉ có một đường mã.

BẪY: lớp con `PreTrainedModel` phải đặt `base_model_prefix = "roberta"` VÀ giữ bộ mã hoá ở thuộc
tính `self.roberta`. Tên khác thì toàn bộ bộ mã hoá bị khởi tạo ngẫu nhiên mà không báo lỗi (chỉ
có bảng "LOAD REPORT" liệt kê các khoá MISSING). `tests/ml/test_phobert_model.py` canh điều này.

BẪY thứ hai: `tokenizer.json` của PhoBERT-base-v2 có 66.119 mục (id tới 66118) nhưng bảng embedding
chỉ có 64.001 hàng. 2.118 mục thừa là mảnh BPE do merge tạo ra mà từ điển gốc không có (vd "đạt_");
tokenizer chậm chính thức đổi chúng thành `<unk>`. Đưa id đó vào mô hình làm sập huấn luyện (chỉ số
ngoài biên), nên: lúc train/eval dùng `encode_ids` (đổi id >= 64001 thành `<unk>`), còn mô hình
xuất ra ONNX được nới bảng embedding tới đủ 66.119 hàng, hàng thừa là bản sao hàng `<unk>`, nhờ vậy
mọi id của tokenizer.json đều chạy được và hành xử như `<unk>`, phía phục vụ không phải xử lý gì.
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
UNK_ID = 3  # <unk> của PhoBERT
MODEL_VOCAB_SIZE = 64001  # số hàng bảng embedding của checkpoint gốc (config.vocab_size)


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


def clamp_ids(ids: list[int]) -> list[int]:
    """Đổi id nằm ngoài bảng embedding gốc (>= 64.001) thành `<unk>`, như tokenizer chính thức."""
    return [UNK_ID if i >= MODEL_VOCAB_SIZE else i for i in ids]


def encode_ids(tokenizer: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Id token cho huấn luyện và chấm điểm bằng torch: luôn nằm trong bảng embedding gốc."""
    return [clamp_ids(encoding.ids) for encoding in tokenizer.encode_batch(texts)]


def expand_word_embeddings(model: MultiHeadPhoBERT, rows: int, fill_id: int = UNK_ID) -> None:
    """Nới bảng embedding từ vựng lên `rows` hàng tại chỗ; hàng mới sao chép hàng `fill_id`.

    Dùng khi xuất ONNX để mọi id của tokenizer.json chạy được và hành xử đúng như `<unk>`.
    """
    old = model.roberta.embeddings.word_embeddings
    if rows < old.num_embeddings:
        raise ValueError(f"chỉ nới được, không thu nhỏ: {old.num_embeddings} -> {rows}")
    new = nn.Embedding(rows, old.embedding_dim, padding_idx=old.padding_idx)
    with torch.no_grad():
        new.weight[: old.num_embeddings] = old.weight
        new.weight[old.num_embeddings :] = old.weight[fill_id]
    model.roberta.embeddings.word_embeddings = new
    model.config.vocab_size = rows
