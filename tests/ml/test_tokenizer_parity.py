"""`tokenizer.json` (train, export, serve dùng chung) phải cho id giống hệt PhobertTokenizer.

Ngoại lệ đã biết: tokenizer.json có 66.119 mục còn bảng embedding chỉ 64.001 hàng; mảnh BPE nằm
ngoài bảng (id >= 64.001) bị tokenizer chính thức đổi thành `<unk>`. Test canh cả hai điều này.
"""

import pytest

from ml.common import load_processed
from ml.phobert_model import (
    BASE_MODEL,
    BASE_REVISION,
    MAX_LEN,
    MODEL_VOCAB_SIZE,
    PAD_ID,
    UNK_ID,
    encode_ids,
    load_tokenizer,
)

pytestmark = pytest.mark.train

RARE_PIECE_TEXT = "thầy truyền_đạt_kiến thức rất tốt ."


@pytest.fixture(scope="module")
def reference():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    except OSError as error:
        pytest.skip(f"không tải được tokenizer tham chiếu: {error}")


def test_reference_special_ids_are_the_ones_the_code_assumes(reference):
    assert reference.pad_token_id == PAD_ID
    assert reference.unk_token_id == UNK_ID
    assert reference.vocab_size + 1 == MODEL_VOCAB_SIZE


def test_rare_piece_text_matches_the_reference_only_after_clamping(reference):
    expected = reference([RARE_PIECE_TEXT], truncation=True, max_length=MAX_LEN)["input_ids"][0]
    raw = load_tokenizer().encode(RARE_PIECE_TEXT).ids
    assert raw != expected and max(raw) >= MODEL_VOCAB_SIZE  # id thô lệch tokenizer chính thức
    assert encode_ids(load_tokenizer(), [RARE_PIECE_TEXT])[0] == expected
    assert UNK_ID in expected


@pytest.mark.data
def test_ids_match_the_reference_tokenizer_on_every_validation_sentence(reference):
    sentences = load_processed("validation")["text_rdr"].tolist()
    expected = reference(sentences, truncation=True, max_length=MAX_LEN)["input_ids"]
    ours = [encoding.ids for encoding in load_tokenizer().encode_batch(sentences)]
    assert len(ours) == len(expected) == 1583
    assert [i for i, (a, b) in enumerate(zip(ours, expected, strict=True)) if a != b] == []


@pytest.mark.data
@pytest.mark.parametrize("split", ["train", "validation"])
def test_clamped_ids_match_the_reference_on_every_sentence(reference, split):
    sentences = load_processed(split)["text_rdr"].tolist()
    expected = reference(sentences, truncation=True, max_length=MAX_LEN)["input_ids"]
    ours = encode_ids(load_tokenizer(), sentences)
    mismatches = [i for i, (a, b) in enumerate(zip(ours, expected, strict=True)) if a != b]
    assert mismatches == []
    assert max(max(ids) for ids in ours) < MODEL_VOCAB_SIZE


@pytest.mark.data
def test_the_raw_tokenizer_does_emit_out_of_table_ids_on_real_training_text():
    """Hồi quy cho lỗi đã làm sập một lần huấn luyện: hai câu train có mảnh "đạt_"."""
    sentences = load_processed("train")["text_rdr"].tolist()
    raw = [e.ids for e in load_tokenizer().encode_batch(sentences)]
    assert sum(max(ids) >= MODEL_VOCAB_SIZE for ids in raw) >= 1
