"""`tokenizer.json` (train, export, serve dùng chung) phải cho id giống hệt PhobertTokenizer."""

import pytest

from ml.common import load_processed
from ml.phobert_model import BASE_MODEL, BASE_REVISION, MAX_LEN, PAD_ID, load_tokenizer

pytestmark = [pytest.mark.data, pytest.mark.train]


def test_ids_match_the_reference_tokenizer_on_every_validation_sentence():
    transformers = pytest.importorskip("transformers")
    try:
        reference = transformers.AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    except OSError as error:
        pytest.skip(f"không tải được tokenizer tham chiếu: {error}")
    sentences = load_processed("validation")["text_rdr"].tolist()
    expected = reference(sentences, truncation=True, max_length=MAX_LEN)["input_ids"]
    ours = [encoding.ids for encoding in load_tokenizer().encode_batch(sentences)]
    assert len(ours) == len(expected) == 1583
    mismatches = [i for i, (a, b) in enumerate(zip(ours, expected, strict=True)) if a != b]
    assert mismatches == []
    assert reference.pad_token_id == PAD_ID
