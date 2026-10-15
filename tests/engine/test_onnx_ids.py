"""Id token ngoài bảng embedding gốc của PhoBERT phải được đọc là <unk>, y như lúc huấn luyện."""

import numpy as np

from edupulse.bundle_schema import OnnxSpec
from edupulse.engine.onnx_model import clamp_unknown_ids

SPEC = OnnxSpec(quantization="test")  # giá trị mặc định = PhoBERT-base-v2: 64001 hàng, <unk> = 3
BASE, UNK = SPEC.base_vocab_size, SPEC.unk_id


def test_defaults_describe_phobert_base_v2():
    assert (BASE, UNK) == (64001, 3)


def test_ids_inside_the_embedding_table_are_untouched():
    ids = np.array([[0, 1, 2, 3, 47001, BASE - 1]], dtype=np.int64)
    assert clamp_unknown_ids(ids.copy(), BASE, UNK).tolist() == ids.tolist()


def test_ids_beyond_the_table_become_unk():
    ids = np.array([[0, BASE, 65771, 66118, 5603, 2]], dtype=np.int64)
    assert clamp_unknown_ids(ids, BASE, UNK).tolist() == [[0, UNK, UNK, UNK, 5603, 2]]


def test_clamping_works_in_place_and_on_empty_input():
    ids = np.full((2, 3), 70000, dtype=np.int64)
    assert clamp_unknown_ids(ids, BASE, UNK) is ids and set(ids.ravel().tolist()) == {UNK}
    assert clamp_unknown_ids(np.empty((0, 0), dtype=np.int64), BASE, UNK).size == 0


def test_the_threshold_comes_from_the_manifest_not_a_constant():
    ids = np.array([[5, 10, 20]], dtype=np.int64)
    assert clamp_unknown_ids(ids, 10, 3).tolist() == [[5, 3, 3]]
