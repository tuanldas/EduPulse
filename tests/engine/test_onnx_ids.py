"""Id token ngoài bảng embedding của PhoBERT phải được đọc là <unk>, y như lúc huấn luyện."""

import numpy as np

from edupulse.engine.onnx_model import EMBEDDING_ROWS, UNK_ID, clamp_unknown_ids


def test_ids_inside_the_embedding_table_are_untouched():
    ids = np.array([[0, 1, 2, 3, 47001, EMBEDDING_ROWS - 1]], dtype=np.int64)
    assert clamp_unknown_ids(ids.copy()).tolist() == ids.tolist()


def test_ids_beyond_the_table_become_unk():
    ids = np.array([[0, EMBEDDING_ROWS, 65771, 66118, 5603, 2]], dtype=np.int64)
    assert clamp_unknown_ids(ids).tolist() == [[0, UNK_ID, UNK_ID, UNK_ID, 5603, 2]]


def test_clamping_works_in_place_and_on_empty_input():
    ids = np.full((2, 3), 70000, dtype=np.int64)
    assert clamp_unknown_ids(ids) is ids and set(ids.ravel().tolist()) == {UNK_ID}
    assert clamp_unknown_ids(np.empty((0, 0), dtype=np.int64)).size == 0


def test_special_ids_the_model_relies_on_stay_inside_the_table():
    # <s>=0, <pad>=1, </s>=2, <unk>=3 và <mask>=64000 đều nằm trong bảng.
    assert max(0, 1, 2, UNK_ID, EMBEDDING_ROWS - 1) < EMBEDDING_ROWS
