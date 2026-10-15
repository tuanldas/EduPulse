"""Suy luận ONNX trên bundle thật: hình dạng đầu ra, thứ tự, chia lô, cắt độ dài, đa luồng.

Int8 động tính scale kích hoạt theo cả tensor nên logits của một câu lệch nhẹ khi đổi các câu cùng
lô; test so với dung sai thay vì `==`, trừ khi cùng cách xếp lô.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from edupulse.engine.bundle import Bundle
from edupulse.engine.onnx_model import OnnxModel
from edupulse.labels import SENTIMENT, TOPIC

pytestmark = pytest.mark.model

# Dài ngắn khác hẳn nhau, để việc sắp theo độ dài thật sự đổi thứ tự câu trong lô.
TEXTS = [
    "thầy dạy rất hay",
    "phòng học nóng , máy chiếu mờ và micro thường_xuyên bị rè nên rất khó nghe bài giảng của thầy",
    "môn này có hai bài kiểm_tra giữa kỳ",
    "thư_viện mới rất rộng_rãi , có máy lạnh và ổ cắm điện ở mọi chỗ ngồi , em rất hài_lòng",
    "ok",
    "thủ_tục xin giấy xác_nhận quá rườm_rà , em phải đi lại nhiều lần",
]
LOGIT_TOLERANCE = 0.25


@pytest.fixture(scope="module")
def model(bundle: Bundle) -> OnnxModel:
    return OnnxModel(bundle, threads=2)


def test_logits_have_one_row_per_sentence_and_one_column_per_label(model: OnnxModel):
    logits = model.predict(TEXTS)
    assert logits.sentiment.shape == (len(TEXTS), len(SENTIMENT))
    assert logits.topic.shape == (len(TEXTS), len(TOPIC))
    assert logits.sentiment.dtype == np.float32
    assert np.isfinite(logits.sentiment).all() and np.isfinite(logits.topic).all()


def test_empty_input_gives_empty_arrays(model: OnnxModel):
    logits = model.predict([])
    assert logits.sentiment.shape == (0, len(SENTIMENT))
    assert logits.topic.shape == (0, len(TOPIC))


def test_batching_and_length_sorting_keep_results_in_input_order(model: OnnxModel):
    batched = model.predict(TEXTS)
    for i, text in enumerate(TEXTS):
        alone = model.predict([text])
        np.testing.assert_allclose(batched.sentiment[i], alone.sentiment[0], atol=LOGIT_TOLERANCE)
        np.testing.assert_allclose(batched.topic[i], alone.topic[0], atol=LOGIT_TOLERANCE)


def test_reversing_the_input_reverses_the_output(model: OnnxModel):
    forward = model.predict(TEXTS)
    backward = model.predict(TEXTS[::-1])
    np.testing.assert_allclose(forward.sentiment, backward.sentiment[::-1], atol=LOGIT_TOLERANCE)
    np.testing.assert_allclose(forward.topic, backward.topic[::-1], atol=LOGIT_TOLERANCE)


def test_small_batches_agree_with_one_big_batch(model: OnnxModel):
    big = model.predict(TEXTS, batch_size=32)
    small = model.predict(TEXTS, batch_size=2)
    np.testing.assert_allclose(big.sentiment, small.sentiment, atol=LOGIT_TOLERANCE)
    np.testing.assert_allclose(big.topic, small.topic, atol=LOGIT_TOLERANCE)


def test_single_sentence_runs_are_reproducible(model: OnnxModel):
    first = model.predict([TEXTS[1]])
    second = model.predict([TEXTS[1]])
    np.testing.assert_array_equal(first.sentiment, second.sentiment)
    np.testing.assert_array_equal(first.topic, second.topic)


def test_progress_reports_sentences_done_after_each_batch(model: OnnxModel):
    reported: list[int] = []
    model.predict(TEXTS, batch_size=4, progress=reported.append)
    assert reported == [4, 6]


def test_input_beyond_max_len_is_truncated_not_rejected(model: OnnxModel, bundle: Bundle):
    max_len = bundle.manifest.max_len
    # Mỗi "rất" là một token; thêm <s> và </s> thì 126 từ vừa đúng max_len.
    exactly_full = " ".join(["rất"] * (max_len - 2))
    far_too_long = " ".join(["rất"] * (max_len * 4))
    full = model.predict([exactly_full])
    cut = model.predict([far_too_long])
    np.testing.assert_array_equal(full.sentiment, cut.sentiment)
    np.testing.assert_array_equal(full.topic, cut.topic)


def test_concurrent_calls_match_sequential_calls(model: OnnxModel):
    sequential = [model.predict([text]) for text in TEXTS]
    with ThreadPoolExecutor(max_workers=6) as pool:
        concurrent = list(pool.map(lambda text: model.predict([text]), TEXTS * 3))
    for i, result in enumerate(concurrent):
        expected = sequential[i % len(TEXTS)]
        np.testing.assert_array_equal(result.sentiment, expected.sentiment)
        np.testing.assert_array_equal(result.topic, expected.topic)
