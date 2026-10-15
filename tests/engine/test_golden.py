"""Đường phục vụ phải tái hiện dự đoán của 20 câu tự viết đã lưu trong bundle (hồi quy).

`golden_predictions.json` ghi xác suất thô (T = 1) của mô hình int8 chạy từng câu một (batch = 1)
trên CPU arm64. Nhãn phải bằng nhau ở mọi nền tảng; xác suất chỉ so chặt khi cùng kiến trúc, vì
nhân int8 của x86 khác nhân arm64.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pytest

from edupulse.bundle_schema import GoldenPredictions
from edupulse.engine.bundle import Bundle
from edupulse.engine.onnx_model import OnnxModel
from edupulse.engine.service import Analyzer
from edupulse.labels import LABELS
from edupulse.scoring import softmax_t
from edupulse.text import NORMALIZER_VERSION, prepare

pytestmark = pytest.mark.model

GOLDEN_SENTENCES = Path(__file__).resolve().parents[1] / "fixtures" / "golden_sentences.json"
PROB_TOLERANCE = 1e-4
SAME_ARCH = platform.machine().lower() in {"arm64", "aarch64"}


@pytest.fixture(scope="module")
def golden(bundle: Bundle) -> GoldenPredictions:
    return GoldenPredictions.model_validate_json(
        (bundle.root / "golden_predictions.json").read_bytes()
    )


@pytest.fixture(scope="module")
def sentences() -> dict[str, str]:
    items = json.loads(GOLDEN_SENTENCES.read_text(encoding="utf-8"))["items"]
    return {item["id"]: item["text"] for item in items}


def test_golden_file_covers_the_twenty_hand_written_sentences(golden, sentences):
    assert golden.normalizer_version == NORMALIZER_VERSION
    assert [item.id for item in golden.items] == list(sentences)
    assert len(golden.items) == 20
    # Điều kiện sinh golden nằm trong file thì mới so sánh được; chưa có trường thì coi là batch=1.
    assert golden.model_dump().get("batch_size", 1) == 1


def test_preprocessing_reproduces_the_stored_model_text(golden, sentences):
    for item in golden.items:
        assert prepare(sentences[item.id], use_ner=False).model_text == item.model_text, item.id


def test_serve_path_predicts_the_same_labels_as_the_golden_file(
    golden: GoldenPredictions, sentences: dict[str, str], analyzer: Analyzer
):
    for item in golden.items:
        analysis = analyzer.analyze(sentences[item.id])
        assert (analysis.sentiment.label, analysis.topic.label) == (item.sentiment, item.topic), (
            item.id
        )


@pytest.mark.skipif(not SAME_ARCH, reason="golden sinh trên arm64; x86 chỉ kỳ vọng nhãn bằng nhau")
def test_raw_probabilities_match_to_four_decimals(golden: GoldenPredictions, bundle: Bundle):
    model = OnnxModel(bundle, threads=2)
    for item in golden.items:
        logits = model.predict([item.model_text])
        np.testing.assert_allclose(
            softmax_t(logits.sentiment[0], 1.0),
            item.p_sentiment,
            atol=PROB_TOLERANCE,
            err_msg=item.id,
        )
        np.testing.assert_allclose(
            softmax_t(logits.topic[0], 1.0), item.p_topic, atol=PROB_TOLERANCE, err_msg=item.id
        )


def test_golden_labels_are_valid_labels(golden: GoldenPredictions):
    for item in golden.items:
        assert item.sentiment in LABELS["sentiment"] and item.topic in LABELS["topic"]
        assert item.sentiment == LABELS["sentiment"][int(np.argmax(item.p_sentiment))]
        assert item.topic == LABELS["topic"][int(np.argmax(item.p_topic))]
