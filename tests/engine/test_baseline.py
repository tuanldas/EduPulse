"""Mô hình cơ sở TF-IDF + LinearSVC: nạp lười, dự đoán, giải thích bằng hệ số, lỗi nạp."""

import threading
from pathlib import Path

import numpy as np
import pytest

from edupulse.config import load_settings
from edupulse.engine.baseline import (
    FILENAME,
    NAME,
    TOP_TERMS,
    Baseline,
    BaselineUnavailable,
)
from edupulse.engine.bundle import load_bundle
from edupulse.labels import LABELS
from edupulse.text import ngrams, prepare

pytestmark = pytest.mark.model

SENTENCES = [
    "Thầy giải thích rõ ràng, em nghe một lần là hiểu.",
    "Phòng học có máy chiếu quá mờ, quạt kêu to.",
    "Giảng viên hay vào lớp muộn và ra về sớm.",
    "Chương trình đào tạo bám sát nhu cầu tuyển dụng.",
    "Em không có ý kiến gì về vấn đề này.",
    "Giáo trình đã cũ, cần cập nhật thêm ví dụ thực tế.",
]


@pytest.fixture(scope="module")
def baseline_dir() -> Path:
    return load_bundle(load_settings().bundle_dir, verify_files=False).root / "baseline"


@pytest.fixture(scope="module")
def baseline(baseline_dir: Path) -> Baseline:
    return Baseline(baseline_dir)


def model_text(text: str) -> str:
    return prepare(text, use_ner=False).model_text


def test_models_are_loaded_only_when_first_asked(baseline_dir: Path):
    fresh = Baseline(baseline_dir)
    assert not fresh.loaded
    fresh.predict(model_text(SENTENCES[0]))
    assert fresh.loaded


def test_concurrent_first_calls_load_the_models_once(baseline_dir: Path):
    fresh = Baseline(baseline_dir)
    text = model_text(SENTENCES[0])
    results, loaded = [], []

    def worker() -> None:
        results.append(fresh.predict(text))
        loaded.append(id(fresh._models))

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(results) == 6 and len(set(loaded)) == 1
    assert len({(r.sentiment.label, r.topic.label) for r in results}) == 1


def test_predictions_use_the_project_labels_and_calibrated_probabilities(baseline: Baseline):
    for text in SENTENCES:
        result = baseline.predict(model_text(text))
        assert result.sentiment.label in LABELS["sentiment"]
        assert result.topic.label in LABELS["topic"]
        assert 1 / 3 <= result.sentiment.confidence <= 1.0
        assert 1 / 4 <= result.topic.confidence <= 1.0
        assert result.elapsed_ms > 0


def test_obvious_sentences_get_the_obvious_labels(baseline: Baseline):
    nice = baseline.predict(model_text(SENTENCES[0]))
    assert (nice.sentiment.label, nice.topic.label) == ("positive", "lecturer")
    room = baseline.predict(model_text(SENTENCES[1]))
    assert (room.sentiment.label, room.topic.label) == ("negative", "facility")


def test_decisive_terms_come_from_the_sentence_and_are_at_most_three(baseline: Baseline):
    for text in SENTENCES:
        prepared = model_text(text)
        grams = set(ngrams(prepared)) | set(prepared.split())
        result = baseline.predict(prepared)
        for head in (result.sentiment, result.topic):
            assert len(head.top_terms) <= TOP_TERMS
            assert set(head.top_terms) <= grams, (text, head.top_terms)
            assert len(set(head.top_terms)) == len(head.top_terms)


def test_decisive_terms_are_the_largest_positive_coefficient_contributions(
    baseline: Baseline,
):
    """Tính lại độc lập: đóng góp = hệ số của lớp dự đoán × giá trị TF-IDF của đặc trưng."""
    text = model_text(SENTENCES[2])
    result = baseline.predict(text)
    models = baseline._ensure_loaded()
    model = models["topic"]
    index = LABELS["topic"].index(result.topic.label)
    row = model.features.transform([text]).toarray()[0][: model.n_word]
    contributions = row * model.classifier.coef_[index, : model.n_word]
    best = np.argsort(-contributions, kind="stable")[:TOP_TERMS]
    expected = tuple(str(model.word_terms[i]) for i in best if contributions[i] > 0)
    assert result.topic.top_terms == expected
    assert expected  # câu này có đặc trưng ủng hộ rõ ràng


def test_a_sentence_with_no_known_words_has_no_decisive_terms(baseline: Baseline):
    result = baseline.predict("qwertyuiop asdfghjkl")
    assert result.sentiment.top_terms == () and result.topic.top_terms == ()


def test_the_comparison_is_much_faster_than_the_main_model(baseline: Baseline):
    baseline.predict(model_text(SENTENCES[0]))  # khởi động
    assert max(baseline.predict(model_text(s)).elapsed_ms for s in SENTENCES) < 100


def test_name_matches_the_model_card(baseline: Baseline):
    assert NAME == "TF-IDF + LinearSVC" and FILENAME.format(task="topic") == "topic_linearsvc.skops"


# --- Lỗi nạp --------------------------------------------------------------------------------------


def test_missing_files_are_reported_as_unavailable(tmp_path: Path):
    with pytest.raises(BaselineUnavailable) as failure:
        Baseline(tmp_path).predict("thầy dạy hay")
    assert failure.value.code == "baseline_unavailable"
    assert "sentiment_linearsvc.skops" in str(failure.value) or "topic" in str(failure.value)


def test_corrupt_files_are_reported_as_unavailable(tmp_path: Path):
    for task in ("sentiment", "topic"):
        (tmp_path / FILENAME.format(task=task)).write_bytes(b"khong phai skops")
    fresh = Baseline(tmp_path)
    with pytest.raises(BaselineUnavailable):
        fresh.predict("thầy dạy hay")
    assert not fresh.loaded  # lần sau vẫn thử lại thay vì giữ trạng thái nửa vời


def test_a_pickle_in_place_of_a_skops_file_is_refused(tmp_path: Path):
    import pickle

    class Evil:
        def __reduce__(self):
            return (print, ("không được chạy",))

    for task in ("sentiment", "topic"):
        (tmp_path / FILENAME.format(task=task)).write_bytes(pickle.dumps(Evil()))
    with pytest.raises(BaselineUnavailable):
        Baseline(tmp_path).predict("thầy dạy hay")


# --- Chất lượng trên val (cần `make data`) --------------------------------------------------------


@pytest.mark.data
def test_baseline_accuracy_on_validation_sentences_is_in_the_expected_range(baseline: Baseline):
    pq = pytest.importorskip("pyarrow.parquet")

    path = load_settings().data_dir / "raw" / "validation.parquet"
    if not path.is_file():
        pytest.skip("chưa có data/raw — chạy `make data`")
    table = pq.read_table(path)
    texts = table.column("sentence").to_pylist()[:300]
    sentiment = table.column("sentiment").to_pylist()[:300]
    topic = table.column("topic").to_pylist()[:300]
    right_sentiment = right_topic = 0
    for text, gold_s, gold_t in zip(texts, sentiment, topic, strict=True):
        result = baseline.predict(model_text(text))
        right_sentiment += result.sentiment.label == LABELS["sentiment"][gold_s]
        right_topic += result.topic.label == LABELS["topic"][gold_t]
    assert right_sentiment / 300 > 0.85  # LinearSVC cảm xúc ~92% trên val
    assert right_topic / 300 > 0.80  # chủ đề ~87%
