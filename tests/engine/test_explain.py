"""Giải thích bằng bỏ từng từ: hiển thị token, chọn từ ủng hộ hay đi ngược, chạy mô hình thật."""

import numpy as np
import pytest

from edupulse.bundle_schema import HeadCalibration
from edupulse.config import XAI_MAX_WORDS
from edupulse.engine.explain import (
    MIN_EFFECT,
    TOP_OPPOSE,
    TOP_SUPPORT,
    ExplanationUnavailable,
    _explain_head,
    display_token,
    is_punctuation,
)
from edupulse.engine.service import Analyzer
from edupulse.scoring import CDF_POINTS, softmax_t
from edupulse.text import prepare


@pytest.mark.parametrize(
    ("token", "shown"),
    [
        ("giảng_viên", "giảng viên"),
        ("cơ_sở_vật_chất", "cơ sở vật chất"),
        ("wzjwz1", "[TÊN]"),
        ("wzjwz12", "[TÊN]"),
        ("colonsmile", ":)"),
        ("colonsmilesmile", ":))"),
        ("colonsad", ":("),
        ("colonbigsmile", ":D"),
        ("colonlove", "<3"),
        ("doubledot", ":"),
        ("vdotv", "v.v"),
        ("dotdotdot", "..."),
        ("3dot0", "3.0"),
        ("9dot5", "9.5"),
        ("uitdotedudotvn", "uit.edu.vn"),
        ("dotnet", ".net"),
        ("thầy", "thầy"),
        (",", ","),
        ("dotation", "dotation"),
    ],
)
def test_display_token_restores_names_and_symbols(token, shown):
    assert display_token(token) == shown


@pytest.mark.java
def test_display_token_inverts_what_the_normalizer_produces():
    # Chuẩn hoá đổi ký hiệu thành token chữ; hiển thị phải đổi ngược lại được.
    model_text = prepare("Em thích :) và điểm 3.0, xem uit.edu.vn nhé :(", use_ner=False).model_text
    shown = " ".join(display_token(token) for token in model_text.split())
    for symbol in (":)", "3.0", "uit.edu.vn", ":("):
        assert symbol in shown


@pytest.mark.parametrize(
    "token", [",", ".", "!", "?", ";", "(", ")", '"', "-", "%", "doubledot", "dotdotdot"]
)
def test_punctuation_tokens_are_recognised(token):
    assert is_punctuation(token)


@pytest.mark.parametrize(
    "token",
    ["thầy", "giảng_viên", "wzjwz1", "colonsmile", "colonsad", "vdotv", "3dot0", "50%", "a,"],
)
def test_words_names_emoticons_and_numbers_are_not_punctuation(token):
    assert not is_punctuation(token)


def _head_calibration(temperature: float = 1.0) -> HeadCalibration:
    return HeadCalibration(temperature=temperature, cdf=list(np.linspace(0.3, 1.0, CDF_POINTS)))


def _logits_with_drops(drops: list[float]) -> np.ndarray:
    """Hàng 0 là câu đủ (lớp 0 thắng); hàng i+1 là câu bỏ từ i, logit lớp 0 giảm đúng `drops[i]`."""
    full = np.array([2.0, 0.0, 0.0])
    rows = [full] + [full + np.array([-d, 0.0, 0.0]) for d in drops]
    return np.stack(rows)


def test_support_and_oppose_pick_the_largest_effects_in_order():
    # Bỏ từ 0,1,2,3,4,5 làm logit lớp thắng đổi -0.5, -2, -1, +1, 0, -0.01.
    shown = ["a", "b", "c", "d", "e", "f"]
    head = _explain_head(
        "sentiment",
        _logits_with_drops([0.5, 2.0, 1.0, -1.0, 0.0, 0.01]),
        _head_calibration(),
        shown,
    )
    assert head.label == "negative"
    assert [t.text for t in head.tokens] == shown
    assert head.support == (1, 2, 0)[:TOP_SUPPORT]  # mạnh nhất trước
    assert head.oppose == (3,)[:TOP_OPPOSE]
    # Từ 4 (không đổi) và từ 5 (đổi dưới ngưỡng) không phải bằng chứng cho hay chống.
    assert 4 not in head.support + head.oppose and 5 not in head.support + head.oppose


def test_delta_is_the_drop_in_probability_of_the_predicted_label():
    logits = _logits_with_drops([1.0, 0.0])
    head = _explain_head("sentiment", logits, _head_calibration(), ["x", "y"])
    probs = softmax_t(logits, 1.0)
    assert head.probability == pytest.approx(probs[0, 0])
    assert head.tokens[0].p_without == pytest.approx(probs[1, 0])
    assert head.tokens[0].delta == pytest.approx(probs[0, 0] - probs[1, 0])
    assert head.tokens[1].delta == pytest.approx(0.0)


def test_temperature_changes_the_probabilities_shown():
    logits = _logits_with_drops([1.0])
    cold = _explain_head("sentiment", logits, _head_calibration(1.0), ["x"])
    hot = _explain_head("sentiment", logits, _head_calibration(3.0), ["x"])
    assert hot.probability < cold.probability  # T > 1 làm xác suất bớt tự tin


def test_ties_prefer_the_earlier_word_and_small_effects_are_ignored():
    head = _explain_head(
        "sentiment", _logits_with_drops([1.0, 1.0, 1.0, 1.0]), _head_calibration(), list("abcd")
    )
    assert head.support == (0, 1, 2)
    assert head.oppose == ()
    tiny = _explain_head("sentiment", _logits_with_drops([0.001]), _head_calibration(), ["a"])
    assert tiny.tokens[0].delta < MIN_EFFECT
    assert tiny.support == () and tiny.oppose == ()


def test_single_token_sentence_has_one_effect():
    head = _explain_head(
        "topic", np.array([[3.0, 0, 0, 0], [0.0, 0, 0, 0]]), _head_calibration(), ["x"]
    )
    assert len(head.tokens) == 1 and head.support == (0,)


def test_tokens_that_are_not_occluded_have_zero_effect_and_are_never_selected():
    # Câu "a , b": chỉ che từ 0 và 2; dấu phẩy (vị trí 1) không được chạy lại.
    logits = _logits_with_drops([2.0, 2.0])
    head = _explain_head("sentiment", logits, _head_calibration(), ["a", ",", "b"], occluded=[0, 2])
    comma = head.tokens[1]
    assert comma.delta == 0.0 and comma.p_without == pytest.approx(head.probability)
    assert head.tokens[0].delta > MIN_EFFECT and head.tokens[2].delta > MIN_EFFECT
    assert head.support == (0, 2)


def test_a_sentence_of_only_punctuation_has_nothing_to_occlude():
    head = _explain_head(
        "sentiment", _logits_with_drops([]), _head_calibration(), ["!", "?"], occluded=[]
    )
    assert [t.delta for t in head.tokens] == [0.0, 0.0]
    assert head.support == () and head.oppose == ()


# --- Trên mô hình thật --------------------------------------------------------------------------


@pytest.mark.model
def test_occlusion_runs_one_forward_per_non_punctuation_word_plus_the_sentence(analyzer: Analyzer):
    text = "Cô hướng dẫn tỉ mỉ, bài tập vừa sức, em học thấy vui :)"
    prepared = prepare(text, use_ner=False)
    tokens = prepared.model_text.split()
    words = [t for t in tokens if not is_punctuation(t)]
    assert len(words) < len(tokens), "câu thử phải có dấu câu để kiểm tra việc bỏ qua"
    explanation = analyzer.explain_prepared(prepared)

    assert explanation.n_forwards == len(words) + 1
    for head in (explanation.sentiment, explanation.topic):
        assert len(head.tokens) == prepared.n_words
        assert 0.0 < head.probability <= 1.0
        assert len(head.support) <= TOP_SUPPORT and len(head.oppose) <= TOP_OPPOSE
        assert all(0 <= i < prepared.n_words for i in head.support + head.oppose)
        assert all(head.tokens[i].delta >= MIN_EFFECT for i in head.support)
        assert all(head.tokens[i].delta <= -MIN_EFFECT for i in head.oppose)
        for token, effect in zip(tokens, head.tokens, strict=True):
            if is_punctuation(token):
                assert effect.delta == 0.0
            assert effect.p_without == pytest.approx(head.probability - effect.delta, abs=1e-6)
        selected = {tokens[i] for i in head.support + head.oppose}
        assert not any(is_punctuation(t) for t in selected)
    assert ":)" in [t.text for t in explanation.sentiment.tokens]


@pytest.mark.model
def test_explanation_reproduces_the_analyze_prediction_exactly(analyzer: Analyzer):
    # Câu đầy đủ chạy lô 1 như analyze nên nhãn và xác suất giống hệt (không lệch theo bạn cùng lô).
    for text in (
        "Thầy giảng bài rất dễ hiểu, em rất thích.",
        "Phòng học nóng, máy chiếu mờ và micro thường xuyên bị rè nên rất khó nghe.",
        "Môn này có hai bài kiểm tra giữa kỳ và một bài thi cuối kỳ.",
    ):
        analysis = analyzer.analyze(text)
        explanation = analyzer.explain(text)
        assert explanation.sentiment.label == analysis.sentiment.label
        assert explanation.sentiment.probability == analysis.sentiment.confidence
        assert explanation.topic.label == analysis.topic.label
        assert explanation.topic.probability == analysis.topic.confidence


@pytest.mark.model
def test_explanation_label_matches_the_analysis_label(analyzer: Analyzer):
    text = "Phòng học nóng, máy chiếu mờ và micro thường xuyên bị rè nên rất khó nghe."
    analysis = analyzer.analyze(text)
    explanation = analyzer.explain(text)
    assert explanation.sentiment.label == analysis.sentiment.label
    assert explanation.topic.label == analysis.topic.label


@pytest.mark.model
def test_sentence_longer_than_the_limit_is_not_explained(analyzer: Analyzer):
    text = " ".join(["giảng viên dạy rất hay"] * 12)
    assert prepare(text, use_ner=False).n_words > XAI_MAX_WORDS
    with pytest.raises(ExplanationUnavailable) as failure:
        analyzer.explain(text)
    assert failure.value.code == "xai_too_long"
    assert failure.value.max_words == XAI_MAX_WORDS


@pytest.mark.model
def test_sentence_at_the_limit_is_still_explained(analyzer: Analyzer):
    words = ["rất"] * (XAI_MAX_WORDS - 2) + ["hay", "."]
    prepared = prepare(" ".join(words), use_ner=False)
    assert prepared.n_words <= XAI_MAX_WORDS
    tokens = prepared.model_text.split()
    expected = sum(not is_punctuation(t) for t in tokens) + 1
    assert analyzer.explain_prepared(prepared).n_forwards == expected
