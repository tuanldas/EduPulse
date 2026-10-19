"""Giới hạn số từ của giải thích: mặc định 25, chỉnh bằng EDUPULSE_XAI_MAX_WORDS."""

import dataclasses

import pytest

from edupulse.config import XAI_MAX_WORDS, load_settings
from edupulse.engine.explain import ExplanationUnavailable
from edupulse.engine.service import Analyzer
from edupulse.text import prepare


def test_default_limit_is_25_words():
    assert XAI_MAX_WORDS == 25
    assert load_settings({}).xai_max_words == 25


def test_environment_overrides_the_limit():
    assert load_settings({"EDUPULSE_XAI_MAX_WORDS": "40"}).xai_max_words == 40
    assert load_settings({"EDUPULSE_XAI_MAX_WORDS": " 12 "}).xai_max_words == 12


def test_blank_environment_value_means_default():
    assert load_settings({"EDUPULSE_XAI_MAX_WORDS": ""}).xai_max_words == XAI_MAX_WORDS


@pytest.mark.parametrize("value", ["0", "-3", "hai muoi", "2.5"])
def test_invalid_limit_names_the_variable(value: str):
    with pytest.raises(ValueError, match="EDUPULSE_XAI_MAX_WORDS"):
        load_settings({"EDUPULSE_XAI_MAX_WORDS": value})


@pytest.mark.model
def test_analyzer_limit_comes_from_the_settings(analyzer: Analyzer, settings):
    assert analyzer.xai_max_words == settings.xai_max_words


@pytest.mark.model
def test_analyzer_refuses_sentences_above_its_own_limit(analyzer: Analyzer):
    text = "Giảng viên dạy rất hay và nhiệt tình ."
    n_words = prepare(text, use_ner=False).n_words
    strict = Analyzer(analyzer.bundle, analyzer._model, xai_max_words=n_words - 1)
    with pytest.raises(ExplanationUnavailable) as failure:
        strict.explain(text)
    assert (failure.value.n_words, failure.value.max_words) == (n_words, n_words - 1)

    lenient = Analyzer(analyzer.bundle, analyzer._model, xai_max_words=n_words)
    assert lenient.explain(text).n_forwards > 1


def test_settings_stay_immutable():
    settings = load_settings({})
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.xai_max_words = 99  # type: ignore[misc]
