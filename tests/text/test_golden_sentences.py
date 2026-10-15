"""20 câu vàng tự viết: phủ 3 cảm xúc x 4 chủ đề và các ca đặc biệt của tiền xử lý."""

import json
from pathlib import Path

import pytest

from edupulse.config import XAI_MAX_WORDS
from edupulse.labels import SENTIMENT, TOPIC
from edupulse.text import prepare

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "golden_sentences.json"
ITEMS = {item["id"]: item for item in json.loads(FIXTURE.read_text(encoding="utf-8"))["items"]}


def test_there_are_twenty_unique_sentences():
    assert len(ITEMS) == 20
    assert len({item["text"] for item in ITEMS.values()}) == 20


def test_every_sentiment_topic_pair_is_covered():
    pairs = {(item["intended"]["sentiment"], item["intended"]["topic"]) for item in ITEMS.values()}
    assert {(s, t) for s in SENTIMENT for t in TOPIC} <= pairs


def test_intended_labels_are_valid():
    for item in ITEMS.values():
        assert item["intended"]["sentiment"] in SENTIMENT
        assert item["intended"]["topic"] in TOPIC


@pytest.mark.java
def test_special_cases_behave_as_documented():
    prepared = {key: prepare(item["text"], use_ner=False) for key, item in ITEMS.items()}
    assert prepared["g13"].no_diacritics
    assert not any(p.no_diacritics for key, p in prepared.items() if key != "g13")
    assert prepared["g14"].pii_types == ("phone",) and "[SĐT]" in prepared["g14"].display_text
    assert prepared["g15"].pii_types == ("name",) and "[TÊN]" in prepared["g15"].display_text
    assert "colonsmile" in prepared["g16"].model_text.split()
    assert prepared["g17"].n_words > XAI_MAX_WORDS
    assert all(p.n_words <= XAI_MAX_WORDS for key, p in prepared.items() if key != "g17")
    assert "11doubledot55" in prepared["g18"].model_text.split()
    for old in ("thủy", "hòa", "khóa"):
        assert old not in prepared["g19"].model_text
    assert "vdotv" in prepared["g20"].model_text.split()
    assert prepared["g20"].model_text.endswith("!")
    assert not any(p.pii_types for key, p in prepared.items() if key not in {"g14", "g15"})
