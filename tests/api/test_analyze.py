"""`/api/analyze` và `/api/explain` qua TestClient với bundle thật (marker `model`).

Cờ, nhãn và độ tin cậy phụ thuộc mô hình nên test chỉ kiểm tra những gì đúng với mọi mô hình hợp
lệ: hợp đồng dữ liệu, luật (ẩn PII, không dấu, giới hạn độ dài) và sự nhất quán giữa hai endpoint.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import get_args

import pytest
from fastapi.testclient import TestClient

from edupulse.api import schemas
from edupulse.config import MAX_CHARS, XAI_MAX_WORDS
from edupulse.labels import LABELS

SENTENCE = "Thầy giảng bài rất dễ hiểu, em rất thích môn này."


def post(client: TestClient, path: str, text: str):
    return client.post(path, json={"text": text})


def test_schema_literals_match_the_label_order_in_code():
    assert get_args(schemas.SentimentLabel) == LABELS["sentiment"]
    assert get_args(schemas.TopicLabel) == LABELS["topic"]


@pytest.mark.model
def test_analyze_returns_both_heads_with_consistent_probabilities(live_client: TestClient):
    response = post(live_client, "/api/analyze", SENTENCE)
    assert response.status_code == 200
    body = schemas.AnalyzeResponse.model_validate(response.json())

    for head, task in ((body.sentiment, "sentiment"), (body.topic, "topic")):
        assert list(head.probs) == list(LABELS[task])
        assert sum(head.probs.values()) == pytest.approx(1.0, abs=1e-3)
        assert head.label == max(head.probs, key=head.probs.get)
        assert head.confidence == pytest.approx(head.probs[head.label], abs=1e-4)
    assert body.display_text == SENTENCE
    assert body.pii_types == [] and not body.no_diacritics
    assert body.n_words > 0 and body.xai_available


@pytest.mark.model
def test_flag_and_its_reasons_agree(live_client: TestClient):
    for text in (SENTENCE, "Môn này ổn.", "Phòng thực hành nằm ở tầng ba của tòa nhà B."):
        flag = schemas.AnalyzeResponse.model_validate(
            post(live_client, "/api/analyze", text).json()
        ).flag
        if not flag.needs_review:
            assert flag.reasons == []
            continue
        (reason,) = flag.reasons
        assert reason.code == "low_confidence" and reason.task in ("sentiment", "topic")
        assert len(reason.candidates) == 2
        assert reason.message.startswith(("Cảm xúc không chắc", "Chủ đề không chắc"))


@pytest.mark.model
@pytest.mark.parametrize("body", [{"text": ""}, {"text": "   \n\t "}, {}, {"text": 5}, {"x": "y"}])
def test_invalid_bodies_are_422_with_the_standard_error_shape(live_client: TestClient, body):
    response = live_client.post("/api/analyze", json=body)
    assert response.status_code == 422
    detail = schemas.ErrorResponse.model_validate(response.json()).detail
    assert detail.code == "validation_error"
    assert detail.fields[0].loc[:2] == ["body", "text"]


@pytest.mark.model
def test_length_limit_is_500_characters(live_client: TestClient):
    ok = post(live_client, "/api/analyze", "a" * MAX_CHARS)
    assert ok.status_code == 200
    too_long = post(live_client, "/api/analyze", "a" * (MAX_CHARS + 1))
    assert too_long.status_code == 422
    assert too_long.json()["detail"]["fields"][0]["type"] == "string_too_long"
    assert "a" * 20 not in too_long.text  # lỗi không lặp lại nội dung đã gửi


@pytest.mark.model
def test_phone_number_is_masked_and_never_returned(live_client: TestClient):
    response = post(live_client, "/api/analyze", "Môn này khó, liên hệ 0912345678 để hỏi thêm.")
    body = response.json()
    assert "[SĐT]" in body["display_text"]
    assert "phone" in body["pii_types"]
    assert "0912345678" not in response.text


@pytest.mark.model
def test_name_after_a_title_is_masked(live_client: TestClient):
    body = post(live_client, "/api/analyze", "Thầy Nam dạy rất nhiệt tình.").json()
    assert body["display_text"].startswith("Thầy [TÊN]")
    assert "name" in body["pii_types"]
    assert "Nam" not in body["display_text"]


@pytest.mark.model
def test_text_without_diacritics_is_always_flagged_with_its_own_reason(live_client: TestClient):
    body = post(
        live_client, "/api/analyze", "thay day rat de hieu va nhiet tinh voi sinh vien"
    ).json()
    assert body["no_diacritics"] is True
    assert body["flag"]["needs_review"] is True
    assert [r["code"] for r in body["flag"]["reasons"]] == ["no_diacritics"]


@pytest.mark.model
def test_sentence_over_the_explain_limit_is_analyzed_but_not_flagged_for_length(
    live_client: TestClient,
):
    long_text = " ".join(["giảng viên dạy rất hay và nhiệt tình"] * 9)
    body = post(live_client, "/api/analyze", long_text).json()
    assert body["n_words"] > XAI_MAX_WORDS
    assert body["xai_available"] is False
    assert all(r["code"] != "too_long" for r in body["flag"]["reasons"])


@pytest.mark.model
def test_explain_covers_every_word_of_the_analyzed_sentence(live_client: TestClient):
    analysis = post(live_client, "/api/analyze", SENTENCE).json()
    response = post(live_client, "/api/explain", SENTENCE)
    assert response.status_code == 200
    explanation = schemas.ExplainResponse.model_validate(response.json())

    assert explanation.n_forwards == analysis["n_words"] + 1
    for head, analysed in (
        (explanation.sentiment, analysis["sentiment"]),
        (explanation.topic, analysis["topic"]),
    ):
        assert len(head.tokens) == analysis["n_words"]
        assert head.label == analysed["label"]
        assert head.probability == pytest.approx(analysed["confidence"], abs=0.03)
        assert all(0 <= i < len(head.tokens) for i in head.support + head.oppose)


@pytest.mark.model
def test_explain_refuses_long_sentences_with_a_stable_code(live_client: TestClient):
    long_text = " ".join(["giảng viên dạy rất hay và nhiệt tình"] * 9)
    response = post(live_client, "/api/explain", long_text)
    assert response.status_code == 422
    detail = schemas.ErrorResponse.model_validate(response.json()).detail
    assert detail.code == "xai_too_long"
    assert str(XAI_MAX_WORDS) in detail.message


HOSTILE_TEXTS = {
    "nul": "a\x00b thầy dạy hay",
    "lone surrogate": "thầy \ude00 dạy",
    "emoji flood": "😀" * 120,
    "right-to-left": "‮ thầy dạy hay",
    "zero width": "​" * 400,
    "combining marks": "a" + "́" * 400,
    "markup": "<script>alert(1)</script> thầy dạy hay",
    "template": "{{7*7}} ${jndi:ldap://x} %s%s%s%n",
    "one huge token": "x" * MAX_CHARS,
    "punctuation only": '!?.,;()"' * 60,
    "digits only": "1" * MAX_CHARS,
    "other scripts": "老师讲得很好 المعلم ممتاز ครูสอนดี",
    "top code point": "\U0010ffff thầy",
    "only a phone number": "0912345678",
    "emoticon flood": ":) " * 150,
    "dots": "." * MAX_CHARS,
    "piece outside the embedding table": "truyền_đạt_kiến thưc tốt",
    "model special tokens typed by the user": "<s> </s> <pad> <unk> <mask> thầy dạy hay",
}


@pytest.mark.model
@pytest.mark.parametrize("path", ["/api/analyze", "/api/explain"])
@pytest.mark.parametrize("text", list(HOSTILE_TEXTS.values()), ids=list(HOSTILE_TEXTS))
def test_hostile_text_is_answered_or_rejected_but_never_a_server_error(
    live_client: TestClient, path: str, text: str
):
    # json.dumps giữ nguyên ký tự lạ (kể cả nửa cặp thay thế) thành \uXXXX như trình duyệt gửi.
    response = live_client.post(
        path, content=json.dumps({"text": text}), headers={"Content-Type": "application/json"}
    )
    assert response.status_code in (200, 422), response.text[:200]
    if response.status_code == 422:
        code = response.json()["detail"]["code"]
        assert code in ("validation_error", "xai_too_long", "no_content")


@pytest.mark.model
@pytest.mark.parametrize("path", ["/api/analyze", "/api/explain"])
@pytest.mark.parametrize("text", ["0912345678", "😀😀", "abc@example.com"])
def test_text_with_nothing_left_after_masking_is_422_no_content(
    live_client: TestClient, path: str, text: str
):
    response = post(live_client, path, text)
    assert response.status_code == 422
    detail = schemas.ErrorResponse.model_validate(response.json()).detail
    assert detail.code == "no_content"
    assert text not in response.text


@pytest.mark.model
def test_wrong_method_on_analyze_is_405_with_allow_post(live_client: TestClient):
    response = live_client.get("/api/analyze")
    assert response.status_code == 405
    assert response.headers["allow"] == "POST"


@pytest.mark.model
def test_health_reports_the_loaded_model(live_client: TestClient):
    health = schemas.Health.model_validate(live_client.get("/api/health").json())
    assert health.model_version and health.stage in ("smoke", "release")
    assert health.rss_mb > 0


@pytest.mark.model
def test_concurrent_requests_all_succeed(live_client: TestClient):
    texts = [SENTENCE, "Phòng học nóng và ồn.", "Thủ tục xin giấy xác nhận rườm rà.", "Ổn."] * 3

    def call(index_text: tuple[int, str]) -> int:
        index, text = index_text
        path = "/api/explain" if index % 3 == 0 else "/api/analyze"
        return post(live_client, path, text).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(call, enumerate(texts)))
    assert statuses == [200] * len(texts)
