"""`Analyzer` trên bundle thật: hiệu chỉnh, điểm cờ, cờ một câu và cả lô, tiến độ."""

import json
import math
from pathlib import Path

import pytest

from edupulse.config import BUDGETS, DEFAULT_BUDGET, Settings
from edupulse.engine.service import Analyzer, NoContent
from edupulse.labels import LABELS

pytestmark = pytest.mark.model

GOLDEN_SENTENCES = Path(__file__).resolve().parents[1] / "fixtures" / "golden_sentences.json"


def _golden_items() -> list[dict]:
    return json.loads(GOLDEN_SENTENCES.read_text(encoding="utf-8"))["items"]


@pytest.fixture(scope="module")
def texts() -> list[str]:
    return [item["text"] for item in _golden_items()]


@pytest.fixture(scope="module")
def typed_texts() -> list[str]:
    """Các câu có dấu: không câu nào bị cờ bắt buộc nên số cờ của lô chỉ do ngân sách quyết định."""
    return [item["text"] for item in _golden_items() if "no_diacritics" not in item["covers"]]


def test_head_results_are_a_calibrated_distribution_over_the_labels(analyzer: Analyzer, texts):
    for analysis in analyzer.infer(analyzer.prepare_many(texts)):
        for task, head in (("sentiment", analysis.sentiment), ("topic", analysis.topic)):
            assert len(head.probs) == len(LABELS[task])
            assert sum(head.probs) == pytest.approx(1.0, abs=1e-6)
            assert head.confidence == pytest.approx(max(head.probs))
            assert head.label == LABELS[task][head.probs.index(max(head.probs))]
            assert 0.0 <= head.quantile <= 1.0
        assert analysis.score == pytest.approx(
            min(analysis.sentiment.quantile, analysis.topic.quantile)
        )


def test_one_sentence_gives_the_same_numbers_by_every_route(analyzer: Analyzer, texts):
    single = analyzer.analyze(texts[0])
    (batched,) = analyzer.analyze_many([texts[0]])
    assert single.sentiment == batched.sentiment and single.topic == batched.topic
    assert single.score == batched.score


def test_batch_size_one_reproduces_single_sentence_results_exactly(analyzer: Analyzer, typed_texts):
    # Int8 động làm kết quả phụ thuộc bạn cùng lô; lô 1 là chế độ tất định khớp analyze và golden.
    batched = analyzer.analyze_many(typed_texts, batch_size=1)
    for text, analysis in zip(typed_texts, batched, strict=True):
        single = analyzer.analyze(text)
        assert (analysis.sentiment, analysis.topic) == (single.sentiment, single.topic), text
        assert analysis.score == single.score


def test_single_sentence_is_flagged_exactly_when_below_the_default_budget_threshold(
    analyzer: Analyzer, texts
):
    tau = analyzer.calibration.tau[DEFAULT_BUDGET]
    for text in texts:
        analysis = analyzer.analyze(text)
        expected = analysis.prepared.no_diacritics or analysis.score < tau
        assert analysis.flag.needs_review == expected, text
        assert bool(analysis.flag.reasons) == analysis.flag.needs_review


def test_batch_flags_round_up_the_budget_over_the_batch(analyzer: Analyzer, typed_texts):
    assert not any(p.no_diacritics for p in analyzer.prepare_many(typed_texts))
    for budget in BUDGETS:
        flagged = sum(a.flag.needs_review for a in analyzer.analyze_many(typed_texts, budget))
        assert flagged == math.ceil(budget * len(typed_texts) / 100), budget


def test_batch_flags_the_lowest_scores_and_names_the_weaker_head(analyzer: Analyzer, typed_texts):
    analyses = analyzer.analyze_many(typed_texts, 20)
    flagged = [a for a in analyses if a.flag.needs_review]
    others = [a for a in analyses if not a.flag.needs_review]
    assert max(a.score for a in flagged) <= min(a.score for a in others)
    for analysis in flagged:
        (reason,) = analysis.flag.reasons
        weaker = "sentiment" if analysis.sentiment.quantile <= analysis.topic.quantile else "topic"
        assert reason.task == weaker


def test_no_diacritics_sentence_is_flagged_first_and_counts_toward_the_budget(
    analyzer: Analyzer, typed_texts
):
    batch = [*typed_texts, "thay day rat de hieu va nhiet tinh voi sinh vien"]
    analyses = analyzer.analyze_many(batch, 10)
    assert analyses[-1].flag.needs_review
    assert [r.code for r in analyses[-1].flag.reasons] == ["no_diacritics"]
    assert sum(a.flag.needs_review for a in analyses) == math.ceil(0.10 * len(batch))


def test_progress_reports_done_and_total_for_the_whole_batch(analyzer: Analyzer, texts):
    seen: list[tuple[int, int]] = []
    analyzer.analyze_many(texts * 3, progress=lambda done, total: seen.append((done, total)))
    assert seen[-1] == (len(texts) * 3, len(texts) * 3)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert {total for _, total in seen} == {len(texts) * 3}


NO_CONTENT_TEXTS = ["0912345678", "😀😀😀", "abc@example.com", "   0912 345 678   "]


@pytest.mark.parametrize("text", NO_CONTENT_TEXTS)
def test_text_with_nothing_left_to_analyze_is_refused_not_guessed(analyzer: Analyzer, text):
    with pytest.raises(NoContent) as failure:
        analyzer.analyze(text)
    assert failure.value.code == "no_content" and failure.value.rows == (0,)
    with pytest.raises(NoContent):
        analyzer.explain(text)


def test_batch_names_the_rows_that_have_no_content(analyzer: Analyzer, typed_texts):
    batch = [typed_texts[0], "0912345678", typed_texts[1], "😀"]
    with pytest.raises(NoContent) as failure:
        analyzer.analyze_many(batch)
    assert failure.value.rows == (1, 3)


def test_callers_can_filter_empty_rows_before_inference(analyzer: Analyzer, typed_texts):
    prepared = analyzer.prepare_many([typed_texts[0], "0912345678", typed_texts[1]])
    kept = [p for p in prepared if p.n_words]
    assert len(kept) == 2
    assert len(analyzer.flag(analyzer.infer(kept))) == 2


def test_name_only_text_still_has_a_token_to_analyze(analyzer: Analyzer):
    analysis = analyzer.analyze("Thầy Nam")
    assert analysis.prepared.n_words >= 1 and analysis.sentiment.probs


def test_empty_batch_is_empty(analyzer: Analyzer):
    assert analyzer.analyze_many([]) == []
    assert analyzer.flag([]) == []


@pytest.mark.data
@pytest.mark.slow
def test_default_threshold_flags_about_ten_percent_of_the_validation_set(
    analyzer: Analyzer, settings: Settings
):
    pq = pytest.importorskip("pyarrow.parquet")
    column = pq.read_table(
        settings.data_dir / "processed" / "validation.parquet", columns=["sentence"]
    )
    sentences = [str(s) for s in column.column(0).to_pylist()]

    analyses = analyzer.infer(analyzer.prepare_many(sentences))
    tau = analyzer.calibration.tau[DEFAULT_BUDGET]
    share = sum(a.score < tau for a in analyses) / len(analyses)
    assert 0.07 <= share <= 0.13, f"{share:.3f} câu val dưới ngưỡng ngân sách {DEFAULT_BUDGET}%"
