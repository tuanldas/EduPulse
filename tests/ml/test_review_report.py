"""Báo cáo ngân sách duyệt trên test: dùng calibration đã đóng băng, đúng luật của engine."""

import math

import numpy as np
import pytest

from edupulse.config import BUDGETS, DEFAULT_BUDGET
from edupulse.scoring import confidence, quantile_rank, softmax_t
from ml import calibrate, preds, review_report
from ml.common import write_json

SIZES = {"sentiment": 3, "topic": 4}


def make_logits(n: int, seed: int, sharp: float = 2.5):
    """Logit có cấu trúc: lớp thật nổi lên theo `sharp` cộng nhiễu, nên độ tin cậy trải rộng."""
    rng = np.random.default_rng(seed)
    y = {task: rng.integers(0, size, size=n) for task, size in SIZES.items()}
    logits = {}
    for task, size in SIZES.items():
        base = np.eye(size)[y[task]] * sharp
        logits[task] = (base + rng.normal(scale=1.6, size=(n, size))).astype(np.float32)
    return logits, y


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """Calibration từ 'val' tổng hợp rồi đóng băng; dự đoán int8 trên 'test' tổng hợp khác."""
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    val_logits, val_y = make_logits(900, seed=1)
    calibration, _ = calibrate.calibrate(val_logits, val_y)
    path = tmp_path / "outputs" / "calibration" / "mtl-s42" / "calibration.json"
    write_json(path, calibration.model_dump(mode="json"))

    test_logits, test_y = make_logits(500, seed=2)
    directory = tmp_path / "preds"
    preds.save_preds(
        directory,
        preds.Preds(
            preds.DEPLOY_KEY,
            test_y,
            {t: test_logits[t].argmax(axis=1) for t in test_y},
            test_logits,
        ),
    )
    write_json(directory / preds.META_FILE, {"split": "test", "n": 500, "deploy_run": "mtl-s42"})
    (directory / preds.DONE_FILE).write_text("xong\n", encoding="utf-8")
    ctx = preds.EvalContext("test", directory, tmp_path / "results", rehearsal=False)
    monkeypatch.setattr(review_report, "context", lambda rehearsal=False: ctx)
    return ctx, calibration, test_logits, test_y


def test_flag_scores_follow_the_frozen_temperature_and_quantile_table(frozen):
    _, calibration, logits, y = frozen
    scores, probs = review_report.flag_scores(logits, calibration)
    heads = {"sentiment": calibration.sentiment, "topic": calibration.topic}
    ranks = {}
    for task in SIZES:
        expected = softmax_t(logits[task].astype(np.float64), heads[task].temperature)
        assert np.allclose(probs[task], expected)
        ranks[task] = quantile_rank(confidence(expected), heads[task].cdf)
    assert np.allclose(scores, np.minimum(ranks["sentiment"], ranks["topic"]))


def test_changing_the_frozen_temperature_changes_the_scores(frozen):
    _, calibration, logits, _ = frozen
    hot = calibration.model_copy(
        update={"sentiment": calibration.sentiment.model_copy(update={"temperature": 3.0})}
    )
    base, _ = review_report.flag_scores(logits, calibration)
    other, probs = review_report.flag_scores(logits, hot)
    assert not np.allclose(base, other)
    assert np.allclose(probs["sentiment"], softmax_t(logits["sentiment"].astype(np.float64), 3.0))


def test_batch_rule_flags_exactly_the_lowest_scoring_ceiling_of_the_budget(frozen):
    _, calibration, logits, y = frozen
    scores, probs = review_report.flag_scores(logits, calibration)
    ok = {t: probs[t].argmax(axis=1) == y[t] for t in SIZES}
    rows = review_report.budget_rows(scores, ok, calibration.tau)
    assert set(rows) == set(BUDGETS)
    for budget, row in rows.items():
        flagged = row["rank_batch"]["flagged_share"]
        assert flagged == pytest.approx(math.ceil(budget * 500 / 100) / 500)
    # Ngân sách lớn hơn thì bắt nhiều lỗi hơn và phần không cờ chính xác hơn.
    caught = [rows[b]["rank_batch"]["caught_sentiment_errors"] for b in BUDGETS]
    assert caught == sorted(caught)
    accepted = [rows[b]["rank_batch"]["accept_accuracy_both"] for b in BUDGETS]
    assert accepted == sorted(accepted)


def test_threshold_rule_uses_the_frozen_tau_not_a_refit(frozen):
    _, calibration, logits, y = frozen
    scores, probs = review_report.flag_scores(logits, calibration)
    ok = {t: probs[t].argmax(axis=1) == y[t] for t in SIZES}
    rows = review_report.budget_rows(scores, ok, calibration.tau)
    for budget in BUDGETS:
        row = rows[budget]["threshold"]
        assert row["tau"] == calibration.tau[budget]
        assert row["flagged_share"] == pytest.approx(
            float((scores < calibration.tau[budget]).mean())
        )


def test_calibration_quality_is_identical_before_and_after_when_t_is_one():
    logits, y = make_logits(300, seed=3)
    same = review_report.calibration_quality(logits["topic"], y["topic"], 1.0)
    assert same["ece_before_t"] == same["ece_after_t"]
    assert same["nll_before_t"] == same["nll_after_t"]
    hotter = review_report.calibration_quality(logits["topic"], y["topic"], 2.5)
    assert hotter["nll_after_t"] != hotter["nll_before_t"]


def test_report_has_the_documented_structure(frozen):
    ctx, calibration, _, _ = frozen
    report = review_report.build_report(ctx, calibration, "mtl-s42")
    assert (report["split"], report["n"], report["deploy_run"]) == ("test", 500, "mtl-s42")
    assert report["default_budget"] == DEFAULT_BUDGET
    assert set(report["budgets"]) == set(BUDGETS)
    assert set(report["budgets"][10]["rank_batch"]) == {
        "flagged_share",
        "caught_sentiment_errors",
        "caught_topic_errors",
        "accept_accuracy_both",
        "all_accuracy_both",
    }
    assert set(report["calibration_quality"]) == set(SIZES)
    assert report["temperature"] == {
        "sentiment": calibration.sentiment.temperature,
        "topic": calibration.topic.temperature,
    }


def test_the_validation_table_reproduces_the_frozen_val_numbers(frozen):
    _, calibration, _, _ = frozen
    text = review_report.table_val(calibration)
    assert text.count("\n|") >= 6 and "(mặc định)" in text
    e = calibration.efficiency[DEFAULT_BUDGET]
    assert f"{review_report.vn(e.caught_sentiment_errors * 100, 1)}%" in text
    assert f"{review_report.vn(calibration.tau[DEFAULT_BUDGET], 4)}" in text


def test_the_test_table_has_the_three_sections(frozen):
    ctx, calibration, _, _ = frozen
    text = review_report.table_test(review_report.build_report(ctx, calibration, "mtl-s42"))
    assert "Luật xếp hạng cả lô" in text and "Luật ngưỡng từng câu" in text
    assert "Hiệu chuẩn xác suất trên cùng tập" in text and "tập test" in text
    assert text.count("(mặc định)") == 1


def test_main_writes_the_report_json_and_both_tables(frozen, capsys):
    ctx, _, _, _ = frozen
    assert review_report.main() == 0
    assert ctx.metrics_path("review_budget_test.json").is_file()
    assert ctx.table_path("review_budget_test.md").read_text("utf-8").startswith("Hiệu quả")
    assert ctx.table_path("review_budget_val.md").is_file()
    assert "Luật xếp hạng cả lô" in capsys.readouterr().out


def test_main_refuses_to_run_before_the_test_was_evaluated(frozen, capsys):
    ctx, _, _, _ = frozen
    (ctx.preds_dir / preds.DONE_FILE).unlink()
    assert review_report.main() == review_report.EXIT_NOT_DONE
    assert "chưa chấm xong" in capsys.readouterr().err
    assert not ctx.metrics_path("review_budget_test.json").exists()


def test_calibrate_cli_dispatches_the_test_report_and_keeps_the_default_path(monkeypatch):
    seen = {}

    def fake_main(rehearsal=False):
        seen["rehearsal"] = rehearsal
        return 0

    monkeypatch.setattr(review_report, "main", fake_main)
    assert calibrate.main(["--report", "test"]) == 0 and seen == {"rehearsal": False}
    assert calibrate.main(["--report", "test", "--rehearsal"]) == 0 and seen == {"rehearsal": True}
    parser = calibrate.build_parser()
    assert parser.parse_args([]).report is None and parser.parse_args([]).run == "best"
