"""Mô phỏng vòng người duyệt: chia R/E, chọn cờ, các nhánh «sau», kiểm định, báo cáo, dòng lệnh.

Dữ liệu là câu tổng hợp tự viết; `load_processed` được thay bằng hàm trả bảng tổng hợp và từ chối
tập test, nên không test nào chạm dữ liệu thật.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from edupulse.labels import LABELS, TASKS
from edupulse.text import NORMALIZER_VERSION
from ml import hitl_simulation as hitl
from ml import retrain
from ml.common import mps_lock, write_json
from ml.plotting import vn

HP = {"sentiment": {"C": 1.0, "class_weight": None}, "topic": {"C": 1.0, "class_weight": None}}
TOPIC_WORDS = {
    0: ("thầy", "cô", "giảng_viên"),
    1: ("chương_trình", "giáo_trình", "môn_học"),
    2: ("phòng_học", "máy_chiếu", "thư_viện"),
    3: ("căng_tin", "bãi_xe", "học_phí"),
}
OPINIONS = {
    0: ("tệ", "kém", "chán"),
    1: ("bình_thường", "tạm_được", "vừa_phải"),
    2: ("tốt", "tuyệt_vời", "hay"),
}


def make_frame(n: int, seed: int, offset: int, noise: float = 0.2) -> pd.DataFrame:
    """Câu tự viết, mỗi câu một mã riêng; một phần nhãn bị đảo để mô hình có lỗi cho cờ bắt."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        topic = int(rng.integers(0, 4))
        sentiment = int(rng.choice(3, p=[0.45, 0.15, 0.40]))
        word = TOPIC_WORDS[topic][int(rng.integers(0, 3))]
        opinion = OPINIONS[sentiment][int(rng.integers(0, 3))]
        if rng.random() < noise:
            sentiment = int(rng.integers(0, 3))
        if rng.random() < noise:
            topic = int(rng.integers(0, 4))
        rows.append(
            {
                "text_rdr": f"{word} {opinion} câu{offset + i} .",
                "sentiment": sentiment,
                "topic": topic,
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def corpus() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train 300 câu; validation 207 câu nên R = 103, E = 104 và ngân sách 10% là ⌈10,3⌉ = 11."""
    return make_frame(300, 1, 0), make_frame(207, 2, 1000)


@pytest.fixture(scope="module")
def simulation(corpus):
    """Một lần chạy đủ (20 lần ngẫu nhiên); ghi lại phản hồi đưa vào từng lần retrain."""
    train, val = corpus
    calls: list[pd.DataFrame] = []
    original = hitl.retrain_baseline

    def spy(train_frame, feedback, hp_by_task, weight=1):
        calls.append(feedback.copy())
        return original(train_frame, feedback, hp_by_task, weight)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(hitl, "retrain_baseline", spy)
        result = hitl.run_simulation(train, val, HP, bootstrap_rounds=200)
    return SimpleNamespace(result=result, calls=calls, train=train, val=val)


# --- Chia dữ liệu ---------------------------------------------------------------------------------


def test_split_is_disjoint_complete_reproducible_and_stratified(corpus):
    _, val = corpus
    review, evaluation = hitl.split_validation(val, seed=0)
    assert (len(review), len(evaluation)) == (103, 104)
    assert not set(review) & set(evaluation)
    assert sorted([*review, *evaluation]) == list(range(len(val)))
    assert (np.diff(review) > 0).all() and (np.diff(evaluation) > 0).all()
    again_review, again_evaluation = hitl.split_validation(val, seed=0)
    assert np.array_equal(review, again_review) and np.array_equal(evaluation, again_evaluation)
    assert not np.array_equal(review, hitl.split_validation(val, seed=1)[0])
    labels = val["sentiment"].to_numpy()
    for label in range(3):
        gap = (labels[review] == label).sum() - (labels[evaluation] == label).sum()
        assert abs(gap) <= 2  # phân tầng làm tròn mỗi lớp trong khoảng dưới 1 câu mỗi nửa


def test_split_of_the_real_validation_size_gives_791_and_792():
    rng = np.random.default_rng(0)
    val = pd.DataFrame({"sentiment": rng.choice(3, size=1583, p=[0.45, 0.05, 0.50])})
    review, evaluation = hitl.split_validation(val)
    assert (len(review), len(evaluation)) == (791, 792)


# --- Chọn cờ --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 7, 25, 100, 791, 1583])
def test_flags_cover_exactly_the_ceiling_of_ten_percent(n):
    rng = np.random.default_rng(n)
    flags, scores = hitl.select_flags(rng.random(n), rng.random(n), 10)
    assert flags.sum() == math.ceil(0.10 * n)
    if (~flags).any():
        assert scores[flags].max() <= scores[~flags].min()  # cờ đúng những câu điểm thấp nhất


def test_flags_pick_the_lowest_scoring_rows_in_the_right_order():
    n = 30
    sentiment, topic = np.full(n, 0.9), np.full(n, 0.9)
    sentiment[[4, 11, 17]] = [0.31, 0.30, 0.32]
    flags, scores = hitl.select_flags(sentiment, topic, 10)
    assert flags.nonzero()[0].tolist() == [4, 11, 17]
    ranking = np.argsort(scores, kind="stable")[:3].tolist()
    assert ranking == [11, 4, 17]  # điểm thấp nhất trước: 0,30 rồi 0,31 rồi 0,32


def test_a_row_is_flagged_when_either_head_is_unsure():
    n = 40
    sentiment, topic = np.full(n, 0.95), np.full(n, 0.95)
    sentiment[[2, 5]] = [0.40, 0.45]  # chỉ cảm xúc kém chắc
    topic[[20, 33]] = [0.40, 0.45]  # chỉ chủ đề kém chắc
    flags, _ = hitl.select_flags(sentiment, topic, 10)
    assert flags.nonzero()[0].tolist() == [2, 5, 20, 33]


def test_ties_are_broken_by_row_order_and_the_budget_is_a_parameter():
    flags, _ = hitl.select_flags(np.full(25, 0.8), np.full(25, 0.8), 10)
    assert flags.nonzero()[0].tolist() == [0, 1, 2]  # ⌈2,5⌉ = 3 câu đầu
    flags, _ = hitl.select_flags(np.full(50, 0.8), np.full(50, 0.8), 20)
    assert flags.sum() == 10


def test_decision_confidence_is_the_largest_softmax_probability_of_the_margins(corpus):
    train, val = corpus
    model = retrain.retrain_baseline(train, _empty(), HP)
    texts = val["text_rdr"].tolist()
    for task in TASKS:
        margins = model[task].decision_function(texts)
        expected = np.exp(margins - margins.max(axis=1, keepdims=True))
        expected = (expected / expected.sum(axis=1, keepdims=True)).max(axis=1)
        got = hitl.decision_confidence(model[task], texts)
        assert np.allclose(got, expected)
        assert (got > 1 / len(LABELS[task]) - 1e-12).all() and (got <= 1).all()


def _empty() -> pd.DataFrame:
    return pd.DataFrame({"text_model": [], "label_s": [], "label_t": []}).astype(
        {"text_model": "str", "label_s": "int64", "label_t": "int64"}
    )


# --- Chỉ số ---------------------------------------------------------------------------------------


def test_summarize_matches_scikit_learn():
    rng = np.random.default_rng(0)
    y = {"sentiment": rng.integers(0, 3, 200), "topic": rng.integers(0, 4, 200)}
    pred = {
        task: np.where(rng.random(200) < 0.7, y[task], rng.integers(0, 3, 200)) for task in TASKS
    }
    summary = hitl.summarize(y, pred)
    for task in TASKS:
        labels = list(range(len(LABELS[task])))
        assert summary[task]["accuracy"] == pytest.approx(accuracy_score(y[task], pred[task]))
        assert summary[task]["macro_f1"] == pytest.approx(
            f1_score(y[task], pred[task], labels=labels, average="macro", zero_division=0)
        )
        assert summary[task]["per_class_f1"] == pytest.approx(
            f1_score(y[task], pred[task], labels=labels, average=None, zero_division=0)
        )


def test_aggregate_runs_uses_the_sample_standard_deviation():
    def run(accuracy, macro, weighted, per_class):
        metrics = {
            "accuracy": accuracy,
            "macro_f1": macro,
            "weighted_f1": weighted,
            "per_class_f1": per_class,
        }
        return {task: metrics for task in TASKS}

    runs = [
        run(0.9, 0.70, 0.8, [0.1, 0.2]),
        run(0.8, 0.74, 0.9, [0.3, 0.4]),
        run(0.7, 0.72, 0.7, [0.5, 0.6]),
    ]
    mean, std = hitl.aggregate_runs(runs)
    for task in TASKS:
        assert mean[task]["macro_f1"] == pytest.approx(0.72)
        assert std[task]["macro_f1"] == pytest.approx(np.std([0.70, 0.74, 0.72], ddof=1))
        assert mean[task]["per_class_f1"] == pytest.approx([0.3, 0.4])
        assert std[task]["per_class_f1"] == pytest.approx([0.2, 0.2])
        assert isinstance(mean[task]["accuracy"], float)
    single_mean, single_std = hitl.aggregate_runs(runs[:1])
    assert single_std["sentiment"]["macro_f1"] == 0.0 and single_mean["topic"]["accuracy"] == 0.9


def test_flagging_report_on_a_hand_made_batch():
    flags = np.array([True, True] + [False] * 8)
    correct = {
        "sentiment": np.array([False, True, True, True, True, True, True, True, False, True]),
        "topic": np.array([False, False] + [True] * 8),
    }
    report = hitl.flagging_report(flags, correct)
    assert (report["n_review"], report["n_flagged"], report["flagged_share"]) == (10, 2, 0.2)
    assert report["accuracy_flagged"] == {"sentiment": 0.5, "topic": 0.0}
    assert report["accuracy_unflagged"] == {"sentiment": 0.875, "topic": 1.0}
    assert report["accuracy_both_flagged"] == 0.0 and report["accuracy_both_unflagged"] == 0.875
    assert report["wrong_any_share_flagged"] == 1.0
    assert report["wrong_any_share_unflagged"] == 0.125
    assert report["errors"]["sentiment"] == {"total": 2, "in_flagged": 1, "caught_share": 0.5}
    assert report["errors"]["topic"] == {"total": 2, "in_flagged": 2, "caught_share": 1.0}


def test_comparison_with_the_deployed_baseline_reports_the_match_rate():
    pred = {"sentiment": np.array([0, 1, 2, 0]), "topic": np.array([0, 1, 2, 3])}
    same = hitl.compare_with_deployed(pred, {t: pred[t].copy() for t in TASKS})
    assert same == {
        "checked": True,
        "identical": True,
        "match_rate": {"sentiment": 1.0, "topic": 1.0},
        "n": 4,
    }
    other = {"sentiment": np.array([0, 1, 2, 1]), "topic": pred["topic"].copy()}
    differs = hitl.compare_with_deployed(pred, other)
    assert differs["identical"] is False
    assert differs["match_rate"] == {"sentiment": 0.75, "topic": 1.0}
    unchecked = hitl.compare_with_deployed(pred, None)
    assert unchecked["checked"] is False and unchecked["identical"] is None


# --- Cả vòng mô phỏng -----------------------------------------------------------------------------


def test_result_is_plain_json_with_the_documented_structure(simulation):
    result = simulation.result
    json.dumps(result, allow_nan=False)  # không còn kiểu numpy, không NaN
    assert {"flagging", "branches", "tests", "phobert_smoke", "parameters", "data"} <= set(result)
    assert set(result["branches"]) == {"before", "flagged", "random", "all_review"}
    assert set(result["tests"]) == {
        "flagged_vs_before",
        "all_review_vs_before",
        "flagged_vs_random",
    }
    for name in ("before", "flagged", "all_review"):
        for task in TASKS:
            assert {"accuracy", "macro_f1", "weighted_f1", "per_class_f1"} <= set(
                result["branches"][name][task]
            )
            assert len(result["branches"][name][task]["per_class_f1"]) == len(LABELS[task])
    assert result["phobert_smoke"]["skipped"] is True
    assert result["parameters"]["seed"] == 0 and result["parameters"]["budget_pct"] == 10
    assert result["parameters"]["hp"]["sentiment"] == {"C": 1.0, "class_weight": None}
    assert result["timing_seconds"]["total"] > 0


def test_review_and_evaluation_halves_are_disjoint_and_sized_as_documented(simulation):
    data, flagging = simulation.result["data"], simulation.result["flagging"]
    assert (data["n_train"], data["n_val"]) == (300, 207)
    assert (data["n_review"], data["n_eval"]) == (103, 104)
    assert data["n_review"] + data["n_eval"] == data["n_val"]
    review_pos, eval_pos = hitl.split_validation(simulation.val)
    flagged = set(flagging["flagged_val_positions"])
    assert flagged <= set(review_pos) and not flagged & set(eval_pos)
    assert flagging["n_flagged"] == len(flagged) == math.ceil(0.10 * 103) == 11
    assert sum(data["eval_support"]["sentiment"]) == 104
    assert sum(data["review_support"]["topic"]) == 103


def test_no_retrain_ever_sees_the_evaluation_half(simulation):
    review_pos, eval_pos = hitl.split_validation(simulation.val)
    evaluation_texts = set(simulation.val.iloc[eval_pos]["text_rdr"])
    review_texts = set(simulation.val.iloc[review_pos]["text_rdr"])
    assert len(simulation.calls) == 1 + 1 + 20 + 1  # trước, cờ, 20 ngẫu nhiên, trần trên
    for feedback in simulation.calls:
        used = set(feedback["text_model"])
        assert used <= review_texts and not used & evaluation_texts


def test_the_before_model_is_fit_on_train_alone_and_flagged_labels_are_the_original_ones(
    simulation,
):
    assert len(simulation.calls[0]) == 0  # «trước»: phản hồi rỗng
    positions = simulation.result["flagging"]["flagged_val_positions"]
    expected = simulation.val.iloc[positions]
    flagged = simulation.calls[1].sort_values("text_model").reset_index(drop=True)
    gold = expected.sort_values("text_rdr").reset_index(drop=True)
    assert flagged["text_model"].tolist() == gold["text_rdr"].tolist()
    assert flagged["label_s"].tolist() == gold["sentiment"].tolist()  # nhãn gốc, không sửa
    assert flagged["label_t"].tolist() == gold["topic"].tolist()


def test_every_feedback_row_carries_the_original_labels_of_its_sentence(simulation):
    gold = simulation.val.set_index("text_rdr")
    for feedback in simulation.calls[1:]:
        rows = gold.loc[feedback["text_model"]]
        assert rows["sentiment"].tolist() == feedback["label_s"].tolist()
        assert rows["topic"].tolist() == feedback["label_t"].tolist()


def test_the_ceiling_branch_reviews_the_whole_review_half(simulation):
    review_pos, _ = hitl.split_validation(simulation.val)
    everything = simulation.calls[-1]
    assert sorted(everything["text_model"]) == sorted(simulation.val.iloc[review_pos]["text_rdr"])


def test_random_control_repeats_twenty_times_with_seeds_zero_to_nineteen(simulation):
    random = simulation.result["branches"]["random"]
    assert random["seeds"] == list(range(20)) and len(random["runs"]) == 20
    assert [run["seed"] for run in random["runs"]] == list(range(20))
    review_pos, _ = hitl.split_validation(simulation.val)
    review = simulation.val.iloc[review_pos].reset_index(drop=True)
    n_flagged = simulation.result["flagging"]["n_flagged"]
    random_calls = simulation.calls[2:22]
    for seed, feedback in enumerate(random_calls):
        pick = np.random.default_rng(seed).choice(len(review), size=n_flagged, replace=False)
        assert sorted(feedback["text_model"]) == sorted(review.iloc[pick]["text_rdr"])
        assert len(feedback) == n_flagged  # cùng cỡ với tập câu cờ
    assert len({tuple(sorted(f["text_model"])) for f in random_calls}) == 20  # 20 lần rút khác nhau
    assert random["n_sentences"] == n_flagged


def test_random_branch_statistics_come_from_its_runs(simulation):
    random = simulation.result["branches"]["random"]
    for task in TASKS:
        values = [run["metrics"][task]["macro_f1"] for run in random["runs"]]
        assert random["mean"][task]["macro_f1"] == pytest.approx(np.mean(values))
        assert random["std"][task]["macro_f1"] == pytest.approx(np.std(values, ddof=1))


def test_branch_metrics_equal_a_fresh_evaluation_on_the_evaluation_half(simulation):
    train, val = simulation.train, simulation.val
    _, eval_pos = hitl.split_validation(val)
    evaluation = val.iloc[eval_pos].reset_index(drop=True)
    y, texts = hitl.labels_of(evaluation), evaluation["text_rdr"].tolist()
    for index, name in ((0, "before"), (1, "flagged"), (22, "all_review")):
        models = retrain.retrain_baseline(train, simulation.calls[index], HP)
        pred = hitl.predict_all(models, texts)
        assert hitl.summarize(y, pred) == simulation.result["branches"][name]


def test_flagging_numbers_match_a_recount_from_the_before_model(simulation):
    train, val = simulation.train, simulation.val
    review_pos, _ = hitl.split_validation(val)
    models = retrain.retrain_baseline(train, simulation.calls[0], HP)
    pred = hitl.predict_all(models, val["text_rdr"].tolist())
    y = hitl.labels_of(val)
    wrong_any = (pred["sentiment"] != y["sentiment"]) | (pred["topic"] != y["topic"])
    flagging = simulation.result["flagging"]
    flagged = np.array(flagging["flagged_val_positions"])
    unflagged = np.setdiff1d(review_pos, flagged)
    assert flagging["wrong_any_share_flagged"] == pytest.approx(wrong_any[flagged].mean())
    assert flagging["wrong_any_share_unflagged"] == pytest.approx(wrong_any[unflagged].mean())
    for task in TASKS:
        wrong = pred[task] != y[task]
        assert flagging["errors"][task]["total"] == wrong[review_pos].sum()
        assert flagging["errors"][task]["in_flagged"] == wrong[flagged].sum()


def test_flagged_rows_are_the_lowest_scoring_rows_of_the_review_half(simulation):
    train, val = simulation.train, simulation.val
    review_pos, _ = hitl.split_validation(val)
    review = val.iloc[review_pos].reset_index(drop=True)
    models = retrain.retrain_baseline(train, simulation.calls[0], HP)
    texts = review["text_rdr"].tolist()
    conf = {t: hitl.decision_confidence(models[t], texts) for t in TASKS}
    flags, scores = hitl.select_flags(conf["sentiment"], conf["topic"], 10)
    assert review_pos[flags].tolist() == simulation.result["flagging"]["flagged_val_positions"]
    assert scores[flags].max() <= scores[~flags].min()


def test_paired_tests_carry_mcnemar_and_bootstrap_on_the_evaluation_half(simulation):
    n_eval = simulation.result["data"]["n_eval"]
    branches = simulation.result["branches"]
    for name, branch in (("flagged_vs_before", "flagged"), ("all_review_vs_before", "all_review")):
        for task in TASKS:
            paired = simulation.result["tests"][name][task]
            assert paired["n"] == n_eval == 104 and paired["rounds"] == 200
            assert 0 <= paired["mcnemar_p"] <= 1 and 0 <= paired["bootstrap_p"] <= 1
            assert paired["ci95_low"] <= paired["ci95_high"]
            assert paired["a_only_correct"] >= 0 and paired["b_only_correct"] >= 0
            assert paired["macro_f1_a"] == pytest.approx(branches[branch][task]["macro_f1"])
            assert paired["macro_f1_b"] == pytest.approx(branches["before"][task]["macro_f1"])
            assert paired["delta_macro_f1"] == pytest.approx(
                branches[branch][task]["macro_f1"] - branches["before"][task]["macro_f1"]
            )


def test_flagged_versus_random_reports_the_difference_and_an_empirical_p(simulation):
    for task in TASKS:
        item = simulation.result["tests"]["flagged_vs_random"][task]
        runs = simulation.result["branches"]["random"]["runs"]
        random_f1 = np.array([run["metrics"][task]["macro_f1"] for run in runs])
        assert item["n_runs"] == 20
        assert item["delta_macro_f1"] == pytest.approx(item["macro_f1_flagged"] - random_f1.mean())
        assert item["runs_flagged_at_least"] == (item["macro_f1_flagged"] >= random_f1).sum()
        assert item["p_empirical"] == pytest.approx(
            (1 + (random_f1 >= item["macro_f1_flagged"]).sum()) / 21
        )
        assert 1 / 21 - 1e-12 <= item["p_empirical"] <= 1
        assert item["macro_f1_random_min"] <= item["macro_f1_random_mean"]
        assert item["macro_f1_random_mean"] <= item["macro_f1_random_max"]


def test_the_simulation_is_reproducible(corpus):
    train, val = corpus
    kwargs = {"random_repeats": 2, "bootstrap_rounds": 50}
    first = hitl.run_simulation(train, val, HP, **kwargs)
    second = hitl.run_simulation(train, val, HP, **kwargs)
    for key in ("flagging", "branches", "tests"):
        assert first[key] == second[key]


def test_deployed_baseline_check_is_recorded_whether_or_not_it_matches(corpus):
    train, val = corpus
    models = retrain.retrain_baseline(train, _empty(), HP)
    deployed = hitl.predict_all(models, val["text_rdr"].tolist())
    kwargs = {"random_repeats": 1, "bootstrap_rounds": 20}
    same = hitl.run_simulation(train, val, HP, deployed_pred=deployed, **kwargs)
    assert same["before_matches_deployed_baseline"] is True
    assert same["deployed_check"]["match_rate"] == {"sentiment": 1.0, "topic": 1.0}
    altered = {task: values.copy() for task, values in deployed.items()}
    altered["topic"][:5] = (altered["topic"][:5] + 1) % 4
    differs = hitl.run_simulation(train, val, HP, deployed_pred=altered, **kwargs)
    assert differs["before_matches_deployed_baseline"] is False  # vẫn chạy hết
    assert differs["deployed_check"]["match_rate"]["topic"] == pytest.approx(1 - 5 / 207)
    assert "branches" in differs
    none = hitl.run_simulation(train, val, HP, **kwargs)
    assert none["before_matches_deployed_baseline"] is None


# --- Báo cáo --------------------------------------------------------------------------------------


def table_blocks(text: str) -> list[list[str]]:
    blocks, current = [], []
    for line in text.splitlines():
        if line.startswith("|"):
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    return [*blocks, current] if current else blocks


def test_report_has_every_section_and_consistent_tables(simulation):
    text = hitl.render_report(simulation.result)
    for heading in (
        "# Mô phỏng vòng người duyệt (HITL) trên validation",
        "## Thiết kế",
        "## Kết quả trên nửa đánh giá E (n = 104)",
        "## Kiểm định trên E",
        "## Hiệu quả gắn cờ trên R (n = 103)",
        "## Cách đọc",
        "## Giới hạn",
        "## Smoke retrain PhoBERT",
    ):
        assert heading in text
    assert "Đây là mô phỏng" in text and "không có người duyệt thật" in text
    blocks = table_blocks(text)
    assert len(blocks) == 4
    for block in blocks:
        widths = {line.count("|") for line in block}
        assert len(widths) == 1, block  # mọi hàng cùng số cột với tiêu đề
    results = blocks[0]
    assert [line.split("|")[1].strip().split(" (")[0] for line in results[2:]] == [
        "Trước",
        "Sau · câu cờ",
        "Sau · ngẫu nhiên",
        "Trần trên · duyệt hết R",
    ]


def test_report_uses_vietnamese_number_format_and_states_the_design_numbers(simulation):
    text = hitl.render_report(simulation.result)
    before = simulation.result["branches"]["before"]["sentiment"]["macro_f1"]
    assert vn(before * 100, 1) in text
    assert "⌈10% · 103⌉ = 11" in text
    assert "lặp 20 lần với seed 0..19" in text
    assert "bootstrap 200 vòng" in text
    for block in table_blocks(text):
        assert not any(re.search(r"\d\.\d", line) for line in block), block  # dấu phẩy thập phân
    assert "chưa kiểm được" in text  # dữ liệu tổng hợp không có baseline đã lưu để đối chiếu


def test_report_states_results_even_without_any_improvement(simulation):
    text = hitl.render_report(simulation.result)
    reading = text.split("## Cách đọc")[1].split("## Giới hạn")[0]
    assert "có thể không có ý nghĩa thống kê" in reading
    assert "kể cả khi không có cải thiện" in reading
    assert reading.count("nhánh cờ so với «trước»") == 2
    assert reading.count("trần trên") >= 2 and "ngẫu nhiên cùng cỡ" in reading
    limits = text.split("## Giới hạn")[1].split("## Smoke")[0]
    for phrase in ("Nhãn gốc thay người duyệt", "Chỉ LinearSVC", "tập test", "không chạy"):
        assert phrase in limits  # lần chạy này bỏ smoke nên mục giới hạn nói rõ là không chạy


PAIRED = {
    "delta_macro_f1": 0.005,
    "ci95_low": -0.01,
    "ci95_high": 0.02,
    "bootstrap_p": 0.4,
    "mcnemar_p": 0.5,
    "a_only_correct": 5,
    "b_only_correct": 3,
}


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({}, ["tăng 0,5 điểm", "không có ý nghĩa thống kê"]),
        ({"delta_macro_f1": -0.012}, ["giảm 1,2 điểm", "không có ý nghĩa thống kê"]),
        ({"delta_macro_f1": 0.0004}, ["gần như không đổi"]),
        (
            {"delta_macro_f1": 0.03, "ci95_low": 0.01, "ci95_high": 0.05, "mcnemar_p": 0.01},
            ["tăng 3,0 điểm", "có ý nghĩa thống kê ở cả hai kiểm định"],
        ),
        ({"mcnemar_p": 0.01}, ["chỉ kiểm định McNemar có ý nghĩa", "bằng chứng yếu"]),
        (
            {"ci95_low": 0.002, "ci95_high": 0.03},
            ["chỉ kiểm định bootstrap có ý nghĩa", "bằng chứng yếu"],
        ),
    ],
)
def test_verdict_describes_direction_and_significance(changes, expected):
    text = hitl.verdict({**PAIRED, **changes})
    for phrase in expected:
        assert phrase in text


def test_p_values_and_signed_points_are_formatted_the_vietnamese_way():
    assert hitl.p_text(0.0004) == "< 0,001" and hitl.p_text(0.0794) == "0,079"
    assert hitl.points(0.0123) == "+1,2" and hitl.points(-0.0123) == "-1,2"
    assert hitl.points(0.00001) == "0,0"  # làm tròn về 0 thì không dấu
    assert hitl.pct(0.9046, 1) == "90,5"


def fake_smoke() -> dict:
    metrics = {"accuracy": 0.9, "macro_f1": 0.8, "weighted_f1": 0.9, "per_class_f1": [0.8] * 3}
    return {
        "skipped": False,
        "base_run": "mtl-s42",
        "steps": 20,
        "batch_size": 32,
        "feedback_size": 80,
        "feedback_per_batch": 16,
        "replay_per_batch": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "clip_norm": 1.0,
        "n_eval": 792,
        "train_seconds": 123.4,
        "total_seconds": 200.0,
        "losses": [1.5, 0.9],
        "before": {"sentiment": metrics, "topic": metrics},
        "after": {"sentiment": metrics, "topic": metrics},
        "delta_macro_f1": {"sentiment": 0.001, "topic": -0.002},
        "label_changes": {"sentiment": 3, "topic": 5},
        "accept_rule": {"accepted": False, "reason": "chủ đề tụt 0,0020 (0,8000 -> 0,7980)"},
    }


def test_report_renders_the_phobert_smoke_section(simulation):
    text = hitl.render_report({**simulation.result, "phobert_smoke": fake_smoke()})
    section = text.split("## Smoke retrain PhoBERT")[1]
    assert "(CPU, 20 bước)" in section and "`mtl-s42`" in section
    assert "AdamW lr 2e-5" in section and "FP32" in section and "123,4 giây" in section
    assert "| Cảm xúc | 90,0 | 90,0 | 80,0 | 80,0 | +0,1 | 3 |" in section
    assert "| Chủ đề | 90,0 | 90,0 | 80,0 | 80,0 | -0,2 | 5 |" in section
    assert "chỉ là smoke" in section
    assert "TỪ CHỐI (chủ đề tụt 0,0020" in section
    assert "chỉ smoke 20 bước" in text.split("## Giới hạn")[1]


def test_accept_rule_of_the_retrain_script_is_applied_to_the_branches(simulation):
    from ml.retrain import accept

    branches, rule = simulation.result["branches"], simulation.result["accept_rule"]
    assert rule["tolerance"] == retrain.ACCEPT_TOLERANCE == 0.0
    for name in ("flagged", "all_review"):
        before = {task: branches["before"][task]["macro_f1"] for task in TASKS}
        after = {task: branches[name][task]["macro_f1"] for task in TASKS}
        accepted, reason = accept(before, after)
        assert rule[name] == {"accepted": accepted, "reason": reason}
    text = hitl.render_report(simulation.result)
    expected = "NHẬN" if rule["flagged"]["accepted"] else "TỪ CHỐI"
    assert f"với nhánh cờ: {expected} (" in text


def test_gate_decision_rejects_any_task_that_drops_and_accepts_ties():
    def metrics(sentiment, topic):
        return {"sentiment": {"macro_f1": sentiment}, "topic": {"macro_f1": topic}}

    assert hitl.gate_decision(metrics(0.8, 0.7), metrics(0.8, 0.7))["accepted"] is True
    assert hitl.gate_decision(metrics(0.8, 0.7), metrics(0.9, 0.7))["accepted"] is True
    worse = hitl.gate_decision(metrics(0.8, 0.7), metrics(0.9, 0.69))
    assert worse["accepted"] is False and "chủ đề tụt" in worse["reason"]
    single = hitl.gate_decision({"topic": {"macro_f1": 0.7}}, {"topic": {"macro_f1": 0.6}})
    assert single["accepted"] is False  # mô hình đơn nhiệm chỉ xét tác vụ của nó


def test_written_files_contain_numbers_but_no_sentence(simulation, tmp_path):
    metrics_path, table_path = tmp_path / "m" / "hitl.json", tmp_path / "t" / "hitl.md"
    hitl.write_outputs(simulation.result, metrics_path, table_path)
    metrics_text = metrics_path.read_text("utf-8")
    table_text = table_path.read_text("utf-8")
    assert json.loads(metrics_text)["data"]["n_eval"] == 104
    assert table_text == hitl.render_report(simulation.result)
    sentences = [*simulation.train["text_rdr"], *simulation.val["text_rdr"]]
    leaked = [s for s in sentences if s in metrics_text or s in table_text]
    assert leaked == []
    assert not re.search(r"câu\d+", metrics_text + table_text)  # cũng không có mã câu tổng hợp


# --- Dòng lệnh ------------------------------------------------------------------------------------


class FakeData:
    """Thay `load_processed`: trả bảng tổng hợp, ghi lại split bị yêu cầu, từ chối tập test."""

    def __init__(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        self.frames = {"train": train, "validation": val}
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, split: str, *, allow_test: bool = False) -> pd.DataFrame:
        self.calls.append((split, allow_test))
        if split == "test" or allow_test:
            raise AssertionError("mô phỏng không được mở tập test")
        return self.frames[split].copy()


@pytest.fixture
def fake_pipeline(corpus, tmp_path, monkeypatch) -> FakeData:
    train, val = corpus
    fake = FakeData(train, val)
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(hitl, "load_processed", fake)
    monkeypatch.setattr(hitl, "load_deployed_baseline_hp", lambda: HP)
    manifest = {"dataset_revision": "rev", "files": {s: {"sha256": s * 3} for s in fake.frames}}
    monkeypatch.setattr(hitl, "processed_manifest", lambda: manifest)
    return fake


def test_main_with_skip_phobert_writes_both_results_and_never_asks_for_test(
    fake_pipeline, tmp_path, capsys
):
    results = tmp_path / "results"
    assert hitl.main(["--skip-phobert", "--results-dir", str(results), "--seed", "3"]) == 0
    assert fake_pipeline.calls == [("train", False), ("validation", False)]
    document = json.loads((results / "metrics" / "hitl_retrain.json").read_text("utf-8"))
    assert document["parameters"]["seed"] == 3 and document["phobert_smoke"]["skipped"] is True
    assert document["data"]["train_sha256"] == "train" * 3
    assert document["data"]["validation_sha256"] == "validation" * 3
    assert document["data"]["dataset_revision"] == "rev"
    assert "test" not in json.dumps(document["data"])  # không có dấu vân tay của tập test
    assert set(document["code"]) == {"git_commit", "git_dirty_files", "sha256", "versions"}
    assert set(document["code"]["sha256"]) == {
        "ml/hitl_simulation.py",
        "ml/retrain.py",
        "ml/train_baseline.py",
        "ml/stats.py",
    }
    table = (results / "tables" / "hitl_retrain.md").read_text("utf-8")
    assert "Mô phỏng vòng người duyệt" in table
    assert "Mô phỏng vòng người duyệt" in capsys.readouterr().out


def test_main_with_phobert_gives_way_when_the_machine_lock_is_held(fake_pipeline, tmp_path, capsys):
    results = tmp_path / "results"
    with mps_lock(note="job khác"):
        assert hitl.main(["--results-dir", str(results)]) == 4
    assert "job khác" in capsys.readouterr().err
    assert fake_pipeline.calls == [] and not results.exists()


# --- Smoke PhoBERT trên checkpoint nhỏ ------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory) -> Path:
    """Run PhoBERT nhỏ (2 lớp, ẩn 32) nhưng từ vựng đủ lớn cho id của tokenizer thật."""
    torch = pytest.importorskip("torch")
    from ml.phobert_model import MultiHeadConfig, MultiHeadPhoBERT, load_tokenizer

    try:
        load_tokenizer()
    except Exception as error:
        pytest.skip(f"không nạp được tokenizer PhoBERT (cần cache Hub hoặc mạng): {error}")
    run = tmp_path_factory.mktemp("tiny") / "mtl-s1"
    torch.manual_seed(0)
    config = MultiHeadConfig(
        task_heads={"sentiment": 3, "topic": 4},
        vocab_size=64001,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=132,
        type_vocab_size=1,
        pad_token_id=1,
    )
    MultiHeadPhoBERT(config).save_pretrained(run)
    write_json(
        run / "history.json",
        {"normalizer_version": NORMALIZER_VERSION, "finished": True, "best_score": 0.5},
    )
    return run


@pytest.mark.train
@pytest.mark.slow
def test_phobert_smoke_measures_before_and_after_and_leaves_no_checkpoint_behind(
    tiny_checkpoint, simulation, tmp_path, monkeypatch
):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    shutil.copytree(tiny_checkpoint, tmp_path / "models" / "phobert" / "mtl-s1")
    models_before = sorted(p.name for p in (tmp_path / "models" / "phobert").rglob("*"))
    train, val = simulation.train, simulation.val
    _, eval_pos = hitl.split_validation(val)
    flagged = simulation.result["flagging"]["flagged_val_positions"]
    smoke = hitl.phobert_smoke(
        "best",
        train,
        hitl.feedback_from(val.iloc[flagged]),
        val.iloc[eval_pos].reset_index(drop=True),
        steps=3,
        seed=0,
    )
    assert smoke["skipped"] is False and smoke["base_run"] == "mtl-s1" and smoke["steps"] == 3
    assert smoke["device"] == "cpu" and smoke["n_eval"] == len(eval_pos)
    assert smoke["feedback_size"] == len(flagged) and len(smoke["losses"]) == 3
    assert smoke["train_seconds"] >= 0 and smoke["total_seconds"] >= smoke["train_seconds"]
    for task in TASKS:
        for side in ("before", "after"):
            assert 0 <= smoke[side][task]["macro_f1"] <= 1
        change = smoke["after"][task]["macro_f1"] - smoke["before"][task]["macro_f1"]
        assert smoke["delta_macro_f1"][task] == pytest.approx(change)
        assert 0 <= smoke["label_changes"][task] <= smoke["n_eval"]
    expected = hitl.gate_decision(smoke["before"], smoke["after"])
    assert smoke["accept_rule"] == expected  # cùng quy tắc nhận mà `ml.retrain` dùng
    json.dumps(smoke, allow_nan=False)
    scratch = tmp_path / "outputs" / "retrain"
    assert list(scratch.iterdir()) == []  # checkpoint tạm đã bị xoá
    assert sorted(p.name for p in (tmp_path / "models" / "phobert").rglob("*")) == models_before
    report = hitl.render_report({**simulation.result, "phobert_smoke": smoke})
    assert "## Smoke retrain PhoBERT (CPU, 3 bước)" in report
