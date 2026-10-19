"""Kiểm định ghép cặp trên dự đoán đã lưu: so sánh đúng cặp, kết luận đúng, ghi đúng tệp."""

import json

import numpy as np
import pytest

from ml import preds, significance
from ml.common import write_json

SIZES = {"sentiment": 3, "topic": 4}


def make_preds(key: str, y: dict, accuracy: float, seed: int) -> preds.Preds:
    """Dự đoán đúng với xác suất `accuracy`, sai thì chọn lớp khác ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    pred, logits = {}, {}
    for task, truth in y.items():
        wrong = (truth + rng.integers(1, SIZES[task], size=len(truth))) % SIZES[task]
        pred[task] = np.where(rng.random(len(truth)) < accuracy, truth, wrong)
        logits[task] = np.eye(SIZES[task])[pred[task]].astype(np.float32)
    return preds.Preds(key, y, pred, logits)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    y = {task: rng.integers(0, size, size=800) for task, size in SIZES.items()}
    quality = {"linearsvc": 0.80, "logreg": 0.78, "complementnb": 0.70, "mtl-s42": 0.92}
    directory = tmp_path / "preds"
    for index, (key, accuracy) in enumerate(quality.items()):
        preds.save_preds(directory, make_preds(key, y, accuracy, seed=index + 1))
    # Bản int8 gần như trùng bản FP32: 5 câu đúng thành sai và 5 câu sai thành đúng (đối xứng),
    # nên hai bên không khác biệt có ý nghĩa theo cả hai kiểm định.
    fp32 = preds.load_preds(directory, "mtl-s42")
    int8_pred = {t: fp32.pred[t].copy() for t in fp32.tasks}
    for task in int8_pred:
        right = np.flatnonzero(fp32.pred[task] == y[task])[:5]
        wrong = np.flatnonzero(fp32.pred[task] != y[task])[:5]
        int8_pred[task][right] = (y[task][right] + 1) % SIZES[task]
        int8_pred[task][wrong] = y[task][wrong]
    preds.save_preds(
        directory, preds.Preds(preds.DEPLOY_KEY, y, int8_pred, {t: fp32.logits[t] for t in y})
    )
    write_json(directory / preds.META_FILE, {"split": "test", "n": 800, "deploy_run": "mtl-s42"})
    (directory / preds.DONE_FILE).write_text("xong\n", encoding="utf-8")
    ctx = preds.EvalContext("test", directory, tmp_path / "results", rehearsal=False)
    monkeypatch.setattr(significance, "context", lambda rehearsal=False: ctx)
    return ctx


def test_pairs_compare_the_deployed_build_with_each_baseline_and_with_its_fp32_run():
    assert significance.pairs("mtl-s42") == [
        ("baseline", "mtl-int8", "linearsvc"),
        ("baseline", "mtl-int8", "logreg"),
        ("baseline", "mtl-int8", "complementnb"),
        ("fp32", "mtl-int8", "mtl-s42"),
    ]


def test_run_produces_one_record_per_pair_and_task(workspace):
    result = significance.run(workspace)
    assert len(result["records"]) == 4 * 2
    assert {(r["group"], r["task"]) for r in result["records"]} == {
        (group, task) for group in ("baseline", "fp32") for task in ("sentiment", "topic")
    }
    assert result["n"] == 800 and result["alpha"] == 0.05
    assert result["bootstrap"] == {"rounds": 2000, "seed": 0}


def test_much_better_model_is_significant_and_a_near_copy_is_not(workspace):
    records = significance.run(workspace)["records"]
    vs_svc = next(r for r in records if r["b"] == "linearsvc" and r["task"] == "sentiment")
    assert vs_svc["delta_macro_f1"] > 0.05 and vs_svc["ci95_low"] > 0
    assert vs_svc["significant_accuracy"] and vs_svc["significant_macro_f1"]
    vs_fp32 = next(r for r in records if r["group"] == "fp32" and r["task"] == "sentiment")
    assert abs(vs_fp32["delta_macro_f1"]) < 0.02
    assert not vs_fp32["significant_accuracy"] and not vs_fp32["significant_macro_f1"]
    assert (vs_fp32["a_only_correct"], vs_fp32["b_only_correct"]) == (5, 5)
    assert vs_fp32["mcnemar_p"] == pytest.approx(1.0)


def test_a_one_directional_tiny_change_is_flagged_by_the_interval_but_not_by_mcnemar():
    """5 câu chỉ xấu đi: McNemar không đủ bằng chứng (p = 0,0625) nhưng hiệu bootstrap luôn <= 0."""
    y = np.tile(np.arange(3), 100)
    pred_a = y.copy()
    pred_b = y.copy()
    pred_b[:5] = (y[:5] + 1) % 3  # chỉ mô hình B sai ở 5 câu
    record = significance.compare(
        "baseline",
        preds.Preds("a", {"sentiment": y}, {"sentiment": pred_a}, {"sentiment": np.eye(3)[pred_a]}),
        preds.Preds("b", {"sentiment": y}, {"sentiment": pred_b}, {"sentiment": np.eye(3)[pred_b]}),
        "sentiment",
    )
    assert record["mcnemar_p"] == pytest.approx(0.0625)
    assert not record["significant_accuracy"] and record["significant_macro_f1"]
    assert record["significant_macro_f1"] == (record["ci95_low"] > 0 or record["ci95_high"] < 0)


def test_records_match_a_direct_paired_difference(workspace):
    from ml import stats

    record = next(
        r
        for r in significance.run(workspace)["records"]
        if r["b"] == "complementnb" and r["task"] == "topic"
    )
    a = preds.load_preds(workspace.preds_dir, "mtl-int8")
    b = preds.load_preds(workspace.preds_dir, "complementnb")
    direct = stats.paired_difference(a.y["topic"], a.pred["topic"], b.pred["topic"], 4)
    for key, value in direct.items():
        assert record[key] == pytest.approx(value)


def test_comparing_models_with_different_labels_is_refused(workspace):
    a = preds.load_preds(workspace.preds_dir, "mtl-int8")
    b = preds.load_preds(workspace.preds_dir, "linearsvc")
    b.y["sentiment"] = (b.y["sentiment"] + 1) % 3
    with pytest.raises(ValueError, match="không cùng nhãn thật"):
        significance.compare("baseline", a, b, "sentiment")


def test_markdown_lists_both_groups_with_vietnamese_numbers(workspace):
    text = significance.markdown(significance.run(workspace))
    assert "Bản triển khai (int8) so với baseline TF-IDF" in text
    assert "Bản triển khai (int8) so với PhoBERT FP32 cùng run" in text
    assert "McNemar" in text and "bootstrap" in text and "tập test" in text
    assert text.count("| Cảm xúc |") == 4 and text.count("| Chủ đề |") == 4
    assert "." not in text.split("| Cảm xúc |")[1].split("|")[3]  # số Việt dùng dấu phẩy thập phân


def test_main_writes_the_json_and_the_table_and_prints_the_table(workspace, capsys):
    assert significance.main([]) == 0
    out = capsys.readouterr().out
    data = json.loads(workspace.metrics_path("significance.json").read_text("utf-8"))
    assert len(data["records"]) == 8 and "Kiểm định ghép cặp" in out
    table = workspace.table_path("significance.md").read_text("utf-8")
    assert table.startswith("Kiểm định ghép cặp trên tập test")


def test_main_refuses_to_run_before_the_evaluation_is_done(workspace, capsys):
    (workspace.preds_dir / preds.DONE_FILE).unlink()
    assert significance.main([]) == significance.EXIT_NOT_DONE
    assert "chưa chấm xong" in capsys.readouterr().err
    assert not workspace.metrics_path("significance.json").exists()
