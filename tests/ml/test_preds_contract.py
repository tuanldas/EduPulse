"""Hợp đồng dự đoán: khoá mô hình, ghi/đọc tệp, ngữ cảnh test và diễn tập."""

import numpy as np
import pytest

from ml import preds
from ml.common import write_json

SIZES = {"sentiment": 3, "topic": 4}


def synthetic(key: str, n: int = 12, seed: int = 0, tasks=("sentiment", "topic")) -> preds.Preds:
    rng = np.random.default_rng(seed)
    logits = {t: rng.normal(size=(n, SIZES[t])).astype(np.float32) for t in tasks}
    return preds.Preds(
        key,
        y={t: rng.integers(0, SIZES[t], size=n) for t in tasks},
        pred={t: logits[t].argmax(axis=1) for t in tasks},
        logits=logits,
    )


def test_valid_model_keys_are_accepted_and_others_rejected():
    for key in ("linearsvc", "logreg", "complementnb", "mtl-s42", "mtl-s7", "mtl-int8"):
        assert preds.check_key(key) == key
    for bad in ("svm", "mtl-s", "mtl-s42-extra", "stl-topic-s42", "mtl-int4", ""):
        with pytest.raises(ValueError, match="khoá mô hình lạ"):
            preds.check_key(bad)


def test_fp32_keys_carry_the_seed_and_nothing_else_does():
    assert preds.is_fp32_key("mtl-s42") and not preds.is_fp32_key("mtl-int8")
    assert preds.seed_of("mtl-s44") == 44
    assert preds.seed_of("linearsvc") is None and preds.seed_of(preds.DEPLOY_KEY) is None


def test_final_keys_are_the_baselines_the_deployed_run_and_its_int8_build():
    assert preds.final_keys("mtl-s42") == [
        "linearsvc",
        "logreg",
        "complementnb",
        "mtl-s42",
        "mtl-int8",
    ]
    with pytest.raises(ValueError):
        preds.final_keys("stl-topic-s42")


def test_canonical_order_is_baselines_then_fp32_by_seed_then_int8():
    shuffled = ["mtl-int8", "mtl-s43", "complementnb", "mtl-s42", "linearsvc", "logreg"]
    assert sorted(shuffled, key=preds.key_order) == [
        "linearsvc",
        "logreg",
        "complementnb",
        "mtl-s42",
        "mtl-s43",
        "mtl-int8",
    ]


def test_save_and_load_roundtrip_keeps_labels_predictions_and_logits(tmp_path):
    original = synthetic("mtl-s42")
    preds.save_preds(tmp_path, original)
    loaded = preds.load_preds(tmp_path, "mtl-s42")
    assert loaded.tasks == original.tasks
    for task in original.tasks:
        assert np.array_equal(loaded.y[task], original.y[task])
        assert np.array_equal(loaded.pred[task], original.pred[task])
        assert np.allclose(loaded.logits[task], original.logits[task])


def test_a_single_task_model_only_stores_its_own_task(tmp_path):
    preds.save_preds(tmp_path, synthetic("mtl-s42", tasks=("topic",)))
    loaded = preds.load_preds(tmp_path, "mtl-s42")
    assert loaded.tasks == ("topic",)
    assert set(np.load(tmp_path / "mtl-s42.npz").files) == {"y_topic", "pred_topic", "logits_topic"}


def test_missing_preds_file_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="chưa có dự đoán"):
        preds.load_preds(tmp_path, "mtl-s42")


def test_inconsistent_arrays_are_refused():
    good = synthetic("mtl-s42")
    with pytest.raises(ValueError, match="cùng tập tác vụ"):
        preds.Preds("mtl-s42", good.y, {"sentiment": good.pred["sentiment"]}, good.logits)
    short = {**good.pred, "topic": good.pred["topic"][:-1]}
    with pytest.raises(ValueError, match="lệch số dòng"):
        preds.Preds("mtl-s42", good.y, short, good.logits)


def test_available_keys_follow_the_canonical_order_not_the_file_order(tmp_path):
    for key in ("mtl-int8", "mtl-s42", "complementnb", "linearsvc"):
        preds.save_preds(tmp_path, synthetic(key))
    assert preds.available_keys(tmp_path) == ["linearsvc", "complementnb", "mtl-s42", "mtl-int8"]


def test_done_marker_and_meta_are_read_back(tmp_path):
    assert not preds.is_done(tmp_path)
    write_json(tmp_path / preds.META_FILE, {"split": "test", "deploy_run": "mtl-s42"})
    (tmp_path / preds.DONE_FILE).write_text("xong\n", encoding="utf-8")
    assert preds.is_done(tmp_path) and preds.read_meta(tmp_path)["deploy_run"] == "mtl-s42"


def test_real_context_points_at_test_preds_and_the_repo_results(monkeypatch, tmp_path):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    ctx = preds.context()
    assert ctx.split == "test" and not ctx.rehearsal and ctx.split_label == "test"
    assert ctx.preds_dir == tmp_path / "outputs" / "test_preds"
    assert ctx.results_dir == preds.REPO_ROOT / "results"
    assert ctx.metrics_path("a.json") == preds.REPO_ROOT / "results" / "metrics" / "a.json"
    assert ctx.table_path("a.md").parent.name == "tables"
    assert ctx.figure_path("a.png").parent.name == "figures"


def test_rehearsal_context_uses_validation_and_never_writes_into_the_repo(monkeypatch, tmp_path):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    ctx = preds.context(rehearsal=True)
    assert ctx.split == "validation" and ctx.rehearsal and ctx.split_label == "val"
    assert tmp_path in ctx.preds_dir.parents and tmp_path in ctx.results_dir.parents
    assert preds.REPO_ROOT not in ctx.results_dir.parents


def test_the_test_split_is_only_opened_by_the_real_context(monkeypatch, tmp_path):
    """Ngữ cảnh test thật mở test qua load_processed(allow_test=True); diễn tập không bao giờ."""
    opened: list[tuple[str, bool]] = []

    def fake_load(split, *, allow_test=False):
        opened.append((split, allow_test))

    monkeypatch.setattr(preds, "load_processed", fake_load)
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    preds.context(rehearsal=True).load_split()
    preds.context().load_split()
    assert opened == [("validation", False), ("test", True)]
