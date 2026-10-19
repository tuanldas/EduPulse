"""Hợp đồng dự đoán: khoá mô hình, ghi/đọc tệp, ngữ cảnh test và diễn tập."""

import numpy as np
import pytest

from ml import preds
from ml.common import write_json


def synthetic(key: str, n: int = 12, seed: int = 0) -> preds.Preds:
    rng = np.random.default_rng(seed)
    tasks = preds.tasks_of(key)
    sizes = {"sentiment": 3, "topic": 4}
    logits = {t: rng.normal(size=(n, sizes[t])).astype(np.float32) for t in tasks}
    return preds.Preds(
        key,
        y={t: rng.integers(0, sizes[t], size=n) for t in tasks},
        pred={t: logits[t].argmax(axis=1) for t in tasks},
        logits=logits,
    )


def test_every_model_key_maps_to_its_tasks():
    assert preds.tasks_of("linearsvc") == ("sentiment", "topic")
    assert preds.tasks_of("mtl-s43") == ("sentiment", "topic")
    assert preds.tasks_of("stl-sentiment-s42") == ("sentiment",)
    assert preds.tasks_of("stl-topic-s44") == ("topic",)
    assert preds.tasks_of(preds.DEPLOY_KEY) == ("sentiment", "topic")


def test_unknown_keys_are_rejected():
    for bad in ("svm", "mtl-s", "stl-other-s42", "mtl-s42-extra"):
        with pytest.raises(ValueError, match="khoá mô hình lạ"):
            preds.tasks_of(bad)
        with pytest.raises(ValueError):
            preds.arch_of(bad)


def test_architecture_and_seed_of_a_key():
    assert preds.arch_of("complementnb") == "baseline"
    assert preds.arch_of("mtl-s44") == "mtl"
    assert preds.arch_of("stl-topic-s42") == "stl-topic"
    assert preds.arch_of(preds.DEPLOY_KEY) == "mtl-int8"
    assert preds.seed_of("mtl-s44") == 44 and preds.seed_of("linearsvc") is None
    assert preds.seed_of(preds.DEPLOY_KEY) is None


def test_canonical_key_order_is_baselines_mtl_stl_then_the_deployed_model():
    keys = preds.all_keys()
    assert len(keys) == 13 and len(set(keys)) == 13
    assert keys[:3] == list(preds.BASELINES) and keys[-1] == preds.DEPLOY_KEY
    assert keys[3:6] == ["mtl-s42", "mtl-s43", "mtl-s44"]
    assert keys[6:9] == [f"stl-sentiment-s{s}" for s in preds.SEEDS]


def test_save_and_load_roundtrip_keeps_labels_predictions_and_logits(tmp_path):
    original = synthetic("mtl-s42")
    preds.save_preds(tmp_path, original)
    loaded = preds.load_preds(tmp_path, "mtl-s42")
    assert loaded.tasks == original.tasks
    for task in original.tasks:
        assert np.array_equal(loaded.y[task], original.y[task])
        assert np.array_equal(loaded.pred[task], original.pred[task])
        assert np.allclose(loaded.logits[task], original.logits[task])


def test_single_task_model_only_stores_its_own_task(tmp_path):
    preds.save_preds(tmp_path, synthetic("stl-topic-s42"))
    loaded = preds.load_preds(tmp_path, "stl-topic-s42")
    assert loaded.tasks == ("topic",)
    assert set(np.load(tmp_path / "stl-topic-s42.npz").files) == {
        "y_topic",
        "pred_topic",
        "logits_topic",
    }


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
    for key in ("mtl-int8", "stl-topic-s42", "linearsvc", "mtl-s43"):
        preds.save_preds(tmp_path, synthetic(key))
    assert preds.available_keys(tmp_path) == ["linearsvc", "mtl-s43", "stl-topic-s42", "mtl-int8"]


def test_done_marker_and_meta_are_read_back(tmp_path):
    assert not preds.is_done(tmp_path)
    write_json(tmp_path / preds.META_FILE, {"split": "test", "deploy_run": "mtl-s43"})
    (tmp_path / preds.DONE_FILE).write_text("xong\n", encoding="utf-8")
    assert preds.is_done(tmp_path) and preds.read_meta(tmp_path)["deploy_run"] == "mtl-s43"


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


def test_the_test_split_cannot_be_opened_by_accident(monkeypatch, tmp_path):
    """Ngữ cảnh test thật mở test qua load_processed(allow_test=True); diễn tập không bao giờ."""
    opened: list[tuple[str, bool]] = []

    def fake_load(split, *, allow_test=False):
        opened.append((split, allow_test))
        return None

    monkeypatch.setattr(preds, "load_processed", fake_load)
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    preds.context(rehearsal=True).load_split()
    preds.context().load_split()
    assert opened == [("validation", False), ("test", True)]
