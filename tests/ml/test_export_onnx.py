"""Xuất ONNX: chạy trọn đường (xuất, so torch, int8, chấm val, câu vàng) trên mô hình nhỏ."""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.train, pytest.mark.data, pytest.mark.slow]

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")

from edupulse.bundle_schema import GoldenPredictions  # noqa: E402
from edupulse.labels import LABELS, TASKS  # noqa: E402
from edupulse.text import NORMALIZER_VERSION  # noqa: E402
from ml import export_onnx  # noqa: E402
from ml.common import write_json  # noqa: E402
from ml.phobert_model import MultiHeadConfig, MultiHeadPhoBERT  # noqa: E402


@pytest.fixture(scope="module")
def tiny_run(tmp_path_factory):
    """Checkpoint nhỏ (2 lớp, ẩn 32) nhưng từ vựng đủ lớn cho id của tokenizer PhoBERT."""
    run = tmp_path_factory.mktemp("models") / "mtl-s1"
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
    write_json(run / "history.json", {"normalizer_version": NORMALIZER_VERSION, "finished": True})
    return run


@pytest.fixture(scope="module")
def exported(tiny_run, tmp_path_factory):
    out = tmp_path_factory.mktemp("onnx")
    gate = export_onnx.export(tiny_run, out, threads=2)
    return out, gate


def test_gate_report_has_the_documented_structure(exported):
    _, gate = exported
    assert gate["run"] == "mtl-s1" and gate["n_val"] == 1583
    assert gate["parity"]["ok"] and gate["parity"]["max_abs_logit_diff"] < 1e-3
    assert set(gate["fp32"]) == set(gate["int8"]) == set(TASKS)
    assert set(gate["delta_macro_f1_points"]) == set(TASKS)
    assert gate["gate"]["max_drop_points"] == 1.0
    assert gate["files"]["int8_bytes"] < gate["files"]["fp32_bytes"]


def test_all_expected_files_are_written(exported):
    out, _ = exported
    for name in (
        "model.fp32.onnx",
        "model.int8.onnx",
        "val_logits_int8.npz",
        "onnx_gate.json",
        "golden_predictions.json",
    ):
        assert (out / name).is_file(), name
    assert not (out / "model.preprocessed.onnx").exists()  # tệp trung gian đã dọn


def test_saved_validation_logits_have_the_right_shapes(exported):
    out, _ = exported
    data = np.load(out / "val_logits_int8.npz")
    assert set(data.files) == {*TASKS, *(f"y_{t}" for t in TASKS)}
    for task in TASKS:
        assert data[task].shape == (1583, len(LABELS[task]))
        assert data[f"y_{task}"].shape == (1583,)


def test_session_exposes_the_documented_input_and_output_names(exported):
    out, _ = exported
    session = export_onnx.make_session(out / "model.int8.onnx", threads=1)
    assert [i.name for i in session.get_inputs()] == ["input_ids", "attention_mask"]
    assert [o.name for o in session.get_outputs()] == ["sentiment_logits", "topic_logits"]
    assert all(i.type == "tensor(int64)" for i in session.get_inputs())
    assert all(o.type == "tensor(float)" for o in session.get_outputs())


def test_dynamic_axes_accept_any_batch_and_length(exported):
    out, _ = exported
    session = export_onnx.make_session(out / "model.int8.onnx", threads=1)
    for batch, seq in ((1, 2), (1, 40), (3, 17), (32, 128)):
        ids = np.full((batch, seq), 5, dtype=np.int64)
        result = session.run(None, {"input_ids": ids, "attention_mask": np.ones_like(ids)})
        assert result[0].shape == (batch, 3) and result[1].shape == (batch, 4)


def test_run_logits_returns_rows_in_input_order(exported):
    out, _ = exported
    session = export_onnx.make_session(out / "model.int8.onnx", threads=1)
    sequences = [[0, 9, 2], [0, 5, 6, 7, 8, 2], [0, 4, 2], [0, 3, 3, 3, 2]]
    together = export_onnx.run_logits(session, sequences, batch_size=2)
    for index, sequence in enumerate(sequences):
        alone = export_onnx.run_logits(session, [sequence], batch_size=1)
        for task in TASKS:
            # Int8 động: lô khác nhau cho logits lệch nhẹ, nhưng thứ tự dòng phải đúng.
            assert together[task][index] == pytest.approx(alone[task][0], abs=0.25)


def test_golden_predictions_follow_the_single_sentence_contract(exported):
    out, _ = exported
    golden = GoldenPredictions.model_validate_json(
        (out / "golden_predictions.json").read_text("utf-8")
    )
    assert golden.batch_size == 1 and golden.normalizer_version == NORMALIZER_VERSION
    assert [item.id for item in golden.items] == [f"g{i:02d}" for i in range(1, 21)]
    session = export_onnx.make_session(out / "model.int8.onnx", threads=1)
    first = golden.items[0]
    ids = export_onnx.encode([first.model_text])
    logits = export_onnx.run_logits(session, ids, batch_size=1)
    probs = np.exp(logits["sentiment"][0] - logits["sentiment"][0].max())
    assert (probs / probs.sum()).tolist() == pytest.approx(first.p_sentiment, abs=1e-4)
    for item in golden.items:
        assert sum(item.p_sentiment) == pytest.approx(1.0) and sum(item.p_topic) == pytest.approx(
            1.0
        )
        assert item.sentiment in LABELS["sentiment"] and item.topic in LABELS["topic"]


def test_gate_json_is_valid_utf8_json(exported):
    out, gate = exported
    assert json.loads((out / "onnx_gate.json").read_text("utf-8"))["run"] == gate["run"]


def test_refuses_runs_made_by_another_normalizer(tiny_run, tmp_path):
    history = json.loads((tiny_run / "history.json").read_text("utf-8"))
    stale = tmp_path / "stale"
    stale.mkdir()
    for name in ("config.json", "model.safetensors"):
        (stale / name).write_bytes((tiny_run / name).read_bytes())
    write_json(stale / "history.json", {**history, "normalizer_version": "0"})
    with pytest.raises(RuntimeError, match="chuẩn hoá"):
        export_onnx.export(stale, tmp_path / "out", threads=1)
