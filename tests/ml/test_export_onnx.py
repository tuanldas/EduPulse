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
    assert gate["gate"]["modes"] == ["batch=1", "batch=32"]
    assert gate["files"]["int8_bytes"] < gate["files"]["fp32_bytes"]


def test_gate_covers_both_serving_modes_and_reports_batch_dependence(exported):
    _, gate = exported
    assert set(gate["int8_batched"]) == set(gate["delta_macro_f1_points_batched"]) == set(TASKS)
    for task in TASKS:
        stats = gate["batch_dependence"][task]
        assert 0 <= stats["median_abs_logit_diff"] <= stats["p95_abs_logit_diff"]
        assert stats["p95_abs_logit_diff"] <= stats["max_abs_logit_diff"]
        assert 0 <= stats["label_flips"] <= gate["n_val"]
    worst = max(
        -delta
        for deltas in (gate["delta_macro_f1_points"], gate["delta_macro_f1_points_batched"])
        for delta in deltas.values()
    )
    assert gate["gate"]["passed"] == (worst <= gate["gate"]["max_drop_points"])


def test_batch_dependence_statistics_on_known_arrays():
    single = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    batched = np.array([[1.1, 0.0], [0.0, 0.8], [0.0, 2.0]])
    stats = export_onnx.batch_dependence(single, batched)
    assert stats["max_abs_logit_diff"] == pytest.approx(2.0)
    assert stats["label_flips"] == 1  # chỉ dòng cuối đổi nhãn
    assert stats["median_abs_logit_diff"] == pytest.approx(0.15)  # sắp xếp: 0 0 0,1 0,2 2 2


def test_reference_validation_logits_are_the_single_sentence_run(exported):
    out, _ = exported
    saved = np.load(out / "val_logits_int8.npz")
    session = export_onnx.make_session(out / "model.int8.onnx", threads=2)
    from ml.common import load_processed

    sentences = load_processed("validation")["text_rdr"].tolist()[:5]
    for index, sequence in enumerate(export_onnx.encode(sentences)):
        alone = export_onnx.run_logits(session, [sequence], batch_size=1)
        for task in TASKS:
            assert saved[task][index] == pytest.approx(alone[task][0], abs=1e-5)


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


# --- Id ngoài bảng embedding gốc ------------------------------------------------------------------

RARE_PIECE_TEXT = "thầy truyền_đạt_kiến thức rất tốt ."


def test_embedding_report_documents_the_expansion(exported):
    _, gate = exported
    report = gate["embedding"]
    assert (report["base_vocab_size"], report["unk_id"]) == (64001, 3)
    assert report["embedding_rows"] == 66119 and report["extra_rows"] == 2118
    assert report["unk_alias_max_diff"]["fp32"] <= 1e-5
    assert report["unk_alias_max_diff"]["int8"] <= 1e-5
    parity = report["rare_piece_parity"]
    assert parity["ok"] and parity["n_out_of_range"] >= 2  # hai câu tự viết, cộng câu của dữ liệu
    assert parity["max_abs_logit_diff"] < parity["tolerance"]


def test_onnx_accepts_the_largest_tokenizer_id_and_treats_it_as_unk(exported):
    out, _ = exported
    for name in ("model.fp32.onnx", "model.int8.onnx"):
        session = export_onnx.make_session(out / name, threads=1)
        last = export_onnx.run_logits(session, [[0, 66118, 5, 2]], batch_size=1)
        unk = export_onnx.run_logits(session, [[0, 3, 5, 2]], batch_size=1)
        for task in TASKS:
            assert np.array_equal(last[task], unk[task]), name
        assert export_onnx.unk_alias_gap(session, 66119) == 0.0


def test_ids_beyond_the_tokenizer_vocabulary_are_still_rejected(exported):
    out, _ = exported
    session = export_onnx.make_session(out / "model.int8.onnx", threads=1)
    with pytest.raises(Exception, match="out of bounds|out of range|bound"):
        export_onnx.run_logits(session, [[0, 66119, 2]], batch_size=1)


def test_torch_and_onnx_agree_on_a_sentence_with_an_out_of_table_piece(exported, tiny_run):
    """Torch nhận id đã đổi thành <unk>, ONNX nhận id thô của tokenizer.json: kết quả phải khớp."""
    from ml.phobert_model import MODEL_VOCAB_SIZE, clamp_ids

    out, _ = exported
    raw = export_onnx.encode([RARE_PIECE_TEXT])
    assert max(raw[0]) >= MODEL_VOCAB_SIZE
    model = MultiHeadPhoBERT.from_pretrained(tiny_run).eval()
    expected = export_onnx.torch_logits(model, [clamp_ids(raw[0])])
    session = export_onnx.make_session(out / "model.fp32.onnx", threads=1)
    actual = export_onnx.run_logits(session, raw, batch_size=1)
    for task in TASKS:
        assert actual[task] == pytest.approx(expected[task], abs=1e-4)


def test_int8_model_runs_on_the_rare_piece_sentence(exported):
    out, _ = exported
    session = export_onnx.make_session(out / "model.int8.onnx", threads=1)
    logits = export_onnx.run_logits(session, export_onnx.encode([RARE_PIECE_TEXT]), batch_size=1)
    assert all(np.isfinite(logits[task]).all() for task in TASKS)
