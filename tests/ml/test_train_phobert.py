"""Vòng huấn luyện: gộp batch, đệm, bảo vệ MPS, và một run CPU nhỏ chạy hết đường."""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.train

torch = pytest.importorskip("torch")

from edupulse.text import NORMALIZER_VERSION  # noqa: E402
from ml import train_phobert  # noqa: E402
from ml.phobert_model import PAD_ID, MultiHeadPhoBERT, load_tokenizer  # noqa: E402


def test_run_names():
    assert train_phobert.run_name("mtl", 42, dry=False) == "mtl-s42"
    assert train_phobert.run_name("sentiment", 43, dry=False) == "stl-sentiment-s43"
    assert train_phobert.run_name("topic", 44, dry=True) == "dry-stl-topic-s44"


def test_epoch_batches_cover_every_sample_once_with_bounded_size():
    lengths = [3 + (i * 7) % 40 for i in range(1000)]
    batches = train_phobert.epoch_batches(lengths, batch_size=32, seed=42, epoch=0)
    flat = [i for batch in batches for i in batch]
    assert sorted(flat) == list(range(1000))
    assert max(len(b) for b in batches) <= 32


def test_epoch_batches_are_deterministic_and_change_between_epochs():
    lengths = [5 + i % 30 for i in range(300)]
    first = train_phobert.epoch_batches(lengths, 32, seed=1, epoch=0)
    assert first == train_phobert.epoch_batches(lengths, 32, seed=1, epoch=0)
    assert first != train_phobert.epoch_batches(lengths, 32, seed=1, epoch=1)
    assert first != train_phobert.epoch_batches(lengths, 32, seed=2, epoch=0)


def test_epoch_batches_group_similar_lengths():
    rng = np.random.default_rng(0)
    lengths = rng.integers(3, 60, size=1600).tolist()
    batches = train_phobert.epoch_batches(lengths, 32, seed=0, epoch=0)
    spread = np.mean([max(lengths[i] for i in b) - min(lengths[i] for i in b) for b in batches])
    assert spread < 10  # ngẫu nhiên hoàn toàn sẽ ~ 55


def test_pad_batch_pads_to_a_multiple_of_eight_and_masks_padding():
    ids, mask = train_phobert.pad_batch([[0, 5, 6, 2], [0, 7, 2]], torch.device("cpu"))
    assert ids.shape == mask.shape == (2, 8)
    assert ids[1].tolist() == [0, 7, 2] + [PAD_ID] * 5
    assert mask.tolist() == [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]]


def test_mps_run_is_refused_while_docker_is_running(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(train_phobert, "pick_device", lambda name: "mps")
    monkeypatch.setattr(train_phobert, "docker_running", lambda: True)
    code = train_phobert.main(["--out", str(tmp_path / "run")])
    assert code == 3
    assert "Docker Desktop" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()  # chưa tạo gì khi bị từ chối


def test_rejects_non_positive_epochs(capsys):
    assert train_phobert.main(["--epochs", "0"]) == 2


def test_full_cpu_run_waits_for_the_machine_wide_lock(monkeypatch, tmp_path, capsys):
    from ml.common import mps_lock

    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))

    def must_not_train(*args, **kwargs):
        raise AssertionError("không được train khi khoá đang do job khác giữ")

    monkeypatch.setattr(train_phobert, "train", must_not_train)
    with mps_lock(note="job khác"):
        code = train_phobert.main(["--device", "cpu", "--out", str(tmp_path / "run")])
    assert code == 4
    assert "job khác" in capsys.readouterr().err


def fake_history() -> dict:
    scores = {"macro_f1": 0.5}
    return {
        "epochs": [{"val": {"sentiment": scores, "topic": scores}}],
        "best_epoch": 1,
        "total_s": 1.0,
        "peak_mb": 0,
    }


def test_lock_held_by_the_queue_is_not_taken_again(monkeypatch, tmp_path):
    from ml.common import mps_lock

    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))
    monkeypatch.setattr(train_phobert, "train", lambda *args, **kwargs: fake_history())
    with mps_lock(note="hàng đợi"):  # tiến trình cha giữ khoá; con chạy với --lock-held
        argv = ["--device", "cpu", "--lock-held", "--out", str(tmp_path / "run")]
        assert train_phobert.main(argv) == 0


def test_lock_held_flag_without_a_holder_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("EDUPULSE_HOME", str(tmp_path))

    def must_not_train(*args, **kwargs):
        raise AssertionError("không được train khi không có ai giữ khoá")

    monkeypatch.setattr(train_phobert, "train", must_not_train)
    argv = ["--device", "cpu", "--lock-held", "--out", str(tmp_path / "run")]
    assert train_phobert.main(argv) == 4
    assert "--lock-held" in capsys.readouterr().err


@pytest.mark.data
@pytest.mark.slow
def test_tiny_cpu_run_writes_loadable_checkpoint_and_is_idempotent(tmp_path):
    out = tmp_path / "run"
    argv = ["--heads", "mtl", "--limit", "64", "--epochs", "1", "--device", "cpu"]
    argv += ["--out", str(out)]
    assert train_phobert.main(argv) == 0

    history = json.loads((out / "history.json").read_text("utf-8"))
    assert history["finished"] is True
    assert history["normalizer_version"] == NORMALIZER_VERSION
    assert history["best_epoch"] == 1 and len(history["epochs"]) == 1
    assert history["args"]["out"] == str(out)  # Path được ghi thành chuỗi
    assert {"sentiment", "topic"} == set(history["epochs"][0]["val"])
    assert (out / "model.safetensors").is_file()
    assert (out / "epoch1" / "model.safetensors").is_file()

    saved = np.load(out / "val_logits.npz")
    assert set(saved.files) == {"sentiment", "topic", "y_sentiment", "y_topic"}
    assert saved["sentiment"].shape == (32, 3) and saved["topic"].shape == (32, 4)

    # Nạp lại checkpoint và dự đoán lại phải ra đúng logits đã lưu.
    from ml.common import load_processed

    val = load_processed("validation").head(32)
    sequences = [e.ids for e in load_tokenizer().encode_batch(val["text_rdr"].tolist())]
    model = MultiHeadPhoBERT.from_pretrained(out)
    again = train_phobert.predict(model, sequences, torch.device("cpu"), ("sentiment", "topic"))
    for task in ("sentiment", "topic"):
        assert np.allclose(again[task], saved[task], atol=1e-4)

    stamp = (out / "history.json").stat().st_mtime_ns
    assert train_phobert.main(argv) == 0  # run đã hoàn tất: không chạy lại
    assert (out / "history.json").stat().st_mtime_ns == stamp


@pytest.mark.data
@pytest.mark.slow
def test_training_survives_sentences_whose_pieces_fall_outside_the_embedding_table(tmp_path):
    """Hồi quy: mảnh "đạt_" (id 65771) trong 800 câu train đầu từng làm sập cả run."""
    from ml.common import load_processed
    from ml.phobert_model import MODEL_VOCAB_SIZE

    first = load_processed("train")["text_rdr"].tolist()[:800]
    raw = [e.ids for e in load_tokenizer().encode_batch(first)]
    assert any(max(ids) >= MODEL_VOCAB_SIZE for ids in raw), (
        "tiền điều kiện của test không còn đúng"
    )

    out = tmp_path / "run"
    argv = ["--heads", "mtl", "--limit", "800", "--epochs", "1", "--device", "cpu"]
    assert train_phobert.main([*argv, "--out", str(out)]) == 0
    history = json.loads((out / "history.json").read_text("utf-8"))
    assert history["finished"] is True and history["n_train"] == 800
