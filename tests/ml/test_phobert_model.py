"""Mô hình nhiều đầu: cấu hình đầu, bộ mã hoá không bị khởi tạo ngẫu nhiên, lưu/nạp giữ logits."""

import numpy as np
import pytest

pytestmark = pytest.mark.train

torch = pytest.importorskip("torch")

from ml.phobert_model import (  # noqa: E402
    BASE_MODEL,
    BASE_REVISION,
    MultiHeadConfig,
    MultiHeadPhoBERT,
    head_sizes,
    load_pretrained,
    load_tokenizer,
)


def tiny_config(task_heads):
    return MultiHeadConfig(
        task_heads=task_heads,
        vocab_size=120,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=70,
        pad_token_id=1,
    )


def test_head_sizes():
    assert head_sizes("mtl") == {"sentiment": 3, "topic": 4}
    assert head_sizes("sentiment") == {"sentiment": 3}
    assert head_sizes("topic") == {"topic": 4}
    with pytest.raises(ValueError, match="heads"):
        head_sizes("both")


def test_class_keeps_the_names_that_make_pretrained_weights_load():
    assert MultiHeadPhoBERT.base_model_prefix == "roberta"
    model = MultiHeadPhoBERT(tiny_config({"sentiment": 3}))
    assert hasattr(model, "roberta")


def test_outputs_are_one_logit_vector_per_head():
    model = MultiHeadPhoBERT(tiny_config({"sentiment": 3, "topic": 4})).eval()
    ids = torch.tensor([[0, 5, 6, 2], [0, 7, 2, 1]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    out = model(input_ids=ids, attention_mask=mask)
    assert list(out) == ["sentiment", "topic"]
    assert out["sentiment"].shape == (2, 3)
    assert out["topic"].shape == (2, 4)


def test_single_head_model_has_only_that_head():
    model = MultiHeadPhoBERT(tiny_config({"topic": 4})).eval()
    assert list(model(input_ids=torch.tensor([[0, 5, 2]]))) == ["topic"]


def test_config_without_heads_is_rejected():
    with pytest.raises(ValueError, match="task_heads"):
        MultiHeadPhoBERT(tiny_config({}))


def test_save_and_reload_give_identical_logits(tmp_path):
    torch.manual_seed(0)
    model = MultiHeadPhoBERT(tiny_config({"sentiment": 3, "topic": 4})).eval()
    ids = torch.tensor([[0, 5, 9, 2]])
    expected = model(input_ids=ids)
    model.save_pretrained(tmp_path)
    reloaded = MultiHeadPhoBERT.from_pretrained(tmp_path).eval()
    assert reloaded.config.task_heads == {"sentiment": 3, "topic": 4}
    again = reloaded(input_ids=ids)
    for name in expected:
        assert torch.equal(expected[name], again[name])


def test_head_initialisation_is_small_and_seeded():
    torch.manual_seed(3)
    first = MultiHeadPhoBERT(tiny_config({"sentiment": 3}))
    torch.manual_seed(3)
    second = MultiHeadPhoBERT(tiny_config({"sentiment": 3}))
    weight = first.heads["sentiment"].weight
    assert torch.equal(weight, second.heads["sentiment"].weight)
    assert weight.std().item() < 0.1
    assert first.heads["sentiment"].bias.abs().sum().item() == 0


@pytest.fixture(scope="module")
def pretrained():
    try:
        return load_pretrained("mtl")
    except OSError as error:  # không có mạng và chưa có trong cache Hub
        pytest.skip(f"không tải được {BASE_MODEL}: {error}")


def test_pretrained_encoder_weights_are_loaded_not_random(pretrained):
    """Bẫy `base_model_prefix`: nếu sai tên, bộ mã hoá âm thầm khởi tạo ngẫu nhiên."""
    from transformers import RobertaModel

    reference = RobertaModel.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, add_pooling_layer=False
    )
    ours = dict(pretrained.roberta.named_parameters())
    for name in (
        "embeddings.word_embeddings.weight",
        "encoder.layer.0.attention.self.query.weight",
        "encoder.layer.11.output.dense.weight",
    ):
        assert torch.equal(ours[name], dict(reference.named_parameters())[name]), name


def test_pretrained_model_reloads_with_identical_logits(pretrained, tmp_path):
    tokenizer = load_tokenizer()
    ids = torch.tensor([tokenizer.encode("thầy dạy rất nhiệt_tình .").ids])
    pretrained.eval()
    with torch.no_grad():
        expected = pretrained(input_ids=ids)
    pretrained.save_pretrained(tmp_path)
    reloaded = MultiHeadPhoBERT.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        again = reloaded(input_ids=ids)
    for name in expected:
        assert torch.allclose(expected[name], again[name], atol=1e-6)


def test_tokenizer_adds_special_tokens_and_truncates():
    tokenizer = load_tokenizer(max_len=16)
    short = tokenizer.encode("thầy dạy rất nhiệt_tình .").ids
    assert short[0] == 0 and short[-1] == 2  # <s> ... </s>
    long = tokenizer.encode(" ".join(["nhiệt_tình"] * 100)).ids
    assert len(long) == 16 and long[-1] == 2
    assert np.all(np.array(long) >= 0)
