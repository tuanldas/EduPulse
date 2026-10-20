"""Nhãn trục kiểu Việt trong hình kết quả: dấu phẩy thập phân, dấu chấm hàng nghìn."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.ticker import FuncFormatter

from ml import eval_tables, plotting, preds


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, "0"),
        (0.0, "0"),
        (-0.0, "0"),
        (1, "1"),
        (0.2, "0,2"),
        (0.25, "0,25"),
        (0.125, "0,125"),
        (0.1 + 0.2, "0,3"),
        (1600, "1.600"),
        (1234.5, "1.234,5"),
        (-0.5, "-0,5"),
        (2.0000001, "2"),
        (1e-9, "0"),
        (11426, "11.426"),
    ],
)
def test_tick_labels_use_a_decimal_comma_and_a_thousands_dot(value, expected):
    assert plotting.format_tick(value) == expected
    assert plotting.format_tick(value, 3) == expected  # chữ ký của FuncFormatter: (giá trị, vị trí)


def test_every_axis_gets_its_own_formatter_instance():
    fig, (first, second) = plt.subplots(1, 2)
    plotting.vietnamese_axes(first)
    plotting.vietnamese_axes(second, "y")
    assert isinstance(first.xaxis.get_major_formatter(), FuncFormatter)
    assert first.xaxis.get_major_formatter() is not first.yaxis.get_major_formatter()
    assert first.yaxis.get_major_formatter() is not second.yaxis.get_major_formatter()
    assert not isinstance(second.xaxis.get_major_formatter(), FuncFormatter)  # chỉ trục y
    plt.close(fig)


def test_a_drawn_linear_axis_shows_commas_not_dots():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 2400])
    ax.set_xlim(0, 1)
    plotting.vietnamese_axes(ax)
    fig.canvas.draw()
    x_labels = [t.get_text() for t in ax.get_xticklabels()]
    y_labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "0,2" in x_labels and "0,8" in x_labels
    assert any(
        label.count(".") == 1 and label.endswith(".000") for label in y_labels
    )  # 1.000, 2.000
    assert not any("." in label for label in x_labels)
    plt.close(fig)


def synthetic_series():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=400)
    logits = (3.0 * np.eye(3)[y] + rng.normal(size=(400, 3))).astype(np.float32)
    model = preds.Preds(
        "mtl-int8", {"sentiment": y}, {"sentiment": logits.argmax(1)}, {"sentiment": logits}
    )
    metrics = eval_tables.preds_metrics(model, "sentiment", reliability=True)
    return {"PhoBERT": metrics, "TF-IDF": metrics}


def test_reliability_figure_axes_use_vietnamese_decimals(monkeypatch, tmp_path):
    captured = {}

    def keep(fig, path):
        captured["fig"] = fig
        return path

    monkeypatch.setattr(eval_tables, "save_figure_to", keep)
    eval_tables.plot_reliability(tmp_path / "r.png", synthetic_series(), "sentiment", "Độ tin cậy")
    fig = captured["fig"]
    fig.canvas.draw()
    top, bottom = fig.axes
    y_labels = [t.get_text() for t in top.get_yticklabels()]
    x_labels = [t.get_text() for t in bottom.get_xticklabels()]
    assert {"0,2", "0,4", "0,6", "0,8"} <= set(y_labels) and {"0,2", "0,8"} <= set(x_labels)
    assert not any("." in label for label in y_labels + x_labels)
    plt.close(fig)


def test_reliability_figure_file_is_still_written(tmp_path):
    path = eval_tables.plot_reliability(
        tmp_path / "r.png", synthetic_series(), "sentiment", "Độ tin cậy"
    )
    assert path.is_file() and path.stat().st_size > 5000


@pytest.mark.train
@pytest.mark.data
def test_length_histograms_of_the_eda_use_the_same_formatter(monkeypatch):
    pytest.importorskip("torch")
    from ml import eda

    texts = [
        "thầy dạy rất hay",
        "phòng máy nóng quá",
        "chương_trình học nặng lý_thuyết",
        "em thấy môn này bổ_ích",
    ] * 10
    frame = pd.DataFrame(
        {
            "text_rdr": texts,
            "sentence": [t.replace("_", " ") for t in texts],
            "n_words": [len(t.split()) for t in texts],
            "split": ["train", "validation", "test", "train"] * 10,
        }
    )
    captured = {}
    monkeypatch.setattr(eda, "save_table", lambda *args, **kwargs: None)
    monkeypatch.setattr(eda, "save_figure", lambda fig, name: captured.setdefault(name, fig))
    eda.length_report(frame)
    fig = captured["eda_length.png"]
    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert isinstance(ax.xaxis.get_major_formatter(), FuncFormatter)
        assert isinstance(ax.yaxis.get_major_formatter(), FuncFormatter)
    plt.close(fig)
