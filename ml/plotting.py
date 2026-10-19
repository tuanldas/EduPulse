"""Kiểu vẽ và định dạng số chung cho hình và bảng trong `results/` (chỉ cần matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Màu ngữ nghĩa của docs/design-guidelines.md mục 2.2 (fill biểu đồ, dùng chung hai theme).
COLORS = {
    "negative": "#C94827",
    "neutral": "#8B909A",
    "positive": "#1B9A6E",
    "lecturer": "#0167BB",
    "training_program": "#944A76",
    "facility": "#B88818",
    "others": "#66837A",
}
INK, MUTED, GRID, BACKGROUND = "#14171E", "#5B6170", "#DCDAD2", "#FAFAF8"
ACCENT = "#0167BB"


def vn(value: float, digits: int = 1) -> str:
    """Số kiểu Việt: dấu phẩy thập phân, dấu chấm hàng nghìn."""
    text = f"{value:,.{digits}f}"
    return text.replace(",", "\0").replace(".", ",").replace("\0", ".")


def setup_style() -> None:
    families = {f.name for f in font_manager.fontManager.ttflist}
    preferred = "Be Vietnam Pro" if "Be Vietnam Pro" in families else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": preferred,
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def save_figure_to(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
