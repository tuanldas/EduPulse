"""Kiểu vẽ và định dạng số chung cho hình và bảng trong `results/` (chỉ cần matplotlib)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

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


def format_tick(value: float, _position: int | None = None) -> str:
    """Nhãn trục kiểu Việt: dấu phẩy thập phân, dấu chấm hàng nghìn (0,2 · 1.600 · 1.234,5)."""
    rounded = round(float(value), 6) + 0.0  # + 0.0 đổi -0.0 thành 0.0
    text = f"{rounded:.6f}".rstrip("0").rstrip(".")
    decimals = len(text.split(".")[1]) if "." in text else 0
    return vn(rounded, decimals)


def vietnamese_axes(ax: plt.Axes, which: str = "both") -> None:
    """Đặt nhãn số của trục tuyến tính theo kiểu Việt; trục log giữ nhãn luỹ thừa 10 mặc định.

    Mỗi trục nhận một FuncFormatter riêng: matplotlib gắn formatter vào đúng một trục.
    """
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(FuncFormatter(format_tick))
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(FuncFormatter(format_tick))


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
