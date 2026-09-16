"""Shared compact layout and visual style for appendix figures."""
from __future__ import annotations

from collections.abc import Sequence
from math import ceil
from common import arm_color, sequence_colors, RCPARAMS

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


INK = "#22252A"
MUTED = "#68707C"
GRID = "#DDE1E7"

SINGLE_COLUMN_WIDTH = 3.4
FULL_WIDTH = 6.9
PANEL_HEIGHT = 1.75
PANEL_TITLE_BAND = 0.24
X_TICK_ROW = 0.21
Y_TICK_COLUMN = 0.32
AXIS_LABEL_BAND = 0.25
FRAME_GAP = 0.08
LEGEND_ROW = 0.18
LEGEND_GAP = 0.05
OUTER_PAD = 0.04
MIN_PANEL_WIDTH = 0.95

ARCHITECTURE_LABELS = {
    "van": "Vanilla",
    "k1": "Operator-1",
    "k2": "Loop-2",
    "dep": "Untied-2",
    "k2_grow": "Loop Grow",
    "dep_grow": "Untied Grow",
}

ARCHITECTURE_COLORS = {arm: arm_color(arm) for arm in ARCHITECTURE_LABELS}


ARCHITECTURE_MARKERS = {
    "van": "o",
    "k1": "s",
    "k2": "^",
    "dep": "D",
    "k2_grow": "P",
    "dep_grow": "v",
}

PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "axes.titleweight": "bold",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.axisbelow": True,
    "text.color": INK,
    "figure.labelsize": 10,
    "figure.labelweight": "normal",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.color": GRID,
    "grid.linewidth": 0.65,
    "grid.alpha": 0.65,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.0,
    "legend.labelspacing": 0.3,
    "legend.borderpad": 0,
    "legend.borderaxespad": 0,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
    "savefig.bbox": None,
    "figure.autolayout": False,
    "figure.constrained_layout.use": False,
}


def apply_style(**overrides) -> None:
    """Install the report style, with optional figure-specific overrides."""
    mpl.rcParams.update(PAPER_STYLE | {key: value for key, value in RCPARAMS.items() if "color" in key or key == "pdf.fonttype"} | overrides)


def style_axis(axis, *, grid: bool = True) -> None:
    """Apply the common axis treatment without changing scales or labels."""
    if grid:
        axis.grid(True, which="major")
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(MUTED)
        spine.set_linewidth(PAPER_STYLE["axes.linewidth"])


def legend_row_count(labels: Sequence[str], *, width: float = FULL_WIDTH, max_rows: int | None = None) -> int:
    """Return the fewest legend rows whose fixed-width entries fit the canvas."""
    labels = tuple(labels)
    if not labels:
        return 0
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("legend labels must be non-empty strings")
    if width <= 2 * OUTER_PAD:
        raise ValueError("legend canvas is narrower than its outer padding")
    if max_rows is None:
        max_rows = len(labels)
    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    max_rows = min(max_rows, len(labels))

    measurement_figure = plt.figure(figsize=(width, 1))
    texts = [measurement_figure.text(0, 0, label, fontsize=mpl.rcParams["legend.fontsize"]) for label in labels]
    measurement_figure.canvas.draw()
    renderer = measurement_figure.canvas.get_renderer()
    text_widths = [text.get_window_extent(renderer).width / measurement_figure.dpi for text in texts]
    plt.close(measurement_figure)

    font_inches = FontProperties(size=mpl.rcParams["legend.fontsize"]).get_size_in_points() / 72
    entry_padding = (mpl.rcParams["legend.handlelength"] + mpl.rcParams["legend.handletextpad"]) * font_inches
    column_spacing = mpl.rcParams["legend.columnspacing"] * font_inches
    border = 2 * mpl.rcParams["legend.borderpad"] * font_inches
    entry_widths = [entry_padding + text_width for text_width in text_widths]
    available = width - 2 * OUTER_PAD
    for rows in range(1, max_rows + 1):
        columns = ceil(len(labels) / rows)
        column_widths = [max(entry_widths[row * columns + column] for row in range(rows) if row * columns + column < len(labels)) for column in range(columns)]
        legend_width = border + sum(column_widths) + (columns - 1) * column_spacing
        if legend_width <= available:
            return rows
    raise ValueError(f"legend does not fit within {max_rows} rows; shorten labels or allow more rows")


def make_grid(
    nrows: int,
    ncols: int,
    *,
    titles: Sequence[str],
    width: float = FULL_WIDTH,
    panel_height: float = PANEL_HEIGHT,
    square_panels: bool = False,
    sharex: bool | str = False,
    sharey: bool | str = False,
    share_xlabel: bool = False,
    share_ylabel: bool = False,
    shared_xlabel: str | None = None,
    shared_ylabel: str | None = None,
    column_xlabels: Sequence[str] | None = None,
    row_ylabels: Sequence[str] | None = None,
    legend_rows: int = 0,
    x_tick_rows: int = 1,
    y_tick_columns: int = 1,
    top_tick_rows: int = 0,
    right_tick_columns: int = 0,
    top_label_rows: int = 0,
    right_label_columns: int = 0,
    squeeze: bool = True,
):
    """Create an exact-size publication grid with every decoration preallocated."""
    counts = {
        "nrows": nrows,
        "ncols": ncols,
        "legend_rows": legend_rows,
        "x_tick_rows": x_tick_rows,
        "y_tick_columns": y_tick_columns,
        "top_tick_rows": top_tick_rows,
        "right_tick_columns": right_tick_columns,
        "top_label_rows": top_label_rows,
        "right_label_columns": right_label_columns,
    }
    for name, value in counts.items():
        minimum = 1 if name in {"nrows", "ncols"} else 0
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    axis_sharing = {"sharex": sharex, "sharey": sharey}
    for name, value in axis_sharing.items():
        if not isinstance(value, bool) and value not in {"none", "all", "row", "col"}:
            raise ValueError(f"{name} must be a boolean or one of: none, all, row, col")
    if not isinstance(share_xlabel, bool) or not isinstance(share_ylabel, bool):
        raise ValueError("label sharing options must be booleans")
    if width <= 0 or panel_height <= 0:
        raise ValueError("width and panel_height must be positive")
    if len(titles) != nrows * ncols:
        raise ValueError(f"expected {nrows * ncols} panel titles, got {len(titles)}")
    if any(not isinstance(title, str) or not title.strip() for title in titles):
        raise ValueError("every panel must have a non-empty title")
    if shared_xlabel is not None and (not isinstance(shared_xlabel, str) or not shared_xlabel.strip()):
        raise ValueError("shared_xlabel must be a non-empty string")
    if shared_ylabel is not None and (not isinstance(shared_ylabel, str) or not shared_ylabel.strip()):
        raise ValueError("shared_ylabel must be a non-empty string")
    if column_xlabels is not None:
        if shared_xlabel is not None:
            raise ValueError("use either shared_xlabel or column_xlabels, not both")
        if len(column_xlabels) != ncols or any(not isinstance(label, str) or not label.strip() for label in column_xlabels):
            raise ValueError(f"column_xlabels must contain {ncols} non-empty strings")
    if row_ylabels is not None:
        if shared_ylabel is not None:
            raise ValueError("use either shared_ylabel or row_ylabels, not both")
        if len(row_ylabels) != nrows or any(not isinstance(label, str) or not label.strip() for label in row_ylabels):
            raise ValueError(f"row_ylabels must contain {nrows} non-empty strings")
    share_xlabel = share_xlabel or shared_xlabel is not None or column_xlabels is not None
    share_ylabel = share_ylabel or shared_ylabel is not None or row_ylabels is not None
    shares_x_between_rows = sharex is True or sharex in {"all", "col"}
    shares_y_between_columns = sharey is True or sharey in {"all", "row"}

    left = OUTER_PAD + y_tick_columns * Y_TICK_COLUMN + AXIS_LABEL_BAND
    right = OUTER_PAD + right_tick_columns * Y_TICK_COLUMN + right_label_columns * AXIS_LABEL_BAND
    column_gap = FRAME_GAP + right_tick_columns * Y_TICK_COLUMN + right_label_columns * AXIS_LABEL_BAND
    if not shares_y_between_columns:
        column_gap += y_tick_columns * Y_TICK_COLUMN
    if not share_ylabel:
        column_gap += AXIS_LABEL_BAND
    panel_width = (width - left - right - (ncols - 1) * column_gap) / ncols
    if panel_width < MIN_PANEL_WIDTH:
        raise ValueError(
            f"layout leaves {panel_width:.2f} in per panel; use fewer columns or a wider figure"
        )
    if square_panels:
        panel_height = panel_width

    bottom = OUTER_PAD + x_tick_rows * X_TICK_ROW + AXIS_LABEL_BAND
    top_decoration = top_tick_rows * X_TICK_ROW + top_label_rows * AXIS_LABEL_BAND
    top = OUTER_PAD + PANEL_TITLE_BAND + top_decoration
    if legend_rows:
        top += legend_rows * LEGEND_ROW + LEGEND_GAP
    row_gap = FRAME_GAP + PANEL_TITLE_BAND + top_decoration
    if not shares_x_between_rows:
        row_gap += x_tick_rows * X_TICK_ROW
    if not share_xlabel:
        row_gap += AXIS_LABEL_BAND
    height = bottom + nrows * panel_height + (nrows - 1) * row_gap + top

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width, height),
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        layout=None,
    )
    figure.set_layout_engine(None)
    figure.subplots_adjust(
        left=left / width,
        right=1 - right / width,
        bottom=bottom / height,
        top=1 - top / height,
        wspace=column_gap / panel_width,
        hspace=row_gap / panel_height,
    )
    title_pad = 4 + top_decoration * 72
    panel_axes = figure.axes[:nrows * ncols]
    for axis, title in zip(panel_axes, titles):
        axis.set_title(title, pad=title_pad)
        style_axis(axis)
    label_y = (OUTER_PAD + AXIS_LABEL_BAND / 2) / height
    label_x = (OUTER_PAD + AXIS_LABEL_BAND / 2) / width
    if shared_xlabel is not None:
        figure.supxlabel(shared_xlabel, x=0.5, y=label_y, ha="center", va="center")
    elif column_xlabels is not None:
        for column, label in enumerate(column_xlabels):
            position = panel_axes[(nrows - 1) * ncols + column].get_position()
            figure.text((position.x0 + position.x1) / 2, label_y, label, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    if shared_ylabel is not None:
        figure.supylabel(shared_ylabel, x=label_x, y=0.5, ha="center", va="center")
    elif row_ylabels is not None:
        for row, label in enumerate(row_ylabels):
            position = panel_axes[row * ncols].get_position()
            figure.text(label_x, (position.y0 + position.y1) / 2, label, rotation=90, ha="center", va="center", fontsize=mpl.rcParams["axes.labelsize"])
    figure._report_legend_rows = legend_rows
    return figure, axes


def add_top_legend(figure, handles, labels, *, rows: int, **kwargs):
    """Place a figure legend in the rows reserved by ``make_grid``."""
    reserved_rows = getattr(figure, "_report_legend_rows", None)
    if reserved_rows is None:
        raise ValueError("figure was not created by make_grid")
    if rows != reserved_rows or rows < 1:
        raise ValueError(f"legend uses {rows} rows but the grid reserves {reserved_rows}")
    handles = tuple(handles)
    labels = tuple(labels)
    if len(handles) != len(labels):
        raise ValueError("legend handles and labels must have the same length")
    if not labels:
        raise ValueError("legend must contain at least one entry")
    layout_options = {"loc", "bbox_to_anchor", "borderaxespad", "ncol", "ncols", "mode", "fontsize", "prop", "handlelength", "handleheight", "handletextpad", "columnspacing", "labelspacing", "borderpad", "title", "title_fontsize"}
    if layout_options.intersection(kwargs):
        raise ValueError("add_top_legend owns legend placement, typography, and spacing")
    legend_row_count(labels, width=figure.get_figwidth(), max_rows=rows)
    columns = ceil(len(labels) / rows)
    order = [row * columns + column for column in range(columns) for row in range(rows) if row * columns + column < len(labels)]
    handles = tuple(handles[index] for index in order)
    labels = tuple(labels[index] for index in order)
    anchor_y = 1 - OUTER_PAD / figure.get_figheight()
    options = {
        "loc": "upper center",
        "bbox_to_anchor": (0.5, anchor_y),
        "borderaxespad": 0,
        "ncols": columns,
    } | kwargs
    return figure.legend(handles, labels, **options)


def ordered_colors(count: int, *, quantity: str) -> tuple[str, ...]:
    """Use the shared palette for the quantity encoded in color."""
    return tuple(sequence_colors(count, quantity=quantity))


def architecture_label(name: str) -> str:
    """Return a publication name while preserving an existing ablation suffix."""
    prefixes = (
        ("K2 Grow", ARCHITECTURE_LABELS["k2_grow"]),
        ("Dep Grow", ARCHITECTURE_LABELS["dep_grow"]),
        ("K1", ARCHITECTURE_LABELS["k1"]),
        ("K2", ARCHITECTURE_LABELS["k2"]),
        ("Dep", ARCHITECTURE_LABELS["dep"]),
    )
    for old, new in prefixes:
        if name == old or name.startswith(old + " ") or name.startswith(old + "-"):
            return new + name[len(old):]
    return name
