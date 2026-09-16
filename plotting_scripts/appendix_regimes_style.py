"""Physical panel layout for the dense data-constrained appendix figures."""
from __future__ import annotations
from collections.abc import Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import RCPARAMS

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

# Compact type sizes keep the original 12–18-panel canvases legible.
STYLE = {**RCPARAMS, "font.size": 9.5, "axes.titlesize": 10.5,
         "axes.labelsize": 10, "legend.fontsize": 8, "xtick.labelsize": 8.5,
         "ytick.labelsize": 8.5, "legend.frameon": False, "axes.titleweight": "bold",
         "grid.color": "#DDE1E7", "grid.alpha": 0.65, "grid.linewidth": 0.65}

def style_axis(axis, *, grid=True):
    axis.grid(grid, which="major")
    axis.set_axisbelow(True)

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
    return figure, axes
