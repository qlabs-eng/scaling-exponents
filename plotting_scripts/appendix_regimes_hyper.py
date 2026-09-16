"""Figure 23: measured weight-decay and hyperparameter sensitivities."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from common import save_figure, MUTED
from appendix_regimes_style import style_axis
import appendix_regimes_grid as paper
MULTIPLIERS = (0.25,0.5,1.0,2.0,4.0)
KNOBS = ("glr","wd","om","rm","emb_alpha")
KNOB_LABELS = {"glr":"GLR", "wd":"WD", "om":"OM", "rm":"RM", "emb_alpha":"EMB_ALPHA"}
FAMILIES = ("k","u")
LOOPS = (2,4,8)
MARKERS = {2:"o",4:"s",8:"D"}
COLORS = {k:paper.LOOP_COLORS[k] for k in LOOPS}

def plot_slices(axes, slices: dict[str, dict[str, dict[int, list[float]]]], *, colors=COLORS) -> None:
    """Draw the sensitivity panels on an existing two-by-five axes array."""
    for row, family in enumerate(FAMILIES):
        for column, knob in enumerate(KNOBS):
            axis = axes[row][column]
            for loops in LOOPS:
                axis.plot(
                    MULTIPLIERS,
                    slices[family][knob][loops],
                    color=colors[loops],
                    marker=MARKERS[loops],
                    linewidth=1.6,
                    markersize=5.0,
                    zorder=3,
                )
            axis.axvline(1.0, color=MUTED, linewidth=0.8, linestyle=":", alpha=0.85, zorder=1)
            axis.set_xscale("log", base=2)
            axis.set_xticks(MULTIPLIERS, labels=("0.25x", "0.5x", "1x", "2x", "4x"))
            axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            axis.margins(y=0.08)
def plot(loop_data, slices, output: int) -> None:
    width, height = paper.GRID_WIDTH, 9.95
    figure = plt.figure(figsize=(width, height))
    rows = []
    for row, (columns, bottom) in enumerate(((6, 7.3), (5, 3.65), (5, 0.65))):
        grid = figure.add_gridspec(
            1, columns, left=0.65 / width, right=1 - 0.1 / width,
            bottom=bottom / height, top=(bottom + paper.SQUARE_PANEL_SIZE) / height,
            wspace=0.20,
        )
        axes = []
        for column in range(columns):
            axis = figure.add_subplot(grid[0, column], sharey=axes[0] if row and column else None)
            style_axis(axis)
            if row and column:
                axis.tick_params(labelleft=False)
            axes.append(axis)
        rows.append(axes)

    paper.plot_wd_row(rows[0], loop_data, mark_optima=False)
    for axis, depth in zip(rows[0], paper.WD_FIT_DEPTHS):
        axis.set_title(f"Depth {depth}")
    rows[0][0].set_ylabel("Loss")

    plot_slices(rows[1:], slices, colors=paper.LOOP_COLORS)
    for axes, family in zip(rows[1:], FAMILIES):
        position = axes[0].get_position()
        axes[0].set_ylabel("Loss")
        figure.text(position.x0, position.y1 + 0.34 / height, "Tied" if family == "k" else "Untied", weight="bold", fontsize=10)
        for axis, knob in zip(axes, KNOBS):
            axis.set_title(KNOB_LABELS[knob])
            axis.set_xlabel("Multiple of center")

    figure.text(0.65 / width, 9.7 / height, "(a) Weight-decay sensitivity · 100M tokens × 10 epochs", weight="bold", fontsize=12)
    figure.text(0.65 / width, 6.48 / height, "(b) Hyperparameter sensitivity · depth 8", weight="bold", fontsize=12)
    figure.text(0.65 / width, 6.18 / height, "Fixed tokens: tied 1.235B · untied 1.466B", fontsize=10)
    rows[0][0].legend(
        [Line2D([], [], color=paper.LOOP_COLORS[loops]) for loops in paper.LOOPS],
        [f"K={loops}" for loops in paper.LOOPS],
        loc="upper right", ncols=2, borderaxespad=0.6,
    )
    rows[1][0].legend(
        [Line2D([], [], color=paper.LOOP_COLORS[loops], marker=MARKERS[loops]) for loops in LOOPS],
        [f"K={loops}" for loops in LOOPS],
        loc="upper center", borderaxespad=0.6,
    )
    save_figure(figure, output, bbox_inches=None)
