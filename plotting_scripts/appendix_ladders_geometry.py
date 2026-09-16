"""Physical sizes for report 05: embed PDFs at their documented width ratios."""
import matplotlib.pyplot as plt
from math import ceil

from appendix_style import apply_style as apply_report_style, legend_row_count, style_axis


FULL_WIDTH = 6.9
PANEL_SIZE = 1.65  # Retained for downstream figures that share report 05 typography.
MARKER_SIZE = 4.0
LINE_WIDTH = 1.1


def apply_style():
    apply_report_style(**{
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "figure.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "lines.markersize": MARKER_SIZE,
        "lines.linewidth": LINE_WIDTH, "lines.markeredgewidth": 0.6,
    })


def figure_width(ncols):
    return FULL_WIDTH if ncols >= 3 else FULL_WIDTH / 2


def make_grid(nrows, ncols, *, titles, legend_rows=0, inset=False,
              shared_xlabel=None, column_xlabels=None, sharex=False,
              legend_row_height=0.18):
    """Use native full/half-column canvases with identical physical typography."""
    apply_style()
    width = 0.4 * FULL_WIDTH if inset else figure_width(ncols)
    pitch = width / ncols
    panel_size = pitch - 0.65
    top = 0.22 + (legend_row_height * legend_rows + 0.02 if legend_rows else 0)
    bottom, row_gap = 0.40, 0.27 if sharex else 0.62
    height = bottom + nrows * panel_size + (nrows - 1) * row_gap + top
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False)
    for index, (axis, title) in enumerate(zip(axes.flat, titles, strict=True)):
        row, column = divmod(index, ncols)
        left = 0.59 + column * pitch
        lower = bottom + (nrows - row - 1) * (panel_size + row_gap)
        axis.set_position((left / width, lower / height,
                           panel_size / width, panel_size / height))
        title = {"Reducible Loss": "Loss", "Compute Multiplier": "Multiplier"}.get(title, title)
        axis.set_title(title, pad=4)
        style_axis(axis)
        xlabel = column_xlabels[column] if column_xlabels else shared_xlabel
        if sharex and row < nrows - 1:
            axis.sharex(axes[-1, column])
            axis.tick_params(axis="x", labelbottom=False)
        elif xlabel:
            axis.set_xlabel(xlabel)
    fig._report_legend_rows = legend_rows
    fig._ladder_legend_height = legend_row_height * legend_rows
    return fig, axes[0] if nrows == 1 else axes


def add_column_legends(figure, columns):
    """Place one legend per column in the grid's reserved top band."""
    rows = figure._report_legend_rows
    legends = []
    for column, handles in enumerate(columns):
        labels = [handle.get_label() for handle in handles]
        count = legend_row_count(labels, width=figure.get_figwidth() / len(columns))
        if count > rows:
            raise ValueError("column legends exceed the reserved legend rows")
        ncols = ceil(len(handles) / count)
        order = [row * ncols + col for col in range(ncols) for row in range(count)
                 if row * ncols + col < len(handles)]
        legends.append(figure.legend(
            [handles[index] for index in order], [labels[index] for index in order],
            loc="lower center", ncols=ncols, borderaxespad=0,
            bbox_to_anchor=((column + 0.5) / len(columns),
                            1 - figure._ladder_legend_height / figure.get_figheight())))
    return legends


def add_stacked_legends(figure, groups):
    """Keep independent legend keys in separate compact rows above the panels."""
    row_offset = 0
    legends = []
    for handles, labels in groups:
        rows = legend_row_count(labels, width=figure.get_figwidth())
        ncols = ceil(len(labels) / rows)
        order = [row * ncols + col for col in range(ncols) for row in range(rows)
                 if row * ncols + col < len(labels)]
        legends.append(figure.legend(
            [handles[index] for index in order], [labels[index] for index in order],
            loc="upper center", ncols=ncols, borderaxespad=0,
            bbox_to_anchor=(0.5, 1 - (0.04 + 0.18 * row_offset) / figure.get_figheight())))
        row_offset += rows
    if row_offset != figure._report_legend_rows:
        raise ValueError("independent legends must use the reserved legend rows")
    return legends
