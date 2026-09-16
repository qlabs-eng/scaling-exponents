"""Common typography and square panels for PDFs embedded at the same width."""
from appendix_style import apply_style as apply_report_style, make_grid as make_report_grid


FULL_WIDTH = 6.9
ANNOTATION_SIZE = 7


def apply_style():
    apply_report_style(**{
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "figure.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "lines.markersize": 4,
        "lines.linewidth": 1.1, "lines.markeredgewidth": 0.6,
    })


def make_grid(nrows, ncols, *, width=FULL_WIDTH, **kwargs):
    apply_style()
    # Leave room for wide y-tick labels and x-ticks at the right canvas edge.
    left_pad, right_pad = 0.12, 0.10
    grid_width = width - left_pad - right_pad
    fig, axes = make_report_grid(
        nrows, ncols, width=grid_width, square_panels=True, **kwargs)
    fig.set_size_inches(width, fig.get_figheight())
    for axis in fig.axes:
        position = axis.get_position()
        axis.set_position(((position.x0 * grid_width + left_pad) / width,
                           position.y0, position.width * grid_width / width,
                           position.height))
    for label in fig.texts:
        label.set_x((label.get_position()[0] * grid_width + left_pad) / width)
    return fig, axes
