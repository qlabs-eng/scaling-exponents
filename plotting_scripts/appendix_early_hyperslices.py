"""Figure 14b measured architecture slices, original quadratic minima and GLR exponent fits."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import save_figure
from appendix_style import INK, MUTED, FULL_WIDTH, add_top_legend, apply_style, legend_row_count, make_grid, ordered_colors
from appendix_early_data import load_data

def load_slices():
    """Restore numeric dictionary keys in the exported measured slice grids."""
    records = load_data("hyper_slices")
    for record in records.values():
        for name in ("scaled", "constant"):
            if record[name] is not None:
                record[name] = {knob: {int(depth): {float(x): loss for x, loss in points.items()}
                                     for depth, points in rungs.items()}
                                for knob, rungs in record[name].items()}
        for name in ("parameters", "constant_parameters"):
            if record[name] is not None:
                record[name] = {int(depth): count for depth, count in record[name].items()}
    return records


SLICE_VARIANT_ROWS = (
    ("Van", "K1"),
    ("K2", "Dep"),
    ("K2 Grow", "Dep Grow"),
)


SLICE_ARCHITECTURES = tuple(
    architecture for pair in SLICE_VARIANT_ROWS for architecture in pair
)


KNOBS = ("glr", "om", "rm", "wd")


OPTIMUM_COLOR = INK


def vertex_beta(points: dict[int, dict[float, float]], npar: dict[int, float]):
    """Vertex at each rung, followed by a line against log2 recorded parameter count."""
    vertices = {}
    for depth, rung in sorted(points.items()):
        if len(rung) < 3:
            continue
        xs = np.array(sorted(rung))
        ys = np.array([rung[x] for x in xs])
        p2, p1, _ = np.polyfit(xs, np.log(ys), 2)
        if p2 > 0:
            vertices[depth] = -p1 / (2 * p2)
    if len(vertices) < 2:
        return None
    depths = list(vertices)
    scale = np.array([np.log2(npar[d] / npar[8]) for d in depths])
    values = np.array([vertices[d] for d in depths])
    if len(vertices) == 2:
        beta = (values[1] - values[0]) / (scale[1] - scale[0])
        return float(beta), np.nan
    (beta, _), cov = np.polyfit(scale, values, 1, cov=True)
    return float(beta), float(np.sqrt(cov[0, 0]))


KNOB_LABEL = {"glr": "GLR", "hlrm": "HLRM", "om": "OM", "rm": "RM",
              "wd": "WD", "emb_alpha": "EMB"}


def _quadratic_slice(rung: dict[float, float]):
    """Fit log loss quadratically in log2 multiplier and return its vertex."""
    if len(rung) < 3:
        return None
    xs = np.array(sorted(rung))
    ys = np.array([rung[x] for x in xs])
    coefficients = np.polyfit(xs, np.log(ys), 2)
    vertex = None
    if coefficients[0] > 0:
        vertex_x = -coefficients[1] / (2 * coefficients[0])
        vertex_y = float(np.exp(np.polyval(coefficients, vertex_x)))
        vertex = (float(vertex_x), vertex_y)
    return xs, ys, coefficients, vertex


DISPLAY_ARCH = {
    "Van": "Vanilla",
    "Van Grow": "Van Grow",
    "K1": "Operator-1",
    "K2": "Loop-2",
    "K2 Grow": "Loop Grow",
    "Dep": "Untied-2",
    "Dep d16": "Untied-2",
    "Dep Grow": "Untied Grow",
    "Dep TPP6": "Untied-2 TPP6",
}


def slice_figure() -> None:
    """Plot paired quadratic d8/d9/d10 slices for the six selected variants."""
    apply_style()
    rungs = (8, 9, 10)
    rung_colors = dict(zip(rungs, ordered_colors(len(rungs), quantity="depth")))
    labels = tuple(
        [f"d{depth}" for depth in rungs]
        + ["Scaled recipe", "Constant", "Optimum"]
    )
    handles = [
        plt.Line2D([], [], color=rung_colors[depth], marker="o", ls="none", label=label)
        for depth, label in zip(rungs, labels[:len(rungs)])
    ]
    handles += [
        plt.Line2D([], [], color=MUTED, ls="-", label=labels[-3]),
        plt.Line2D([], [], color=MUTED, ls="--", label=labels[-2]),
        plt.Line2D([], [], color=OPTIMUM_COLOR, marker="*", ls="none",
                   label=labels[-1]),
    ]
    legend_rows = legend_row_count(labels)
    titles = tuple(
        (f"{DISPLAY_ARCH.get(architecture, architecture)} · {KNOB_LABEL[knob]}"
         if column == 0 else KNOB_LABEL[knob])
        for pair in SLICE_VARIANT_ROWS
        for architecture in pair
        for column, knob in enumerate(KNOBS)
    )
    fig, axes = make_grid(
        len(SLICE_VARIANT_ROWS),
        2 * len(KNOBS),
        titles=titles,
        width=2 * FULL_WIDTH,
        sharex=True,
        sharey="row",
        shared_xlabel="Multiplier",
        row_ylabels=("Loss",) * len(SLICE_VARIANT_ROWS),
        legend_rows=legend_rows,
        squeeze=False,
    )
    for architecture_index, architecture in enumerate(SLICE_ARCHITECTURES):
        i, variant_column = divmod(architecture_index, 2)
        column_offset = variant_column * len(KNOBS)
        record = load_slices()[architecture]
        points, npar = record["scaled"], record["parameters"]
        constant_points = record["constant"]
        if constant_points is None:
            glr_beta = record["adopted_beta"]
        else:
            fitted = vertex_beta(constant_points["glr"], record["constant_parameters"])
            if fitted is None:
                raise ValueError(f"fewer than two convex GLR slices for {architecture}")
            glr_beta = -fitted[0]
        plotted_in_row = False
        plotted_by_knob = {knob: False for knob in KNOBS}
        for j, knob in enumerate(KNOBS):
            series = ((constant_points, "--", True), (points, "-", False))
            for recipe_points, linestyle, hollow in series:
                if recipe_points is None:
                    continue
                for depth in rungs:
                    # The scaled recipes share their d8 constant anchor exactly.
                    if hollow and depth == 8:
                        continue
                    fitted = _quadratic_slice(recipe_points[knob].get(depth, {}))
                    if fitted is None:
                        continue
                    xs_log, ys, coefficients, vertex = fitted
                    color = rung_colors[depth]
                    curve_x = np.linspace(xs_log.min(), xs_log.max(), 160)
                    axis = axes[i][column_offset + j]
                    axis.plot(
                        2 ** curve_x,
                        np.exp(np.polyval(coefficients, curve_x)),
                        color=color,
                        ls=linestyle,
                        alpha=0.82,
                        zorder=2,
                    )
                    axis.plot(
                        2 ** xs_log,
                        ys,
                        color=color,
                        marker="o",
                        mfc="white" if hollow else color,
                        ls="none",
                        zorder=3,
                    )
                    if vertex is not None and not hollow:
                        axis.plot(
                            2 ** vertex[0],
                            vertex[1],
                            color=OPTIMUM_COLOR,
                            marker="*",
                            mfc=OPTIMUM_COLOR,
                            mec=OPTIMUM_COLOR,
                            ms=7,
                            ls="none",
                            zorder=4,
                        )
                    plotted_in_row = True
                    plotted_by_knob[knob] = True
        for j, knob in enumerate(KNOBS):
            axis = axes[i][column_offset + j]
            axis.axvline(1.0, color=MUTED, lw=0.9, ls=":", alpha=0.8, zorder=1)
            axis.set_xscale("log", base=2)
            # Leave a small margin around the measured quarter-to-four-times grid.
            axis.set_xlim(2 ** -2.12, 2 ** 2.12)
            axis.set_xticks(
                (0.25, 0.5, 1, 2, 4),
                labels=("0.25×", "0.5×", "1×", "2×", "4×"),
            )
            # All columns share the multiplier scale. Alternating the two visible
            # label rows keeps the five manual ticks legible at full-width size.
            axis.tick_params(labelbottom=(i == len(SLICE_VARIANT_ROWS) - 1
                                          and j % 2 == 0))
            if not plotted_by_knob[knob]:
                axis.text(0.5, 0.5, "Not run", transform=axis.transAxes,
                          ha="center", va="center", color=MUTED)
        axes[i][column_offset].text(
            0.97,
            0.91,
            rf"$\beta={glr_beta:.2f}$",
            transform=axes[i][column_offset].transAxes,
            ha="right",
            va="top",
            color=INK,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                  "pad": 1.0},
            zorder=5,
        )
        if not plotted_in_row:
            raise FileNotFoundError(f"no slices found for {architecture}")
    add_top_legend(fig, handles, labels, rows=legend_rows)
    save_figure(fig, 14, panel="b")
