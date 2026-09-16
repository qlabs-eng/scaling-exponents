"""Figure 14a one-dimensional hyperparameter measurements and bowl fits."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import save_figure
from appendix_style import MUTED, OUTER_PAD, AXIS_LABEL_BAND, Y_TICK_COLUMN, add_top_legend, legend_row_count, make_grid, ordered_colors
from appendix_early_data import load_data


def slices(rows):
    """{(rung, knob): {value: loss}}, {rung: centre loss}, {(rung, knob): centre value}.

    A rung's centre cell is a point on every slice that rung swept -- and only those: the
    odd rungs carry the knobs whose optimum moves with scale, so which knobs they swept is
    read off the data rather than assumed.
    """
    cells, centres, centre_vals = {}, {}, {}
    swept = {}
    for r in rows:
        if r["knob"]:
            swept.setdefault(int(r["depth"]), set()).add(r["knob"])
    for r in rows:
        rung, loss = int(r["depth"]), float(r["val_loss"])
        if r["knob"]:
            cells.setdefault((rung, r["knob"]), {})[float(r[r["knob"]])] = loss
        else:
            centres[rung] = loss
            for knob in swept.get(rung, ()):
                cells.setdefault((rung, knob), {})[float(r[knob])] = loss
                centre_vals[(rung, knob)] = float(r[knob])
    return cells, centres, centre_vals


ORDER = ("glr", "elrm", "hlrm", "om", "rm", "wd", "wte", "uis",
         "warmup", "wdr", "beta2", "eps")


CATEGORICAL = ("warmup", "wdr", "beta2", "eps")


XLABEL = {
    **{knob: "Multiple of center" for knob in ORDER if knob not in CATEGORICAL},
    "warmup": "Warmup steps",
    "wdr": "WDR",
    "beta2": r"$\beta_2$",
    "eps": r"$\epsilon$",
}


TITLE = {
    "glr": "GLR",
    "elrm": "ELRM",
    "hlrm": "HLRM",
    "om": "OM",
    "rm": "RM",
    "wd": "WD",
    "wte": "WTE init std",
    "uis": "UIS",
    "warmup": "Warmup",
    "wdr": "WDR",
    "beta2": r"Adam $\beta_2$",
    "eps": r"Adam $\epsilon$",
}


MARKERS = {8: "o", 9: "s", 10: "^", 11: "D", 12: "v"}


MULTIPLIER_LABELS = {
    0.25: "¼",
    0.5: "½",
}


def plot_figure(cells, centres, centre_vals, rungs, lo, hi, pad):
    colors = dict(zip(rungs, ordered_colors(len(rungs), quantity="depth")))
    labels = [f"d{rung}" for rung in rungs] + ["Fixed recipe"]
    # Keep the 2x6 dashboard while giving horizontal tick labels room between columns.
    figure_width = 10.0
    column_gap = 0.20
    panel_width = (
        figure_width
        - (OUTER_PAD + Y_TICK_COLUMN + AXIS_LABEL_BAND)
        - OUTER_PAD
        - 5 * column_gap
    ) / 6
    legend_rows = legend_row_count(labels, width=figure_width)
    figure, axes = make_grid(
        2,
        6,
        titles=[TITLE[knob] for knob in ORDER],
        width=figure_width,
        panel_height=panel_width,
        sharey=True,
        shared_ylabel="Loss",
        legend_rows=legend_rows,
        x_tick_rows=1,
        squeeze=False,
    )
    figure.subplots_adjust(wspace=column_gap / panel_width)

    for axis, knob in zip(axes.flat, ORDER):
        categorical = knob in CATEGORICAL
        ticks = sorted({
            value if categorical else value / centre_vals[(rung, knob)]
            for rung in rungs
            for value in cells.get((rung, knob), {})
        })
        positions = {value: index for index, value in enumerate(ticks)}
        for rung in rungs:
            series = cells.get((rung, knob))
            if not series:
                continue
            x_values = sorted(series)
            y_values = np.array([min(series[value], hi + pad) for value in x_values])
            clipped = np.array([series[value] > hi + pad for value in x_values])
            plot_x = (
                [positions[value] for value in x_values]
                if categorical
                else [value / centre_vals[(rung, knob)] for value in x_values]
            )
            color = colors[rung]
            axis.plot(
                plot_x,
                y_values,
                marker=MARKERS[rung],
                color=color,
                zorder=3,
            )
            if clipped.any():
                axis.plot(
                    np.array(plot_x)[clipped],
                    y_values[clipped],
                    marker="^",
                    markersize=7,
                    linestyle="none",
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.3,
                    zorder=4,
                )
            axis.axhline(
                centres[rung],
                color=color,
                linewidth=0.9,
                linestyle="--",
                alpha=0.55,
                zorder=1,
            )
        if categorical:
            axis.set_xticks(range(len(ticks)), labels=[f"{value:g}" for value in ticks])
            axis.set_xlim(-0.35, len(ticks) - 0.65)
        else:
            axis.set_xscale("log", base=2)
            axis.xaxis.set_major_locator(FixedLocator(ticks))
            axis.xaxis.set_major_formatter(
                FuncFormatter(lambda value, _: MULTIPLIER_LABELS.get(value, f"{value:g}"))
            )
            axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_xlabel(XLABEL[knob])
        axis.tick_params(axis="both", colors="#111111")
        axis.set_yscale("log")
        axis.set_ylim(lo - pad, hi + pad)
        for set_formatter in (axis.yaxis.set_major_formatter, axis.yaxis.set_minor_formatter):
            set_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))

    handles = [
        Line2D([], [], color=colors[rung], marker=MARKERS[rung], label=f"d{rung}")
        for rung in rungs
    ]
    handles.append(Line2D([], [], color=MUTED, linewidth=0.9, linestyle="--", label="Fixed recipe"))
    add_top_legend(figure, handles, labels, rows=legend_rows)
    save_figure(figure, 14, panel="a")


def main():
    cells, centres, centre_vals = slices(load_data("hyper1d"))
    rungs = sorted({r for r, _ in cells})

    # A knob whose value has run away takes the loss with it. Clip the axis to the band
    # the sweep is actually resolving and mark what was clipped, so one diverged cell
    # cannot flatten every bowl in the figure.
    every = [l for s in cells.values() for l in s.values()]
    lo = min(every)
    hi = min(max(every), max(centres.values()) + 0.25)
    pad = 0.04 * (hi - lo)

    plot_figure(cells, centres, centre_vals, rungs, lo, hi, pad)
    print(f"plotted {sum(len(series) for series in cells.values())} points across d{rungs[0]}–d{rungs[-1]}")
