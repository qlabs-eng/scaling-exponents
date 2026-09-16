"""Figure 21: data regimes and K1-selected weight decay."""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, NullFormatter, ScalarFormatter
from common import sequence_colors, save_figure, MUTED
from appendix_regimes_style import make_grid, LEGEND_ROW, OUTER_PAD
from appendix_regimes_data import Point, load_joint_grid

REGIMES = ("fresh", "replay250m4", "replay100m10")
LOOPS = (1, 2, 3, 4, 6)
DEPTHS = {1: (6,8,10,12,14,16,18), 2: (6,8,10,12,14,16), 3: (6,8,10,12,14,16), 4: (6,8,10,12,14), 6: (4,6,8,10,12,14)}
JOINT_DEPTHS = {**DEPTHS, 8: (8,10,12,14), 12: (8,10,12)}
JOINT_LOOPS = tuple(JOINT_DEPTHS)
BASE_BUDGETS = tuple(value * 1e18 for value in (0.846120,1.419447,2.381258,3.994787,6.701636))
BUDGET_VALUES = sorted(BASE_BUDGETS + tuple(math.sqrt(a*b) for a,b in zip(BASE_BUDGETS, BASE_BUDGETS[1:])))
BUDGETS = tuple((f"C={value:.2e}", value, color) for value,color in zip(BUDGET_VALUES, sequence_colors(len(BUDGET_VALUES), quantity="compute")))
LOOP_COLORS = dict(zip(JOINT_LOOPS, sequence_colors(len(JOINT_LOOPS), quantity="recurrence")))
OVERFIT_DEPTHS = tuple(range(6,19,2))
OVERFIT_DEPTH_COLORS = dict(zip(OVERFIT_DEPTHS, sequence_colors(len(OVERFIT_DEPTHS), quantity="depth")))
DELTA_DEPTHS = (6,8,10,12,14)
DEPTH_MARKER = {4: "X",6: "o",8: "s",10: "^",12: "D",14: "P",16: "v",18: "<"}
WIDTH = 10.35
PANEL_HEIGHT = 2.05

def select_k1_wd_series(grid):
    """At each depth select K1's best WD, then use it for every recurrence."""
    candidates = {
        depth: [(point, wd) for wd, by_loop in grid.items() for point in by_loop[1] if point.depth == depth]
        for depth in DEPTHS[1]
    }
    selected = {}
    series = {}
    for depth, choices in candidates.items():
        baseline, wd = min(choices, key=lambda pair: (pair[0].loss, pair[1]))
        selected[depth] = wd
        series[depth] = [baseline] + [
            point for loops, points in grid.get(wd, {}).items() if loops > 1
            for point in points if point.depth == depth
        ]
        series[depth].sort(key=lambda point: point.compute)
    return selected, series

def interpolate_loss(points: list[Point], compute: float) -> float:
    for point in points:
        if math.isclose(point.compute, compute, rel_tol=1e-12):
            return point.loss
    for lower, upper in zip(points, points[1:]):
        if lower.compute <= compute <= upper.compute:
            weight = (compute - lower.compute) / (upper.compute - lower.compute)
            return lower.loss + weight * (upper.loss - lower.loss)
    raise ValueError(f"compute {compute:g} is outside K{points[0].loops} ladder range")

def iso_losses(series: dict[int, list[Point]], compute: float, *, require_original: bool = True) -> dict[int, float]:
    losses = {
        loops: interpolate_loss(points, compute)
        for loops, points in series.items()
        if points and points[0].compute <= compute <= points[-1].compute
    }
    if require_original and not set(LOOPS) <= set(losses):
        raise ValueError(f"budget {compute:g} not bracketed by every original loop count")
    return losses

def fit_loop_optimum(losses: dict[int, float]) -> tuple[np.ndarray, float | None, float | None]:
    loops = np.asarray(sorted(losses), dtype=float)
    values = np.asarray([losses[int(loop)] for loop in loops])
    coefficients = np.polyfit(np.log(loops), np.log(values), 2)
    curvature, slope = coefficients[:2]
    if curvature <= 0:
        return coefficients, None, None
    optimum = math.exp(-slope / (2 * curvature))
    if not loops.min() < optimum < loops.max():
        return coefficients, None, None
    loss = math.exp(float(np.polyval(coefficients, math.log(optimum))))
    return coefficients, optimum, loss

def setup_compute_axis(axis: plt.Axes) -> None:
    for _, budget, color in BUDGETS:
        axis.axvline(budget, color=color, linewidth=1.4, linestyle=(0, (2, 3)), alpha=0.72, zorder=1)
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.38e18, 13e18)
    axis.xaxis.set_major_locator(FixedLocator(tuple(value * 1e18 for value in (0.5, 1, 2, 4, 8))))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axis.xaxis.set_major_formatter(formatter)
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("Compute (FLOPs)", loc="left")
    axis.margins(y=0.08)

def setup_loop_axis(axis: plt.Axes, loops=LOOPS) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.86, max(loops) * 1.12)
    axis.xaxis.set_major_locator(FixedLocator(loops))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("Recurrence")
    axis.margins(y=0.08)

def row_major(handles: list[Line2D], labels: list[str], ncols: int) -> tuple[list[Line2D], list[str]]:
    rows = math.ceil(len(labels) / ncols)
    order = [row * ncols + column for column in range(ncols) for row in range(rows) if row * ncols + column < len(labels)]
    return [handles[index] for index in order], [labels[index] for index in order]

def add_column_legends(figure: plt.Figure, axis: plt.Axes, groups: list[tuple[list[Line2D], list[str], int]]) -> None:
    """Stack legends in the top band, centered over a panel column."""
    axes = [axis] if isinstance(axis, plt.Axes) else axis
    positions = [item.get_position() for item in axes]
    center = (min(position.x0 for position in positions) + max(position.x1 for position in positions)) / 2
    top = 1 - OUTER_PAD / figure.get_figheight()
    for handles, labels, ncols in groups:
        ordered_handles, ordered_labels = row_major(handles, labels, ncols)
        figure.legend(handles=ordered_handles, labels=ordered_labels, loc="upper center", bbox_to_anchor=(center, top), ncols=ncols, borderaxespad=0, columnspacing=0.8, handlelength=1.3, handletextpad=0.4)
        top -= math.ceil(len(labels) / ncols) * LEGEND_ROW / figure.get_figheight()

def add_row_legend(figure: plt.Figure, axes: list[plt.Axes] | np.ndarray,
                   handles: list[Line2D], labels: list[str]) -> None:
    positions = [axis.get_position() for axis in axes]
    center = (min(position.x0 for position in positions) + max(position.x1 for position in positions)) / 2
    bottom = max(position.y1 for position in positions) + 0.035 / figure.get_figheight()
    figure.legend(handles=handles, labels=labels, loc="lower center", bbox_to_anchor=(center, bottom), ncols=len(labels), borderaxespad=0, columnspacing=0.8, handlelength=1.3, handletextpad=0.4)

def plot_regime_delta(axis: plt.Axes, regime_series: dict[int, list[Point]]) -> None:
    by_cell = {(point.depth, loops): point.loss for loops, points in regime_series.items() for point in points}
    axis.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    for depth in DELTA_DEPTHS:
        baseline = by_cell[depth, 1]
        available = [loops for loops in sorted(regime_series) if (depth, loops) in by_cell]
        values = [by_cell[depth, loops] - baseline for loops in available]
        axis.plot(available, values, color=OVERFIT_DEPTH_COLORS[depth], marker=DEPTH_MARKER[depth], linewidth=1.6, markersize=4.8, zorder=3)
    setup_loop_axis(axis, tuple(sorted(regime_series)))

def plot_regime_scaling(compute_axis, regime_series):
    for loop_count in sorted(regime_series):
        points = regime_series[loop_count]
        compute_axis.plot(
            [point.compute for point in points],
            [point.loss for point in points],
            color=LOOP_COLORS[loop_count],
            linewidth=1.6,
            zorder=2,
        )
        for point in points:
            compute_axis.plot(point.compute, point.loss, color=LOOP_COLORS[loop_count], marker=DEPTH_MARKER[point.depth], linestyle="none", markersize=5.0, zorder=3)
    setup_compute_axis(compute_axis)

def plot_regime_optima(optimum_axis, regime_series, *, require_original=True):
    summaries = []
    for _, budget, color in BUDGETS:
        losses = iso_losses(regime_series, budget, require_original=require_original)
        measured_loops = np.asarray(sorted(losses), dtype=float)
        measured_losses = np.asarray([losses[int(loop)] for loop in measured_loops])
        coefficients, optimum, optimum_loss = fit_loop_optimum(losses)
        fit_loops = np.geomspace(measured_loops.min(), measured_loops.max(), 240)
        optimum_axis.plot(fit_loops, np.exp(np.polyval(coefficients, np.log(fit_loops))), color=color, linewidth=1.8, zorder=2)
        optimum_axis.plot(measured_loops, measured_losses, color=color, marker="o", markersize=5.2, linestyle="none", zorder=3)
        if optimum is None:
            best_index = int(np.argmin(measured_losses))
            optimum_axis.plot(measured_loops[best_index], measured_losses[best_index], marker="*", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0, markersize=10, linestyle="none", zorder=4)
            summaries.append((None, float(measured_losses[best_index])))
        else:
            optimum_axis.plot(optimum, optimum_loss, color=color, marker="*", markeredgecolor="white", markeredgewidth=0.6, markersize=10, linestyle="none", zorder=4)
            summaries.append((optimum, float(optimum_loss)))
    setup_loop_axis(optimum_axis, tuple(sorted(regime_series)))
    return summaries

def plot_regimes(series: dict[str, dict[int, list[Point]]]) -> dict[str, list[tuple[float | None, float]]]:
    _, selected = select_k1_wd_series(load_joint_grid())
    k1_wd_series = {
        loops: sorted((point for points in selected.values() for point in points if point.loops == loops), key=lambda point: point.compute)
        for loops in JOINT_LOOPS
    }
    regimes = (*REGIMES, "replay100m10_k1_wd")
    series = {**series, regimes[-1]: k1_wd_series}
    figure, axes = make_grid(
        3,
        4,
        titles=(
            "Fresh data", "250M × 4 epochs", "100M × 10 epochs", "100M × 10, K1-opt. WD",
            "Fresh-data optima", "250M × 4 optima", "100M × 10 optima", "K1-opt. WD optima",
            "1B difference to K1", "250M × 4 difference to K1", "100M × 10 difference to K1", "K1-opt. WD difference to K1",
        ),
        width=WIDTH * 4 / 3,
        panel_height=PANEL_HEIGHT,
        legend_rows=4,
        top_label_rows=1,
        squeeze=False,
    )
    fit_summaries = {}
    for column, regime in enumerate(regimes):
        regime_series = series[regime]
        compute_axis, optimum_axis = axes[0, column], axes[1, column]
        plot_regime_scaling(compute_axis, regime_series)
        fit_summaries[regime] = plot_regime_optima(optimum_axis, regime_series, require_original=regime in REGIMES)

    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel("Loss (Interpolated)")
    for axis in axes[2, 1:]:
        axis.sharey(axes[2, 0])
        axis.tick_params(axis="y", labelleft=False)
    for axis, regime in zip(axes[2], regimes):
        plot_regime_delta(axis, series[regime])
    axes[2, 0].set_ylabel(r"$L(K)-L(K=1)$")

    available = sorted({loops for regime_series in series.values() for loops in regime_series})
    loop_handles = [Line2D([], [], color=LOOP_COLORS[value]) for value in available]
    loop_labels = [f"K={value}" for value in available]
    slice_handle = Line2D([], [], color=BUDGETS[len(BUDGETS) // 2][2], linestyle=(0, (2, 3)), linewidth=1.8)
    depth_handles = [Line2D([], [], color=MUTED, marker=DEPTH_MARKER[depth], linestyle="none") for depth in DEPTH_MARKER]
    depth_labels = [f"d{depth}" for depth in DEPTH_MARKER]
    add_column_legends(figure, axes[0], [
        (loop_handles + [slice_handle], loop_labels + ["compute slices"], 4),
        (depth_handles, depth_labels, 4),
    ])
    delta_handles = [Line2D([], [], color=OVERFIT_DEPTH_COLORS[depth], marker=DEPTH_MARKER[depth]) for depth in DELTA_DEPTHS]
    delta_labels = [f"d{depth}" for depth in DELTA_DEPTHS]
    add_row_legend(figure, axes[2], delta_handles, delta_labels)
    save_figure(figure, 21, bbox_inches=None)
    return fit_summaries
