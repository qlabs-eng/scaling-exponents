"""Figures 22–24: full WD grid and architecture comparisons."""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, NullFormatter, ScalarFormatter
from common import sequence_colors, save_figure, INK, MUTED
from appendix_regimes_style import make_grid, style_axis, LEGEND_ROW, OUTER_PAD
from appendix_regimes_data import Point
import appendix_regimes as regime_figure
import appendix_regimes_architectures as vanilla_figure

LOOPS = regime_figure.JOINT_LOOPS
WEIGHT_DECAYS = (0.05,0.2,0.4,0.8,1.2,1.6)
BUDGET_VALUES = (("B1",0.846120),("B2",1.419447),("B3",2.381258),("B4",3.994787),("B5",6.701636))
BUDGETS = tuple((*budget,color) for budget,color in zip(BUDGET_VALUES,sequence_colors(len(BUDGET_VALUES), quantity="compute")))
DEPTHS = regime_figure.JOINT_DEPTHS
ALL_DEPTHS = tuple(sorted({d for depths in DEPTHS.values() for d in depths}))
DEPTH_MARKERS = regime_figure.DEPTH_MARKER
LOOP_COLORS = regime_figure.LOOP_COLORS
WD_FIT_DEPTHS = (6,8,10,12,14,16)
DELTA_DEPTHS = (6,8,10,12,14)
DELTA_DEPTH_COLORS = dict(zip(DELTA_DEPTHS,sequence_colors(len(DELTA_DEPTHS), quantity="depth")))
SQUARE_PANEL_SIZE = 2.0
GRID_WIDTH = 14.65

def interpolate_loss(points: list[dict], budget: float) -> float:
    for point in points:
        if math.isclose(point["compute"], budget, rel_tol=1e-12):
            return point["loss"]
    for lower, upper in zip(points, points[1:]):
        if lower["compute"] <= budget <= upper["compute"]:
            fraction = ((budget - lower["compute"])
                        / (upper["compute"] - lower["compute"]))
            return lower["loss"] + fraction * (upper["loss"] - lower["loss"])
    raise ValueError(
        f"budget {budget:g} outside [{points[0]['compute']:g}, {points[-1]['compute']:g}]"
    )

def iso_losses(series: dict[int, list[dict]], budget: float) -> dict[int, float]:
    return {loops: interpolate_loss(points, budget) for loops, points in series.items()
            if points and points[0]["compute"] <= budget <= points[-1]["compute"]}

def setup_compute_axis(axis: plt.Axes) -> None:
    for _, budget, color in BUDGETS:
        axis.axvline(budget, color=color, linewidth=1.4,
                     linestyle=(0, (2, 3)), alpha=0.72, zorder=1)
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.38, 13.0)
    axis.xaxis.set_major_locator(FixedLocator((0.5, 1, 2, 4, 8)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Compute (EFLOP)", loc="left")
    axis.margins(y=0.08)
    style_axis(axis)

def compute_axis_in_flops(axis):
    """Convert the study's compute units to FLOPs for display."""
    limits = axis.get_xlim()
    for line in axis.lines:
        line.set_xdata(np.asarray(line.get_xdata()) * 1e18)
    axis.set_xlim(*(value * 1e18 for value in limits))
    axis.xaxis.set_major_locator(FixedLocator(tuple(value * 1e18 for value in (0.5, 1, 2, 4, 8))))
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 0))
    axis.xaxis.set_major_formatter(formatter)
    axis.set_xlabel("Compute (FLOPs)", loc="left")

def setup_k_axis(axis: plt.Axes) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.86, 13.4)
    axis.xaxis.set_major_locator(FixedLocator(LOOPS))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Recurrence")
    axis.margins(y=0.08)
    style_axis(axis)

def save(fig: plt.Figure, output: int) -> None:
    save_figure(fig, output, bbox_inches=None)

def row_major(handles: list[Line2D], labels: list[str], ncols: int) -> tuple[list[Line2D], list[str]]:
    rows = math.ceil(len(labels) / ncols)
    order = [row * ncols + column for column in range(ncols) for row in range(rows) if row * ncols + column < len(labels)]
    return [handles[index] for index in order], [labels[index] for index in order]

def add_span_legends(figure: plt.Figure, axes: list[plt.Axes] | np.ndarray, groups: list[tuple[list[Line2D], list[str], int]]) -> None:
    positions = [axis.get_position() for axis in axes]
    center = (min(position.x0 for position in positions) + max(position.x1 for position in positions)) / 2
    top = 1 - OUTER_PAD / figure.get_figheight()
    for handles, labels, ncols in groups:
        ordered_handles, ordered_labels = row_major(handles, labels, ncols)
        figure.legend(handles=ordered_handles, labels=ordered_labels, loc="upper center", bbox_to_anchor=(center, top), ncols=ncols, borderaxespad=0, columnspacing=0.8, handlelength=1.3, handletextpad=0.4)
        top -= math.ceil(len(labels) / ncols) * LEGEND_ROW / figure.get_figheight()

def add_row_legend(figure: plt.Figure, axes: list[plt.Axes] | np.ndarray,
                   handles: list[Line2D], labels: list[str], ncols: int) -> None:
    positions = [axis.get_position() for axis in axes]
    center = (min(position.x0 for position in positions) + max(position.x1 for position in positions)) / 2
    bottom = max(position.y1 for position in positions) + 0.035 / figure.get_figheight()
    ordered_handles, ordered_labels = row_major(handles, labels, ncols)
    figure.legend(handles=ordered_handles, labels=ordered_labels, loc="lower center", bbox_to_anchor=(center, bottom), ncols=ncols, borderaxespad=0, columnspacing=0.8, handlelength=1.3, handletextpad=0.4)

def plot_complete_grid(loop_data: dict[float, dict[int, list[dict]]], output: int) -> None:
    titles = (
        tuple(f"WD = {weight_decay:g}" for weight_decay in WEIGHT_DECAYS)
        + tuple(f"WD = {weight_decay:g} optima" for weight_decay in WEIGHT_DECAYS)
        + tuple(f"WD = {weight_decay:g}: Δ to K1" for weight_decay in WEIGHT_DECAYS)
    )
    figure, axes = make_grid(
        3,
        len(WEIGHT_DECAYS),
        titles=titles,
        width=GRID_WIDTH,
        panel_height=SQUARE_PANEL_SIZE,
        sharey=False,
        row_ylabels=("Loss", "Loss (Interpolated)", r"$L(K)-L(K=1)$"),
        legend_rows=2,
        top_label_rows=1,
        squeeze=False,
    )
    for index, weight_decay in enumerate(WEIGHT_DECAYS):
        compute_axis = axes[0, index]
        iso_axis = axes[1, index]
        delta_axis = axes[2, index]
        series = loop_data[weight_decay]

        for loops in LOOPS:
            points = series[loops]
            compute_axis.plot(
                [point["compute"] for point in points],
                [point["loss"] for point in points],
                color=LOOP_COLORS[loops],
                linewidth=1.6,
                zorder=2,
            )
            for point in points:
                compute_axis.plot(point["compute"], point["loss"], color=LOOP_COLORS[loops], marker=DEPTH_MARKERS[int(point["depth"])], linestyle="none", markersize=5.0, zorder=3)
        setup_compute_axis(compute_axis)
        compute_axis_in_flops(compute_axis)

        for _, budget, color in BUDGETS:
            losses = iso_losses(series, budget)
            available = sorted(losses)
            iso_axis.plot(available, [losses[loops] for loops in available], color=color, marker="o", markersize=5.0, linewidth=1.6, zorder=3)
        setup_k_axis(iso_axis)

        delta_axis.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
        for depth in DELTA_DEPTHS:
            baseline = cell_loss(loop_data, weight_decay, 1, depth)
            available = [loops for loops in LOOPS if any(int(point["depth"]) == depth for point in series[loops])]
            values = [cell_loss(loop_data, weight_decay, loops, depth) - baseline for loops in available]
            delta_axis.plot(available, values, color=DELTA_DEPTH_COLORS[depth], marker=DEPTH_MARKERS[depth], linewidth=1.6, markersize=4.8, zorder=3)
        setup_k_axis(delta_axis)

    loop_handles = [Line2D([], [], color=LOOP_COLORS[loops]) for loops in LOOPS]
    loop_labels = [f"K={loops}" for loops in LOOPS]
    depth_handles = [Line2D([], [], color=MUTED, marker=DEPTH_MARKERS[depth], linestyle="none") for depth in ALL_DEPTHS]
    depth_labels = [f"d{depth}" for depth in ALL_DEPTHS]
    add_span_legends(figure, axes[0], [
        (loop_handles, loop_labels, len(loop_handles)),
        (depth_handles, depth_labels, len(depth_handles)),
    ])

    delta_handles = [Line2D([], [], color=DELTA_DEPTH_COLORS[depth], marker=DEPTH_MARKERS[depth]) for depth in DELTA_DEPTHS]
    delta_labels = [f"d{depth}" for depth in DELTA_DEPTHS]
    add_row_legend(figure, axes[2], delta_handles, delta_labels, len(delta_handles))
    save(figure, output)

def cell_loss(loop_data: dict[float, dict[int, list[dict]]], weight_decay: float, loops: int, depth: int) -> float:
    return float(next(point["loss"] for point in loop_data[weight_decay][loops] if int(point["depth"]) == depth))

def setup_wd_axis(axis: plt.Axes, *, upper_y: float | None = None) -> None:
    ticks = WEIGHT_DECAYS
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.04, 1.8)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_yscale("log")
    axis.margins(y=0.08)
    if upper_y is not None:
        axis.set_ylim(top=upper_y)
    lower, upper = axis.get_ylim()
    axis.yaxis.set_major_locator(FixedLocator(np.geomspace(lower, upper, 4)))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Weight decay")
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    style_axis(axis)

def plot_wd_row(axes: list[plt.Axes] | np.ndarray,
                loop_data: dict[float, dict[int, list[dict]]], *, mark_optima: bool = True) -> None:
    for axis, depth in zip(axes, WD_FIT_DEPTHS):
        panel_losses = []
        for loops in LOOPS:
            if depth not in DEPTHS[loops]:
                continue
            available = [wd for wd in WEIGHT_DECAYS if any(int(point["depth"]) == depth for point in loop_data[wd][loops])]
            if not available:
                continue
            measured_wd = np.asarray(available, dtype=float)
            losses = np.asarray([cell_loss(loop_data, weight_decay, loops, depth) for weight_decay in available])
            panel_losses.extend(losses)
            axis.plot(measured_wd, losses, color=LOOP_COLORS[loops], marker="o", linewidth=1.6, markersize=4.3, zorder=3)
            if mark_optima:
                optimum_index = int(np.argmin(losses))
                axis.plot(measured_wd[optimum_index], losses[optimum_index], marker="o", markerfacecolor="none", markeredgecolor=LOOP_COLORS[loops], markeredgewidth=1.5, markersize=9.5, linestyle="none", zorder=4)
        upper_y = min(panel_losses) + 0.2 if depth >= 8 else None
        setup_wd_axis(axis, upper_y=upper_y)

def plot_vanilla_row(axes: list[plt.Axes] | np.ndarray, series, tied_grid) -> tuple[list[Line2D], list[str], Line2D]:
    definitions = (
        ("vanilla", "tied"),
        ("vanilla", "untied"),
        ("recurrent", "untied"),
    )
    frontier = [
        point
        for point in vanilla_figure.lower_log_compute_convex_envelope({
            loops: [Point(point["compute"], point["loss"], int(point["depth"]), loops) for point in points]
            for loops, points in tied_grid.items()
        })
        if not (point.loops == 6 and point.depth == 4)
    ]
    for column, (role, family) in enumerate(definitions):
        family_series = series[(role, family)]
        compute_axis, optimum_axis = axes[column], axes[column + 3]
        available = tuple(sorted(family_series))
        for loop_count in available:
            points = family_series[loop_count]
            compute_axis.plot(
                [point.compute for point in points],
                [point.loss for point in points],
                color=vanilla_figure.LOOP_COLORS[loop_count],
                linewidth=1.6,
                zorder=2,
            )
            for point in points:
                compute_axis.plot(point.compute, point.loss, color=vanilla_figure.LOOP_COLORS[loop_count], marker=vanilla_figure.DEPTH_MARKER[point.depth], linestyle="none", markersize=5.0, zorder=3)
        vanilla_figure.setup_compute_axis(compute_axis)
        compute_axis.set_xlim(0.38, 13.0)
        compute_axis.plot([point.compute for point in frontier], [point.loss for point in frontier], color=INK, linewidth=1.8, linestyle=(0, (4, 2)), zorder=5)

        for _, budget, color in vanilla_figure.BUDGETS:
            losses = vanilla_figure.iso_losses(family_series, budget)
            measured_loops = np.asarray(sorted(losses), dtype=float)
            measured_losses = np.asarray([losses[int(loop)] for loop in measured_loops])
            coefficients, optimum, optimum_loss = regime_figure.fit_loop_optimum(losses)
            fit_loops = np.geomspace(measured_loops.min(), measured_loops.max(), 240)
            optimum_axis.plot(fit_loops, np.exp(np.polyval(coefficients, np.log(fit_loops))), color=color, linewidth=1.4, zorder=2)
            optimum_axis.plot(measured_loops, measured_losses, color=color, marker="o", markersize=3.8, linestyle="none", zorder=3)
            if optimum is None:
                best_index = int(np.argmin(measured_losses))
                optimum_axis.plot(measured_loops[best_index], measured_losses[best_index], marker="*", markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.0, markersize=9, linestyle="none", zorder=4)
            else:
                optimum_axis.plot(optimum, optimum_loss, color=color, marker="*", markeredgecolor="white", markeredgewidth=0.6, markersize=9, linestyle="none", zorder=4)
        vanilla_figure.setup_loop_axis(optimum_axis, available)
        compute_axis_in_flops(compute_axis)

    loop_handles = [Line2D([], [], color=vanilla_figure.LOOP_COLORS[value]) for value in vanilla_figure.VANILLA_UNTIED_LOOPS]
    loop_labels = [f"K={value}" for value in vanilla_figure.VANILLA_UNTIED_LOOPS]
    frontier_handle = Line2D([], [], color=INK, linestyle=(0, (4, 2)))
    return loop_handles, loop_labels, frontier_handle

def plot_other_architectures(vanilla_series, tied_grid, output):
    figure, axes = make_grid(
        2, 3,
        titles=("Tied Vanilla scaling", "Untied Vanilla scaling", "Untied-2 scaling",
                "Tied Vanilla optima", "Untied Vanilla optima", "Untied-2 optima"),
        width=regime_figure.WIDTH, panel_height=SQUARE_PANEL_SIZE,
        row_ylabels=("Loss", "Loss (Interpolated)"), legend_rows=2, top_label_rows=1,
    )
    axes = axes.ravel()
    loop_handles, loop_labels, frontier_handle = plot_vanilla_row(axes, vanilla_series, tied_grid)
    depth_handles = [Line2D([], [], color=MUTED, marker=DEPTH_MARKERS[depth], linestyle="none") for depth in ALL_DEPTHS]
    add_span_legends(figure, axes, [
        (loop_handles + [frontier_handle], loop_labels + ["Tied frontier"], len(loop_handles) + 1),
        (depth_handles, [f"d{depth}" for depth in ALL_DEPTHS], len(depth_handles)),
    ])
    save(figure, output)
