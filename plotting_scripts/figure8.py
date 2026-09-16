"""Figure 8: regenerate offline from the bundled paper measurements."""
from __future__ import annotations

import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import save_figure, flop_tick
from appendix_style import MUTED, FULL_WIDTH, OUTER_PAD, LEGEND_ROW, apply_style, legend_row_count, make_grid, ordered_colors, style_axis
from appendix_early_data import load_data

@dataclass(frozen=True)
class Point:
    compute: float
    loss: float
    depth: int
    loops: int

FIGURE_WIDTH = 2 * FULL_WIDTH
LINE_WIDTH, MARKER_SIZE, OPTIMUM_SIZE = 1.4, 4.0, 8.0

def compute_label(compute):
    return rf"$C={compute / 1e18:.2f}\times10^{{18}}$"

def format_compute_axis(axis, *, constrained=False):
    axis.set_xscale("log")
    axis.set_xlim(0.38e18, 10.7e18)
    axis.xaxis.set_major_locator(FixedLocator((0.5e18, 1e18, 2e18, 5e18, 1e19)))
    axis.xaxis.set_major_formatter(FuncFormatter(flop_tick))
    axis.xaxis.set_minor_formatter(NullFormatter())


def add_legend_row(figure: plt.Figure, handles: list[Line2D], labels: list[str], row: int) -> None:
    legend_row_count(labels, width=figure.get_figwidth(), max_rows=1)
    anchor_y = 1 - (OUTER_PAD + row * LEGEND_ROW) / figure.get_figheight()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, anchor_y), borderaxespad=0, ncols=len(labels))


LOOPS = (1, 2, 3, 4, 6)


DEPTH_MARKERS = {4: "X", 6: "o", 8: "s", 10: "^", 12: "D", 14: "P", 16: "v", 18: "<"}


BUDGETS = tuple(zip(
    (0.846120e18, 1.419447e18, 2.381258e18, 3.994787e18, 6.701636e18),
    ordered_colors(5, quantity="compute"),
))


def loss_formatter(value: float, _: int) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


COLORS = dict(zip(LOOPS, ordered_colors(len(LOOPS), quantity="recurrence")))


def plot_ladders(axis: plt.Axes, ladders: dict[int, list[Point]]) -> None:
    for compute, _ in BUDGETS:
        axis.axvline(compute, color=MUTED, linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.35)
    for loops in LOOPS:
        points = ladders[loops]
        axis.plot([point.compute for point in points], [point.loss for point in points], color=COLORS[loops], label=f"K={loops}", zorder=2)
        for point in points:
            axis.plot(point.compute, point.loss, linestyle="none", color=COLORS[loops], marker=DEPTH_MARKERS[point.depth], zorder=3)
    format_compute_axis(axis, constrained=True)
    axis.yaxis.set_major_formatter(FuncFormatter(loss_formatter))
    axis.margins(y=0.08)
    style_axis(axis)


def fit_loop_optimum(loops: np.ndarray, losses: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit log(loss) as a quadratic in log(loop count)."""
    coefficients = np.polyfit(np.log(loops), np.log(losses), 2)
    curvature, slope = coefficients[:2]
    if curvature <= 0:
        raise ValueError("loop fit does not have a minimum")
    optimum = math.exp(-slope / (2 * curvature))
    if not loops.min() <= optimum <= loops.max():
        raise ValueError(f"loop optimum K={optimum:.3f} is outside the measured range")
    optimum_loss = math.exp(float(np.polyval(coefficients, math.log(optimum))))
    return coefficients, optimum, optimum_loss


def interpolate_loss(points: list[Point], compute: float) -> float:
    """Linearly interpolate loss at a fixed amount of compute."""
    for point in points:
        if math.isclose(point.compute, compute, rel_tol=1e-12):
            return point.loss
    for lower, upper in zip(points, points[1:]):
        if lower.compute <= compute <= upper.compute:
            weight = (compute - lower.compute) / (upper.compute - lower.compute)
            return lower.loss + weight * (upper.loss - lower.loss)
    raise ValueError(f"compute {compute:g} is outside the ladder range")


def plot_optima(axis: plt.Axes, ladders: dict[int, list[Point]], family: str) -> None:
    loops = np.asarray(LOOPS, dtype=float)
    fit_loops = np.geomspace(loops.min(), loops.max(), 300)
    plotted_losses = []
    for compute, color in BUDGETS:
        losses = np.asarray([interpolate_loss(ladders[value], compute) for value in LOOPS])
        coefficients, optimum, optimum_loss = fit_loop_optimum(loops, losses)
        axis.plot(fit_loops, np.exp(np.polyval(coefficients, np.log(fit_loops))), color=color, linewidth=LINE_WIDTH, zorder=2)
        axis.plot(loops, losses, linestyle="none", color=color, marker="o", markersize=MARKER_SIZE, zorder=3)
        axis.plot(optimum, optimum_loss, linestyle="none", color=color, marker="*", markeredgecolor="white", markeredgewidth=0.6, markersize=OPTIMUM_SIZE, zorder=4)
        plotted_losses.extend(losses)
        plotted_losses.append(optimum_loss)
        print(f"{family} {compute:.3e} FLOPs: K*={optimum:.3f}, loss={optimum_loss:.6f}")
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.88, 7.55)
    axis.xaxis.set_major_locator(FixedLocator(LOOPS))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(FuncFormatter(loss_formatter))
    axis.set_ylim(min(plotted_losses) - 0.018, max(plotted_losses) + 0.018)
    style_axis(axis)


def plot_figure(families: dict[str, dict[int, list[Point]]]) -> None:
    apply_style()
    loop_handles = [Line2D([], [], color=COLORS[loops])
                    for loops in LOOPS]
    loop_labels = [f"K={loops}" for loops in LOOPS]
    size_handles = [Line2D([], [], color=MUTED, marker=DEPTH_MARKERS[depth], linestyle="none")
                    for depth in DEPTH_MARKERS]
    size_labels = [f"d{depth}" for depth in DEPTH_MARKERS]
    budget_handles = [Line2D([], [], color=color) for _, color in BUDGETS]
    budget_labels = [compute_label(compute) for compute, _ in BUDGETS]
    figure, axes = make_grid(
        1,
        4,
        titles=("Loop-2 scaling", "Loop-2 optima", "Untied-2 scaling", "Untied-2 optima"),
        width=FIGURE_WIDTH,
        square_panels=True,
        column_xlabels=("Compute (FLOPs)", "Recurrence", "Compute (FLOPs)", "Recurrence"),
        legend_rows=3,
        squeeze=False,
    )
    for family_index, family in enumerate(("tied", "untied")):
        scaling_axis = axes[0, 2 * family_index]
        optimum_axis = axes[0, 2 * family_index + 1]
        plot_ladders(scaling_axis, families[family])
        plot_optima(optimum_axis, families[family], family)
        scaling_axis.set_ylabel("Loss")
        optimum_axis.set_ylabel("Loss (Interpolated)")
    add_legend_row(figure, loop_handles, loop_labels, 0)
    add_legend_row(figure, size_handles, size_labels, 1)
    add_legend_row(figure, budget_handles, budget_labels, 2)
    save_figure(figure, 8)


def load_fresh_data():
    return {family: {int(k): [Point(**point) for point in points]
                     for k, points in ladders.items()}
            for family, ladders in load_data("fresh_data").items()}


def main():
    plot_figure(load_fresh_data())


if __name__ == "__main__":
    main()
