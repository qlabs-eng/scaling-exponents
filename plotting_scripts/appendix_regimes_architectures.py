"""Figure 24: architecture scaling and the tied reference frontier."""
from __future__ import annotations

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, NullFormatter
from common import sequence_colors
from appendix_regimes_style import style_axis
from appendix_regimes_data import Point
from appendix_regimes import interpolate_loss

VANILLA_UNTIED_LOOPS = (1,2,3,4,6,8,12,16,20)
LOOP_COLORS = dict(zip(VANILLA_UNTIED_LOOPS, sequence_colors(len(VANILLA_UNTIED_LOOPS), quantity="recurrence")))
DEPTH_MARKER = {4: "X",6: "o",8: "s",10: "^",12: "D",14: "P",16: "v",18: "<"}
BUDGET_VALUES = (0.846120,1.419447,2.381258,3.994787,6.701636)
BUDGETS = tuple((f"B{i+1}",value,color) for i,(value,color) in enumerate(zip(BUDGET_VALUES, sequence_colors(len(BUDGET_VALUES), quantity="compute"))))

def iso_losses(series: dict[int, list[Point]], compute: float) -> dict[int, float]:
    return {
        loops: interpolate_loss(points, compute)
        for loops, points in series.items()
        if points[0].compute <= compute <= points[-1].compute
    }

def setup_compute_axis(axis: plt.Axes) -> None:
    for _, budget, color in BUDGETS:
        axis.axvline(budget, color=color, linewidth=1.4, linestyle=(0, (2, 3)), alpha=0.72, zorder=1)
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.38, 11.2)
    axis.xaxis.set_major_locator(FixedLocator((0.5, 1, 2, 4, 8)))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("Compute (EFLOP)", loc="left")
    axis.margins(y=0.08)
    style_axis(axis)

def setup_loop_axis(axis: plt.Axes, loops: tuple[int, ...]) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.86, max(loops) * 1.12)
    axis.xaxis.set_major_locator(FixedLocator(loops))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("Recurrence")
    axis.margins(y=0.08)
    style_axis(axis)
def lower_log_compute_convex_envelope(series: dict[int, list[Point]]) -> list[Point]:
    """Return the lower convex envelope in the coordinates used by the plot."""
    ordered = sorted((point for points in series.values() for point in points), key=lambda point: point.compute)
    pareto = []
    best_loss = math.inf
    for point in ordered:
        if point.loss < best_loss:
            pareto.append(point)
            best_loss = point.loss

    def cross(first: Point, second: Point, third: Point) -> float:
        first_compute, second_compute, third_compute = (math.log(point.compute) for point in (first, second, third))
        return (second_compute - first_compute) * (third.loss - first.loss) - (second.loss - first.loss) * (third_compute - first_compute)

    envelope = []
    for point in pareto:
        while len(envelope) >= 2 and cross(envelope[-2], envelope[-1], point) <= 0:
            envelope.pop()
        envelope.append(point)
    return envelope
