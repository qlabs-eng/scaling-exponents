"""Figure 7: regenerate offline from the bundled paper measurements."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FormatStrFormatter, FuncFormatter, LogLocator, NullFormatter

from common import save_figure
from appendix_style import MUTED, OUTER_PAD, LEGEND_ROW, apply_style, make_grid, ordered_colors, style_axis
from appendix_early_data import load_data

import figure8 as loop_source
import appendix_early_growth as opt_grow_source
import appendix_early_stored as stored_tpp

@dataclass(frozen=True)
class ShellPoint:
    depth: int
    shell: int
    executed: int
    compute: float
    loss: float


def frontier_loss(hull: list[ShellPoint], compute: float) -> float:
    knots_x = [math.log(point.compute) for point in hull]
    x = math.log(compute)
    if x <= knots_x[0]:
        lo, hi = 0, 1
    elif x >= knots_x[-1]:
        lo, hi = len(hull) - 2, len(hull) - 1
    else:
        hi = next(index for index, knot in enumerate(knots_x) if knot >= x)
        lo = hi - 1
    slope = (hull[hi].loss - hull[lo].loss) / (knots_x[hi] - knots_x[lo])
    return hull[lo].loss + slope * (x - knots_x[lo])


def frontier_points(points: list[ShellPoint], *, allow_skipped_depths: bool = False) -> list[ShellPoint]:
    depths = sorted({point.depth for point in points})
    by_depth = {depth: [point for point in points if point.depth == depth] for depth in depths}
    groups = [by_depth[depth] + ([None] if allow_skipped_depths else []) for depth in depths]
    best_chain = None
    best_score = -math.inf
    for candidates in itertools.product(*groups):
        chain = sorted((point for point in candidates if point is not None), key=lambda point: point.compute)
        if len(chain) < 2:
            continue
        knots_x = [math.log(point.compute) for point in chain]
        if any(right <= left for left, right in zip(knots_x, knots_x[1:])):
            continue
        slopes = [(right.loss - left.loss) / (right_x - left_x) for left, right, left_x, right_x in zip(chain, chain[1:], knots_x, knots_x[1:])]
        if any(right + 1e-12 < left for left, right in zip(slopes, slopes[1:])):
            continue
        predicted = [frontier_loss(chain, point.compute) for point in points]
        if any(value > point.loss + 1e-12 for value, point in zip(predicted, points)):
            continue
        score = sum(predicted)
        if score > best_score + 1e-12:
            best_chain, best_score = chain, score
    if best_chain is None:
        requirement = "at most one run" if allow_skipped_depths else "one run"
        raise ValueError(f"no convex frontier uses {requirement} per stored depth and stays below every run")
    return best_chain


SHELL_DEPTHS = (6, 7, 8, 9, 10)


SHELL_DEPTH_COLORS = dict(zip(SHELL_DEPTHS, ordered_colors(len(SHELL_DEPTHS), quantity="depth")))


def loss_formatter(value: float, _: int) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


CORE_MARKERS = {2: "o", 3: "s", 4: "^", 5: "D", 6: "v", 7: "<", 8: "P"}


def plot_shell_ladders(axis: plt.Axes, points: list[ShellPoint], hull: list[ShellPoint], *, depth_colors: dict[int, str] = SHELL_DEPTH_COLORS) -> None:
    compute_grid = np.geomspace(min(point.compute for point in points), max(point.compute for point in points), 200)
    axis.plot(compute_grid, [frontier_loss(hull, value) for value in compute_grid], color="black", linestyle=(0, (4, 3)), linewidth=1.2, alpha=0.7, zorder=1)
    for depth in sorted({point.depth for point in points}):
        series = sorted((point for point in points if point.depth == depth), key=lambda point: point.compute)
        axis.plot([point.compute for point in series], [point.loss for point in series], color=depth_colors[depth], zorder=2)
        for point in series:
            core_size = point.depth - point.shell
            axis.plot(point.compute, point.loss, linestyle="none", color=depth_colors[depth], marker=CORE_MARKERS[core_size], markersize=4.8, zorder=3)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    axis.yaxis.set_major_formatter(FuncFormatter(loss_formatter))
    axis.margins(y=0.08)
    style_axis(axis)


def shell_regrets(points: list[ShellPoint], hull: list[ShellPoint]) -> np.ndarray:
    regrets = np.asarray([point.loss - frontier_loss(hull, point.compute) for point in points])
    if np.any(regrets < -1e-10):
        raise ValueError("convex frontier rises above an observed shell run")
    regrets[np.abs(regrets) < 1e-12] = 0
    return regrets


def plot_core_target(axis: plt.Axes, points: list[ShellPoint], hull: list[ShellPoint], *, depth_colors: dict[int, str] = SHELL_DEPTH_COLORS, xlim: tuple[float, float] = (0.18, 0.82)) -> dict[int, tuple[np.ndarray, float]]:
    regrets = shell_regrets(points, hull)
    axis.axhline(0, color=MUTED, linestyle=(0, (2, 3)), linewidth=0.8, alpha=0.55)
    fits = {}
    for depth in sorted({point.depth for point in points}):
        indices = sorted((index for index, point in enumerate(points) if point.depth == depth), key=lambda index: (points[index].depth - points[index].shell) / points[index].depth)
        for index in indices:
            point = points[index]
            core_size = point.depth - point.shell
            core_fraction = core_size / point.depth
            axis.plot(core_fraction, regrets[index] * 1e3, linestyle="none", color=depth_colors[depth], marker=CORE_MARKERS[core_size], markersize=4.8, zorder=3)
        fractions = np.asarray([(points[index].depth - points[index].shell) / points[index].depth for index in indices])
        depth_regrets = regrets[indices] * 1e3
        coefficients = np.polyfit(fractions, depth_regrets, 2)
        if coefficients[0] <= 0:
            raise ValueError(f"d{depth} shell-regret fit is not convex")
        minimum_fraction = -coefficients[1] / (2 * coefficients[0])
        if not fractions.min() <= minimum_fraction <= fractions.max():
            raise ValueError(f"d{depth} shell-regret fit has an unbracketed minimum")
        fit_fractions = np.linspace(fractions.min(), fractions.max(), 200)
        axis.plot(fit_fractions, np.polyval(coefficients, fit_fractions), color=depth_colors[depth], linestyle=(0, (1, 2)), linewidth=1.4, alpha=0.9, zorder=2)
        axis.plot(minimum_fraction, np.polyval(coefficients, minimum_fraction), marker="*", linestyle="none", color=depth_colors[depth], markeredgecolor="white", markeredgewidth=0.6, markersize=9, zorder=4)
        fits[depth] = (coefficients, minimum_fraction)
    axis.set_xlim(*xlim)
    axis.set_xticks(np.arange(0.2, 0.81, 0.1))
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.margins(y=0.08)
    style_axis(axis)
    return fits


def row_major(handles: list[Line2D], labels: list[str], ncols: int) -> tuple[list[Line2D], list[str]]:
    rows = math.ceil(len(labels) / ncols)
    order = [row * ncols + column for column in range(ncols) for row in range(rows) if row * ncols + column < len(labels)]
    return [handles[index] for index in order], [labels[index] for index in order]


def add_column_legends(figure: plt.Figure, axis: plt.Axes, groups: list[tuple[list[Line2D], list[str], int]]) -> None:
    """Stack figure-level legends in the top band, centered over one panel column."""
    position = axis.get_position()
    center = (position.x0 + position.x1) / 2
    top = 1 - OUTER_PAD / figure.get_figheight()
    for handles, labels, ncols in groups:
        ordered_handles, ordered_labels = row_major(handles, labels, ncols)
        figure.legend(handles=ordered_handles, labels=ordered_labels, loc="upper center", bbox_to_anchor=(center, top), ncols=ncols, borderaxespad=0, columnspacing=0.6, handlelength=1.1, handletextpad=0.4, fontsize=7)
        top -= math.ceil(len(labels) / ncols) * LEGEND_ROW / figure.get_figheight()


GROW_BUDGETS = (2.0, 2.5, 3.0, 3.5, 4.0)


GROW_TARGETS = (2, 3, 4, 6, 8)


GROW_BUDGET_COLORS = dict(zip(GROW_BUDGETS, ordered_colors(len(GROW_BUDGETS), quantity="compute")))


def plot_grow_target(axis: plt.Axes, points: dict[tuple[float, int], float], *, baseline_line: bool, loss_ticks: bool) -> None:
    baseline = points[2.0, 2]
    if baseline_line:
        axis.axhline(baseline, color=MUTED, linewidth=1.2, linestyle=":", zorder=1)
    axis.scatter([2], [baseline], s=54, facecolor="white", edgecolor=GROW_BUDGET_COLORS[2.0], linewidth=2.0, zorder=4)
    for budget in GROW_BUDGETS[1:]:
        color = GROW_BUDGET_COLORS[budget]
        cells = sorted((target, loss) for (cell_budget, target), loss in points.items() if cell_budget == budget)
        xs = [target for target, _ in cells]
        ys = [loss for _, loss in cells]
        axis.plot(xs, ys, color=color, linewidth=2.0, zorder=3)
        grown = [(target, loss) for target, loss in cells if target > budget]
        fixed = [(target, loss) for target, loss in cells if target == budget]
        if grown:
            axis.scatter(*zip(*grown), s=48, color=color, zorder=4)
        if fixed:
            axis.scatter(*zip(*fixed), s=54, facecolor="white", edgecolor=color, linewidth=2.0, zorder=4)
    axis.set_xscale("log")
    axis.set_xticks(GROW_TARGETS, labels=[str(target) for target in GROW_TARGETS])
    axis.xaxis.set_minor_formatter(NullFormatter())
    if loss_ticks:
        axis.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axis.margins(y=0.08)
    style_axis(axis)


def grow_legend() -> tuple[list[Line2D], tuple[str, ...]]:
    labels = tuple(rf"$B={budget:g}$" for budget in GROW_BUDGETS)
    handles = [Line2D([], [], color=GROW_BUDGET_COLORS[budget]) for budget in GROW_BUDGETS]
    return handles, labels


def plot_growth_timing(axis: plt.Axes, source, rows: list[dict]) -> None:
    """Plot the measured Loop-2 TPP-6 diagonal from the growth grid."""
    for depth in source.DEPTHS:
        series = source.series_for(rows, "k2", float(depth), depth)
        expected = source.expected_cells("k2", float(depth), depth)
        if len(series) != expected:
            raise ValueError(f"Loop-2 d{depth} TPP-6 growth series has {len(series)} cells; expected {expected}")
        tpp = source.k2_stored_tpp(series, "k2", depth)
        if not math.isclose(tpp, 6.0, rel_tol=0, abs_tol=0.002):
            raise ValueError(f"Loop-2 d{depth} diagonal has TPP {tpp:.3f}, expected 6")
        x = np.asarray([row["rho"] for row in series])
        y = np.asarray([row["loss"] for row in series])
        color = stored_tpp.BUDGET_COLOR[depth]
        axis.scatter(x, y, s=24, marker="o", color=color, edgecolor="white", linewidth=0.4, zorder=4)
        fit = source.quadratic(series)
        if fit is None:
            raise ValueError(f"Loop-2 d{depth} TPP-6 growth fit is not convex and bracketed")
        coefficients, rho_star, loss_star = fit
        grid = np.linspace(x.min(), x.max(), 200)
        axis.plot(grid, np.polyval(coefficients, grid), color=color, linewidth=1.35, alpha=0.9, zorder=2)
        axis.scatter([rho_star], [loss_star], marker="*", s=62, color=color, edgecolor="white", linewidth=0.45, zorder=5)
    axis.set_xlim(-0.012, 0.412)
    axis.set_xticks(np.arange(0, 0.41, 0.1))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel(r"$K=4$ training fraction $\rho$")
    axis.set_ylabel("Validation loss")
    style_axis(axis)


def plot_growth_tpp_refit(axis: plt.Axes, source) -> tuple[float, float]:
    """Draw the Loop-2 stored-parameter TPP refit from the growth study."""
    optimum = stored_tpp.draw(axis, stored_tpp.ARMS[0], "initial")
    y_min, y_max = axis.get_ylim()
    loss_ticks = np.arange(np.ceil(y_min / 0.05) * 0.05, np.floor(y_max / 0.05) * 0.05 + 0.025, 0.05)
    axis.yaxis.set_major_locator(FixedLocator(loss_ticks))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.set_xlabel("Initial stored TPP")
    axis.set_ylabel("Validation loss")
    return optimum


COMBINED_WIDTH = 10.35


COMBINED_PANEL_HEIGHT = 2.05


PLOT_SHELL_DEPTHS = (6, 7, 8, 9, 10)


CORE_SIZES = (2, 3, 4, 5, 6, 7, 8)


GROW_KL_THRESHOLD = 2.0


GROW_PRELUDE, GROW_CORE, GROW_CODA = 2, 3, 3


def grow_kl_effective_depth(point: dict) -> int:
    lens = point["block_lens"]
    effective_depth = int(point["effective_depth"])
    target = int(point["target"])
    if effective_depth != GROW_PRELUDE + target * GROW_CORE + GROW_CODA or len(lens) != effective_depth + 1:
        raise ValueError(f"{point['cell']}: effective depth does not match the decoded layers")
    kl = np.asarray([item["kl_to_final"] for item in lens], dtype=float)
    if not np.all(np.isfinite(kl)) or np.any(kl < 0):
        raise ValueError(f"{point['cell']}: invalid KL values")
    peak_index = int(np.argmax(kl))
    below = np.flatnonzero(kl[peak_index + 1:] < GROW_KL_THRESHOLD)
    if not len(below):
        raise ValueError(f"{point['cell']}: KL never falls below {GROW_KL_THRESHOLD:g}")
    depth = int(peak_index + 1 + below[0])
    if int(point["kl_effective_depth"]) != depth:
        raise ValueError(f"{point['cell']}: stored KL effective depth is inconsistent")
    return depth


def plot_combined_figure(shell_points: list[ShellPoint], grow_points: dict[tuple[float, int], float], grow_lens: dict[tuple[float, int], dict]) -> None:
    shell_points = [point for point in shell_points if point.depth in PLOT_SHELL_DEPTHS]
    figure, axes = make_grid(
        2,
        4,
        titles=("Where should layers be allocated?", "How much recurrence?", "What should recurrence grow to?", "When should we grow?", "Frontier regret", "Fixed-compute optima", r"KL effective depth $L_{\mathrm{eff}}^{\mathrm{KL}}$", "Loop-2 TPP refit"),
        width=COMBINED_WIDTH,
        panel_height=COMBINED_PANEL_HEIGHT,
        legend_rows=4,
        squeeze=False,
    )
    shell_axes = axes[:, 0]
    loop_axes = axes[:, 1]
    grow_axes = axes[:, 2]
    timing_axes = axes[:, 3]

    shell_axes[0].set_xlabel("Compute (FLOPs)")
    shell_axes[0].set_ylabel("Validation loss")
    shell_axes[1].set_xlabel("Core size / physical layers")
    shell_axes[1].set_ylabel(r"Regret to frontier ($\times 10^{-3}$)")
    hull = frontier_points(shell_points)
    plot_shell_ladders(shell_axes[0], shell_points, hull)
    shell_fits = plot_core_target(shell_axes[1], shell_points, hull)
    shell_depth_handles = [Line2D([], [], color=SHELL_DEPTH_COLORS[depth]) for depth in PLOT_SHELL_DEPTHS]
    shell_depth_labels = [f"d{depth}" for depth in PLOT_SHELL_DEPTHS]
    core_handles = [Line2D([], [], color=MUTED, marker=CORE_MARKERS[core_size], linestyle="none") for core_size in CORE_SIZES]
    core_labels = [f"C={core_size}" for core_size in CORE_SIZES]
    add_column_legends(figure, shell_axes[0], [
        (shell_depth_handles, shell_depth_labels, 5),
        (core_handles, core_labels, 4),
    ])

    recurrence_ladders = loop_source.load_fresh_data()["tied"]
    loop_source.plot_ladders(loop_axes[0], recurrence_ladders)
    # Three major ticks fit this narrow panel without changing its compute range.
    loop_axes[0].xaxis.set_major_locator(FixedLocator((0.5e18, 2e18, 1e19)))
    for line, (_, color) in zip(loop_axes[0].lines[:len(loop_source.BUDGETS)], loop_source.BUDGETS):
        line.set_color(color)
        line.set_alpha(0.8)
        line.set_linewidth(1.6)
    loop_source.plot_optima(loop_axes[1], recurrence_ladders, "recurrence")
    loop_axes[0].set_xlabel("Compute (FLOPs)")
    loop_axes[0].set_ylabel("Loss")
    loop_axes[1].set_xlabel("Recurrence")
    loop_axes[1].set_ylabel("Loss (Interpolated)")
    loop_handles = [Line2D([], [], color=loop_source.COLORS[loops]) for loops in loop_source.LOOPS]
    loop_labels = [f"K={loops}" for loops in loop_source.LOOPS]
    depth_handles = [Line2D([], [], color=loop_source.MUTED, marker=loop_source.DEPTH_MARKERS[depth], linestyle="none") for depth in loop_source.DEPTH_MARKERS]
    depth_labels = [f"d{depth}" for depth in loop_source.DEPTH_MARKERS]
    slice_handle = Line2D([], [], color=MUTED, linestyle=(0, (2, 1.4)), linewidth=1.2)
    add_column_legends(figure, loop_axes[0], [
        (loop_handles, loop_labels, 5),
        (depth_handles, depth_labels, 4),
        ([slice_handle], ["compute slice"], 1),
    ])

    grow_axes[0].set_xlabel(r"Target loop count $K$")
    grow_axes[0].set_ylabel("Validation loss")
    grow_axes[1].set_xlabel(r"Target loop count $K$")
    grow_axes[1].set_ylabel(r"$L_{\mathrm{eff}}^{\mathrm{KL}}$")
    plot_grow_target(grow_axes[0], grow_points, baseline_line=True, loss_ticks=True)
    depth_points = {key: float(grow_kl_effective_depth(point)) for key, point in grow_lens.items()}
    plot_grow_target(grow_axes[1], depth_points, baseline_line=False, loss_ticks=False)
    grow_handles, grow_labels = grow_legend()
    fixed_handle = Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", markeredgewidth=1.8, linestyle="none")
    add_column_legends(figure, grow_axes[0], [
        (grow_handles, list(grow_labels), 3),
        ([fixed_handle], ["fixed K"], 1),
    ])

    opt_grow_rows = opt_grow_source.load_rows()
    if not opt_grow_rows:
        raise FileNotFoundError("no completed optgrow3d result records")
    plot_growth_timing(timing_axes[0], opt_grow_source, opt_grow_rows)
    tpp_optimum = plot_growth_tpp_refit(timing_axes[1], opt_grow_source)
    depth_handles = [
        Line2D([], [], color=color, marker="o")
        for color in stored_tpp.BUDGET_COLOR.values()
    ]
    depth_labels = [f"d{budget}" for budget in stored_tpp.BUDGET_COLOR]
    add_column_legends(figure, timing_axes[0], [
        (depth_handles, depth_labels, 3),
    ])

    save_figure(figure, 7)
    print("shell fit minima: " + ", ".join(f"d{depth}={minimum:.3f}" for depth, (_, minimum) in shell_fits.items()))
    print(f"Loop-2 stored-TPP refit: growth={tpp_optimum[0]:.3f}, fixed={tpp_optimum[1]:.3f}")


def main():
    apply_style()
    shells = [ShellPoint(**point) for point in load_data("shell")]
    targets = {(point["budget"], point["target"]): point["loss"] for point in load_data("grow_targets")}
    lenses = {(point["budget"], point["target"]): point for point in load_data("grow_lens")}
    plot_combined_figure(shells, targets, lenses)


if __name__ == "__main__":
    main()
