#!/usr/bin/env python3
"""Single epoch against multi-epoch. Left: the FineWeb compute-optimal ladders and their compute multiplier over Vanilla. Right: the
100M-token x 10-epoch loop grid at weight decay 0.8, and Operator-1 with tuned weight decay against scaling the loop count.

    python plotting_scripts/figure2.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from common import (ARM_LABEL, FIGURES, INK, MUTED, MAIN_RCPARAMS, MAIN_FIGURE_WIDTH, MAIN_SMALL_FONT, add_source_arguments, fit_power_law, fit_power_law_fixed_irreducible, ladders_from,
                    quantity_colors, plot_multipliers, save, style_axis, architecture_colors, zoom_inset)
from loop_grid import LoopGradientHandle, LoopGradientHandler, clip_after, compute_at_loss, compute_axis, convex_envelope, load_cells, series

ARMS = ("vanilla", "operator_1", "loop_2", "untied_2", "loop_grow", "untied_grow")  # legend order
LOOPS = (1, 2, 3, 4, 6, 8, 12)
LOOP_DEPTHS = {1: range(6, 19, 2), 2: range(6, 17, 2), 3: range(6, 17, 2), 4: range(6, 15, 2), 6: range(4, 15, 2), 8: (8, 10, 12), 12: (8, 10, 12)}  # Loop-8 d14 left out
ZOOM_FROM = 2e19


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_arguments(parser)
    parser.add_argument("--output", type=Path, default=FIGURES / "figure2.png")
    args = parser.parse_args()
    ladders = ladders_from(args)
    cells = [cell for cell in load_cells(args) if cell.loops in LOOPS and cell.depth in LOOP_DEPTHS[cell.loops]]
    colors = architecture_colors(ARMS)
    loop_colors = quantity_colors(LOOPS, quantity="recurrence")
    irreducible = fit_power_law([p.compute for p in ladders["vanilla"]], [p.loss for p in ladders["vanilla"]]).irreducible
    fits = {arm: fit_power_law_fixed_irreducible([p.compute for p in ladders[arm]], [p.loss for p in ladders[arm]], irreducible) for arm in ARMS}

    plt.rcParams.update(MAIN_RCPARAMS | {"legend.fontsize": 11, "legend.title_fontsize": 11})
    fig, ((loss_axis, loop_axis), (multiplier_axis, strategy_axis)) = plt.subplots(2, 2, figsize=(MAIN_FIGURE_WIDTH, 9), sharex="col")

    # Single epoch: fitted ladders with the exponents in the legend and an inset on the largest budgets, then the compute multiplier.
    for arm in ARMS:
        compute = np.array([p.compute for p in ladders[arm]])
        grid = np.geomspace(compute.min(), compute.max(), 300)
        loss_axis.plot(grid, fits[arm](grid), color=colors[arm], label=rf"{ARM_LABEL[arm]}  ($\gamma = {fits[arm].exponent:.3f}$)")
        loss_axis.plot(compute, [p.loss for p in ladders[arm]], "o", color=colors[arm], markeredgewidth=0, zorder=3)
    loss_axis.set_xscale("log")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("Single Epoch")
    low, high = loss_axis.get_ylim()
    loss_axis.set_ylim(low - 0.12, high)  # keep the larger legend below the measured curves
    loss_axis.legend(loc="lower left", labelspacing=0.3, borderaxespad=0.5, handlelength=1.5, handletextpad=0.4)
    zoomed = [p for arm in ARMS for p in ladders[arm] if p.compute >= ZOOM_FROM]
    inset = zoom_inset(loss_axis, [0.56, 0.52, 0.42, 0.46], (ZOOM_FROM, max(p.compute for p in zoomed) * 1.15),
                       (min(p.loss for p in zoomed) - 0.01, max(p.loss for p in zoomed) + 0.01), (3e19, 1e20), labelsize=MAIN_SMALL_FONT)
    for arm in ARMS:
        points = [p for p in ladders[arm] if p.compute >= ZOOM_FROM]
        grid = np.geomspace(ZOOM_FROM, points[-1].compute, 100)
        inset.plot(grid, fits[arm](grid), color=colors[arm], linewidth=1.8)
        inset.plot([p.compute for p in points], [p.loss for p in points], "o", color=colors[arm], markersize=5.5, markeredgewidth=0, zorder=3)
    final = plot_multipliers(multiplier_axis, ladders, ARMS, colors)
    multiplier_axis.yaxis.set_major_locator(MultipleLocator(0.1))
    multiplier_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.1f}×"))
    multiplier_axis.set_xlabel("Compute (FLOPs)")

    # Multi-epoch: the loop grid at weight decay 0.8, one ladder per loop count.
    fixed = series(cells)
    for index, loops in enumerate(LOOPS):
        points = fixed[loops]
        loop_axis.plot([p.compute for p in points], [p.loss for p in points], "o-", color=loop_colors[loops], markeredgewidth=0, zorder=3 + 0.01 * (len(LOOPS) - index))
    compute_axis(loop_axis)
    loop_axis.set_xlabel(None)
    loop_axis.set_ylabel("Loss")
    loop_axis.set_title("Multi-Epoch")
    handles = [Line2D([], [], color=loop_colors[loops], marker="o", markeredgewidth=0) for loops in LOOPS]
    loop_axis.legend(handles, [str(loops) for loops in LOOPS], title="Loop count", ncol=len(LOOPS), loc="upper right", columnspacing=0.6, handlelength=1.1, handletextpad=0.3, borderaxespad=0.5)

    # Strategies: Operator-1 at its best weight decay per depth; the loop-count frontier at weight decay 0.8, coloured by the winning
    # loop count; the frontier over every weight decay, dashed. Both frontiers run to the first cell at or beyond Operator-1's largest budget.
    k1_points = [min((c for c in cells if c.loops == 1 and c.depth == depth), key=lambda c: c.loss) for depth in LOOP_DEPTHS[1]]
    pool = [c for c in cells if c.compute >= k1_points[0].compute * (1 - 1e-9)]
    scale = clip_after(convex_envelope([c for c in pool if math.isclose(c.weight_decay, 0.8)]), k1_points[-1].compute)
    tuned = clip_after(convex_envelope(pool + k1_points), k1_points[-1].compute)
    strategy_axis.plot([p.compute for p in k1_points], [p.loss for p in k1_points], "o-", color=loop_colors[1], label="Operator-1", zorder=4)
    for lower, upper in zip(scale, scale[1:]):
        strategy_axis.plot([lower.compute, upper.compute], [lower.loss, upper.loss], color=loop_colors[upper.loops], solid_capstyle="round", zorder=3)
    for point in scale:
        strategy_axis.plot(point.compute, point.loss, "o", color=loop_colors[point.loops], zorder=4)
    strategy_axis.plot([p.compute for p in tuned], [p.loss for p in tuned], color=MUTED, linestyle=(0, (4, 2.5)), linewidth=1.3, label="Scale Loop Count, tuned WD", zorder=2)
    compute_axis(strategy_axis)
    strategy_axis.set_ylabel("Loss")
    handles, labels = strategy_axis.get_legend_handles_labels()
    handles.insert(1, LoopGradientHandle(loop_colors))
    labels.insert(1, "Scale Loop Count")
    strategy_axis.legend(handles, labels, handler_map={LoopGradientHandle: LoopGradientHandler()}, loc="upper right")
    target = k1_points[-1].loss
    matched = compute_at_loss(scale, target)
    gain = k1_points[-1].compute / matched
    strategy_axis.plot(matched, target, "x", color=MUTED, markeredgewidth=1.4, zorder=5)
    strategy_axis.annotate("", xy=(k1_points[-1].compute, target), xytext=(matched, target), arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 0.9}, zorder=4)
    strategy_axis.annotate(f"{gain:.1f}× compute\nefficiency gain", xy=(math.sqrt(k1_points[-1].compute * matched), target), xytext=(0, 40), textcoords="offset points",
                           ha="center", va="bottom", fontsize=MAIN_SMALL_FONT, color=INK, bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})

    for axis in (loss_axis, loop_axis, multiplier_axis, strategy_axis):
        style_axis(axis)
        axis.set_box_aspect(0.72)
    fig.subplots_adjust(left=0.08, right=0.93, bottom=0.09, top=0.95, wspace=0.30, hspace=0.16)
    save(fig, args.output)
    for arm in ARMS:
        print(f"  {ARM_LABEL[arm]:12s} gamma = {fits[arm].exponent:.4f}   final multiplier {final[arm]:.3f}×")
    print("  Operator-1 weight decay per depth: " + ", ".join(f"d{p.depth} {p.weight_decay:g}" for p in k1_points))
    print(f"  Scale Loop Count frontier: " + ", ".join(f"K{p.loops} d{p.depth}" for p in scale))
    print(f"  compute-efficiency gain at Operator-1's largest budget: {gain:.2f}×")


if __name__ == "__main__":
    main()
