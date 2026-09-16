#!/usr/bin/env python3
"""The 100M-token x 10-epoch grid in four panels: matched-compute cuts over loop count, the marginal gain over Operator-1 per depth,
the scaling recipes (Operator-1 and the loop-count frontier, each at weight decay 0.8 and with tuned weight decay), and the best weight decay per cell.

    python plotting_scripts/figure5.py
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

from common import FIGURES, MUTED, MAIN_RCPARAMS, MAIN_FIGURE_WIDTH, MAIN_SMALL_FONT, add_source_arguments, quantity_colors, save, style_axis
from loop_grid import (FINE_BUDGETS, WEIGHT_DECAYS, best_weight_decays, clip_after, compute_axis, convex_envelope, load_cells, loop_axis, plot_cuts,
                       plot_heatmap, series)

LOOPS = (1, 2, 3, 4, 6, 8, 12)
GAIN_DEPTHS = (6, 8, 10, 12, 14)  # the depths measured at four or more loop counts
HEATMAP_DEPTHS = (6, 8, 10, 12, 14)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_arguments(parser)
    parser.add_argument("--output", type=Path, default=FIGURES / "figure5.png")
    args = parser.parse_args()
    cells = [cell for cell in load_cells(args) if round(cell.weight_decay, 3) in WEIGHT_DECAYS]
    ladders = series(cells)
    # Compact inline legends; extra headroom keeps measured curves visible.
    legend = dict(loc="upper right", fontsize=10, title_fontsize=10,
                  labelspacing=0.25, borderaxespad=0.4, borderpad=0.35,
                  handlelength=1.1, handletextpad=0.35)

    plt.rcParams.update(MAIN_RCPARAMS)
    fig, (cut_axis, gain_axis, recipe_axis, wd_axis) = plt.subplots(1, 4, figsize=(MAIN_FIGURE_WIDTH, 3.5))

    plot_cuts(cut_axis, ladders, quantity_colors(FINE_BUDGETS, quantity="compute"))
    loop_axis(cut_axis, LOOPS)
    cut_axis.set_ylabel("Interpolated loss")
    cut_axis.set_title("Matched-compute cuts")
    low, high = cut_axis.get_ylim()
    cut_axis.set_ylim(low, high + (high - low) * 0.8)
    handles, _ = cut_axis.get_legend_handles_labels()
    cut_axis.legend(handles, [f"{budget / 1e18:.2g}" for budget in FINE_BUDGETS],
                    title=r"Compute ($10^{18}$ FLOPs)", ncol=3, columnspacing=0.6, **legend)

    # Marginal gain: loss at loop count K minus the Operator-1 loss at the same depth, at weight decay 0.8.
    loss = {(cell.depth, cell.loops): cell.loss for cell in cells if math.isclose(cell.weight_decay, 0.8)}
    gain_axis.axhline(0.0, color=MUTED, linewidth=0.8, zorder=1)
    for depth, color in quantity_colors(GAIN_DEPTHS, quantity="depth").items():
        loops = [k for k in LOOPS if (depth, k) in loss]
        gain_axis.plot(loops, [loss[depth, k] - loss[depth, 1] for k in loops], "o-", color=color, markeredgewidth=0, label=f"d{depth}", zorder=3)
    loop_axis(gain_axis, LOOPS)
    gain_axis.set_ylabel(r"$L(K) - L(K{=}1)$")
    gain_axis.set_title("Marginal gain")
    low, high = gain_axis.get_ylim()
    gain_axis.set_ylim(low - (high - low) * 0.05, (high - low) * 0.06)
    gain_axis.legend(title="Depth", ncol=2, columnspacing=0.8, **legend)

    # Scaling recipes. Tuned weight decay: for each depth, the weight decay that is best for Operator-1, applied to every loop count at
    # that depth. Frontiers are lower convex envelopes in log compute, cut at the first cell at or beyond Operator-1's largest budget.
    k1_fixed = ladders[1]
    k1_tuned = [min((c for c in cells if c.loops == 1 and c.depth == depth), key=lambda c: c.loss) for depth in sorted({p.depth for p in k1_fixed})]
    tuned_pool = [c for best in k1_tuned for c in cells if c.depth == best.depth and math.isclose(c.weight_decay, best.weight_decay)]
    cap = k1_fixed[-1].compute
    scale_fixed = clip_after(convex_envelope([c for points in ladders.values() for c in points if not (c.loops == 6 and c.depth == 4)], log_compute=True), cap)
    scale_tuned = clip_after(convex_envelope(tuned_pool, log_compute=True), cap)
    recipe_colors = quantity_colors(("k1_fixed", "k1_tuned", "scale_fixed", "scale_tuned"), quantity="strategy")
    for name, points, label in (("k1_fixed", k1_fixed, "Operator-1 · 0.8"), ("k1_tuned", k1_tuned, "Operator-1 · tuned"),
                                ("scale_fixed", scale_fixed, "Scale loops · 0.8"), ("scale_tuned", scale_tuned, "Scale loops · tuned")):
        recipe_axis.plot([p.compute for p in points], [p.loss for p in points], "o-", color=recipe_colors[name], markeredgewidth=0,
                         label=label)
    compute_axis(recipe_axis, max(p.compute for p in scale_fixed + scale_tuned) * 1.3)
    recipe_axis.set_ylabel("Loss")
    recipe_axis.set_title("Scaling recipes")
    low, high = recipe_axis.get_ylim()
    recipe_axis.set_ylim(low, high + (high - low) * 0.85)
    recipe_axis.legend(title="Strategy · WD", **legend)
    recipe_axis.xaxis.set_major_locator(FixedLocator((1e18, 1e19)))

    plot_heatmap(wd_axis, best_weight_decays(cells, LOOPS, HEATMAP_DEPTHS), LOOPS, HEATMAP_DEPTHS, text_size=MAIN_SMALL_FONT)
    wd_axis.set_title("Optimal WD")

    for axis in (cut_axis, gain_axis):
        # The axis label already identifies loop count; omit repeated K prefixes.
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))

    for axis in (cut_axis, gain_axis, recipe_axis):
        style_axis(axis)
    for axis in (cut_axis, gain_axis, recipe_axis, wd_axis):
        axis.set_box_aspect(1)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.20, top=0.88, wspace=0.6)
    save(fig, args.output)
    print("  Operator-1 weight decay per depth: " + ", ".join(f"d{p.depth} {p.weight_decay:g}" for p in k1_tuned))
    print("  Scale Loop Count, WD 0.8: " + ", ".join(f"K{p.loops} d{p.depth}" for p in scale_fixed))
    print("  Scale Loop Count, tuned WD: " + ", ".join(f"K{p.loops} d{p.depth}" for p in scale_tuned))


if __name__ == "__main__":
    main()
