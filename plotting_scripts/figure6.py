#!/usr/bin/env python3
"""Logit-KL effective depth: the first block after the KL peak whose decoded stream is within 2 nats of the final prediction. Left, the FineWeb
compute-optimal ladders; right, Operator-1 with tuned weight decay against scaling the loop count on the 100M-token x 10-epoch grid.

    python plotting_scripts/figure6.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import ARM_LABEL, DATA, FIGURES, MAIN_RCPARAMS, MAIN_FIGURE_WIDTH, flop_tick, quantity_colors, save, style_axis, architecture_colors

ARMS = ("vanilla", "deep_vanilla", "operator_1", "loop_2", "loop_grow", "untied_2", "untied_grow")  # legend order
STRATEGIES = ("Operator-1", "Scale Loop Count")  # the two ends of the magma loop-count scale of the single-vs-multi-epoch figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=FIGURES / "figure6.png")
    args = parser.parse_args()
    data = json.loads((DATA / "effective_depth.json").read_text())
    colors = architecture_colors(ARMS)
    strategy_colors = quantity_colors(STRATEGIES, quantity="recurrence")
    legend = dict(loc="upper left", fontsize=11, labelspacing=0.35, borderaxespad=0.6, handlelength=1.8, handletextpad=0.5)

    plt.rcParams.update(MAIN_RCPARAMS)
    fig, (ladder_axis, strategy_axis) = plt.subplots(1, 2, figsize=(MAIN_FIGURE_WIDTH, 5.6))
    for arm in ARMS:
        rows = data["ladders"][arm]
        ladder_axis.plot([r["compute_flops"] for r in rows], [r["effective_depth"] for r in rows], "o-", color=colors[arm], markeredgewidth=0, label=ARM_LABEL[arm], zorder=3)
    ladder_axis.set_xscale("log")
    ladder_axis.legend(**legend)

    multi = data["multi_epoch"]
    flops = multi["compute_flops"]
    strategy_axis.plot(flops, multi["operator_1_tuned_wd"], "o-", color=strategy_colors["Operator-1"], markeredgewidth=0, label="Operator-1", zorder=4)
    strategy_axis.plot(flops, multi["scale_loop_count"], "o-", color=strategy_colors["Scale Loop Count"], markeredgewidth=0, label="Scale Loop Count", zorder=3)
    strategy_axis.set_xscale("log")
    strategy_axis.set_xlim(flops[0] / 1.25, flops[-1] * 1.25)
    strategy_axis.xaxis.set_major_locator(FixedLocator((1e18, 5e18)))
    strategy_axis.xaxis.set_major_formatter(FuncFormatter(flop_tick))
    strategy_axis.xaxis.set_minor_formatter(NullFormatter())
    strategy_axis.legend(**legend)

    for axis, title in ((ladder_axis, "Single Epoch"), (strategy_axis, "Multi-Epoch")):
        axis.set_title(title)
        axis.set_xlabel("Compute (FLOPs)")
        axis.set_ylabel(r"$L_{\mathrm{eff}}^{\mathrm{KL}}$")
        low, high = axis.get_ylim()
        axis.set_ylim(low, low + (high - low) * 1.12)
        style_axis(axis)
        axis.set_box_aspect(0.82)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.91, wspace=0.22)
    save(fig, args.output)
    print("  Scale Loop Count: " + ", ".join(f"{c / 1e18:.2f}e18 {d:.1f}" for c, d in zip(flops, multi["scale_loop_count"])))


if __name__ == "__main__":
    main()
