#!/usr/bin/env python3
"""Compute-optimal scaling ladders on FineWeb: loss with power-law fits, compute multiplier over Vanilla, and the fitted coefficients.

    python plotting_scripts/figure3.py                               # bundled paper measurements
    python plotting_scripts/figure3.py --csv                          # the bundled numbers, data/fineweb_ladders.csv
    python plotting_scripts/figure3.py --runs-dir runs                # your own ladder_scripts/ runs, from their result.json files
    python plotting_scripts/figure3.py --wandb-entity YOU --wandb-project PROJECT --wandb-group scaling_ladders
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common import (ARM_LABEL, C_REF, FIGURES, MAIN_RCPARAMS, MAIN_FIGURE_WIDTH, MAIN_SMALL_FONT, add_source_arguments, fit_power_law, fit_power_law_fixed_irreducible,
                    ladders_from, load_ladders_runs, plot_multipliers, save, style_axis, architecture_colors, zoom_inset)

ARMS = ("vanilla", "deep_vanilla", "deep_vanilla_grow", "operator_1", "loop_2", "loop_grow", "untied_2", "untied_grow")  # legend order
ZOOM_FROM = 2e19  # the inset covers the largest budgets, where the ladders separate most


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_source_arguments(parser)
    parser.add_argument("--runs-dir", type=Path, help="directory of ladder_scripts/ runs to plot instead of W&B")
    parser.add_argument("--output", type=Path, default=FIGURES / "figure3.png")
    args = parser.parse_args()
    ladders = load_ladders_runs(args.runs_dir) if args.runs_dir else ladders_from(args)
    arms = tuple(arm for arm in ARMS if arm in ladders)
    colors = architecture_colors(ARMS)

    # Vanilla's irreducible loss E comes from a fit with E free; every arm then gets A and gamma with that E held fixed.
    vanilla = ladders["vanilla"]
    irreducible = fit_power_law([p.compute for p in vanilla], [p.loss for p in vanilla]).irreducible
    fits = {arm: fit_power_law_fixed_irreducible([p.compute for p in ladders[arm]], [p.loss for p in ladders[arm]], irreducible) for arm in arms}

    plt.rcParams.update(MAIN_RCPARAMS)
    fig, (loss_axis, multiplier_axis, coefficient_axis) = plt.subplots(1, 3, figsize=(MAIN_FIGURE_WIDTH, 5))

    # Loss: fitted curves through the measured points, exponents in the legend, inset on the largest budgets.
    for arm in arms:
        compute = np.array([p.compute for p in ladders[arm]])
        grid = np.geomspace(compute.min(), compute.max(), 300)
        loss_axis.plot(grid, fits[arm](grid), color=colors[arm],
                       label=rf"{ARM_LABEL[arm]}  ($\gamma = {fits[arm].exponent:.3f}$)")
        loss_axis.plot(compute, [p.loss for p in ladders[arm]], "o", color=colors[arm], markeredgewidth=0, zorder=3)
    loss_axis.set_xscale("log")
    loss_axis.set_title("Loss")
    loss_axis.set_ylabel("Loss")
    # One shared legend keeps the larger text clear of both curves and coefficients.
    handles, labels = loss_axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4,
               frameon=False, labelspacing=0.4, columnspacing=0.8, handlelength=1.2, handletextpad=0.4)
    zoomed = [p for arm in arms for p in ladders[arm] if p.compute >= ZOOM_FROM]
    inset = zoom_inset(loss_axis, [0.60, 0.60, 0.38, 0.38], (ZOOM_FROM, max(p.compute for p in zoomed) * 1.15),
                       (min(p.loss for p in zoomed) - 0.01, max(p.loss for p in zoomed) + 0.01), (3e19, 1e20), labelsize=MAIN_SMALL_FONT)
    for arm in arms:
        points = [p for p in ladders[arm] if p.compute >= ZOOM_FROM]
        if not points:
            continue
        grid = np.geomspace(ZOOM_FROM, points[-1].compute, 100)
        inset.plot(grid, fits[arm](grid), color=colors[arm], linewidth=1.8)
        inset.plot([p.compute for p in points], [p.loss for p in points], "o", color=colors[arm], markersize=5.5, markeredgewidth=0, zorder=3)

    final = plot_multipliers(multiplier_axis, ladders, arms, colors)
    multiplier_axis.set_title("Compute Multiplier")

    # Scaling coefficients: gamma against log A, the fitted line log(L - E) = -gamma log(C / C0) + log A with C0 = Vanilla d8's compute.
    reference = next(p.compute for p in vanilla if p.depth == 8)
    for arm in arms:
        log_a = np.log(fits[arm].amplitude) - fits[arm].exponent * np.log(reference / C_REF)
        coefficient_axis.plot(log_a, fits[arm].exponent, "o", color=colors[arm], markersize=8.5, markeredgewidth=0, label=ARM_LABEL[arm], zorder=3)
    coefficient_axis.margins(x=0.14, y=0.16)
    coefficient_axis.invert_xaxis()  # lower loss at C0 to the right
    coefficient_axis.xaxis.set_major_locator(MultipleLocator(0.01))
    coefficient_axis.set_title("Scaling coefficients")
    coefficient_axis.set_xlabel(r"$\log A$ (Constant)")
    coefficient_axis.set_ylabel(r"$\gamma$ (Exponent)")
    coefficient_axis.text(0.97, 0.05, "$\\log(L - E) =$\n$-\\gamma \\log(C / C_0) + \\log A$", transform=coefficient_axis.transAxes,
                          ha="right", va="bottom", fontsize=MAIN_SMALL_FONT)

    for axis in (loss_axis, multiplier_axis):
        axis.set_xlabel("Compute (FLOPs)")
    for axis in (loss_axis, multiplier_axis, coefficient_axis):
        style_axis(axis)
        axis.set_box_aspect(1)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.14, top=0.80, wspace=0.48)
    save(fig, args.output)
    print(f"Vanilla irreducible loss E = {irreducible:.4f}")
    for arm in arms:
        print(f"  {ARM_LABEL[arm]:18s} gamma = {fits[arm].exponent:.4f} ± {fits[arm].exponent_stderr:.4f}   final multiplier {final[arm]:.3f}×")


if __name__ == "__main__":
    main()
