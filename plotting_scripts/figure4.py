#!/usr/bin/env python3
"""Untied-Grow against Vanilla on FineWeb-Edu, extrapolated to the 7.4B d26 run: loss, CORE NLL, and CORE accuracy against compute.
The CORE evaluation is not part of this release, so this figure renders from the bundled numbers.

    python plotting_scripts/figure4.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import brentq, least_squares

from common import C_REF, DATA, FIGURES, INK, MAIN_RCPARAMS, MAIN_FIGURE_WIDTH, MAIN_SMALL_FONT, fit_power_law, fit_power_law_fixed_irreducible, huber_scale, save, style_axis, architecture_colors, zoom_inset

ARMS = ("vanilla", "untied_grow")
LABEL = {"vanilla": "Vanilla", "untied_grow": "Untied-Grow"}
GPT3_TARGETS = (("2.7B", 0.329), ("6.7B", 0.361), ("13B", 0.385), ("175B", 0.427))  # GPT-3 CORE scores


def task_power_law(compute, floor, scale, exponent):
    return floor + scale * (np.asarray(compute, dtype=float) / C_REF) ** (-exponent)


def fit_task_power_law(compute, nll):
    """Bounded Huber fit of a task's NLL law E + A (C / C_REF)^-gamma from several starting exponents."""
    compute = np.asarray(compute, dtype=float)
    nll = np.asarray(nll, dtype=float)

    def residual(params):
        return task_power_law(compute, *params) - nll

    bounds = ([0.0, 0.0, 0.005], [nll.min() - 1e-8, 50.0, 2.0])
    starts = []
    for exponent in (0.05, 0.1, 0.2, 0.4, 0.8):
        floor = 0.5 * nll.min()
        scale = max(0.01, (nll.max() - floor) * (compute.min() / C_REF) ** exponent)
        starts.append(least_squares(residual, [floor, scale, exponent], bounds=bounds, max_nfev=30_000))
    ordinary = min(starts, key=lambda fit: np.sum(residual(fit.x) ** 2))
    return least_squares(residual, ordinary.x, bounds=bounds, loss="huber", f_scale=huber_scale(residual(ordinary.x), 0.002), max_nfev=30_000).x


def sigmoid(nll, slope, intercept):
    return 1.0 / (1.0 + np.exp(np.clip(slope * np.asarray(nll, dtype=float) - intercept, -700, 700)))


def fit_task_sigmoid(nll, accuracy):
    """Bounded Huber fit of a task's accuracy-from-NLL sigmoid."""
    nll = np.asarray(nll, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)

    def residual(params):
        return sigmoid(nll, *params) - accuracy

    mean = float(np.clip(accuracy.mean(), 0.001, 0.999))
    bounds = ([0.0, -20.0], [20.0, 20.0])
    ordinary = least_squares(residual, [1.0, np.log(mean / (1.0 - mean)) + np.median(nll)], bounds=bounds, max_nfev=30_000)
    return least_squares(residual, ordinary.x, bounds=bounds, loss="huber", f_scale=huber_scale(residual(ordinary.x), 0.005), max_nfev=30_000).x


class CoreModel:
    """CORE accuracy from compute: a per-arm NLL power law per task, a per-task sigmoid from NLL to accuracy shared by both arms,
    the mean over the selected tasks, and one fitted ratio from that mean to the full CORE score."""

    def __init__(self, data: dict):
        self.tasks = data["tasks"]
        records = data["arms"]
        self.nll_fits = {arm: {task: fit_task_power_law([r["compute_flops"] for r in runs], [r["task_nll"][task] for r in runs])
                               for task in self.tasks} for arm, runs in records.items()}
        pooled = [r for runs in records.values() for r in runs]
        self.sigmoids = {task: fit_task_sigmoid([r["task_nll"][task] for r in pooled], [r["task_accuracy"][task] for r in pooled]) for task in self.tasks}
        filtered = np.array([np.mean([r["task_accuracy"][task] for task in self.tasks]) for r in pooled])
        full = np.array([r["core_accuracy"] for r in pooled])
        self.ratio = float(filtered @ full / (filtered @ filtered))

    def predict(self, arm: str, compute):
        return self.ratio * np.mean([sigmoid(task_power_law(compute, *self.nll_fits[arm][task]), *self.sigmoids[task]) for task in self.tasks], axis=0)

    def compute_for(self, arm: str, target: float) -> float:
        return 10 ** brentq(lambda log_compute: float(self.predict(arm, 10 ** log_compute)) - target, 15.0, 40.0)


def mark_d26(axis, compute, forecast, observed, color, std=None) -> None:
    """Hollow square at the d26 run's observed value, joined to the fit's forecast by a dotted line."""
    axis.plot([compute, compute], [forecast, observed], ":", color=color, alpha=0.8)
    axis.errorbar(compute, observed, yerr=std, linestyle="none", marker="s", markerfacecolor="white", markeredgecolor=color,
                  markeredgewidth=1.8, color=color, capsize=2.0, zorder=5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=FIGURES / "figure4.png")
    args = parser.parse_args()
    data = json.loads((DATA / "fineweb_edu_ladders.json").read_text())
    d26 = data["d26"]
    c26 = d26["compute_flops"]
    series = {arm: {key: np.array([r[key] for r in data["arms"][arm]]) for key in ("compute_flops", "val_loss", "core_nll", "core_accuracy")} for arm in ARMS}

    # Loss: Vanilla's E from a fit with E free, shared with Untied-Grow as in the ladder figure. CORE NLL: E free per arm.
    loss_fits = {"vanilla": fit_power_law(series["vanilla"]["compute_flops"], series["vanilla"]["val_loss"])}
    loss_fits["untied_grow"] = fit_power_law_fixed_irreducible(series["untied_grow"]["compute_flops"], series["untied_grow"]["val_loss"], loss_fits["vanilla"].irreducible)
    nll_fits = {arm: fit_power_law(series[arm]["compute_flops"], series[arm]["core_nll"]) for arm in ARMS}
    core = CoreModel(data)
    targets = {arm: {label: core.compute_for(arm, score) for label, score in GPT3_TARGETS} for arm in ARMS}
    plot_max = max(c26, max(compute for arm in targets.values() for compute in arm.values()))
    plot_min = min(series[arm]["compute_flops"].min() for arm in ARMS)
    colors = architecture_colors(ARMS)

    plt.rcParams.update(MAIN_RCPARAMS)
    fig, (loss_axis, nll_axis, core_axis) = plt.subplots(1, 3, figsize=(MAIN_FIGURE_WIDTH, 4.5))

    # Loss and CORE NLL: solid fits over the measured ladders, dashed beyond them, the d26 run against its forecast.
    for axis, metric, fits in ((loss_axis, "val_loss", loss_fits), (nll_axis, "core_nll", nll_fits)):
        for arm in ARMS:
            compute = series[arm]["compute_flops"]
            measured = np.geomspace(compute.min(), compute.max(), 200)
            extrapolated = np.geomspace(compute.max(), plot_max, 300)
            axis.plot(measured, fits[arm](measured), color=colors[arm])
            axis.plot(extrapolated, fits[arm](extrapolated), "--", color=colors[arm])
            axis.plot(compute, series[arm][metric], "o", color=colors[arm], markeredgewidth=0, zorder=3)
        mark_d26(axis, c26, float(fits["untied_grow"](c26)), d26[metric], colors["untied_grow"])
        lower = min(min(float(fits[arm](plot_max)) for arm in ARMS), d26[metric])
        upper = max(series[arm][metric].max() for arm in ARMS)
        axis.set_ylim(lower - 0.10, upper + 0.16)

    # Zoom on the d26 run, with the compute-efficiency gain at its budget: Vanilla's fitted compute for Untied-Grow's fitted loss there.
    zoom_x = (c26 / 2.2, c26 * 3.4)
    zoom_y = (min(d26["val_loss"], float(loss_fits["untied_grow"](zoom_x[1]))) - 0.02, float(loss_fits["vanilla"](zoom_x[0])) + 0.02)
    inset = zoom_inset(loss_axis, [0.50, 0.50, 0.47, 0.46], zoom_x, zoom_y, (1e21, 3e21), labelsize=MAIN_SMALL_FONT)
    grid = np.geomspace(*zoom_x, 200)
    for arm in ARMS:
        inset.plot(grid, loss_fits[arm](grid), "--", color=colors[arm], linewidth=1.8)
    forecast = float(loss_fits["untied_grow"](c26))
    mark_d26(inset, c26, forecast, d26["val_loss"], colors["untied_grow"])
    vanilla_compute = brentq(lambda compute: float(loss_fits["vanilla"](compute)) - forecast, c26, c26 * 1e6)
    inset.plot(c26, forecast, "x", color=colors["untied_grow"], markeredgewidth=1.4, zorder=5)
    inset.annotate("", xy=(vanilla_compute, forecast), xytext=(c26, forecast), arrowprops={"arrowstyle": "<->", "color": INK, "linewidth": 0.9}, zorder=4)
    inset.text(0.96, 0.93, f"{vanilla_compute / c26:.1f}× compute\nefficiency gain", transform=inset.transAxes, ha="right", va="top", fontsize=MAIN_SMALL_FONT, color=INK)

    # CORE accuracy: the taskwise model, GPT-3 scores as horizontal lines with each arm's crossing starred and the compute ratio between them.
    grid = np.geomspace(plot_min, plot_max, 700)
    for arm in ARMS:
        measured_max = series[arm]["compute_flops"].max()
        core_axis.plot(grid[grid <= measured_max], core.predict(arm, grid[grid <= measured_max]), color=colors[arm])
        core_axis.plot(grid[grid >= measured_max], core.predict(arm, grid[grid >= measured_max]), "--", color=colors[arm])
        core_axis.plot(series[arm]["compute_flops"], series[arm]["core_accuracy"], "o", color=colors[arm], markeredgewidth=0, zorder=3)
        for label, score in GPT3_TARGETS:
            core_axis.plot(targets[arm][label], score, "*", color=colors[arm], markersize=11, markeredgecolor="white", markeredgewidth=0.5, zorder=4)
    text_box = {"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.9}
    for label, score in GPT3_TARGETS:
        core_axis.axhline(score, color=INK, linewidth=0.75, linestyle="--", alpha=0.65, zorder=1)
        core_axis.text(0.02, score, label, transform=core_axis.get_yaxis_transform(), ha="left", va="center", fontsize=MAIN_SMALL_FONT, color=INK, bbox=text_box, zorder=6)
        core_axis.text(0.35, score, f"{targets['vanilla'][label] / targets['untied_grow'][label]:.1f}×", transform=core_axis.get_yaxis_transform(),
                       ha="center", va="center", fontsize=MAIN_SMALL_FONT, color=INK, bbox=text_box, zorder=7)
    mark_d26(core_axis, c26, float(core.predict("untied_grow", c26)), d26["core_accuracy"], colors["untied_grow"], d26["core_accuracy_std"])
    upper = max(d26["core_accuracy"], max(series[arm]["core_accuracy"].max() for arm in ARMS), max(float(core.predict(arm, plot_max)) for arm in ARMS), GPT3_TARGETS[-1][1])
    core_axis.set_ylim(0.045, upper + 0.04)

    for axis, title in ((loss_axis, "Loss"), (nll_axis, "CORE NLL"), (core_axis, "CORE accuracy")):
        axis.set_xscale("log")
        axis.set_xlim(plot_min / 1.35, plot_max * 1.08)
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.set_xlabel("Compute (FLOPs)")
        style_axis(axis)
        axis.set_box_aspect(1)
    handles = [Line2D([], [], color=colors[arm], marker="o", label=LABEL[arm]) for arm in ARMS]
    handles.append(Line2D([], [], linestyle="none", marker="s", markerfacecolor="white", markeredgecolor=colors["untied_grow"], markeredgewidth=1.8,
                          label=f"d26 ({d26['parameters'] / 1e9:.1f}B)"))
    loss_axis.legend(handles=handles, loc="lower left", labelspacing=0.35, borderaxespad=0.6, handlelength=1.8, handletextpad=0.5)
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.9, wspace=0.38)
    save(fig, args.output)
    print(f"  shared irreducible loss E = {loss_fits['vanilla'].irreducible:.4f}; gamma Vanilla {loss_fits['vanilla'].exponent:.4f}, Untied-Grow {loss_fits['untied_grow'].exponent:.4f}")
    print(f"  compute-efficiency gain at the d26 budget: {vanilla_compute / c26:.2f}×")
    for label, score in GPT3_TARGETS:
        print(f"  GPT-3 {label} CORE {score:.3f}: Vanilla {targets['vanilla'][label]:.3e}  Untied-Grow {targets['untied_grow'][label]:.3e}  ratio {targets['vanilla'][label] / targets['untied_grow'][label]:.2f}×")
    print(f"  d26 forecasts: loss {forecast:.4f} (observed {d26['val_loss']:.4f}), CORE NLL {float(nll_fits['untied_grow'](c26)):.4f} (observed {d26['core_nll']:.4f}), "
          f"CORE {float(core.predict('untied_grow', c26)):.4f} (observed {d26['core_accuracy']:.4f})")


if __name__ == "__main__":
    main()
