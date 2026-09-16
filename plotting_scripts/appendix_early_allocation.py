"""Figure 10d token allocation coordinates and ablation; fits recomputed from measured inputs."""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from common import sequence_colors, save_figure
from appendix_style import INK, MUTED, FULL_WIDTH, OUTER_PAD, legend_row_count, make_grid, style_axis
from appendix_early_data import load_data

import appendix_early_ablation as ablation
ARMS = (SimpleNamespace(key="k2", title="Loop-2"), SimpleNamespace(key="dep", title="Untied-2"))
BUDGET_COLOR = dict(zip(range(8, 13), sequence_colors(5, quantity="compute")))

def load_prescriptions(arm):
    return load_data("token_allocation")[arm.key]["growth"]


def load_controls(arm):
    return load_data("token_allocation")[arm.key]["controls"]


METRIC = "six_t_squared_over_c"


def quadratic(series: list[dict]):
    if len(series) < 3:
        return None
    x = np.asarray([row[METRIC] for row in series], dtype=float)
    y = np.asarray([row["loss"] for row in series], dtype=float)
    log_x = np.log(x)
    log_y = np.log(y)
    coefficients = np.polyfit(log_x, log_y, 2)
    a, b, _ = coefficients
    if a <= 0:
        return None
    log_x_star = float(-b / (2 * a))
    if not log_x.min() <= log_x_star <= log_x.max():
        return None
    x_star = float(np.exp(log_x_star))
    loss_star = float(np.exp(np.polyval(coefficients, log_x_star)))
    return coefficients, x_star, loss_star


def draw(axis: plt.Axes, arm) -> tuple[float, float]:
    rows = load_prescriptions(arm)
    controls = load_controls(arm)
    if len(rows) != 30:
        raise ValueError(f"expected 30 completed {arm.key} prescription cells; found {len(rows)}")

    growth_optima = []
    fixed_optima = []
    for budget, color in BUDGET_COLOR.items():
        reference = [row for row in controls if row["budget"] == budget]
        reference_fit = quadratic(reference)
        if reference_fit is not None:
            fixed_optima.append(reference_fit[1])
        axis.plot(
            [row[METRIC] for row in reference],
            [row["loss"] for row in reference],
            color=color, linestyle=":", linewidth=1.0, alpha=0.25, zorder=1,
        )
        axis.scatter(
            [row[METRIC] for row in reference],
            [row["loss"] for row in reference],
            s=24, marker="o", facecolors="none", edgecolors=color,
            linewidth=1.0, alpha=0.32, zorder=2,
        )

        series = [row for row in rows if row["budget"] == budget]
        fit = quadratic(series)
        if fit is not None:
            coefficients, optimum, loss_star = fit
            grid = np.geomspace(series[0][METRIC], series[-1][METRIC], 200)
            axis.plot(
                grid, np.exp(np.polyval(coefficients, np.log(grid))),
                color=color, linewidth=1.25, alpha=0.9, zorder=3,
            )
            axis.scatter(
                [optimum], [loss_star], marker="*", s=70, color=color,
                edgecolor="white", linewidth=0.5, zorder=5,
            )
            growth_optima.append(optimum)
        axis.scatter(
            [row[METRIC] for row in series],
            [row["loss"] for row in series],
            s=32, marker="o", color=color, edgecolor="white",
            linewidth=0.5, zorder=4,
        )

    mean_growth = float(np.mean(growth_optima))
    mean_fixed = float(np.mean(fixed_optima))
    axis.axvline(mean_fixed, color=MUTED, linestyle="--", linewidth=0.9, alpha=0.35, zorder=1)
    axis.text(
        mean_fixed, 0.87, f"Fixed {mean_fixed:.2f}",
        transform=axis.get_xaxis_transform(), ha="center", va="top",
        fontsize=matplotlib.rcParams["legend.fontsize"], color=MUTED, alpha=0.75,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
    )
    axis.axvline(mean_growth, color=INK, linestyle="--", linewidth=1.05, zorder=2)
    axis.text(
        mean_growth, 0.98, f"Growth {mean_growth:.2f}",
        transform=axis.get_xaxis_transform(), ha="center", va="top",
        fontsize=matplotlib.rcParams["legend.fontsize"], color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )
    style_axis(axis)
    axis.set_xscale("log")
    return mean_growth, mean_fixed


def main() -> None:
    budget_handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=color,
               markerfacecolor=color, label=rf"$d={budget}$")
        for budget, color in BUDGET_COLOR.items()
    ]
    control_handle = Line2D(
        [0], [0], marker="o", linestyle=":", color=MUTED,
        markerfacecolor="none", alpha=0.45, label="Fixed controls",
    )
    fit_handle = Line2D(
        [0], [0], marker="*", linestyle="-", color=MUTED,
        markerfacecolor=MUTED, label="Fit + optimum",
    )
    handles = [*budget_handles, control_handle, fit_handle]
    labels = [handle.get_label() for handle in handles]
    ablation_handles = []
    for key in ablation.LADDERS[1:]:
        color, marker, _, label = ablation.ladder_style(key)
        ablation_handles.append(Line2D(
            [0], [0], color=color, marker=marker,
            markerfacecolor="white" if key.endswith("rho0p30") else color,
            label=label,
        ))
    # Reserve independent legends over the sweep pair and the ablation panel.
    legend_rows = max(legend_row_count(labels, width=FULL_WIDTH * 2 / 3), 3)
    figure, axes = make_grid(
        1,
        3,
        titles=[ARMS[0].title, ARMS[1].title, "Untied Grow"],
        legend_rows=legend_rows,
    )
    sweep_axes = axes[:2]
    optima = [draw(axis, arm) for axis, arm in zip(sweep_axes, ARMS[:2])]
    for axis in sweep_axes:
        axis.set_xlabel(r"$6T^2/C$")
        axis.set_ylabel("Loss")
    ablation.draw_compute_multiplier(axes[2], ablation.load())
    axes[2].set_ylabel("Compute Multiplier")

    y_min = min(axis.get_ylim()[0] for axis in sweep_axes)
    y_max = max(axis.get_ylim()[1] for axis in sweep_axes)
    tick_step = 0.05
    loss_ticks = np.arange(
        np.ceil(y_min / tick_step) * tick_step,
        np.floor(y_max / tick_step) * tick_step + tick_step / 2,
        tick_step,
    )
    for axis in sweep_axes:
        axis.set_ylim(y_min, y_max)
        axis.yaxis.set_major_locator(FixedLocator(loss_ticks))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    for axis in axes:
        x_min, x_max = axis.get_xlim()
        visible_ticks = [label for label in axis.get_xticklabels()
                         if x_min <= label.get_position()[0] <= x_max]
        visible_ticks[0].set_ha("left")
        visible_ticks[-1].set_ha("right")
    legend_y = 1 - OUTER_PAD / figure.get_figheight()
    figure.legend(handles, labels, loc="upper center", ncols=3,
                  bbox_to_anchor=((axes[0].get_position().x0 + axes[1].get_position().x1) / 2, legend_y))
    figure.legend(ablation_handles, [handle.get_label() for handle in ablation_handles],
                  loc="upper center", ncols=1,
                  bbox_to_anchor=((axes[2].get_position().x0 + axes[2].get_position().x1) / 2, legend_y))
    save_figure(figure, 10, panel="d")
