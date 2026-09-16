"""Initial stored-parameter token allocation for Figure 7."""
from __future__ import annotations

from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from appendix_style import INK, MUTED, ordered_colors, style_axis
from appendix_early_data import load_data

ARMS = (SimpleNamespace(key="k2", title="Loop-2"),)

def load_prescription(arm, basis):
    if arm.key != "k2" or basis != "initial":
        raise ValueError("Figure 7 uses initial Loop-2 stored TPP")
    return load_data("stored_tpp")["growth"]


def load_controls(arm, basis):
    return load_data("stored_tpp")["controls"]


def style(axis: plt.Axes) -> None:
    style_axis(axis)
    axis.set_xscale("log")


BUDGETS = (8, 9, 10, 11, 12)


BUDGET_COLOR = dict(zip(BUDGETS, ordered_colors(len(BUDGETS), quantity="compute")))


def quadratic(series: list[dict]):
    if len(series) < 3:
        return None
    x = np.asarray([row["tpp"] for row in series], dtype=float)
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


def draw(axis: plt.Axes, arm, basis: str) -> tuple[float, float]:
    rows = load_prescription(arm, basis)
    controls = load_controls(arm, basis)
    if len(rows) != 30:
        raise ValueError(f"expected 30 completed {arm.key} prescription cells; found {len(rows)}")

    optima = []
    original_optima = []
    for budget, color in BUDGET_COLOR.items():
        reference = [row for row in controls if row["budget"] == budget]
        reference_fit = quadratic(reference)
        if reference_fit is not None:
            original_optima.append(reference_fit[1])
        axis.plot(
            [row["tpp"] for row in reference],
            [row["loss"] for row in reference],
            color=color, linestyle=":", linewidth=1.0, alpha=0.25, zorder=1,
        )
        axis.scatter(
            [row["tpp"] for row in reference],
            [row["loss"] for row in reference],
            s=24, marker="o", facecolors="none", edgecolors=color,
            linewidth=1.0, alpha=0.32, zorder=2,
        )

        series = [row for row in rows if row["budget"] == budget]
        fit = quadratic(series)
        if fit is not None:
            coefficients, tpp_star, loss_star = fit
            grid = np.geomspace(series[0]["tpp"], series[-1]["tpp"], 200)
            axis.plot(
                grid, np.exp(np.polyval(coefficients, np.log(grid))),
                color=color, linewidth=1.25, alpha=0.9, zorder=3,
            )
            axis.scatter(
                [tpp_star], [loss_star], marker="*", s=70, color=color,
                edgecolor="white", linewidth=0.5, zorder=5,
            )
            optima.append(tpp_star)
        axis.scatter(
            [row["tpp"] for row in series],
            [row["loss"] for row in series],
            s=32, marker="o", color=color, edgecolor="white",
            linewidth=0.5, zorder=4,
        )

    mean_optimum = float(np.mean(optima))
    mean_original_optimum = float(np.mean(original_optima))
    axis.axvline(
        mean_original_optimum, color=MUTED, linestyle="--", linewidth=0.9,
        alpha=0.35, zorder=1,
    )
    axis.text(
        mean_original_optimum, 0.87,
        rf"Fixed {mean_original_optimum:.2f}",
        transform=axis.get_xaxis_transform(), ha="center", va="top",
        fontsize=matplotlib.rcParams["legend.fontsize"], color=MUTED, alpha=0.75,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
    )
    axis.axvline(mean_optimum, color=INK, linestyle="--", linewidth=1.05, zorder=2)
    axis.text(
        mean_optimum, 0.98,
        rf"Growth {mean_optimum:.2f}",
        transform=axis.get_xaxis_transform(), ha="center", va="top",
        fontsize=matplotlib.rcParams["legend.fontsize"], color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )
    style(axis)
    return mean_optimum, mean_original_optimum
