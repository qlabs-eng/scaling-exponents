"""Measured Untied Grow token and growth-fraction ablation for Figure 10d."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import sequence_colors
from appendix_style import MUTED, style_axis
from appendix_early_data import load_data


TPP8_COLOR = sequence_colors(2, quantity="tokens_per_parameter")[1]


TPP6_COLOR = sequence_colors(2, quantity="tokens_per_parameter")[0]


def ladder_style(ladder: str) -> tuple[str, str, str, str]:
    tpp = 8 if ladder.startswith("tpp8") else 6
    fitted = ladder.endswith("fitrho")
    color = TPP8_COLOR if tpp == 8 else TPP6_COLOR
    marker = "o" if fitted else "s"
    linestyle = "-" if fitted else "--"
    label = rf"TPP {tpp}, " + (r"fitted $\rho$" if fitted else r"$\rho=0.30$")
    return color, marker, linestyle, label


LADDERS = (
    "tpp8_fitrho",
    "tpp8_rho0p30",
    "tpp6_fitrho",
    "tpp6_rho0p30",
)


DEPTHS = (6, 8, 10, 12, 14, 16)


def interpolate_compute(series: list[dict], target_loss: float) -> float:
    """Estimate compute at target loss from the two nearest log-loss points."""
    target_log_loss = np.log(target_loss)
    nearest = sorted(
        series,
        key=lambda point: abs(np.log(float(point["validation_loss"]))
                              - target_log_loss),
    )[:2]
    if len(nearest) != 2:
        raise ValueError("compute interpolation requires two ladder points")
    log_loss = np.log([float(point["validation_loss"]) for point in nearest])
    log_compute = np.log([float(point["scheduled_matrix_flops"])
                          for point in nearest])
    if log_loss[0] == log_loss[1]:
        raise ValueError("compute interpolation points have identical loss")
    estimated_log_compute = (
        log_compute[0]
        + (target_log_loss - log_loss[0])
        * (log_compute[1] - log_compute[0])
        / (log_loss[1] - log_loss[0])
    )
    return float(np.exp(estimated_log_compute))


def compute_multipliers(points: dict[tuple[str, int], dict]) -> tuple[list[dict], dict[str, np.ndarray]]:
    baseline = sorted(
        (point for (ladder, _), point in points.items()
         if ladder == "tpp8_fitrho" and int(point["depth"]) in DEPTHS),
        key=lambda point: int(point["depth"]),
    )
    multipliers = {}
    for ladder in LADDERS[1:]:
        series = sorted(
            (point for (key, _), point in points.items() if key == ladder),
            key=lambda point: float(point["scheduled_matrix_flops"]),
        )
        multipliers[ladder] = np.asarray([
            float(base["scheduled_matrix_flops"])
            / interpolate_compute(series, float(base["validation_loss"]))
            for base in baseline
        ])
    return baseline, multipliers


def style(axis: plt.Axes) -> None:
    style_axis(axis)


def draw_compute_multiplier(axis: plt.Axes, points: dict) -> None:
    baseline, multipliers = compute_multipliers(points)
    x = np.asarray([float(point["scheduled_matrix_flops"])
                    for point in baseline])
    for ladder in LADDERS[1:]:
        color, marker, _linestyle, label = ladder_style(ladder)
        fixed = ladder.endswith("rho0p30")
        axis.plot(
            x, multipliers[ladder], color=color, marker=marker,
            linestyle="-", linewidth=1.25, markersize=4.2,
            markerfacecolor="white" if fixed else color,
            markeredgecolor=color, markeredgewidth=0.9, label=label, zorder=3,
        )
    axis.axhline(1.0, color=MUTED, linestyle=":", linewidth=0.9)
    axis.set_xscale("log")
    axis.set_ylabel(r"$C_{\mathrm{base}}/C$")
    all_multipliers = np.concatenate(list(multipliers.values()))
    axis.set_ylim(all_multipliers.min() - 0.005,
                             all_multipliers.max() + 0.025)
    style(axis)
    axis.set_xlabel("Compute (FLOPs)")


def load():
    return {(p["ladder"], p["depth"]): p for p in load_data("ablation")}
