"""Figure 13: regenerate offline from the bundled paper measurements."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from common import save_figure
from appendix_style import INK, add_top_legend, apply_style, legend_row_count, make_grid, ordered_colors

from appendix_early_growth import DEPTHS, DEPTH_MARKER, expected_cells, load_deepvan_rows, load_rows, quadratic, series_for

ARMS = ("deepvan", "k2", "dep")


TITLES = ("Deep Vanilla Grow", "Loop Grow", "Untied Grow")


COLORS = dict(zip(DEPTHS, ordered_colors(len(DEPTHS), quantity="depth")))


def make_figure(fits):
    apply_style()
    labels = tuple(f"d{depth} budget / model" for depth in DEPTHS)
    legend_rows = legend_row_count(labels)
    fig, axes = make_grid(
        1, 3, titles=TITLES, square_panels=True, share_ylabel=True,
        legend_rows=legend_rows,
    )
    for ax, arm in zip(axes, ARMS):
        losses = []
        for depth, fit in fits[arm].items():
            x = np.asarray([row["rho"] for row in fit["series"]])
            y = np.asarray([row["loss"] for row in fit["series"]])
            grid = np.linspace(x.min(), x.max(), 200)
            fitted_loss = np.polyval(fit["coeffs"], grid)
            losses.extend(y)
            losses.extend(fitted_loss)
            ax.plot(x, y, marker=DEPTH_MARKER[depth], lw=0, color=COLORS[depth], zorder=3)
            ax.plot(grid, fitted_loss, color=COLORS[depth], alpha=0.75, zorder=2)
            ax.axvline(fit["rho_star"], color=COLORS[depth], lw=0.9, ls=":", alpha=0.75, zorder=1)
        mean = float(np.mean([fit["rho_star"] for fit in fits[arm].values()]))
        ax.axvline(mean, color=INK, ls="--", zorder=4)
        ax.text(0.97, 0.95, rf"Mean $\rho^*={mean:.3f}$", transform=ax.transAxes,
                ha="right", va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
        # Preserve each sweep's measured domain, with room for endpoint markers.
        xmax = 0.8 if arm == "deepvan" else 0.4
        ax.set_xlim(-0.03 * xmax, 1.03 * xmax)
        ax.set_xticks(np.linspace(0, xmax, 5))
        lo, hi = min(losses), max(losses)
        ax.set_ylim(lo - 0.06 * (hi - lo), hi + 0.28 * (hi - lo))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        ax.set_xlabel(r"Growth fraction ($\rho$)")
    axes[0].set_ylabel("Loss")
    handles = tuple(plt.Line2D([], [], color=COLORS[depth], marker=DEPTH_MARKER[depth], lw=0)
                    for depth in DEPTHS)
    add_top_legend(fig, handles, labels, rows=legend_rows)
    save_figure(fig, 13)


def fitted_series() -> dict:
    rows = load_rows()
    deepvan = load_deepvan_rows()
    fits = {}
    for arm in ARMS:
        fits[arm] = {}
        for depth in DEPTHS:
            series = ([row for row in deepvan if row["depth"] == depth]
                      if arm == "deepvan" else series_for(rows, arm, depth, depth))
            expected = 9 if arm == "deepvan" else expected_cells(arm, depth, depth)
            if len(series) != expected:
                raise ValueError(f"{arm} d{depth}: expected {expected} cells, got {len(series)}")
            if not all(np.isfinite(row["loss"]) for row in series):
                raise ValueError(f"{arm} d{depth}: nonfinite loss")
            fit = quadratic(series)
            if fit is None:
                raise ValueError(f"{arm} d{depth}: no convex, bracketed minimum")
            coeffs, rho_star, _ = fit
            fits[arm][depth] = dict(series=series, coeffs=coeffs, rho_star=rho_star)
    return fits


def main():
    make_figure(fitted_series())


if __name__ == "__main__":
    main()
