"""Growth sweep measurements and original quadratic fits shared by Figures 7, 10 and 13."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator

from common import save_figure
from appendix_style import INK, MUTED, add_top_legend, legend_row_count, make_grid, ordered_colors, style_axis
from appendix_early_data import load_data

def counts_at(depth, mode, recurrence):
    """Historical stored and active parameter counts, exported from the paper model."""
    for row in load_data("counts"):
        if (row["depth"], row["mode"], row["recurrence"]) == (depth, mode, recurrence):
            return row["stored"], row["active"]
    raise ValueError((depth, mode, recurrence))


def budgets_for(arm: str, depth: int) -> tuple[float, ...]:
    if arm == "k2":
        return (8, 8.5, 9, 9.5, 10, 10.5)
    return tuple(sorted((float(depth), 9.5, 10.5)))


def expected_cells(arm: str, budget: float, depth: int) -> int:
    # The original diagonal used the coarse five-rho grid; every crossed series uses 9.
    return 5 if budget == depth and budget in (8, 9, 10) and arm == "dep" else 9


def series_for(rows: list[dict], arm: str, budget: float, depth: int) -> list[dict]:
    return [r for r in rows if r["arm"] == arm and r["budget"] == budget
            and r["depth"] == depth]


def quadratic_coeffs(series: list[dict]):
    if len(series) < 5:
        return None
    x = np.array([r["rho"] for r in series], float)
    y = np.array([r["loss"] for r in series], float)
    return np.polyfit(x, y, 2)


def quadratic(series: list[dict]):
    coeffs = quadratic_coeffs(series)
    if coeffs is None:
        return None
    x = np.array([r["rho"] for r in series], float)
    a, b, _ = coeffs
    if a <= 0:
        return None
    optimum = -b / (2 * a)
    if not x.min() <= optimum <= x.max():
        return None
    return coeffs, float(optimum), float(np.polyval(coeffs, optimum))


ARM_MODE = {"k2": "loop", "dep": "dep"}


def k2_stored_tpp(series: list[dict], arm: str, depth: int):
    """K2-anchored stored TPP: D_K2 / Ns2."""
    controls = [r for r in series if r["rho"] == 0]
    if not controls:
        return None
    control = controls[0]
    ns2, _ = counts_at(depth, ARM_MODE[arm], 2)
    return control["steps"] * control["batch"] / ns2


def growth_stored_tpp(series: list[dict], arm: str, depth: int,
                      rho_star: float) -> float:
    """Growth tokens at fitted rho divided by the K2 stored count."""
    ordered = sorted(series, key=lambda r: r["rho"])
    tokens = float(np.interp(
        rho_star, [r["rho"] for r in ordered],
        [r["steps"] * r["batch"] for r in ordered]))
    ns2, _ = counts_at(depth, ARM_MODE[arm], 2)
    return tokens / ns2


def relative_compute_expansion(arm: str, depth: int,
                               rho_star: float) -> float:
    """E_Krho / E_2: mean compute-active expansion relative to K2."""
    _, nc2 = counts_at(depth, ARM_MODE[arm], 2)
    _, nc4 = counts_at(depth, ARM_MODE[arm], 4)
    return ((1 - rho_star) * nc2 + rho_star * nc4) / nc2


DEPTHS = (8, 9, 10)


def style_axes(ax):
    style_axis(ax)


BUDGETS = (8, 8.5, 9, 9.5, 10, 10.5)


BUDGET_COLOR = dict(zip(BUDGETS, ordered_colors(len(BUDGETS), quantity="compute")))


def plot_curves(rows: list[dict], arm: str, axes):
    for col_i, depth in enumerate(DEPTHS):
        ax = axes[col_i]
        for budget in budgets_for(arm, depth):
            series = series_for(rows, arm, budget, depth)
            if not series:
                continue
            expected = expected_cells(arm, budget, depth)
            color = BUDGET_COLOR[budget]
            x = np.array([r["rho"] for r in series])
            y = np.array([r["loss"] for r in series])
            ax.scatter(x, y, s=22, marker="o", color=color, edgecolor="white",
                       linewidth=0.4, zorder=4, label=rf"$d={budget:g}$")
            coeffs = quadratic_coeffs(series)
            if coeffs is None:
                continue
            grid = np.linspace(x.min(), x.max(), 200)
            ax.plot(grid, np.polyval(coeffs, grid), color=color, lw=1.25,
                    ls="-" if len(series) == expected else "--", alpha=0.88,
                    zorder=2)
            fit = quadratic(series)
            if fit is None:
                continue
            _, rho_star, loss_star = fit
            ax.scatter([rho_star], [loss_star], marker="*", s=64, color=color,
                       edgecolor="white", linewidth=0.45, zorder=5)
        ax.set_xlim(-0.012, 0.412)
        ax.set_xticks(np.arange(0, 0.41, 0.1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.set_xlabel(r"$\rho$")
        style_axes(ax)
        if col_i == 0:
            ax.set_ylabel("Loss", labelpad=0)


DEPTH_MARKER = {8: "o", 9: "s", 10: "^"}


def plot_deepvan_curves(rows: list[dict], axes):
    for col_i, depth in enumerate(DEPTHS):
        ax = axes[col_i]
        series = [row for row in rows if row["depth"] == depth]
        x = np.asarray([row["rho"] for row in series])
        y = np.asarray([row["loss"] for row in series])
        coeffs = np.polyfit(x, y, 2)
        grid = np.linspace(x.min(), x.max(), 240)
        rho_star = float(-coeffs[1] / (2 * coeffs[0]))
        loss_star = float(np.polyval(coeffs, rho_star))
        color = BUDGET_COLOR[depth]
        ax.plot(grid, np.polyval(coeffs, grid), color=color, lw=1.25,
                alpha=0.9, zorder=2)
        ax.scatter(x, y, s=22, marker=DEPTH_MARKER[depth], color=color,
                   edgecolor="white", linewidth=0.4, zorder=4)
        ax.axvline(rho_star, color=color, lw=0.8, ls=":", alpha=0.75,
                   zorder=1)
        ax.scatter([rho_star], [loss_star], marker="*", s=64, color=color,
                   edgecolor="white", linewidth=0.45, zorder=5)
        ax.text(
            0.97, 0.95,
            rf"$\rho^\star={rho_star:.3f}$" + "\n"
            + rf"$L^\star={loss_star:.4f}$",
            transform=ax.transAxes, ha="right", va="top", color=color,
            fontsize=matplotlib.rcParams["legend.fontsize"],
        )
        ax.set_xlim(-0.025, 0.825)
        ax.set_xticks(np.arange(0, 0.81, 0.2))
        ax.get_xticklabels()[0].set_ha("left")
        ax.get_xticklabels()[-1].set_ha("right")
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.3f}")
        )
        ax.set_xlabel(r"Growth fraction ($\rho$)", labelpad=0)
        style_axes(ax)
        if col_i == 0:
            ax.set_ylabel("Loss", labelpad=0)


ARMS = ("k2", "dep")


def optimum_rows(rows: list[dict]) -> list[dict]:
    out = []
    for arm in ARMS:
        for depth in DEPTHS:
            for budget in budgets_for(arm, depth):
                series = series_for(rows, arm, budget, depth)
                fit = quadratic(series)
                tpp_k2 = k2_stored_tpp(series, arm, depth)
                if fit is None or tpp_k2 is None:
                    continue
                _, rho_star, loss_star = fit
                control = next((r["loss"] for r in series if r["rho"] == 0), None)
                expected = expected_cells(arm, budget, depth)
                out.append(dict(arm=arm, depth=depth, budget=budget,
                                tpp_k2=tpp_k2,
                                tpp_growth=growth_stored_tpp(
                                    series, arm, depth, rho_star),
                                expansion=relative_compute_expansion(
                                    arm, depth, rho_star),
                                rho_star=rho_star, complete=len(series) == expected,
                                cells=len(series), loss_star=loss_star,
                                gain=None if control is None else control - loss_star))
    return out


DEPTH_COLOR = dict(zip(DEPTHS, ordered_colors(len(DEPTHS), quantity="depth")))


def plot_tpp(rows: list[dict], arm: str, ax, ratio_ax, residual_ax=None):
    optima = [r for r in optimum_rows(rows) if r["arm"] == arm]
    coeffs = None
    if len(optima) >= 3:
        x = np.array([r["tpp_k2"] for r in optima])
        y = np.array([r["tpp_growth"] for r in optima])
        coeffs = np.polyfit(x, y, 1)
    for depth in DEPTHS:
        series = sorted((r for r in optima if r["depth"] == depth),
                        key=lambda r: r["tpp_k2"])
        if not series:
            continue
        color, marker = DEPTH_COLOR[depth], DEPTH_MARKER[depth]
        ax.plot([r["tpp_k2"] for r in series],
                [r["tpp_growth"] for r in series],
                color=color, lw=0.9, alpha=0.55, zorder=2)
        ratio_ax.plot([r["tpp_k2"] for r in series],
                      [r["expansion"] for r in series],
                      color=color, lw=0.9, alpha=0.55, zorder=2)
        for r in series:
            ax.scatter([r["tpp_k2"]], [r["tpp_growth"]], s=42, marker=marker,
                       facecolor=color if r["complete"] else "white",
                       edgecolor=color, linewidth=1.8, zorder=4)
            ratio_ax.scatter([r["tpp_k2"]], [r["expansion"]], s=42,
                             marker=marker,
                             facecolor=color if r["complete"] else "white",
                             edgecolor=color, linewidth=1.8, zorder=4)
        if coeffs is not None and residual_ax is not None:
            residuals = [r["tpp_growth"]
                         - np.polyval(coeffs, r["tpp_k2"])
                         for r in series]
            residual_ax.plot([r["tpp_k2"] for r in series], residuals,
                             color=color, lw=0.9, alpha=0.55, zorder=2)
            for r, residual in zip(series, residuals):
                residual_ax.scatter([r["tpp_k2"]], [residual], s=42,
                                    marker=marker,
                                    facecolor=color if r["complete"] else "white",
                                    edgecolor=color, linewidth=1.8, zorder=4)
    if coeffs is not None:
        grid = np.linspace(x.min(), x.max(), 100)
        fit_line, = ax.plot(grid, np.polyval(coeffs, grid), color=INK, ls="-",
                            lw=1.4, zorder=3,
                            label="Affine fit")
        identity_line, = ax.plot(grid, grid, color=MUTED, ls=":", lw=1.1,
                                 label="Identity", zorder=1)
        # With both TPPs anchored to Ns2, x/y = E_Krho / E_2.
        implied_ratio = grid / np.polyval(coeffs, grid)
        ratio_fit_line, = ratio_ax.plot(
            grid, implied_ratio, color=INK, lw=1.4, zorder=3,
            label=r"Affine-fit implication")
    tpp_axis_label = r"$\mathrm{TPP}_{K2}$"
    ax.set_xlabel(tpp_axis_label)
    ax.set_ylabel(r"$\mathrm{TPP}_{K_{\rho^\star}}$")
    style_axes(ax)
    depth_handles = [
        Line2D([0], [0], marker=DEPTH_MARKER[depth], ls="none",
               markerfacecolor=DEPTH_COLOR[depth], markeredgecolor=DEPTH_COLOR[depth],
               markersize=5, label=rf"$d={depth}$")
        for depth in DEPTHS
    ]
    line_handles = ([fit_line, identity_line]
                    if coeffs is not None else [])
    ax.legend(handles=depth_handles + line_handles, frameon=False,
              loc="upper left", handlelength=1.7)
    ratio_ax.set_xlabel(tpp_axis_label)
    ratio_ax.set_ylabel(r"$E_{K_{\rho^\star}}/E_2$")
    style_axes(ratio_ax)
    if coeffs is not None:
        ratio_ax.legend(handles=[ratio_fit_line], frameon=False,
                        loc="best")
    if residual_ax is not None:
        residual_ax.axhline(0, color=INK, ls="--", lw=1.2, zorder=1)
        residual_ax.set_xlabel(tpp_axis_label)
        residual_ax.set_ylabel("Affine-fit residual")
        style_axes(residual_ax)
        residual_ax.legend(handles=depth_handles, frameon=False,
                           loc="best")
    return optima


def plot_rho(axis, optima: list[dict]):
    for depth in DEPTHS:
        series = sorted((row for row in optima if row["depth"] == depth),
                        key=lambda row: row["tpp_k2"])
        if not series:
            continue
        color, marker = DEPTH_COLOR[depth], DEPTH_MARKER[depth]
        axis.plot([row["tpp_k2"] for row in series],
                  [row["rho_star"] for row in series],
                  color=color, lw=0.9, alpha=0.55, zorder=2)
        for row in series:
            axis.scatter(
                [row["tpp_k2"]], [row["rho_star"]], s=42, marker=marker,
                facecolor=color if row["complete"] else "white",
                edgecolor=color, linewidth=1.8, zorder=4,
            )
    x = np.asarray([row["tpp_k2"] for row in optima])
    y = np.asarray([row["tpp_growth"] for row in optima])
    affine_coeffs = np.polyfit(x, y, 1)
    predicted_rho = []
    for row in optima:
        _, nc2 = counts_at(row["depth"], ARM_MODE[row["arm"]], 2)
        _, nc4 = counts_at(row["depth"], ARM_MODE[row["arm"]], 4)
        implied_ratio = (
            row["tpp_k2"]
            / np.polyval(affine_coeffs, row["tpp_k2"])
        )
        predicted_rho.append(
            (implied_ratio - 1.0) / (nc4 / nc2 - 1.0))
    axis.scatter(
        x, predicted_rho, marker="*", s=58, color=INK, edgecolor="white",
        linewidth=0.45, label=r"Affine-fit $\rho^\star$", zorder=5,
    )
    axis.set_xlabel(r"$\mathrm{TPP}_{K2}$")
    axis.set_ylabel(r"$\rho^\star$")
    axis.legend(frameon=False, loc="best", handlelength=1.2)
    style_axes(axis)


def make_arm_figure(rows: list[dict], arm: str) -> list[dict]:
    handles = [
        Line2D([0], [0], marker="o", ls="-", color=BUDGET_COLOR[budget],
               label=rf"$d={budget:g}$")
        for budget in BUDGETS
    ]
    labels = [handle.get_label() for handle in handles]
    legend_rows = legend_row_count(labels)
    figure, axes = make_grid(
        2, 3,
        titles=(r"$d=8$", r"$d=9$", r"$d=10$", "TPP", "Expansion", "Growth"),
        legend_rows=legend_rows,
    )
    plot_curves(rows, arm, axes[0])
    optima = plot_tpp(rows, arm, axes[1, 0], axes[1, 1])
    plot_rho(axes[1, 2], optima)
    add_top_legend(figure, handles, labels, rows=legend_rows)
    save_figure(figure, 10, panel={"k2": "a", "dep": "b"}[arm])
    return optima


def make_deepvan_figure(rows: list[dict]) -> None:
    figure, axes = make_grid(
        1, 3, titles=(r"$d=8$", r"$d=9$", r"$d=10$"),
    )
    plot_deepvan_curves(rows, axes)
    save_figure(figure, 10, panel="c")


def load_rows():
    return load_data("growth")


def load_deepvan_rows():
    return load_data("deepvan_growth")
