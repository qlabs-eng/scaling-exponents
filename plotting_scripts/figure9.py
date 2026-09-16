"""Figure 9: regenerate offline from the bundled paper measurements."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

from common import sequence_colors, save_figure
from appendix_style import ARCHITECTURE_LABELS, ARCHITECTURE_COLORS, INK, MUTED, FULL_WIDTH, OUTER_PAD, LEGEND_ROW, LEGEND_GAP, FRAME_GAP, AXIS_LABEL_BAND, X_TICK_ROW, Y_TICK_COLUMN, PANEL_TITLE_BAND, add_top_legend, apply_style, legend_row_count, style_axis
from appendix_early_data import load_data

FIGURE_WIDTH = 2 * FULL_WIDTH
LINE_WIDTH, MARKER_SIZE, OPTIMUM_SIZE = 1.4, 4.0, 8.0
BUDGET_COLORS = dict(zip(range(8, 15), sequence_colors(7, quantity="compute")))

def budget_of(rows):
    """Group cells into budgets by their compute, which is matched by construction. The
    label is the TPP-fit rung that names the budget, recovered by nearest ratio."""
    known = {int(b): value for b, value in load_data("tpl_budgets").items()}
    out = {}
    for r in rows:
        out[id(r)] = min(known, key=lambda b: abs(np.log(known[b] / r["C"])))
    return out


def fitted_tpp_points(fits):
    """Return TPP and compute-active expansion at each bracketed optimum."""
    out = {}
    for (anchor, family), by_budget in fits.items():
        points = []
        for budget, fit in by_budget.items():
            if fit["k"] is None or not fit["interior"] or fit["base"] is None:
                continue
            recurrence_values = np.asarray([point["k"] for point in fit["series"]])
            loop_values = np.log(recurrence_values)
            tpp_values = np.log([point["tpp"] for point in fit["series"]])
            k1 = next(point for point in fit["series"] if point["k"] == 1)
            tpp_star = float(
                np.exp(np.interp(np.log(fit["k"]), loop_values, tpp_values))
            )
            nc_star = float(np.interp(
                fit["k"], recurrence_values,
                [point["Nc"] for point in fit["series"]],
            ))
            points.append({"x": k1["tpp"], "y": tpp_star,
                           "expansion": nc_star / k1["Nc"], "k": fit["k"],
                           "budget": budget})
        if points:
            out[(anchor, family)] = sorted(points, key=lambda point: point["x"])
    return out


MARKERS = {8: "o", 9: "s", 10: "^", 11: "D", 12: "v", 13: "P", 14: "X"}


def budget_legend():
    handles = [Line2D([], [], color=BUDGET_COLORS[budget], marker=MARKERS[budget])
               for budget in range(8, 15)]
    labels = [f"Budget d{budget}" for budget in range(8, 15)]
    return handles, labels


ANCHOR_MARKERS = {8: "o", 9: "^", 10: "s", 11: "D", 12: "v"}


def anchor_legend(points):
    anchors = sorted({anchor for anchor, _ in points})
    handles = [Line2D([], [], linestyle="none", marker=ANCHOR_MARKERS[anchor],
                      color=MUTED) for anchor in anchors]
    labels = [f"Anchor d{anchor}" for anchor in anchors]
    return handles, labels


def make_overview_grid(titles, legend_rows):
    """Keep the margin and relation layout with equally sized square panels."""
    nrows = 2
    ncols = 7
    width = FIGURE_WIDTH
    relation_panel_ratio = 1.0

    bottom = OUTER_PAD + X_TICK_ROW + AXIS_LABEL_BAND
    top = OUTER_PAD + PANEL_TITLE_BAND + legend_rows * LEGEND_ROW + LEGEND_GAP
    row_gap = FRAME_GAP + PANEL_TITLE_BAND

    # Margin panels share their y axis. Each relation panel has its own quantity, so
    # its left-hand gap also reserves a tick column and an axis-label band.
    narrow_gap = FRAME_GAP
    labelled_gap = FRAME_GAP + Y_TICK_COLUMN + AXIS_LABEL_BAND
    gaps = [narrow_gap] * 4 + [labelled_gap] * 2
    left = OUTER_PAD + Y_TICK_COLUMN + AXIS_LABEL_BAND
    right = OUTER_PAD
    panel_space = width - left - right - sum(gaps)
    margin_panel_width = panel_space / (5 + 2 * relation_panel_ratio)
    relation_panel_width = relation_panel_ratio * margin_panel_width
    panel_widths = [margin_panel_width] * 5 + [relation_panel_width] * 2
    panel_height = margin_panel_width
    height = bottom + nrows * panel_height + row_gap + top

    width_ratios = []
    for column, panel_width in enumerate(panel_widths):
        width_ratios.append(panel_width)
        if column < ncols - 1:
            width_ratios.append(gaps[column])

    figure = plt.figure(figsize=(width, height), layout=None)
    figure.set_layout_engine(None)
    grid = figure.add_gridspec(
        nrows,
        2 * ncols - 1,
        left=left / width,
        right=1 - right / width,
        bottom=bottom / height,
        top=1 - top / height,
        wspace=0,
        hspace=row_gap / panel_height,
        width_ratios=width_ratios,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for row in range(nrows):
        for column in range(ncols):
            options = {}
            if row:
                options["sharex"] = axes[0, column]
            if 0 < column < 5:
                options["sharey"] = axes[row, 0]
            axis = figure.add_subplot(grid[row, 2 * column], **options)
            axis.set_title(titles[row * ncols + column], pad=4)
            style_axis(axis)
            axes[row, column] = axis
    for axis in axes[0]:
        axis.tick_params(labelbottom=False)
    for axis in axes[:, 1:5].flat:
        axis.tick_params(labelleft=False)

    label_y = (OUTER_PAD + AXIS_LABEL_BAND / 2) / height
    margin_left = axes[1, 0].get_position().x0
    margin_right = axes[1, 4].get_position().x1
    relation_left = axes[1, 5].get_position().x0
    relation_right = axes[1, 6].get_position().x1
    figure.text((margin_left + margin_right) / 2, label_y, "Recurrence",
                ha="center", va="center", fontsize=matplotlib.rcParams["axes.labelsize"])
    figure.text((relation_left + relation_right) / 2, label_y, r"$\mathrm{TPP}_{K1}$",
                ha="center", va="center", fontsize=matplotlib.rcParams["axes.labelsize"])
    figure.supylabel(r"Loss $-$ K1 loss ($10^{-3}$)",
                     x=(OUTER_PAD + AXIS_LABEL_BAND / 2) / width,
                     y=0.5, ha="center", va="center")
    figure._report_legend_rows = legend_rows
    return figure, axes


MIN_PER_SERIES = 3          # a quadratic needs three points


def bowl(series):
    """Quadratic in (log K, log loss); returns the vertex and the fitted coefficients.

    Returns (k_star, coeffs, interior). k_star is None when the quadratic opens downward,
    which is what a row that rises monotonically in K looks like -- the optimum is then at
    or below K1 and is censored, not measured. interior is False when the vertex falls
    outside the measured span. The coefficients come back either way so the caller can
    still show the data.
    """
    x = np.log(np.array([r["k"] for r in series], float))
    y = np.log(np.array([r["L"] for r in series], float))
    coeffs = np.polyfit(x, y, 2)
    a, b = coeffs[0], coeffs[1]
    if a <= 0:                       # opens downward: no minimum to read
        return None, coeffs, False
    xstar = -b / (2 * a)
    # A vertex is BRACKETED only if the measured argmin is not itself an endpoint. The
    # quadratic landing inside the span is not enough: a series that rises monotonically
    # from K1 puts its vertex at the boundary, where the fit reports whatever its tail
    # does rather than a minimum anyone measured. Same rule the 1-D hyperparameter
    # sweeps use to call a knob "on edge".
    ks = [r["k"] for r in series]
    argmin = min(series, key=lambda r: r["L"])["k"]
    bracketed = argmin not in (ks[0], ks[-1])
    return float(np.exp(xstar)), coeffs, bool(bracketed and x.min() <= xstar <= x.max())


def fits_for(rows, budgets, anchor, family):
    out = {}
    for b in sorted({budgets[id(r)] for r in rows
                     if r["anchor"] == anchor and r["family"] == family}):
        s = sorted((r for r in rows if r["anchor"] == anchor and r["family"] == family
                    and budgets[id(r)] == b), key=lambda r: r["k"])
        if len(s) < MIN_PER_SERIES:
            continue
        k_star, coeffs, interior = bowl(s)
        base = next((r["L"] for r in s if r["k"] == 1), None)
        out[b] = dict(series=s, k=k_star, coeffs=coeffs, interior=interior, base=base,
                      C=float(np.mean([r["C"] for r in s])))
    return out


def fit_capacity_relation(points, family):
    xs = np.asarray([
        point["x"]
        for (anchor, family_), series in points.items()
        if family_ == family
        for point in series
    ])
    ratios = np.asarray([
        point["expansion"]
        for (anchor, family_), series in points.items()
        if family_ == family
        for point in series
    ])
    slope, intercept = np.polyfit(np.log(xs), np.log(ratios), 1)
    return xs, ratios, float(slope), float(intercept)


def format_tpp_relation_axis(axis, *, compact=False):
    # The compact four-column view keeps the same domain with fewer visible labels.
    ticks = [7, 15, 30, 70] if compact else [5, 7, 10, 15, 20, 30, 50, 70, 100]
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_minor_locator(FixedLocator([]))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def power_law_annotation(xs, ys, slope, intercept):
    log_y = np.log(ys)
    residuals = log_y - (intercept + slope * np.log(xs))
    r_squared = 1 - np.sum(residuals ** 2) / np.sum((log_y - log_y.mean()) ** 2)
    return (rf"$y = {np.exp(intercept):.2f}x^{{{slope:.2f}}}$" "\n"
            rf"$R^2 = {r_squared:.3f}$")


FAMILY_COLORS = {"loop": ARCHITECTURE_COLORS["k2"], "dep": ARCHITECTURE_COLORS["dep"]}


def panel_capacity_ratio(axis, points, family, *, compact=False):
    xs, ratios, slope, intercept = fit_capacity_relation(points, family)
    fitted_ratios = ratios
    for (anchor, family_), series in sorted(points.items()):
        if family_ != family:
            continue
        x_values = np.asarray([point["x"] for point in series])
        ratios = np.asarray([point["expansion"] for point in series])
        if len(series) > 1:
            axis.plot(x_values, ratios, linestyle="--", linewidth=LINE_WIDTH,
                      color=FAMILY_COLORS[family], alpha=0.4, zorder=2)
        axis.scatter(x_values, ratios, s=MARKER_SIZE ** 2, marker=ANCHOR_MARKERS[anchor],
                     color=FAMILY_COLORS[family], zorder=4)

    fit_x = np.geomspace(xs.min() * 0.85, xs.max() * 1.15, 160)
    predicted_ratio = np.exp(intercept) * fit_x ** slope
    axis.plot(fit_x, predicted_ratio, linewidth=LINE_WIDTH,
              color=INK, zorder=3)
    format_tpp_relation_axis(axis, compact=compact)
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(FixedLocator([1, 1.1, 1.2, 1.3, 1.5, 2]))
    axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_ylabel(r"$N_{c,K^\star}/N_{c,1}$")
    padding = 0.08 * (fitted_ratios.max() - fitted_ratios.min())
    axis.set_ylim(min(fitted_ratios.min(), predicted_ratio.min()) - padding,
                  max(fitted_ratios.max(), predicted_ratio.max()) + padding)
    annotation = (f"slope {slope:.2f}\nintercept {intercept:.2f}" if compact else
                  power_law_annotation(xs, fitted_ratios, slope, intercept))
    axis.text(0.04, 0.96, annotation,
              transform=axis.transAxes, ha="left", va="top", color=MUTED)


def fit_tpp_relation(points, family):
    xs = np.asarray([
        point["x"]
        for (anchor, family_), series in points.items()
        if family_ == family
        for point in series
    ])
    ys = np.asarray([
        point["y"]
        for (anchor, family_), series in points.items()
        if family_ == family
        for point in series
    ])
    if len(xs) < 2:
        raise ValueError(f"not enough {family} fitted-TPP points")
    slope, intercept = np.polyfit(np.log(xs), np.log(ys), 1)
    return xs, ys, float(slope), float(intercept)


def panel_fitted_tpp(axis, points, family):
    xs, ys, slope, intercept = fit_tpp_relation(points, family)
    for (anchor, family_), series in sorted(points.items()):
        if family_ != family:
            continue
        x_values = [point["x"] for point in series]
        y_values = [point["y"] for point in series]
        if len(series) > 1:
            axis.plot(x_values, y_values, linestyle="--", linewidth=LINE_WIDTH,
                      color=FAMILY_COLORS[family], alpha=0.4, zorder=2)
        axis.scatter(x_values, y_values, s=MARKER_SIZE ** 2, marker=ANCHOR_MARKERS[anchor],
                     color=FAMILY_COLORS[family], zorder=4)

    fit_x = np.asarray([xs.min() * 0.85, xs.max() * 1.15])
    axis.plot(fit_x, np.exp(intercept) * fit_x ** slope, linewidth=LINE_WIDTH,
              color=INK, zorder=3)
    format_tpp_relation_axis(axis)
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(FixedLocator([5, 7, 10, 15, 20, 30, 50, 70]))
    axis.yaxis.set_minor_locator(FixedLocator([]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.set_ylabel(r"$\mathrm{TPP}_{K^\star}$")
    axis.text(0.04, 0.96, power_law_annotation(xs, ys, slope, intercept),
              transform=axis.transAxes, ha="left", va="top", color=MUTED)


def panel_margin(axis, fits, anchor, family, plain, handles, seen):
    axis.axhline(0, color=INK, lw=1.1, alpha=0.6, zorder=1)
    for budget, fit in fits[(anchor, family)].items():
        color = BUDGET_COLORS.get(budget, MUTED)
        if fit["base"] is None:
            continue
        handle, = axis.plot([point["k"] for point in fit["series"]],
                            [1000 * (point["L"] - fit["base"])
                             for point in fit["series"]],
                            marker=MARKERS.get(budget, "o"), lw=LINE_WIDTH,
                            markerfacecolor=color if fit["interior"] else "white",
                            color=color, zorder=3, label=f"budget {budget}")
        if budget not in seen:
            seen.add(budget)
            handles.append(handle)
        if fit["k"] is not None:
            xs = np.linspace(np.log(min(point["k"] for point in fit["series"])),
                             np.log(max(point["k"] for point in fit["series"])), 200)
            axis.plot(np.exp(xs),
                      1000 * (np.exp(np.polyval(fit["coeffs"], xs)) - fit["base"]),
                      color=color, lw=LINE_WIDTH, alpha=0.75, zorder=2)
            if fit["interior"]:
                axis.scatter([fit["k"]],
                             [1000 * (np.exp(np.polyval(fit["coeffs"],
                                                        np.log(fit["k"]))) - fit["base"])],
                             color=color, marker="*", s=OPTIMUM_SIZE ** 2,
                             edgecolors="white", linewidths=0.5, zorder=4)
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(FixedLocator([1, 2, 3, 4, 6]))
    axis.xaxis.set_minor_locator(FixedLocator([]))
    axis.xaxis.set_major_formatter(plain)


ARCH_LABEL = {"loop": ARCHITECTURE_LABELS["k2"], "dep": ARCHITECTURE_LABELS["dep"]}


def main():
    apply_style()

    rows = load_data("tpl")
    if not rows:
        raise SystemExit("no cells with a val_loss")
    budgets = budget_of(rows)
    anchors = sorted({r["anchor"] for r in rows})
    families = ("loop", "dep")
    fits = {(a, f): fits_for(rows, budgets, a, f) for a in anchors for f in families}

    plain = FuncFormatter(lambda v, _: f"{v:g}")
    fitted_points = fitted_tpp_points(fits)
    budget_handles, budget_labels = budget_legend()
    budget_rows = legend_row_count(budget_labels)
    anchor_handles, anchor_labels = anchor_legend(fitted_points)
    overview_labels = budget_labels + anchor_labels
    overview_handles = budget_handles + anchor_handles
    overview_rows = legend_row_count(overview_labels, width=FIGURE_WIDTH)
    overview_titles = [
        title
        for family in families
        for title in (
            *(f"{ARCH_LABEL[family]} · d{anchor}" for anchor in anchors),
            f"{ARCH_LABEL[family]} expansion",
            f"{ARCH_LABEL[family]} TPP",
        )
    ]
    overview_fig, overview_axes = make_overview_grid(overview_titles, overview_rows)
    overview_handles_seen, overview_budgets_seen = [], set()
    for row, family in enumerate(families):
        for column, anchor in enumerate(anchors):
            panel_margin(overview_axes[row, column], fits, anchor, family, plain,
                         overview_handles_seen, overview_budgets_seen)
        panel_capacity_ratio(overview_axes[row, 5], fitted_points, family)
        panel_fitted_tpp(overview_axes[row, 6], fitted_points, family)
        for axis in overview_axes[row, 5:]:
            # Square relation panels need fewer labels to keep the ticks distinct.
            axis.xaxis.set_major_locator(FixedLocator([7, 15, 30, 70]))
            axis.xaxis.set_minor_locator(FixedLocator([5, 10, 20, 50, 100]))
            axis.tick_params(axis="x", which="minor", labelbottom=False)
    for axis in overview_axes[:, :5].flat:
        axis.set_ylim(top=50)
    add_top_legend(overview_fig, overview_handles, overview_labels, rows=overview_rows)
    save_figure(overview_fig, 9)


if __name__ == "__main__":
    main()
