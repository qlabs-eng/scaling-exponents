"""Figure 16: scaling recipes, operator recipe, shape, recurrence, width, optimizer."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from common import save_figure
from appendix_style import ARCHITECTURE_COLORS, ARCHITECTURE_LABELS, ARCHITECTURE_MARKERS, GRID, add_top_legend, legend_row_count
from appendix_ladders_geometry import MARKER_SIZE, figure_width, apply_style, make_grid

import appendix_ladders as ladder
from appendix_ladders_geometry import LINE_WIDTH, add_stacked_legends
ARMS = ("van", "k2", "dep", "k2_grow", "k2_grow_random", "dep_grow")


BASELINE_ARM = "k2_grow"


LABELS = {
    "van": ARCHITECTURE_LABELS["van"],
    "k2": ARCHITECTURE_LABELS["k2"],
    "dep": ARCHITECTURE_LABELS["dep"],
    "dep_grow": ARCHITECTURE_LABELS["dep_grow"],
    "k2_grow": ARCHITECTURE_LABELS["k2_grow"],
    "k2_grow_random": "Rand. Loop",
}


COLORS = {
    "van": ARCHITECTURE_COLORS["van"],
    "k2": ARCHITECTURE_COLORS["k2"],
    "dep": ARCHITECTURE_COLORS["dep"],
    "dep_grow": ARCHITECTURE_COLORS["dep_grow"],
    "k2_grow": ARCHITECTURE_COLORS["k2_grow"],
    "k2_grow_random": ARCHITECTURE_COLORS["k2_grow"],
}


MARKERS = {
    "van": ARCHITECTURE_MARKERS["van"],
    "k2": ARCHITECTURE_MARKERS["k2"],
    "dep": ARCHITECTURE_MARKERS["dep"],
    "dep_grow": ARCHITECTURE_MARKERS["dep_grow"],
    "k2_grow": ARCHITECTURE_MARKERS["k2_grow"],
    "k2_grow_random": "X",
}


LINESTYLES = {
    "van": "-",
    "k2": "-",
    "dep": "-",
    "dep_grow": "-",
    "k2_grow": "-",
    "k2_grow_random": (0, (4, 2)),
}


def draw_markers(axis, x, y, points: list[ladder.Point], arm: str,
                 *, fixed_size: float | None = MARKER_SIZE) -> None:
    sizes = (
        [fixed_size ** 2] * len(points) if fixed_size is not None else
        [ladder.model_marker_size(point.parameters) ** 2 for point in points]
    )
    options = {
        "s": sizes,
        "marker": MARKERS[arm],
        "edgecolors": COLORS[arm],
        "linewidths": 0.6,
        "zorder": 4 if arm == "k2_grow_random" else 3,
    }
    if arm == "k2_grow_random":
        options["facecolors"] = "white"
    else:
        options["facecolors"] = COLORS[arm]
    axis.scatter(x, y, **options)


def plot_random(data: dict[str, list[ladder.Point]]) -> None:
    apply_style()
    fits = ladder.fit_vanilla_baseline_power_laws(data, ARMS)
    legend_labels = tuple(
        f"{LABELS[arm]} ($C^{{-{fits[arm].alpha:.4f}}}$)"
        for arm in ARMS
    )
    legend_rows = legend_row_count(legend_labels, width=figure_width(2))
    figure, (loss_axis, multiplier_axis) = make_grid(
        1, 2, titles=("Reducible Loss", "Compute Multiplier"),
        shared_xlabel="Compute (FLOPs)", legend_rows=legend_rows,
    )

    for arm in ARMS:
        points = data[arm]
        measured_compute = np.asarray([point.compute for point in points])
        measured_loss = np.asarray([point.loss for point in points])
        fit_compute = np.geomspace(measured_compute.min(), measured_compute.max(), 300)
        loss_axis.plot(
            fit_compute, fits[arm].predict(fit_compute) - fits[arm].irreducible,
            color=COLORS[arm], linestyle=LINESTYLES[arm], linewidth=1.1,
            zorder=3 if arm == "k2_grow_random" else 2,
        )
        draw_markers(
            loss_axis, measured_compute, measured_loss - fits[arm].irreducible,
            points, arm, fixed_size=MARKER_SIZE,
        )

    baseline = data[BASELINE_ARM]
    baseline_compute = np.asarray([point.compute for point in baseline])
    for arm in ARMS:
        multipliers = np.asarray([
            1.0 if arm == BASELINE_ARM else ladder.interpolate_compute_multiplier(
                data[arm], point.loss, point.compute)
            for point in baseline
        ])
        finite = np.isfinite(multipliers)
        multiplier_axis.plot(
            baseline_compute[finite], multipliers[finite], color=COLORS[arm],
            linestyle=LINESTYLES[arm], zorder=3 if arm == "k2_grow_random" else 2,
        )
        draw_markers(
            multiplier_axis, baseline_compute[finite], multipliers[finite],
            [point for point, keep in zip(baseline, finite) if keep], arm,
        )
        values = ", ".join(
            f"d{point.depth}={value:.3f}x"
            for point, value in zip(baseline, multipliers) if np.isfinite(value)
        )
        print(f"{LABELS[arm]} multipliers: {values}")

    loss_axis.set_xscale("log")
    loss_axis.set_yscale("log")
    loss_axis.set_ylabel(r"$L - E_{\mathrm{Van}}$")
    for formatter in (loss_axis.yaxis.set_major_formatter,
                      loss_axis.yaxis.set_minor_formatter):
        formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    multiplier_axis.set_xscale("log")
    multiplier_axis.set_ylabel("Compute Multiplier")
    multiplier_axis.set_ylim(0.73, 1.17)
    multiplier_axis.set_yticks((0.75, 0.85, 0.95, 1.00, 1.05, 1.15))
    multiplier_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:g}×"))
    multiplier_axis.axhline(1.0, color=GRID, linewidth=0.9, zorder=1)

    handles = tuple(
        Line2D(
            [], [], color=COLORS[arm], marker=MARKERS[arm],
            linestyle=LINESTYLES[arm], markersize=MARKER_SIZE,
            markerfacecolor="white" if arm == "k2_grow_random" else COLORS[arm],
            label=legend_label,
        )
        for arm, legend_label in zip(ARMS, legend_labels)
    )
    add_top_legend(figure, handles, legend_labels, rows=legend_rows)
    save_figure(figure, 16, "d", bbox_inches=None)

VARIANTS = {
    "deep_van": ("Deep Vanilla", ladder.ARM_COLORS["deep_van"], "h"),
    "dep": ("Untied-2", ladder.ARM_COLORS["dep"], "D"),
    "muon": ("Muon", plt.get_cmap("tab10")(0), "o"),
    "adam": ("Adam", plt.get_cmap("tab10")(3), "s"),
}


def plot_width(data, fits, variants, comparisons, filename, *, scaling_labels=("Width", "Both")):
    arms = tuple(arm for arm, _ in comparisons)
    styles = {}
    variant_handles, variant_labels = [], []
    for variant, width_arm, both_arm in variants:
        name, color, marker = VARIANTS[variant]
        for arm, hollow in ((width_arm, True), (both_arm, False)):
            styles[arm] = dict(color=color, marker=marker,
                               markerfacecolor="white" if hollow else color,
                               markeredgecolor=color, markersize=MARKER_SIZE,
                               linewidth=LINE_WIDTH)
        variant_handles.append(Line2D([], [], **styles[both_arm]))
        variant_labels.append(
            f"{name} (Width $C^{{-{fits[width_arm].alpha:.4f}}}$; Both $C^{{-{fits[both_arm].alpha:.4f}}}$)")
    scaling_handles = [Line2D([], [], color="#3F4752", marker="o",
                             markerfacecolor=fill, markersize=MARKER_SIZE,
                             linewidth=LINE_WIDTH)
                       for fill in ("white", "#3F4752")]
    groups = [(variant_handles, variant_labels), (scaling_handles, scaling_labels)]
    rows = sum(legend_row_count(labels, width=figure_width(2)) for _, labels in groups)
    figure, (loss_axis, multiplier_axis) = make_grid(
        1, 2, titles=("Reducible Loss", "Compute Multiplier"),
        shared_xlabel="Compute (FLOPs)", legend_rows=rows,
    )
    for arm in arms:
        points, fit = data[arm], fits[arm]
        compute = np.array([p.compute for p in points])
        loss = np.array([p.loss for p in points])
        smooth = np.geomspace(compute.min(), compute.max(), 300)
        style = styles[arm]
        loss_axis.plot(smooth, fit.predict(smooth) - fit.irreducible,
                       color=style["color"], linewidth=LINE_WIDTH)
        loss_axis.plot(compute, loss - fit.irreducible, linestyle="none", **style)

    for arm, baseline in comparisons:
        reference = [p for p in data[baseline] if baseline != "van" or 8 <= p.depth <= 20]
        values = np.array([1.0 if arm == baseline else ladder.interpolate_compute_multiplier(
            data[arm], p.loss, p.compute) for p in reference])
        multiplier_axis.plot([p.compute for p in reference], values, **styles[arm])
        print(f"{arm} vs {baseline}: " + ", ".join(
            f"{p.depth}={value:.4f}x" for p, value in zip(reference, values) if np.isfinite(value)))

    loss_axis.set(xscale="log", yscale="log", ylabel=r"$L - E_{\mathrm{Van}}$")
    for setter in (loss_axis.yaxis.set_major_formatter, loss_axis.yaxis.set_minor_formatter):
        setter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    multiplier_axis.set(xscale="log", ylabel="Compute Multiplier")
    multiplier_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    multiplier_axis.axhline(1.0, color=GRID, linewidth=0.9, zorder=0)
    add_stacked_legends(figure, groups)
    save_figure(figure, 16, filename, bbox_inches=None)

def main():
    data = ladder.load_points()
    ladder.plot_vanilla_scaling(data)
    for panel, arms, fit_arms in (("b", ladder.K1_ARMS, ladder.ALL_VARIANT_ARMS),
                                  ("c", ladder.SHAPE_LADDER_ARMS, ladder.SHAPE_LADDER_ARMS)):
        ladder.plot_figure(data, arms, 16, panel, interpolation_multiplier=True,
            fit_arms=fit_arms, subtract_vanilla_irreducible=True,
            hollow_arms=("k1_van_recipe",) if panel == "b" else ("deeper_van", "deeper_k1"))
    plot_random(ladder.load_points("random"))
    data = ladder.load_points("width_optimizer")
    fits = ladder.fit_vanilla_baseline_power_laws(data, tuple(data))
    plot_width(data, fits, (("deep_van", "deep_width", "deep_van"), ("dep", "dep_width", "dep")),
        (("deep_width", "deep_width"), ("dep_width", "deep_width"),
         ("deep_van", "deep_van"), ("dep", "deep_van")), "e")
    plot_width(data, fits, (("muon", "deep_width", "van"), ("adam", "deep_width_adam", "van_adam")),
        (("van", "van"), ("van_adam", "van"), ("deep_width", "deep_width"),
         ("deep_width_adam", "deep_width")), "f", scaling_labels=("Width, Deep Vanilla", "Both, Vanilla"))

if __name__ == "__main__":
    main()
