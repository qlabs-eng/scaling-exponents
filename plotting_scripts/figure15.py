"""Figure 15: architecture ablations, measured runtime, and constant recipes."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from common import save_figure
from appendix_style import GRID, legend_row_count
from appendix_ladders_geometry import MARKER_SIZE, figure_width, apply_style, make_grid

import appendix_ladders as ladder
from appendix_ladders_geometry import add_column_legends
PANELS = (
    ("Untied-2", ("dep", "dep_nci", "dep_nonorm", "dep_noinject", "deep_van", "van")),
    ("Loop-2", ("van", "k2", "k2_nci", "k2_fixed_pc")),
    ("Operator-1", ("van", "k1_1c2", "k1_glr")),
)


LINESTYLES = {"dep_nonorm": "--", "dep_noinject": ":", "dep_nci": "-.", "k2_nci": "--", "k2_fixed_pc": ":"}


LABELS = ladder.ARM_LABEL | {
    "dep_nonorm": "Deep Vanilla + Injection",
    "dep_noinject": "Deep Vanilla + Norm",
}


COLORS = ladder.ARM_COLORS | {
    "dep_nonorm": ladder.ARM_COLORS["deep_van"],
    "dep_noinject": ladder.ARM_COLORS["deep_van"],
}


COMPACT_LABELS = LABELS | {
    "dep_nonorm": "Deep + Inj.",
    "dep_noinject": "Deep + Norm",
    "dep_nci": "No Coda Inj.",
    "k2_nci": "No Coda Inj.",
    "k2_fixed_pc": "NCI 2/C/3",
    "k1_1c2": "1/C/2",
}


def plot_architectures(data):
    apply_style()
    panels = PANELS
    arms = tuple(dict.fromkeys(arm for _, panel in panels for arm in panel))
    legend_rows = max(legend_row_count([COMPACT_LABELS[arm] for arm in panel],
                                      width=figure_width(len(panels)) / len(panels))
                      for _, panel in panels)
    fits = ladder.fit_vanilla_baseline_power_laws(data, arms)
    vanilla = [point for point in data["van"] if 8 <= point.depth <= 20]
    if not vanilla:
        raise ValueError("missing Vanilla d8–d20 reference measurements")
    fig, axes = make_grid(
        2, len(panels),
        titles=tuple(title for title, _ in panels) + ("Reducible Loss",) * 3,
        legend_rows=legend_rows,
        legend_row_height=0.15,
        shared_xlabel="Compute (FLOPs)",
        sharex=True,
    )
    axes = np.asarray(axes).reshape(2, len(panels))
    handles = {}
    for row_index, row in enumerate(axes):
        for axis in row[1:]:
            if row_index == 1:
                axis.sharex(row[0])
            axis.sharey(row[0])
    for column, (_, panel) in enumerate(panels):
        multiplier_axis = axes[0, column]
        loss_axis = axes[1, column]
        for arm in panel:
            fit = fits[arm]
            style = dict(color=COLORS[arm], linestyle=LINESTYLES.get(arm, "-"))
            marker = ladder.ARM_MARKERS[arm]
            multiplier = np.asarray([
                1.0 if arm == "van" else ladder.interpolate_compute_multiplier(data[arm], point.loss, point.compute)
                for point in vanilla
            ])
            if not np.any(np.isfinite(multiplier)):
                raise ValueError(f"no measured loss overlap with Vanilla for {arm}")
            multiplier_axis.plot([point.compute for point in vanilla], multiplier, **style, marker=marker, markersize=MARKER_SIZE)
            points = data[arm]
            compute = np.asarray([point.compute for point in points])
            loss = np.asarray([point.loss for point in points]) - fit.irreducible
            fit_compute = np.geomspace(compute.min(), compute.max(), 300)
            loss_axis.plot(fit_compute, fit.predict(fit_compute) - fit.irreducible, **style, linewidth=1.1)
            loss_axis.plot(compute, loss, linestyle="none", color=style["color"], marker=marker, markersize=MARKER_SIZE)
            handles[arm] = Line2D([], [], **style, marker=marker, markersize=MARKER_SIZE,
                                  label=COMPACT_LABELS[arm])
        multiplier_axis.set_xscale("log")
        multiplier_axis.axhline(1.0, color=GRID, linewidth=0.9, zorder=1)
        multiplier_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        multiplier_axis.set_ylabel("Compute Multiplier")
        # Shared limits leave a small margin around every column's measurements.
        multiplier_axis.set_ylim(0.95, 1.36)
        multiplier_axis.set_yticks(np.arange(1.0, 1.36, 0.1))
        loss_axis.set_xscale("log")
        loss_axis.set_yscale("log")
        loss_axis.set_ylabel(r"$L-E_{\mathrm{Van}}$")
        loss_axis.set_xlabel("Compute (FLOPs)")
        for formatter in (loss_axis.yaxis.set_major_formatter, loss_axis.yaxis.set_minor_formatter):
            formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    add_column_legends(fig, [[handles[arm] for arm in panel] for _, panel in panels])
    save_figure(fig, 15, "a", bbox_inches=None)


def main():
    data = ladder.load_points()
    plot_architectures(data)
    floor = ladder.fit_shared_irreducible(data, ("van",))["van"].irreducible
    ladder.plot_figure(ladder.load_points("timing"), ladder.LADDER_MAJOR_ARMS, 15, "b",
        interpolation_multiplier=True, fit_arms=ladder.LADDER_MAJOR_ARMS,
        subtract_vanilla_irreducible=True, x_label="Training time (h)",
        scaling_variable="H", fixed_irreducible=floor)
    ladder.plot_figure(ladder.load_points("constant"), ladder.CONSTANT_LADDER_ARMS, 15, "c",
        interpolation_multiplier=True, fit_arms=ladder.CONSTANT_LADDER_ARMS,
        subtract_vanilla_irreducible=True)

if __name__ == "__main__":
    main()
