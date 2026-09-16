"""Figure 17: evaluation recurrence minus the trained-recurrence loss."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from common import save_figure
from appendix_style import INK, add_top_legend, legend_row_count

from appendix_style import apply_style, make_grid, ordered_colors
from appendix_ladders import load_json
DEPTHS = tuple(range(6, 21, 2))


DEPTH_COLORS = dict(zip(DEPTHS, ordered_colors(len(DEPTHS), quantity="depth")))


DEPTH_LABELS = tuple(f"d{depth}" for depth in DEPTHS)


EXIT_ARMS = ("k2", "dep", "k2_grow", "k2_grow_random", "dep_grow")


EXIT_TITLES = (
    "Loop-2",
    "Untied-2",
    "Loop Grow",
    "Random recur.",
    "Untied Grow",
)


EXIT_YLIMS = {
    "k2": (-0.005, 0.06),
    "dep": (-0.1, 2.0),
    "k2_grow": (-0.005, 0.05),
    "k2_grow_random": (-0.002, 0.03),
    "dep_grow": (-0.1, 2.3),
}


def plot_exit_lens() -> None:
    apply_style()
    arms = EXIT_ARMS
    target_style = dict(marker="*", ms=8, mfc="#68707C", mec="white", mew=0.4, linestyle="none")
    legend_rows = legend_row_count((*DEPTH_LABELS[:-1], r"$K_{\mathrm{train}}$"), width=8.4)
    fig, axes = make_grid(
        1, len(arms), titles=EXIT_TITLES, width=8.4, square_panels=True, sharey=False,
        shared_xlabel=r"Evaluation recurrence $k$",
        row_ylabels=(r"$L(k)-L(K_{\mathrm{train}})$",), legend_rows=legend_rows,
    )
    for axis, arm in zip(axes, arms):
        axis.grid(False)
        axis.grid(axis="y", color="#DDE1E7", lw=0.5, alpha=0.6)
        axis.tick_params(length=2.5, width=0.6)
        for spine in axis.spines.values():
            spine.set_color("#A5ABB3")
            spine.set_linewidth(0.6)
        axis.axhline(0, color="#A5ABB3", lw=0.6, ls=(0, (2, 2)), zorder=1)
        for point in load_json("recurrence")[arm]:
            colour = DEPTH_COLORS[point["depth"]]
            losses = {item["k"]: item["ce"] for item in point["exit_lens"]}
            ks = sorted(losses)
            trained = int(float(point["core_repetitions"]))
            baseline = losses[trained]
            differences = [losses[recurrence] - baseline for recurrence in ks]
            axis.plot(ks, differences, marker="o", ms=4, mew=0, lw=1.5, solid_capstyle="round", color=colour,
                      label=f"d{point['depth']}")
        axis.plot([trained], [0], **target_style, zorder=4)
        axis.xaxis.get_major_locator().set_params(integer=True)
        axis.set_ylim(*EXIT_YLIMS[arm])
        if arm == "k2_grow_random":
            # Retain integer recurrence marks without crowding the narrow panel.
            axis.set_xticks((1, 4, 8, 12))
            axis.set_xticks([k for k in range(1, 13) if k not in (1, 4, 8, 12)], minor=True)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], **target_style))
    labels.append(r"$K_{\mathrm{train}}$")
    add_top_legend(fig, handles, labels, rows=legend_rows, labelcolor=INK)
    save_figure(fig, 17, bbox_inches=None)


if __name__ == "__main__":
    plot_exit_lens()
