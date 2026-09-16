"""Figure 11: regenerate offline from the bundled paper measurements."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

from common import arm_color, save_figure
from appendix_style import ARCHITECTURE_LABELS, MUTED, add_top_legend, apply_style, legend_row_count, make_grid
from appendix_early_data import load_data


CHAINS = {
    "van": "chain_vanilla",
    "van-d11": "chain_deep_vanilla",
    "k1": "chain_operator_1",
    "k2": "chain_loop_2",
    "dep": "chain_untied_2",
    "dep-noinject": "chain_untied_2_no_inject",
    "dep-no-coda-inj": "chain_untied_2_no_coda_inject",
    "dep-no-norm": "chain_untied_2_no_norm",
    "k2-no-coda-inj": "chain_loop_2_no_coda_inject",
}


PLOT_ARMS = tuple(arm for arm in CHAINS if arm not in {"dep-noinject", "dep-no-norm"})


PHASES = ["glr", "elrm", "hlrm", "rm", "om", "emb_alpha", "wd", "wte", "uis",
          "schedule", "adam"]


ARM_LABEL = {
    **ARCHITECTURE_LABELS,
    "van-d11": "Deep Vanilla",
    "dep-noinject": "Untied-2 No-Inject",
    "dep-no-coda-inj": "Untied-2 No-Coda-Inj",
    "dep-no-norm": "Untied-2 No-Norm",
    "k2-no-coda-inj": "Loop-2 No-Coda-Inj",
}


FIGURE_WIDTH = 5.0


MULTI = {"schedule", "adam"}


NOISE = 1e-3


def trajectory(cells, confirm_loss):
    """[(slot, loss, measured_in_chain)] replaying recipe selection over the cells.

    A joint adoption leaves the incumbent unmeasured: mid-chain that phase gets no
    point (the next phase's probe re-measures it); on the last phase the final-recipes
    confirmation cell trained exactly that recipe, so it stands in, marked unmeasured.
    """
    first_glr = cells[(1, PHASES[0])]
    initial_loss = next(cell["loss"] for cell in first_glr if cell["probe"])
    pts, incumbent = [(-1, initial_loss, True)], None
    for rnd in (1, 2):
        for slot, phase in enumerate(PHASES, start=(rnd - 1) * len(PHASES)):
            group = cells.get((rnd, phase))
            if not group:
                continue                      # van / dep-noinject carry no emb_alpha
            if phase in MULTI:
                probe = next((c["loss"] for c in group if c["probe"]), None)
                base = probe if probe is not None else incumbent
                adopted = []
                for knob in {c["knob"] for c in group if c["knob"]}:
                    best = min(c["loss"] for c in group if c["knob"] == knob)
                    if base - best > NOISE:
                        adopted.append(best)
                incumbent = adopted[0] if len(adopted) == 1 else (
                    base if not adopted else None)
            else:
                probe = next(c["loss"] for c in group if c["probe"])
                best = min(c["loss"] for c in group)
                incumbent = probe if probe - best <= NOISE else best
            if incumbent is not None:
                pts.append((slot, incumbent, True))
    if incumbent is None:                     # joint adoption on the last phase
        pts.append((2 * len(PHASES) - 1, confirm_loss, False))
    return pts


def group_cells(rows, group):
    """{(round, phase): [cell dicts]} for one chain's group."""
    out = {}
    for r in rows:
        if r["series"] != group or r["kind"] != "chain" or not r["val_loss"]:
            continue
        out.setdefault((int(r["round"]), r["phase"]), []).append(
            dict(loss=float(r["val_loss"]), probe=r["probe"] == "True",
                 knob=r["knob"] or None))
    return out


MARKER = {"van": "o", "van-d11": "h", "k1": "s", "k2": "^", "dep": "D", "dep-noinject": "v",
          "dep-no-coda-inj": "P", "dep-no-norm": "X", "k2-no-coda-inj": "*"}


COLOR = {"van": arm_color("van"), "van-d11": arm_color("deepvan"), "k1": arm_color("k1"), "k2": arm_color("k2"), "dep": arm_color("dep"), "dep-noinject": arm_color("dep"), "dep-no-norm": arm_color("dep"), "dep-no-coda-inj": arm_color("dep"), "k2-no-coda-inj": arm_color("k2")}


def main():
    rows = load_data("tune")
    confirm = {r["arm"]: float(r["val_loss"]) for r in rows if r["kind"] == "final"}

    apply_style(**{"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 11,
                   "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                   "figure.labelsize": 11, "legend.fontsize": 9.5})
    legend_labels = [ARM_LABEL[arm] for arm in PLOT_ARMS]
    legend_rows = legend_row_count(legend_labels, width=FIGURE_WIDTH)
    fig, ax = make_grid(
        1,
        1,
        titles=["Chain tuning"],
        width=FIGURE_WIDTH,
        panel_height=3.2,
        shared_ylabel="Loss",
        legend_rows=legend_rows,
        x_tick_rows=4,
        y_tick_columns=2,
        top_tick_rows=1,
    )
    for arm in PLOT_ARMS:
        group = CHAINS[arm]
        pts = trajectory(group_cells(rows, group), confirm.get(arm))
        for rnd in (1, 2):
            lo, hi = (rnd - 1) * len(PHASES), rnd * len(PHASES)
            seg = [p for p in pts if lo <= p[0] < hi]
            if rnd == 1:
                seg = [pts[0]] + seg          # connect the initial recipe
            else:                             # connect the rounds
                seg = [max(p for p in pts if p[0] < lo)] + seg
            ax.plot([p[0] for p in seg], [p[1] for p in seg],
                    color=COLOR[arm], zorder=2)
        filled = [p for p in pts if p[2]]
        hollow = [p for p in pts if not p[2]]
        ax.scatter([p[0] for p in filled], [p[1] for p in filled], s=30,
                   marker=MARKER[arm], color=COLOR[arm], zorder=3)
        if hollow:
            ax.scatter([p[0] for p in hollow], [p[1] for p in hollow], s=36,
                       marker=MARKER[arm], facecolors="none",
                       edgecolors=COLOR[arm], linewidths=1.4, zorder=3)
        print(arm, " ".join(f"{p[1]:.4f}" for p in pts))

    phase_labels = ["GLR", "ELRM", "HLRM", "RM", "OM", r"$\alpha_{\mathrm{emb}}$",
                    "WD", "WTE", "UIS", "Schedule", "Adam"]
    n = len(PHASES)
    ax.axvline(n - 0.5, color=MUTED, lw=1, ls=":")
    round_axis = ax.secondary_xaxis("top")
    round_axis.set_xticks([n / 2 - 0.5, 1.5 * n - 0.5], ["Round 1", "Round 2"])
    round_axis.tick_params(length=0)
    round_axis.spines["top"].set_visible(False)
    ax.set_xticks([-1, *range(2 * n)], ["Initial", *phase_labels, *phase_labels], rotation=90,
                  ha="center", va="top")
    ax.set_xlabel("Hyperparameter")
    ax.grid(False)
    ax.grid(axis="y")
    legend_handles = [
        Line2D([], [], color=COLOR[arm], marker=MARKER[arm],
               markerfacecolor=COLOR[arm], markeredgewidth=0, label=ARM_LABEL[arm])
        for arm in PLOT_ARMS
    ]
    add_top_legend(fig, legend_handles, legend_labels, rows=legend_rows)

    save_figure(fig, 11)

if __name__ == "__main__":
    main()
