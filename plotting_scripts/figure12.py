"""Figure 12: regenerate offline from the bundled paper measurements."""
from __future__ import annotations


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter

from common import save_figure
from appendix_style import ARCHITECTURE_LABELS, INK, MUTED, FULL_WIDTH, add_top_legend, apply_style, legend_row_count, make_grid, ordered_colors
from appendix_early_data import load_data


ARMS = ["van", "deep_van", "k1", "k2", "dep"]


BUDGETS = (8, 9, 10, 11, 12)


BUDGET_COLORS = dict(zip(BUDGETS, ordered_colors(len(BUDGETS), quantity="compute")))


MARKERS = {8: "o", 9: "s", 10: "^", 11: "D", 12: "v"}


def budget_legend():
    labels = tuple(f"d{budget} budget" for budget in BUDGETS)
    handles = tuple(
        plt.Line2D([], [], color=BUDGET_COLORS[budget], marker=MARKERS[budget], lw=0)
        for budget in BUDGETS
    )
    return handles, labels


def loss_panel(ax, arm_fit):
    """Plot the iso-compute bowls and the architecture's mean optimal ratio."""
    fits = arm_fit["fits"]
    for b, f in fits.items():
        col = BUDGET_COLORS.get(b, MUTED)
        pts = f["series"]
        ax.plot([p["tpp"] for p in pts], [p["L"] for p in pts], marker=MARKERS.get(b, "o"),
                lw=0, color=col, zorder=3)
        xs = np.linspace(np.log(min(p["tpp"] for p in pts)),
                         np.log(max(p["tpp"] for p in pts)), 200)
        ax.plot(np.exp(xs), np.exp(np.polyval(f["coeffs"], xs)), color=col,
                alpha=0.75, zorder=2)
        ax.axvline(f["tpp"], color=col, lw=0.9, ls=":", alpha=0.75, zorder=1)
    ax.axvline(arm_fit["tpp_mean"], color=INK, ls="--", zorder=4)
    ax.text(0.97, 0.95, rf"$\overline{{\mathrm{{TPP}}^*}}={arm_fit['tpp_mean']:.2f}$",
            transform=ax.transAxes, ha="right", va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.set_xscale("log"); ax.set_yscale("log")
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    x_ticks = [2, 5, 10, 20]
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.xaxis.set_major_formatter(plain)
    # The series stack across a fraction of a decade, so a log axis puts no major tick in
    # range at all. Place them on a round step chosen from the data.
    every = [p["L"] for f in fits.values() for p in f["series"]]
    lo, hi = min(every), max(every)
    step = next((s for s in (0.05, 0.1, 0.2, 0.25, 0.5) if (hi - lo) / s <= 6), 1.0)
    ax.yaxis.set_major_locator(FixedLocator(np.arange(np.floor(lo / step) * step,
                                                      hi + step, step)))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_formatter(plain)
    ax.set_ylim(top=np.exp(np.log(hi) + 0.28 * (np.log(hi) - np.log(lo))))


ARM_LABEL = ARCHITECTURE_LABELS | {"deep_van": "Deep Vanilla"}


def summary_figure(results):
    """One row of five square architecture panels."""
    curve_arms = ARMS
    missing = [arm for arm in curve_arms if arm not in results]
    if missing:
        raise ValueError(f"missing standalone TPP-curve arms: {missing}")

    budget_handles, budget_labels = budget_legend()
    curve_width = FULL_WIDTH
    curve_legend_rows = legend_row_count(budget_labels, width=curve_width)
    curve_fig, curve_axes = make_grid(
        1,
        5,
        titles=tuple(ARM_LABEL[arm] for arm in curve_arms),
        width=curve_width,
        square_panels=True,
        sharey=True,
        share_ylabel=True,
        legend_rows=curve_legend_rows,
    )
    curve_axes = curve_axes.ravel()
    every_tpp = [p["tpp"] for arm in curve_arms
                 for fit in results[arm]["fits"].values() for p in fit["series"]]
    for axis, arm in zip(curve_axes, curve_arms):
        loss_panel(axis, results[arm])
        # Common bounds include all measured cells with room for endpoint markers.
        axis.set_xlim(min(every_tpp) / 1.10, max(every_tpp) * 1.10)
        axis.set_xlabel("TPP")
    every_loss = [p["L"] for arm in curve_arms
                  for fit in results[arm]["fits"].values() for p in fit["series"]]
    lo, hi = min(every_loss), max(every_loss)
    # A common loss scale makes architectures comparable and leaves annotation room.
    curve_axes[0].set_ylim(lo / 1.01, np.exp(np.log(hi) + 0.45 * np.log(hi / lo)))
    curve_axes[0].set_ylabel("Loss")
    add_top_legend(curve_fig, budget_handles, budget_labels, rows=curve_legend_rows)
    save_figure(curve_fig, 12)


def compute_params(rows):
    """Nc ~ c N^k over the architecture's measured depths: compute-bearing parameters
    against the total. The embedding carries parameters but no matmul, and its share
    shrinks as the model grows, so k > 1."""
    seen = {r["N"]: r["Nc"] for r in rows}
    N = np.array(sorted(seen), float)
    Nc = np.array([seen[n] for n in N], float)
    k, c = np.polyfit(np.log(N), np.log(Nc), 1)
    return float(k), float(np.exp(c))


MIN_PER_BUDGET = 3          # a quadratic needs three points


def bowl(series):
    """Quadratic in (log TPP, log loss); the vertex is that budget's optimal ratio.

    Returns (tpp_star, coeffs, interior) with interior False when the vertex falls outside
    the measured span, i.e. the optimum is an extrapolation rather than a bracketed
    minimum.
    """
    x = np.log(np.array([r["tpp"] for r in series], float))
    y = np.log(np.array([r["L"] for r in series], float))
    coeffs = np.polyfit(x, y, 2)
    a, b, _ = coeffs
    if a <= 0:                       # opens downward: no minimum to read
        return None, None, False
    xstar = -b / (2 * a)
    return float(np.exp(xstar)), coeffs, bool(x.min() <= xstar <= x.max())


def split_at(C, tpp_star, k, c):
    """The (N, D) split at compute C and ratio tpp_star, from C = 6 c N^k D and D = t N."""
    N = (C / (6 * c * tpp_star)) ** (1.0 / (k + 1))
    return N, tpp_star * N


def fit_arm(rows):
    """Per-budget bowls plus the two power laws, for one arm."""
    k, c = compute_params(rows)
    fits = {}
    for b in sorted({r["budget"] for r in rows}):
        series = sorted((r for r in rows if r["budget"] == b), key=lambda r: r["tpp"])
        if len(series) < MIN_PER_BUDGET:
            continue
        tpp_star, coeffs, interior = bowl(series)
        if tpp_star is None:
            continue
        C = float(np.mean([r["C"] for r in series]))
        N, D = split_at(C, tpp_star, k, c)
        # The vertex loss: what the arm reaches at this compute once the split is chosen
        # optimally, read off the same quadratic that gave the ratio.
        L = float(np.exp(np.polyval(coeffs, np.log(tpp_star))))
        fits[b] = dict(series=series, tpp=tpp_star, coeffs=coeffs, C=C, N=N, D=D, L=L,
                       interior=interior)
    Cs = np.array([f["C"] for f in fits.values()], float)
    a1 = a2 = aL = None
    if len(fits) >= 2:
        a1 = float(np.polyfit(np.log(Cs), np.log([f["N"] for f in fits.values()]), 1)[0])
        a2 = float(np.polyfit(np.log(Cs), np.log([f["D"] for f in fits.values()]), 1)[0])
        aL = float(np.polyfit(np.log(Cs), np.log([f["L"] for f in fits.values()]), 1)[0])
    return dict(fits=fits, k=k, c=c, a1=a1, a2=a2, aL=aL,
                tpp_mean=float(np.mean([f["tpp"] for f in fits.values()])))


def main():
    apply_style()
    rows = load_data("tpp")
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    results = {a: fit_arm([r for r in rows if r["arm"] == a]) for a in arms}

    for arm in arms:
        f = results[arm]
        print(f"{arm}: Nc ~ N^{f['k']:.4f}, mean TPP* {f['tpp_mean']:.2f}, "
              f"a1={f['a1']:.4f} a2={f['a2']:.4f} sum={f['a1'] + f['a2']:.4f}, "
              f"L ~ C^{f['aL']:.4f}")
        for b, fit in f["fits"].items():
            print(f"   d{b}: C={fit['C']:.4g}  TPP*={fit['tpp']:.2f}  "
                  f"N*={fit['N']:.4g}  D*={fit['D']:.4g}  L*={fit['L']:.4f}"
                  f"{'' if fit['interior'] else '   (EXTRAPOLATED)'}")

    summary_figure(results)

if __name__ == "__main__":
    main()
