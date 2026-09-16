"""Shared pieces for the paper figures: data loading, power-law fits, and plot style."""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator, NullFormatter
from scipy.optimize import least_squares
from scipy.stats import linregress

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PNG = HERE / "png"
PDF = HERE / "pdf"
FIGURES = PNG
C_REF = 1e18  # compute unit in the power laws

ARM_LABEL = {
    "vanilla": "Vanilla", "deep_vanilla": "Deep Vanilla", "deep_vanilla_grow": "Deep Vanilla Grow",
    "operator_1": "Operator-1", "loop_2": "Loop-2", "loop_grow": "Loop-Grow",
    "untied_2": "Untied-2", "untied_grow": "Untied-Grow",
}
INK, MUTED = "#202020", "#666666"
RCPARAMS = {
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 10,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "lines.linewidth": 2.2, "lines.markersize": 6.5, "legend.frameon": True, "legend.fancybox": True,
    "legend.framealpha": 0.9, "legend.edgecolor": "#CCCCCC", "pdf.fonttype": 42,
}

# Figures 2–6 share a 13-inch canvas width so these sizes also match when the
# paper scales each image to the same column width. Insets/annotations are smaller.
MAIN_FIGURE_WIDTH = 13.0
MAIN_SMALL_FONT = 11
MAIN_RCPARAMS = RCPARAMS | {
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 12, "legend.title_fontsize": 12,
}


@dataclass(frozen=True)
class Point:
    compute: float  # training FLOPs
    loss: float  # final validation loss
    depth: int


def by_compute(ladders: dict[str, list[Point]]) -> dict[str, list[Point]]:
    return {arm: sorted(points, key=lambda point: point.compute) for arm, points in ladders.items()}


def load_ladders_csv(path: Path) -> dict[str, list[Point]]:
    ladders: dict[str, list[Point]] = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            ladders.setdefault(row["arm"], []).append(Point(float(row["compute_flops"]), float(row["val_loss"]), int(row["depth"])))
    return by_compute(ladders)


def load_ladders_runs(runs_dir: Path) -> dict[str, list[Point]]:
    """Read <runs_dir>/<arm>_d<depth>/result.json as written by train.py for the ladder_scripts/ run names."""
    ladders: dict[str, list[Point]] = {}
    for path in sorted(runs_dir.glob("*_d[0-9][0-9]/result.json")):
        arm, depth = path.parent.name.rsplit("_d", 1)
        result = json.loads(path.read_text())
        ladders.setdefault(arm, []).append(Point(float(result["estimated_training_flops"]), float(result["final_val_loss"]), int(depth)))
    return by_compute(ladders)


PADDED_VOCAB, SEQUENCE_LENGTH = 50_304, 2_048


def training_flops(config: dict) -> float:
    """Training FLOPs from a run's config, as the trainer estimates them: 6 x (block + head parameters) per executed layer
    plus causal attention, per token; a grow run counts its low-K and final-K phases separately."""
    width, depth = int(config["n_embd"]), int(config["n_layer"])
    hidden = 256 * ((8 * width // 3 + 255) // 256)
    block = 4 * width * width + 3 * width * hidden

    def per_token(layers: int) -> float:
        return 6.0 * (layers * block + width * PADDED_VOCAB) + layers * 6.0 * width * (SEQUENCE_LENGTH + 1)

    if config["depth_scale_mode"] == "none":
        def layers(_):
            return depth
    else:
        base, remainder = divmod(depth, 3)  # the trainer's default split: remainder to the core, then the coda
        prelude = int(config.get("prelude_layer_count") or base)
        core = int(config.get("core_layer_count") or base + (remainder >= 1))
        coda = int(config.get("coda_layer_count") or base + (remainder >= 2))

        def layers(repetitions):
            return prelude + repetitions * core + coda

    steps, batch = int(config["max_train_steps"]), int(config["total_batch_size"])
    final = int(config.get("core_repetitions") or 1)
    if config.get("core_repetition_schedule") == "late":
        crossover = int(float(config["core_repetition_crossover_fraction"]) * int(config.get("lr_schedule_steps") or steps))
        low = int(config.get("core_repetition_low") or 2)
        return (per_token(layers(low)) * crossover + per_token(layers(final)) * (steps - crossover)) * batch
    return per_token(layers(final)) * steps * batch


def load_ladders_wandb(entity: str, project: str, group: str | None = None, *, refresh: bool = False,
                       cache: Path = DATA / "wandb_runs.json") -> dict[str, list[Point]]:
    """Pull <arm>_d<depth> runs from an explicit W&B source; newest finished run wins."""
    table = {arm: rf"{arm}_d(\d\d)" for arm in ARM_LABEL}
    source = {"entity": entity, "project": project, "group": group}
    runs = None
    if cache.exists() and not refresh:
        cached = json.loads(cache.read_text())
        if isinstance(cached, dict) and cached.get("source") == source:
            runs = cached["runs"]
    if runs is None:
        import wandb

        api = wandb.Api(timeout=60)
        runs = []
        for run in api.runs(f"{entity}/{project}", filters={"group": group} if group else {}, per_page=500):
            summary = run.summary
            loss = summary.get("final_val_loss", summary.get("val/final_step_loss", summary.get("val/loss")))
            if run.state != "finished" or loss is None or not any(re.fullmatch(pattern, run.name) for pattern in table.values()):
                continue
            runs.append({"name": run.name, "created_at": str(run.created_at), "loss": float(loss),
                         "compute": training_flops(run.config)})
        if not runs:
            raise RuntimeError(f"no finished ladder runs in {entity}/{project} for group {group!r}")
        cache.write_text(json.dumps({"source": source, "runs": runs}, indent=1) + "\n")
        print(f"pulled {len(runs)} runs from {entity}/{project} into {cache}")
    newest: dict[tuple[str, int], dict] = {}
    for run in runs:
        for arm, pattern in table.items():
            match = re.fullmatch(pattern, run["name"])
            if match and run["created_at"] > newest.get((arm, int(match.group(1))), {}).get("created_at", ""):
                newest[arm, int(match.group(1))] = run
    ladders: dict[str, list[Point]] = {}
    for (arm, depth), run in newest.items():
        ladders.setdefault(arm, []).append(Point(run["compute"], run["loss"], depth))
    return by_compute(ladders)


@dataclass(frozen=True)
class PowerLaw:
    """L(C) = E + A (C / C_REF)^-gamma."""
    amplitude: float
    exponent: float
    irreducible: float
    exponent_stderr: float | None = None

    def __call__(self, compute):
        return self.irreducible + self.amplitude * (np.asarray(compute, dtype=float) / C_REF) ** (-self.exponent)


def huber_scale(residual, minimum: float = 1e-6) -> float:
    mad = np.median(np.abs(residual - np.median(residual)))
    return max(float(1.345 * mad / 0.67448975), minimum)


def fit_power_law(compute, values) -> PowerLaw:
    """Huber fit of E + A (C / C_REF)^-gamma with E free: a grid over E picks the start, bounded least squares finishes."""
    x = np.asarray(compute, dtype=float) / C_REF
    y = np.asarray(values, dtype=float)

    def residual(params):
        amplitude, exponent, irreducible = params
        return irreducible + amplitude * x ** (-exponent) - y

    best = None
    for irreducible in np.linspace(0.0, y.min() * 0.995, 600):
        slope, intercept = np.polyfit(np.log(x), np.log(y - irreducible), 1)
        if slope >= 0:
            continue
        candidate = np.array([np.exp(intercept), -slope, irreducible])
        error = float(np.sum(residual(candidate) ** 2))
        if best is None or error < best[0]:
            best = error, candidate
    bounds = ([0.0, 0.0, 0.0], [np.inf, np.inf, np.nextafter(y.min(), 0.0)])
    ordinary = least_squares(residual, best[1], bounds=bounds)
    robust = least_squares(residual, ordinary.x, bounds=bounds, loss="huber", f_scale=huber_scale(residual(ordinary.x)))
    amplitude, exponent, irreducible = (float(value) for value in robust.x)
    return PowerLaw(amplitude, exponent, irreducible)


def fit_power_law_fixed_irreducible(compute, values, irreducible: float) -> PowerLaw:
    """Regression of log(L - E) on log(C / C_REF); the exponent's standard error is the slope's."""
    x = np.log(np.asarray(compute, dtype=float) / C_REF)
    y = np.log(np.asarray(values, dtype=float) - irreducible)
    regression = linregress(x, y)
    return PowerLaw(float(np.exp(regression.intercept)), float(-regression.slope), irreducible, float(regression.stderr))


def compute_multiplier(points: list[Point], target_loss: float, baseline_compute: float) -> float:
    """baseline_compute over the compute at which the ladder reaches target_loss, interpolated in log compute
    between the two bracketing models. NaN when the target lies outside the measured losses."""
    candidates = []
    for lower, upper in zip(points, points[1:]):
        if upper.loss < lower.loss and upper.loss <= target_loss <= lower.loss:
            fraction = (target_loss - lower.loss) / (upper.loss - lower.loss)
            candidates.append(math.exp(math.log(lower.compute) + fraction * (math.log(upper.compute) - math.log(lower.compute))))
    return baseline_compute / min(candidates) if candidates else float("nan")


def add_source_arguments(parser) -> None:
    parser.add_argument("--csv", action="store_true", help="use the bundled data/ files only, without W&B")
    parser.add_argument("--wandb-entity", help="W&B account or team; requires --wandb-project")
    parser.add_argument("--wandb-project", help="W&B project; requires --wandb-entity")
    parser.add_argument("--wandb-group", help="optional group filter within the selected W&B project")
    parser.add_argument("--refresh", action="store_true", help="pull from W&B again instead of using the cached pull")


def use_wandb(args) -> bool:
    """Bundled inputs by default; W&B requires an explicit account and project."""
    if args.csv:
        return False
    requested = args.wandb_entity or args.wandb_project or args.wandb_group or args.refresh
    if requested and not (args.wandb_entity and args.wandb_project):
        raise ValueError("W&B access requires both --wandb-entity and --wandb-project")
    return bool(requested)


def ladders_from(args) -> dict[str, list[Point]]:
    """Bundled compute-optimal ladders, optionally replaced by an explicit W&B source."""
    if use_wandb(args):
        try:
            return load_ladders_wandb(args.wandb_entity, args.wandb_project, getattr(args, "wandb_group", None), refresh=args.refresh)
        except Exception as error:  # no credentials, no access, or no network
            print(f"W&B unavailable ({error}); using the bundled data/fineweb_ladders.csv")
    return load_ladders_csv(DATA / "fineweb_ladders.csv")


# Architecture categories use eight distinct tab10 hues across all figures.
ARM_COLORS = {
    "vanilla": "#7f7f7f", "deep_vanilla": "#8c564b", "deep_vanilla_grow": "#bcbd22",
    "operator_1": "#9467bd", "loop_2": "#d62728", "loop_grow": "#ff7f0e",
    "untied_2": "#1f77b4", "untied_grow": "#17becf",
}


def architecture_colors(arms) -> dict[str, str]:
    return {arm: ARM_COLORS[arm] for arm in arms}


ARM_ALIASES = {
    "van": "vanilla", "deep_van": "deep_vanilla", "deepvan": "deep_vanilla",
    "deepvan_grow": "deep_vanilla_grow", "deep_van_grow": "deep_vanilla_grow",
    "k1": "operator_1", "k1_glr": "operator_1", "inject_1": "operator_1",
    "k2": "loop_2", "k2_grow": "loop_grow", "dep": "untied_2", "dep_grow": "untied_grow",
}


def arm_color(arm: str) -> str:
    """Use the same architecture colors for the main figures and appendix."""
    name = arm.lower().replace("-", "_").replace(" ", "_")
    return ARM_COLORS[ARM_ALIASES.get(name, name)]


QUANTITY_PALETTES = {
    "recurrence": ("magma", 0.08, 0.82),
    "compute": ("inferno", 0.12, 0.82),
    "depth": ("viridis", 0.16, 0.78),
    "weight_decay": ("cividis", 0.08, 0.94),
    "tokens_per_parameter": ("cividis", 0.18, 0.78),
    "strategy": ("magma", 0.12, 0.78),
}


def sequence_colors(count: int, *, quantity: str) -> list[str]:
    """Choose a scale by the quantity encoded in color, not by figure number."""
    name, lower, upper = QUANTITY_PALETTES[quantity]
    colormap = plt.get_cmap(name)
    return [to_hex(colormap(position)) for position in np.linspace(lower, upper, count)]


def quantity_colors(values, *, quantity: str) -> dict:
    return dict(zip(values, sequence_colors(len(values), quantity=quantity)))


def plot_multipliers(axis, ladders: dict[str, list[Point]], arms, colors: dict[str, str]) -> dict[str, float]:
    """Compute multiplier over Vanilla at each Vanilla budget from d8 on; returns each arm's final multiplier."""
    budgets = [p for p in ladders["vanilla"] if p.depth >= 8]
    final = {}
    for arm in arms:
        curve = [(p.compute, 1.0 if arm == "vanilla" else compute_multiplier(ladders[arm], p.loss, p.compute)) for p in budgets]
        curve = [(compute, value) for compute, value in curve if np.isfinite(value)]
        axis.plot([c for c, _ in curve], [m for _, m in curve], "o-", color=colors[arm], markeredgewidth=0, zorder=3 + (arm != "vanilla"))
        final[arm] = curve[-1][1]
    axis.set_xscale("log")
    axis.axhline(1.0, color=MUTED, linewidth=0.7, zorder=1)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    axis.set_ylabel("Compute Multiplier")
    axis.margins(y=0.12)
    best = max(arms, key=final.get)  # its final multiplier is called out on the right-hand axis
    axis.axhline(final[best], color=colors[best], linestyle=":", linewidth=2, alpha=0.5, zorder=1)
    right = axis.twinx()
    right.set_ylim(axis.get_ylim())
    right.set_yticks([final[best]])
    right.set_yticklabels([f"{final[best]:.2f} ×"])
    right.get_yticklabels()[0].set_color(colors[best])
    right.tick_params(length=0)
    for spine in right.spines.values():
        spine.set_visible(False)
    return final


def flop_tick(value: float, _: float) -> str:
    if value <= 0:
        return ""
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / 10 ** exponent
    return rf"$10^{{{exponent}}}$" if math.isclose(mantissa, 1.0) else rf"${mantissa:g}\times10^{{{exponent}}}$"


def style_axis(axis) -> None:
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)
    axis.tick_params(colors="black", direction="out")


def zoom_inset(axis, rect, xlim, ylim, xticks, ytick_step: float = 0.1, labelsize: float = 9.5):
    """Inset over a log-compute window, with a box on the parent axis and dashed lines from its top corners to the inset's bottom corners."""
    inset = axis.inset_axes(rect)
    inset.set_xscale("log")
    inset.set_xlim(*xlim)
    inset.set_ylim(*ylim)
    inset.xaxis.set_major_locator(FixedLocator(xticks))
    inset.xaxis.set_major_formatter(FuncFormatter(flop_tick))
    inset.xaxis.set_minor_formatter(NullFormatter())
    inset.yaxis.set_major_locator(MultipleLocator(ytick_step))
    inset.tick_params(labelsize=labelsize, length=2, pad=1.5, colors="black")
    for spine in inset.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)
    _, connectors = axis.indicate_inset_zoom(inset, edgecolor=MUTED, linewidth=0.7, alpha=0.9)
    for connector in connectors:
        connector.set_visible(False)
    for box_x, corner in ((xlim[0], (0.0, 0.0)), (xlim[1], (1.0, 0.0))):
        axis.add_artist(ConnectionPatch(xyA=(box_x, ylim[1]), coordsA=axis.transData, xyB=corner, coordsB=inset.transAxes,
                                        color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), alpha=0.9, zorder=1))
    return inset


def save(fig, path: Path, *, bbox_inches: str | None = "tight", dpi: int = 300) -> None:
    """Save both formats and close the figure; PNG/PDF paths mirror each other.

    A custom --output outside these trees retains the old sibling-file behavior.
    """
    path = Path(path).resolve()
    if path.suffix not in (".png", ".pdf"):
        raise ValueError("figure output must end in .png or .pdf")
    if path.is_relative_to(PNG):
        png_path = path.with_suffix(".png")
        pdf_path = (PDF / path.relative_to(PNG)).with_suffix(".pdf")
    elif path.is_relative_to(PDF):
        pdf_path = path.with_suffix(".pdf")
        png_path = (PNG / path.relative_to(PDF)).with_suffix(".png")
    else:
        png_path, pdf_path = path.with_suffix(".png"), path.with_suffix(".pdf")
    for output in (png_path, pdf_path):
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches=bbox_inches, facecolor="white", dpi=dpi)
    plt.close(fig)
    print(f"wrote {png_path} and {pdf_path}")


def save_figure(fig, number: int, panel: str | None = None, *, bbox_inches: str | None = "tight", dpi: int = 300) -> None:
    """Save and close a numbered figure, optionally as a separate subfigure."""
    path = PNG / (f"figure{number}.png" if panel is None else f"figure{number}/{panel}.png")
    save(fig, path, bbox_inches=bbox_inches, dpi=dpi)
