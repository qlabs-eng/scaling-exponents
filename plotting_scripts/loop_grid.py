"""The data-constrained loop grid: 100M tokens replayed for 10 epochs, over loop count, depth, and weight decay."""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from common import DATA, PADDED_VOCAB, flop_tick, quantity_colors, use_wandb

WEIGHT_DECAYS = (0.05, 0.2, 0.4, 0.8, 1.2, 1.6)
BUDGETS = tuple(value * 1e18 for value in (0.846120, 1.419447, 2.381258, 3.994787, 6.701636))  # the study's five compute budgets
FINE_BUDGETS = tuple(sorted(BUDGETS + tuple(math.sqrt(a * b) for a, b in zip(BUDGETS, BUDGETS[1:]))))  # with their log midpoints
# Grid run names encode depth, loop count, and optionally weight decay.
RUN_NAME = re.compile(r"(?:loop|k1wd)_d(\d\d)_k(\d+)(?:_wd[0-9p]+)?")
EPOCHS, TOKENS = 10, 100_000_000


@dataclass(frozen=True)
class Cell:
    depth: int
    loops: int
    weight_decay: float
    compute: float  # training FLOPs
    loss: float  # final validation loss


def loop_label(loops: int) -> str:
    return "Operator-1" if loops == 1 else f"Loop-{loops}"


def training_flops(depth: int, loops: int, steps: int, batch: int = 524_288) -> float:
    """The grid's compute convention: 6 x (block + head parameters) per executed layer, without attention."""
    width = 128 * depth
    hidden = 256 * ((8 * width // 3 + 255) // 256)
    block = 4 * width * width + 3 * width * hidden
    base, remainder = divmod(depth, 3)  # prelude, core, coda; the remainder goes to the core, then the coda
    layers = base + loops * (base + (remainder >= 1)) + base + (remainder >= 2)
    return 6.0 * (layers * block + width * PADDED_VOCAB) * steps * batch


def key(cell: Cell) -> tuple:
    return cell.depth, cell.loops, round(cell.weight_decay, 3)


def pull(entity: str, project: str, group: str | None = None, *, refresh: bool,
         cache=DATA / "wandb_loop_grid.json") -> list[Cell]:
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
            match = RUN_NAME.fullmatch(run.name)
            config, summary = run.config, run.summary
            loss = summary.get("final_val_loss", summary.get("val/final_step_loss", summary.get("val/loss")))
            in_regime = int(config.get("num_epochs") or 1) == EPOCHS and int(config.get("train_token_limit") or 0) == TOKENS
            if run.state != "finished" or not match or not in_regime or loss is None:
                continue
            depth, loops = int(match.group(1)), int(match.group(2))
            runs.append({"depth": depth, "loops": loops, "weight_decay": float(config["weight_decay"]), "created_at": str(run.created_at),
                         "compute": training_flops(depth, loops, int(config["max_train_steps"]), int(config["total_batch_size"])), "loss": float(loss)})
        if not runs:
            raise RuntimeError(f"no finished loop-grid runs in {entity}/{project}")
        cache.write_text(json.dumps({"source": source, "runs": runs}, indent=1) + "\n")
        print(f"pulled {len(runs)} loop-grid runs from {entity}/{project} into {cache}")
    newest: dict[tuple, dict] = {}
    for run in runs:
        cell = Cell(run["depth"], run["loops"], run["weight_decay"], run["compute"], run["loss"])
        if run["created_at"] > newest.get(key(cell), {}).get("created_at", ""):
            newest[key(cell)] = {**run, "cell": cell}
    return [run["cell"] for run in newest.values()]


def load_cells(args) -> list[Cell]:
    """Load the bundled grid, optionally replacing cells from an explicit W&B source."""
    cells = {}
    with open(DATA / "loop_grid.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            cell = Cell(int(row["depth"]), int(row["loops"]), float(row["weight_decay"]), float(row["compute_flops"]), float(row["val_loss"]))
            cells[key(cell)] = cell
    if use_wandb(args):
        try:
            for cell in pull(args.wandb_entity, args.wandb_project, args.wandb_group, refresh=args.refresh):
                cells[key(cell)] = cell
        except Exception as error:  # no credentials, no access, or no network
            print(f"W&B unavailable ({error}); using the bundled data/loop_grid.csv")
    return list(cells.values())


def series(cells: list[Cell], weight_decay: float = 0.8) -> dict[int, list[Cell]]:
    """One ladder per loop count, sorted by compute."""
    ladders: dict[int, list[Cell]] = {}
    for cell in cells:
        if math.isclose(cell.weight_decay, weight_decay):
            ladders.setdefault(cell.loops, []).append(cell)
    return {loops: sorted(points, key=lambda cell: cell.compute) for loops, points in sorted(ladders.items())}


def interpolate_loss(points: list[Cell], compute: float) -> float:
    """Loss at a budget, linear in compute between the two bracketing models."""
    for lower, upper in zip(points, points[1:]):
        if lower.compute <= compute <= upper.compute:
            fraction = (compute - lower.compute) / (upper.compute - lower.compute)
            return lower.loss + fraction * (upper.loss - lower.loss)
    raise ValueError(f"compute {compute:g} is outside the K{points[0].loops} ladder")


def iso_losses(ladders: dict[int, list[Cell]], compute: float) -> dict[int, float]:
    """Interpolated loss at a budget for every loop count whose ladder brackets it."""
    return {loops: interpolate_loss(points, compute) for loops, points in ladders.items() if points[0].compute <= compute <= points[-1].compute}


def fit_loop_optimum(losses: dict[int, float]):
    """Quadratic in log loss against log loop count, and its vertex."""
    loops = np.array(sorted(losses), dtype=float)
    coefficients = np.polyfit(np.log(loops), np.log([losses[int(k)] for k in loops]), 2)
    curvature, slope = coefficients[:2]
    if curvature <= 0:
        raise ValueError("no quadratic minimum in loop count")
    optimum = math.exp(-slope / (2 * curvature))
    return coefficients, optimum, math.exp(float(np.polyval(coefficients, math.log(optimum))))


def compute_at_loss(points: list[Cell], target: float) -> float:
    """Compute at which a decreasing frontier reaches a loss, linear in compute between the bracketing points."""
    for lower, upper in zip(points, points[1:]):
        if upper.loss < lower.loss and upper.loss <= target <= lower.loss:
            fraction = (target - lower.loss) / (upper.loss - lower.loss)
            return lower.compute + fraction * (upper.compute - lower.compute)
    raise ValueError(f"loss {target:g} is not bracketed")


def convex_envelope(points: list[Cell], *, log_compute: bool = False) -> list[Cell]:
    """Lower convex envelope of the Pareto front, in raw or log compute."""
    pareto, best = [], math.inf
    for point in sorted(points, key=lambda cell: (cell.compute, cell.loss)):
        if point.loss < best:
            pareto.append(point)
            best = point.loss
    x = (lambda cell: math.log(cell.compute)) if log_compute else (lambda cell: cell.compute)

    def turns_left(a, b, c):
        return (x(b) - x(a)) * (c.loss - a.loss) - (b.loss - a.loss) * (x(c) - x(a)) <= 0

    envelope = []
    for point in pareto:
        while len(envelope) >= 2 and turns_left(envelope[-2], envelope[-1], point):
            envelope.pop()
        envelope.append(point)
    return envelope


def clip_after(points: list[Cell], cap: float) -> list[Cell]:
    """Keep points up to the first one at or beyond a compute cap."""
    kept = []
    for point in points:
        kept.append(point)
        if point.compute >= cap:
            break
    return kept


def compute_axis(axis, upper: float = 12.5e18) -> None:
    axis.set_xscale("log")
    axis.set_xlim(0.38e18, upper)
    axis.xaxis.set_major_locator(FixedLocator((0.5e18, 1e18, 2e18, 5e18, 1e19)))
    axis.xaxis.set_major_formatter(FuncFormatter(flop_tick))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Compute (FLOPs)")


def loop_axis(axis, loops) -> None:
    axis.set_xscale("log", base=2)
    axis.set_xlim(0.86, max(loops) * 1.12)
    axis.xaxis.set_major_locator(FixedLocator(loops))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"K{value:g}"))
    axis.xaxis.set_minor_formatter(NullFormatter())
    axis.set_xlabel("Loop count")


def plot_cuts(axis, ladders, colors) -> None:
    """Interpolated loss against loop count at each budget, with a quadratic fit and its minimum starred."""
    for budget in FINE_BUDGETS:
        losses = iso_losses(ladders, budget)
        loops = np.array(sorted(losses), dtype=float)
        coefficients, optimum, optimum_loss = fit_loop_optimum(losses)
        grid = np.geomspace(loops.min(), loops.max(), 240)
        axis.plot(grid, np.exp(np.polyval(coefficients, np.log(grid))), color=colors[budget], label=rf"${budget / 1e18:#.2g}\times10^{{18}}$")
        axis.plot(loops, [losses[int(k)] for k in loops], "o", color=colors[budget], markeredgewidth=0, zorder=3)
        axis.plot(optimum, optimum_loss, "*", color=colors[budget], markeredgecolor="white", markeredgewidth=0.6, markersize=13, zorder=4)


def best_weight_decays(cells, loops, depths) -> dict[tuple[int, int], float]:
    """The weight decay with the lowest loss for each (loop count, depth) cell that was swept."""
    best = {}
    for cell in cells:
        if cell.loops in loops and cell.depth in depths and round(cell.weight_decay, 3) in WEIGHT_DECAYS:
            if (cell.loops, cell.depth) not in best or cell.loss < best[cell.loops, cell.depth][0]:
                best[cell.loops, cell.depth] = (cell.loss, cell.weight_decay)
    return {key: weight_decay for key, (_, weight_decay) in best.items()}


def plot_heatmap(axis, best, loops, depths, *, text_size: float) -> None:
    colormap = ListedColormap(list(quantity_colors(WEIGHT_DECAYS, quantity="weight_decay").values()))
    colormap.set_bad("#ECEFF3")  # cells that were not trained
    categories = np.full((len(loops), len(depths)), np.nan)
    for (k, depth), weight_decay in best.items():
        categories[loops.index(k), depths.index(depth)] = WEIGHT_DECAYS.index(round(weight_decay, 3))
    # Keep the categorical cells as vector geometry in the PDF export.
    axis.pcolormesh(np.arange(len(depths) + 1) - 0.5, np.arange(len(loops) + 1) - 0.5,
                    np.ma.masked_invalid(categories), cmap=colormap,
                    vmin=-0.5, vmax=len(WEIGHT_DECAYS) - 0.5, shading="flat",
                    edgecolors="face", linewidth=0.25, antialiased=False, rasterized=False)
    axis.set_xlim(-0.5, len(depths) - 0.5)
    axis.set_ylim(len(loops) - 0.5, -0.5)
    for (k, depth), weight_decay in best.items():
        row, column = loops.index(k), depths.index(depth)
        red, green, blue = colormap(int(categories[row, column]))[:3]
        lightness = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        axis.text(column, row, f"{weight_decay:g}", ha="center", va="center", fontsize=text_size,
                  color="white" if lightness < 0.55 else "black")
    axis.set_xticks(range(len(depths)), [f"d{depth}" for depth in depths])
    axis.set_yticks(range(len(loops)), [loop_label(k) for k in loops])
    axis.set_xlabel("Depth")
    axis.tick_params(length=0)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


class LoopGradientHandle:
    """Legend entry for a frontier coloured by loop count: one segment per colour."""

    def __init__(self, colors: dict[int, str]):
        self.colors = colors


class LoopGradientHandler(HandlerBase):
    def create_artists(self, legend, handle, xdescent, ydescent, width, height, fontsize, trans):
        edges = np.linspace(xdescent, xdescent + width, len(handle.colors) + 1)
        y = ydescent + height / 2
        return [Line2D([start, stop], [y, y], color=color, linewidth=1.6, solid_capstyle="butt", transform=trans)
                for color, start, stop in zip(handle.colors.values(), edges, edges[1:])]
