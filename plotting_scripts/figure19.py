"""Figure 19: downstream scaling and validation-loss calibration."""
from __future__ import annotations
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares
from common import save_figure
from appendix_style import GRID, MUTED, add_top_legend, legend_row_count

from appendix_ladders import load_json, ARM_COLORS, ARM_LABEL, ARM_MARKERS, LADDER_MAJOR_ARMS
from appendix_style import make_grid
from appendix_ladders_geometry import FULL_WIDTH, PANEL_SIZE, apply_style as apply_paper_style


@dataclass(frozen=True)
class CoreMetrics:
    full_core: float
    answer_nll: float
    filtered_core: float
    filtered_core_std: float


@dataclass(frozen=True)
class LadderEvaluation:
    compute: float
    loss: float
    depth: int
    metrics: CoreMetrics
    task_nll: dict[str, float]


def load_ladder_evaluations() -> dict[str, list[LadderEvaluation]]:
    """Read the downstream measurements shared by all three panels."""
    evaluations = {}
    for arm, points in load_json("downstream").items():
        evaluations[arm] = [
            LadderEvaluation(
                compute=point["compute"],
                loss=point["loss"],
                depth=point["depth"],
                metrics=CoreMetrics(**point["metrics"]),
                task_nll=point["task_nll"],
            )
            for point in points
        ]
    return evaluations

NORMAL_TITLES = (
    "Loss",
    "CORE accuracy",
    "CORE NLL",
    "Filtered tasks",
    "Loss",
    "CORE accuracy",
    "CORE NLL",
    "Filtered tasks",
)


SHARED_XLABEL = "Compute (FLOPs)"


NORMAL_LEGEND_ROWS = 2


NORMAL_WIDTH = FULL_WIDTH


X_TICK_ROWS = 1


Y_TICK_COLUMNS = 1


TOP_TICK_ROWS = 0


RIGHT_TICK_COLUMNS = 0


def fit_loss_power_law(x, y, compute_d8: float) -> tuple[float, float, float]:
    """Huber-fit L = L0 * (C / C_d8) ** -alpha + E in loss space."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    compute_ratio = x / compute_d8

    def residual(params):
        loss0, alpha, irreducible = params
        return loss0 * compute_ratio ** (-alpha) + irreducible - y

    # Scan E for a stable starting point.  Conditional on E, the other two
    # parameters come from a log-log line.
    initial = None
    for irreducible in np.linspace(0.0, y.min() * 0.995, 400):
        slope, intercept = np.polyfit(
            np.log(compute_ratio), np.log(y - irreducible), 1)
        if slope >= 0:
            continue
        candidate = np.array([np.exp(intercept), -slope, irreducible])
        squared_error = float(np.sum(residual(candidate) ** 2))
        if initial is None or squared_error < initial[0]:
            initial = (squared_error, candidate)
    if initial is None:
        raise ValueError("loss series does not admit a decreasing power-law fit")

    bounds = ([0.0, 0.0, 0.0],
              [np.inf, np.inf, np.nextafter(y.min(), 0.0)])
    ordinary_fit = least_squares(residual, initial[1], bounds=bounds)
    ordinary_residual = residual(ordinary_fit.x)
    mad = np.median(np.abs(ordinary_residual - np.median(ordinary_residual)))
    huber_scale = max(1.345 * mad / 0.67448975, 1e-6)
    robust_fit = least_squares(
        residual, ordinary_fit.x, bounds=bounds, loss="huber",
        f_scale=huber_scale,
    )
    return tuple(float(value) for value in robust_fit.x)


def plot_loss_power_law(ax, arm: str, x, y, compute_d8: float,
                        fit=None, show_equation: bool = True) -> None:
    """Draw measured loss markers and their Huber power-law fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    colour = ARM_COLORS[arm]
    if fit is None:
        fit = fit_loss_power_law(x, y, compute_d8)
    loss0, alpha, irreducible = fit
    fit_x = np.geomspace(x.min(), x.max(), 200)
    fit_y = loss0 * (fit_x / compute_d8) ** (-alpha) + irreducible
    equation = (
        rf"{ARM_LABEL[arm]}: $L={loss0:.2f}"
        rf"(C/C_{{d8}})^{{-{alpha:.2f}}}+{irreducible:.2f}$"
    )
    label = equation if show_equation else ARM_LABEL[arm]
    ax.plot(fit_x, fit_y, lw=1.9 if show_equation else 0.9,
            color=colour, label=label, zorder=2)
    ax.plot(x, y, linestyle="none", marker=ARM_MARKERS[arm],
            ms=6.5 if show_equation else 3.5,
            color=colour, zorder=3)


def compute_multiplier(points: list[tuple[float, float]], target: float,
                       vanilla_compute: float,
                       lower_is_better: bool) -> float:
    """Return C_van/C_arm from improving observed brackets without extrapolation."""
    points = sorted(points)
    tolerance = 1e-12 * max(1.0, abs(target))
    candidates = []
    for compute, value in points:
        if abs(value - target) <= tolerance:
            candidates.append(compute)
    for (compute0, value0), (compute1, value1) in zip(points, points[1:]):
        improving = value1 < value0 if lower_is_better else value1 > value0
        if not improving or value1 == value0:
            continue
        if not min(value0, value1) <= target <= max(value0, value1):
            continue
        fraction = (target - value0) / (value1 - value0)
        log_compute = np.log(compute0) + fraction * (
            np.log(compute1) - np.log(compute0))
        candidates.append(float(np.exp(log_compute)))
    if not candidates:
        return float("nan")
    return vanilla_compute / min(candidates)
def plot_ladders():
    """Plot all 58 major-ladder checkpoints and their measured multipliers."""
    apply_paper_style()
    arms = LADDER_MAJOR_ARMS
    evaluations = load_ladder_evaluations()
    fig, axes = make_grid(
        2, 4, titles=NORMAL_TITLES, width=NORMAL_WIDTH, square_panels=True,
        sharex=True, sharey=False, shared_xlabel=SHARED_XLABEL,
        legend_rows=NORMAL_LEGEND_ROWS, x_tick_rows=X_TICK_ROWS,
        y_tick_columns=Y_TICK_COLUMNS, top_tick_rows=TOP_TICK_ROWS,
        right_tick_columns=RIGHT_TICK_COLUMNS,
    )
    loss_axis, core_axis, nll_axis, filtered_axis = axes[0]
    for arm in arms:
        points = evaluations[arm]
        compute = [point.compute for point in points]
        loss = [point.loss for point in points]
        compute_d8 = next(point.compute for point in points if point.depth == 8)
        plot_loss_power_law(loss_axis, arm, compute, loss, compute_d8, show_equation=False)
        for axis, field in ((core_axis, "full_core"), (nll_axis, "answer_nll"),
                            (filtered_axis, "filtered_core")):
            axis.plot(compute, [getattr(point.metrics, field) for point in points],
                      marker=ARM_MARKERS[arm], ms=3.5, lw=0.9,
                      color=ARM_COLORS[arm], label=ARM_LABEL[arm], zorder=3)
        filtered_axis.errorbar(
            compute, [point.metrics.filtered_core for point in points],
            yerr=[point.metrics.filtered_core_std for point in points],
            fmt="none", ecolor=ARM_COLORS[arm], elinewidth=0.9,
            capsize=2.5, alpha=0.75, zorder=2,
        )
    loss_axis.set(xscale="log", yscale="log", ylabel="Loss")
    for setter in (loss_axis.yaxis.set_major_formatter, loss_axis.yaxis.set_minor_formatter):
        setter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    for axis, label in zip(axes[0, 1:], ("CORE accuracy", "CORE NLL", "CORE accuracy")):
        axis.set(xscale="log", ylabel=label)

    definitions = (("loss", True, 0.985), ("full_core", False, 0.65),
                   ("answer_nll", True, 0.985), ("filtered_core", False, 0.985))
    reference_compute = np.asarray(sorted(point.compute for point in evaluations["van"]
                                          if point.depth >= 8))
    for axis, (field, lower_is_better, lower_bound) in zip(axes[1], definitions):
        series = {
            arm: [(point.compute, point.loss if field == "loss" else getattr(point.metrics, field))
                  for point in points]
            for arm, points in evaluations.items()
        }
        reference = dict(series["van"])
        for arm in arms:
            multiplier = [1.0 if arm == "van" else compute_multiplier(
                series[arm], reference[compute], compute, lower_is_better)
                for compute in reference_compute]
            axis.plot(reference_compute, multiplier, marker=ARM_MARKERS[arm], ms=3.5,
                      lw=0.9, color=ARM_COLORS[arm], label=ARM_LABEL[arm],
                      zorder=3 if arm != "van" else 2)
        axis.set(xscale="log", ylabel="Compute Multiplier")
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
        axis.axhline(1.0, color=GRID, lw=0.9, zorder=1)
        finite = np.concatenate([line.get_ydata() for line in axis.lines[:-1]])
        finite = finite[np.isfinite(finite)]
        axis.set_ylim(bottom=min(lower_bound, float(finite.min()) * 0.95))
    handles, labels = loss_axis.get_legend_handles_labels()
    add_top_legend(fig, handles, labels, rows=NORMAL_LEGEND_ROWS)
    save_figure(fig, 19, "a", bbox_inches=None)


def plot_pooled_residuals() -> None:
    apply_paper_style()
    major_data = load_ladder_evaluations()
    labels = tuple(ARM_LABEL[arm] for arm in LADDER_MAJOR_ARMS)
    data = {
        arm: (
            np.asarray([point.loss for point in points]),
            np.asarray([point.metrics.answer_nll for point in points]),
        )
        for arm, points in major_data.items()
    }
    all_x = np.concatenate([x for x, _ in data.values()])
    all_y = np.concatenate([y for _, y in data.values()])
    if not (np.isfinite(all_x).all() and np.isfinite(all_y).all()):
        raise ValueError("non-finite validation loss or CORE NLL")
    # One ordinary least-squares line across all architectures, equal weight per checkpoint.
    slope, intercept = np.polyfit(all_x, all_y, 1)
    width = FULL_WIDTH / 3
    legend_rows = legend_row_count((*labels, "Pooled linear fit"), width=width)
    fig, axes = make_grid(
        2, 1,
        titles=("Major variants", "Linear-fit residuals"),
        width=width,
        square_panels=True,
        sharex=True,
        shared_xlabel="Validation loss",
        legend_rows=legend_rows,
        y_tick_columns=1,
    )
    axes[0].set_ylabel("CORE NLL")
    axes[1].set_ylabel("NLL − linear prediction")

    for arm, label in zip(LADDER_MAJOR_ARMS, labels):
        x, y = data[arm]
        for axis, values in zip(axes, (y, y - (slope * x + intercept))):
            axis.plot(
                x, values,
                color=ARM_COLORS[arm],
                linestyle="none",
                marker=ARM_MARKERS[arm],
                label=label,
                zorder=3,
            )

    fit_x = np.asarray([all_x.min(), all_x.max()])
    axes[0].plot(fit_x, slope * fit_x + intercept, color=MUTED,
                 linestyle="--", linewidth=0.9, label="Pooled linear fit", zorder=2)
    axes[1].axhline(0, color=MUTED, linestyle="--", linewidth=0.9, zorder=2)
    axes[0].invert_xaxis()
    for axis in axes:
        axis.invert_yaxis()
    handles, plotted_labels = axes[0].get_legend_handles_labels()
    add_top_legend(fig, handles, plotted_labels, rows=legend_rows)
    save_figure(fig, 19, "b", bbox_inches=None)

ARMS = LADDER_MAJOR_ARMS


CATEGORY_LABELS = {
    "world_knowledge": "World knowledge",
    "symbolic_problem_solving": "Symbolic problem\nsolving",
    "commonsense_reasoning": "Commonsense\nreasoning",
    "language_understanding": "Language understanding",
    "reading_comprehension": "Reading\ncomprehension",
}


def load_comparison():
    return {
        arm: sorted((point.loss, point.task_nll) for point in points)
        for arm, points in load_ladder_evaluations().items()
    }


def summarize_residuals(data, tasks):
    """Fit each task to Vanilla checkpoints; average measured residuals per arm."""
    all_x = np.asarray([val_loss for arm in ARMS for val_loss, _ in data[arm]])
    lower, upper = float(all_x.min()), float(all_x.max())
    intervals = (("All ladder points", lower, upper),
                 ("Lower-loss points", lower, (lower + upper) / 2))
    masks = {
        title: {
            arm: np.asarray([lo <= val_loss <= hi for val_loss, _ in data[arm]])
            for arm in ARMS
        }
        for title, lo, hi in intervals
    }
    if any(not mask.any() for arms in masks.values() for mask in arms.values()):
        raise ValueError("every architecture must have measured points in both selections")
    fits = {}
    residuals = {arm: [] for arm in ARMS}
    vanilla_x = np.asarray([val_loss for val_loss, _ in data["van"]])
    for task in tasks:
        vanilla_y = np.asarray([losses[task] for _, losses in data["van"]])
        slope, intercept = np.polyfit(vanilla_x, vanilla_y, 1)
        fits[task] = (float(slope), float(intercept))
        for arm in ARMS:
            residuals[arm].append(np.asarray([
                losses[task] - (slope * val_loss + intercept)
                for val_loss, losses in data[arm]
            ]))
    means = {
        title: {
            arm: np.asarray([values[masks[title][arm]].mean() for values in residuals[arm]])
            for arm in ARMS
        }
        for title, _, _ in intervals
    }
    return intervals, masks, fits, means


def plot_bars(groups, tasks, means, titles,
              ylabel="Mean (NLL − linear prediction), nats per token"):
    """Draw grouped residual bars with the common task categories and arm order."""
    apply_paper_style()
    legend_rows = legend_row_count(tuple(ARM_LABEL[arm] for arm in ARMS), width=FULL_WIDTH)
    fig, axes = make_grid(
        2, 1,
        titles=titles,
        width=FULL_WIDTH,
        panel_height=PANEL_SIZE,
        sharex=True,
        sharey=True,
        shared_xlabel="CORE task (Vanilla-filtered)",
        shared_ylabel=ylabel,
        legend_rows=legend_rows,
        x_tick_rows=6,
    )
    positions_by_task = {}
    group_bounds = {}
    next_position = 0.0
    for category, group in groups.items():
        for index, task in enumerate(group):
            positions_by_task[task] = next_position + index
        group_bounds[category] = (next_position - 0.43, next_position + len(group) - 0.57)
        next_position += len(group) + 0.8
    x = np.asarray([positions_by_task[task] for task in tasks])
    bar_width = 0.8 / len(ARMS)
    all_values = np.concatenate([values for arms in means.values() for values in arms.values()])
    padding = 0.08 * np.ptp(all_values)
    for axis, title in zip(axes, means):
        bounds = tuple(group_bounds.values())
        for (_, right), (left, _) in zip(bounds, bounds[1:]):
            axis.axvline((right + left) / 2, color=MUTED, linewidth=0.7, alpha=0.4, zorder=1)
        axis.axhline(0, color=MUTED, linestyle="--", linewidth=0.8, zorder=2)
        for index, arm in enumerate(ARMS):
            positions = x + (index - (len(ARMS) - 1) / 2) * bar_width
            axis.bar(positions, means[title][arm], width=bar_width * 0.94,
                     color=ARM_COLORS[arm], label=ARM_LABEL[arm], zorder=3)
        axis.set_xlim(x[0] - 0.6, x[-1] + 0.6)
        axis.grid(False, axis="x")
    axes[0].set_ylim(float(all_values.max() + padding), float(all_values.min() - padding))
    axes[-1].set_xticks(x, [TASK_LABELS[task] for task in tasks], rotation=50, ha="right")
    category_axis = axes[-1]
    transform = category_axis.get_xaxis_transform()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tick_bottom = min(label.get_window_extent(renderer).y0
                      for label in category_axis.get_xticklabels())
    bracket_y = category_axis.transAxes.inverted().transform(
        (0, tick_bottom - 0.06 * fig.dpi)
    )[1]
    for category, (left, right) in group_bounds.items():
        category_axis.plot([left, left, right, right],
                           [bracket_y + 0.025, bracket_y, bracket_y, bracket_y + 0.025],
                           transform=transform, clip_on=False, color=MUTED, linewidth=0.8)
        is_last = category == next(reversed(groups))
        category_axis.text(right if is_last else (left + right) / 2,
                           bracket_y - 0.025, CATEGORY_LABELS[category],
                           transform=transform, ha="right" if is_last else "center", va="top", fontsize=7,
                           fontweight="bold", clip_on=False)
    handles, labels = axes[0].get_legend_handles_labels()
    add_top_legend(fig, handles, labels, rows=legend_rows)
    save_figure(fig, 19, "c", bbox_inches=None)

TASK_LABELS = load_json("tasks")["labels"]

def main():
    plot_ladders()
    plot_pooled_residuals()
    data = load_comparison()
    metadata = load_json("tasks")
    intervals, masks, fits, means = summarize_residuals(data, metadata["tasks"])
    titles = tuple(f"{title} · validation loss {lo:.3f}–{hi:.3f}" for title, lo, hi in intervals)
    plot_bars(metadata["groups"], metadata["tasks"], means, titles,
              ylabel="Mean (NLL − Vanilla loss fit), nats per token")

if __name__ == "__main__":
    main()
