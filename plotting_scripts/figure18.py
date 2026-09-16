"""Figure 18: dataset comparison and Untied Grow corpus extrapolation."""
from __future__ import annotations
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares
from common import save_figure
from appendix_style import ARCHITECTURE_COLORS, ARCHITECTURE_LABELS, ARCHITECTURE_MARKERS, INK, MUTED, add_top_legend, legend_row_count

from scipy.optimize import brentq
from matplotlib.ticker import FixedLocator, NullFormatter
import figure20 as taskwise
from appendix_ladders_corpus_style import FULL_WIDTH, ANNOTATION_SIZE, apply_style, make_grid
C_REF = 1e18


INLINE_FONT_SIZE = 6


CORPUS_MARKERS = {"FineWeb": "v", "FineWeb-Edu": "o"}


D26_CHECK = {
    "parameters": 7_424_049_152,
    "compute": 1.2252577235512496e21,
    "validation_loss": 2.1371877569901314,
    "core_nll": 1.9900844955083101,
    "core": 0.3865253860573819,
    "core_std": 0.0014668547290330187,
}


@dataclass(frozen=True)
class Series:
    name: str
    compute: np.ndarray
    validation_loss: np.ndarray
    core_nll: np.ndarray
    core: np.ndarray
    filtered_core: np.ndarray


@dataclass(frozen=True)
class SaturatingPowerLawFit:
    amplitude: float
    alpha: float
    irreducible: float
    r_squared: float
    rms_residual: float

    def value_at_compute(self, compute):
        return self.irreducible + self.amplitude * (
            np.asarray(compute) / C_REF) ** (-self.alpha)


def series_from_model(
        model: taskwise.TaskwiseModel, arm: str, name: str) -> Series:
    records = model.records[arm]
    return Series(
        name=name,
        compute=np.asarray([record["compute"] for record in records]),
        validation_loss=np.asarray([
            record["raw"]["val_loss"] for record in records
        ]),
        core_nll=np.asarray([
            record["raw"]["core_loss"] for record in records
        ]),
        core=np.asarray([record["full_accuracy"] for record in records]),
        filtered_core=np.asarray([
            record["filtered_accuracy"] for record in records
        ]),
    )


def fit_saturating_power_law(
        compute: np.ndarray, values: np.ndarray,
        name: str) -> SaturatingPowerLawFit:
    """Huber-fit y = E + A (C / 1e18)^-alpha in metric space."""
    compute_ratio = np.asarray(compute, dtype=float) / C_REF
    values = np.asarray(values, dtype=float)

    def residual(params):
        amplitude, alpha, irreducible = params
        return irreducible + amplitude * compute_ratio ** (-alpha) - values

    initial = None
    for irreducible in np.linspace(0.0, values.min() * 0.995, 600):
        slope, intercept = np.polyfit(
            np.log(compute_ratio), np.log(values - irreducible), 1)
        if slope >= 0:
            continue
        candidate = np.asarray([np.exp(intercept), -slope, irreducible])
        squared_error = float(np.sum(residual(candidate) ** 2))
        if initial is None or squared_error < initial[0]:
            initial = squared_error, candidate
    if initial is None:
        raise ValueError(f"{name} has no decreasing saturating power-law fit")

    bounds = (
        [0.0, 0.0, 0.0],
        [np.inf, np.inf, np.nextafter(values.min(), 0.0)],
    )
    ordinary = least_squares(residual, initial[1], bounds=bounds)
    ordinary_residual = residual(ordinary.x)
    mad = np.median(np.abs(ordinary_residual - np.median(ordinary_residual)))
    huber_scale = max(1.345 * mad / 0.67448975, 1e-6)
    robust = least_squares(
        residual, ordinary.x, bounds=bounds,
        loss="huber", f_scale=huber_scale)

    amplitude, alpha, irreducible = (float(value) for value in robust.x)
    prediction = irreducible + amplitude * compute_ratio ** (-alpha)
    denominator = float(np.sum((values - values.mean()) ** 2))
    r_squared = 1.0 - float(
        np.sum((prediction - values) ** 2)) / denominator
    rms_residual = float(np.sqrt(np.mean((prediction - values) ** 2)))
    return SaturatingPowerLawFit(
        amplitude=amplitude,
        alpha=alpha,
        irreducible=irreducible,
        r_squared=r_squared,
        rms_residual=rms_residual,
    )


def fit_fixed_floor_power_law(
        compute: np.ndarray, values: np.ndarray,
        irreducible: float) -> SaturatingPowerLawFit:
    """Regress log(y - E) on log(C / 1e18) with a fixed floor."""
    compute = np.asarray(compute, dtype=float)
    values = np.asarray(values, dtype=float)
    excess = values - irreducible
    if not (np.all(np.isfinite(compute)) and np.all(compute > 0)
            and np.all(np.isfinite(excess)) and np.all(excess > 0)):
        raise ValueError("Log regression requires positive compute and loss above floor")
    slope, intercept = np.polyfit(np.log(compute / C_REF), np.log(excess), 1)
    amplitude, alpha = float(np.exp(intercept)), float(-slope)
    prediction = irreducible + amplitude * (compute / C_REF) ** (-alpha)
    squared_error = float(np.sum((prediction - values) ** 2))
    return SaturatingPowerLawFit(
        amplitude=amplitude,
        alpha=alpha,
        irreducible=irreducible,
        r_squared=1.0 - squared_error / float(np.sum((values - values.mean()) ** 2)),
        rms_residual=float(np.sqrt(squared_error / len(values))),
    )


def fit_metrics(
        series: tuple[Series, ...], *, baseline: str | None = None,
        ) -> dict[str, dict[str, SaturatingPowerLawFit]]:
    if baseline is not None:
        reference = next(item for item in series if item.name == baseline)
        floors = {
            metric: fit_saturating_power_law(
                reference.compute, getattr(reference, metric),
                f"{reference.name} {metric}").irreducible
            for metric in ("validation_loss", "core_nll")
        }
        return {
            item.name: {
                metric: fit_fixed_floor_power_law(
                    item.compute, getattr(item, metric), floor)
                for metric, floor in floors.items()
            }
            for item in series
        }
    return {
        item.name: {
            "validation_loss": fit_saturating_power_law(
                item.compute, item.validation_loss,
                f"{item.name} validation loss"),
            "core_nll": fit_saturating_power_law(
                item.compute, item.core_nll,
                f"{item.name} CORE NLL"),
        }
        for item in series
    }


def plot_loss_metric(
        axis, series: tuple[Series, ...], fits, metric: str,
        colors: dict[str, str], markers: dict[str, str],
        plot_maximum: float, d26_series: str | None) -> float | None:
    for item in series:
        fit = fits[item.name][metric]
        measured_compute = np.geomspace(
            item.compute.min(), item.compute.max(), 200)
        extrapolated_compute = np.geomspace(
            item.compute.max(), plot_maximum, 300)
        axis.plot(
            measured_compute, fit.value_at_compute(measured_compute),
            color=colors[item.name])
        axis.plot(
            extrapolated_compute, fit.value_at_compute(extrapolated_compute),
            color=colors[item.name], linestyle="--")
        axis.plot(
            item.compute, getattr(item, metric), linestyle="none",
            marker=markers[item.name], color=colors[item.name], zorder=3)

    forecast = None
    check_value = None
    if d26_series is not None:
        check_compute = D26_CHECK["compute"]
        check_value = D26_CHECK[metric]
        forecast = float(
            fits[d26_series][metric].value_at_compute(check_compute))
        axis.plot(
            [check_compute, check_compute], [forecast, check_value],
            color=colors[d26_series], linestyle=":", alpha=0.8)
        axis.plot(
            check_compute, check_value, linestyle="none", marker="s",
            markerfacecolor="white", markeredgecolor=colors[d26_series],
            markeredgewidth=1.8, zorder=5)

    minimum = min(
        float(fits[item.name][metric].value_at_compute(plot_maximum))
        for item in series
    )
    maximum = max(float(getattr(item, metric).max()) for item in series)
    axis.set_xscale("log")
    axis.set_xlim(min(item.compute.min() for item in series) / 1.35,
                  plot_maximum * 1.08)
    lower = minimum if check_value is None else min(minimum, check_value)
    axis.set_ylim(lower - 0.10, maximum + 0.16)
    return forecast


def predict_core(
        model: taskwise.TaskwiseModel,
        calibration: taskwise.TaskwiseCalibration | taskwise.TaskwiseModel,
        arm: str, compute):
    task_predictions = [
        taskwise.task_sigmoid(
            taskwise.power_law(
                compute, *model.nll_fits[arm][task]),
            *calibration.accuracy_fits[task],
        )
        for task in model.selected_tasks
    ]
    filtered = np.mean(np.asarray(task_predictions), axis=0)
    return calibration.full_to_filtered_ratio * filtered


def compute_for_target(
        model: taskwise.TaskwiseModel,
        calibration: taskwise.TaskwiseCalibration | taskwise.TaskwiseModel,
        arm: str, target: float) -> float:
    root = brentq(
        lambda log_compute: (
            float(predict_core(model, calibration, arm, 10**log_compute))
            - target
        ),
        15.0,
        40.0,
    )
    return 10**root


def plot_core_metric(
        axis, series: tuple[Series, ...], models: dict[str, taskwise.TaskwiseModel],
        model_arms: dict[str, str], colors: dict[str, str],
        markers: dict[str, str], plot_maximum: float,
        calibration: taskwise.TaskwiseCalibration | taskwise.TaskwiseModel,
        *, show_targets: bool, d26_series: str | None,
        efficiency_series: tuple[str, str] | None = None,
        label_targets: bool = False,
        ) -> tuple[dict[str, dict[str, float]], float | None]:
    curve_compute = np.logspace(
        np.log10(min(item.compute.min() for item in series)),
        np.log10(plot_maximum),
        700,
    )
    targets = {item.name: {} for item in series}
    for item in series:
        model = models[item.name]
        arm = model_arms[item.name]
        observed_maximum = float(item.compute.max())
        fitted_domain = curve_compute <= observed_maximum
        extrapolated_domain = curve_compute >= observed_maximum
        axis.plot(
            curve_compute[fitted_domain],
            predict_core(
                model, calibration, arm, curve_compute[fitted_domain]),
            color=colors[item.name])
        axis.plot(
            curve_compute[extrapolated_domain],
            predict_core(
                model, calibration, arm, curve_compute[extrapolated_domain]),
            color=colors[item.name], linestyle="--")
        axis.plot(
            item.compute, item.core, linestyle="none",
            marker=markers[item.name], color=colors[item.name], zorder=3)
        if show_targets:
            for label, target in taskwise.GPT_TARGETS:
                compute = compute_for_target(
                    model, calibration, arm, target)
                targets[item.name][label] = compute
                axis.plot(
                    compute, target,
                    linestyle="none", marker="*", ms=7,
                    color=colors[item.name], markeredgecolor="white",
                    markeredgewidth=0.5, zorder=4)

    if show_targets:
        for index, (_label, target) in enumerate(taskwise.GPT_TARGETS):
            axis.axhline(
                target,
                color=INK, linewidth=0.75,
                linestyle="--" if label_targets or index % 2 == 0 else ":",
                alpha=0.65,
                zorder=1)
            if label_targets:
                axis.text(
                    0.02, target, _label.replace("GPT-3 ", ""),
                    transform=axis.get_yaxis_transform(),
                    ha="left", va="center", fontsize=INLINE_FONT_SIZE, color=INK,
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "pad": 0.6, "alpha": 0.88},
                    zorder=6)

    if efficiency_series is not None:
        efficient_series, baseline_series = efficiency_series
        for label, target in taskwise.GPT_TARGETS:
            efficient_compute = targets[efficient_series][label]
            baseline_compute = targets[baseline_series][label]
            ratio = baseline_compute / efficient_compute
            axis.text(
                0.35, target,
                f"{ratio:.1f}×",
                transform=axis.get_yaxis_transform(),
                ha="center", va="center",
                fontsize=INLINE_FONT_SIZE, color=INK,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": 0.8, "alpha": 0.9},
                zorder=7)

    forecast = None
    check_value = None
    if d26_series is not None:
        check_compute = D26_CHECK["compute"]
        check_value = D26_CHECK["core"]
        check_model = models[d26_series]
        check_arm = model_arms[d26_series]
        forecast = float(predict_core(
            check_model, calibration, check_arm, check_compute))
        axis.plot(
            [check_compute, check_compute], [forecast, check_value],
            color=colors[d26_series], linestyle=":", alpha=0.8)
        axis.errorbar(
            check_compute, check_value, yerr=D26_CHECK["core_std"],
            linestyle="none", marker="s", markerfacecolor="white",
            markeredgecolor=colors[d26_series], markeredgewidth=1.8,
            color=colors[d26_series], capsize=2.0, zorder=5)

    axis.set_xscale("log")
    axis.set_xlim(min(item.compute.min() for item in series) / 1.35,
                  plot_maximum * 1.08)
    upper = max(
        check_value if check_value is not None else 0.0,
        max(float(item.core.max()) for item in series),
        max(float(predict_core(
            models[item.name], calibration, model_arms[item.name],
            plot_maximum)) for item in series),
        max((target for _label, target in taskwise.GPT_TARGETS), default=0.0)
        if show_targets else 0.0,
    )
    axis.set_ylim(0.045, upper + (0.04 if efficiency_series else 0.025))
    return targets, forecast


def make_three_panel(legend_labels: tuple[str, ...], *, vertical=False):
    apply_style()
    width = FULL_WIDTH / 3 if vertical else FULL_WIDTH
    legend_rows = legend_row_count(legend_labels, width=width)
    grid = make_grid(
        3 if vertical else 1,
        1 if vertical else 3,
        width=width,
        sharex=vertical,
        titles=("Loss", "CORE NLL", "CORE accuracy"),
        shared_xlabel="Compute (FLOPs)",
        legend_rows=legend_rows,
    )
    return grid, legend_rows


def plot_corpus_comparison(
        fineweb_model: taskwise.TaskwiseModel,
        edu_model: taskwise.TaskwiseModel,
        *, vertical=False,
        ) -> tuple[dict, dict, taskwise.TaskwiseCalibration]:
    series = (
        series_from_model(fineweb_model, "Dep Grow", "FineWeb"),
        series_from_model(edu_model, "Dep Grow", "FineWeb-Edu"),
    )
    if fineweb_model.selected_tasks != edu_model.selected_tasks:
        raise ValueError("corpus models use different task filters")
    calibration = taskwise.fit_taskwise_calibration(
        fineweb_model.selected_tasks,
        (fineweb_model.records["Dep Grow"], edu_model.records["Dep Grow"]),
    )
    fits = fit_metrics(series)
    target_compute = {
        item.name: {
            label: compute_for_target(
                fineweb_model if item.name == "FineWeb" else edu_model,
                calibration, "Dep Grow", target)
            for label, target in taskwise.GPT_TARGETS
        }
        for item in series
    }
    plot_maximum = max(
        compute for values in target_compute.values() for compute in values.values())
    labels = ("FineWeb", "FineWeb-Edu", "GPT-3 capability level")
    (fig, axes), legend_rows = make_three_panel(labels, vertical=vertical)
    validation_axis, nll_axis, core_axis = axes
    plot_loss_metric(
        validation_axis, series, fits, "validation_loss",
        CORPUS_COLORS, CORPUS_MARKERS, plot_maximum, None)
    plot_loss_metric(
        nll_axis, series, fits, "core_nll",
        CORPUS_COLORS, CORPUS_MARKERS, plot_maximum, None)
    _targets, _core_forecast = plot_core_metric(
        core_axis, series,
        {"FineWeb": fineweb_model, "FineWeb-Edu": edu_model},
        {"FineWeb": "Dep Grow", "FineWeb-Edu": "Dep Grow"},
        CORPUS_COLORS, CORPUS_MARKERS, plot_maximum, calibration,
        show_targets=True, d26_series=None,
        efficiency_series=("FineWeb-Edu", "FineWeb"), label_targets=True)
    validation_axis.set_ylabel("Loss")
    nll_axis.set_ylabel("CORE NLL")
    core_axis.set_ylabel("CORE accuracy")
    handles = (
        Line2D([], [], color=CORPUS_COLORS["FineWeb"],
               marker=CORPUS_MARKERS["FineWeb"]),
        Line2D([], [], color=CORPUS_COLORS["FineWeb-Edu"],
               marker=CORPUS_MARKERS["FineWeb-Edu"]),
        Line2D([], [], color=INK, linewidth=0.75, linestyle="--", alpha=0.65),
    )
    add_top_legend(fig, handles, labels, rows=legend_rows)
    save_figure(fig, 18, "b", bbox_inches=None)
    return fits, {
        "core": float(predict_core(
            edu_model, calibration, "Dep Grow", D26_CHECK["compute"]))
    }, calibration

CORPUS_COLORS = {"FineWeb": plt.get_cmap("tab10")(2), "FineWeb-Edu": plt.get_cmap("tab10")(0)}
CORPORA = {"fineweb": "FineWeb", "fineweb_edu": "FineWeb-Edu"}


ARCHITECTURES = {"Vanilla": "van", "Dep Grow": "dep_grow"}


METRICS = (("val_loss", False), ("core_loss", False), ("core_metric", True))


def compute_multipliers(records, metric: str, *, increasing: bool):
    """Match Vanilla scores on improving Untied segments, in log compute."""
    untied = records["Dep Grow"]
    baseline_compute, multipliers = [], []
    for vanilla in records["Vanilla"]:
        target = vanilla["raw"][metric]
        candidates = [point["compute"] for point in untied
                      if np.isclose(point["raw"][metric], target, rtol=0, atol=1e-12)]
        for left, right in zip(untied, untied[1:]):
            y0, y1 = left["raw"][metric], right["raw"][metric]
            if (y1 <= y0 if increasing else y1 >= y0):
                continue
            if not min(y0, y1) <= target <= max(y0, y1):
                continue
            fraction = (target - y0) / (y1 - y0)
            candidates.append(np.exp(
                np.log(left["compute"]) + fraction
                * np.log(right["compute"] / left["compute"])))
        baseline_compute.append(vanilla["compute"])
        multipliers.append(vanilla["compute"] / min(candidates)
                           if candidates else np.nan)
    return np.asarray(baseline_compute), np.asarray(multipliers)


def plot_data_comparison() -> None:
    selected_tasks = tuple(taskwise.load_json("filter")["selected_tasks"])
    fig, axes = make_grid(
        3, 3, titles=(*CORPORA.values(), "Compute Multiplier") * 3,
        sharex=True,
        column_xlabels=("Compute (FLOPs)", "Compute (FLOPs)",
                        "Vanilla compute (FLOPs)"),
        legend_rows=1,
    )
    for row, label in enumerate((r"Loss above floor, $L-E$", "CORE NLL", "CORE Accuracy")):
        axes[row, 1].sharey(axes[row, 0])
        for axis in axes[row, :2]:
            axis.set_ylabel(label)
        axes[row, 2].set_ylabel("Compute Multiplier")
        axes[row, 2].axhline(1.0, color=MUTED, linestyle="--", linewidth=1.0, zorder=1)
        axes[row, 2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}×"))
    print("| corpus | series | Vanilla-fitted E | log-log slope | log-space R² |")
    print("| --- | --- | ---: | ---: | ---: |")
    for column, (corpus, title) in enumerate(CORPORA.items()):
        axis = axes[0, column]
        records = taskwise.load_corpus_records(corpus, selected_tasks)
        vanilla_compute = np.asarray([point["compute"] for point in records["Vanilla"]])
        vanilla_loss = np.asarray([point["raw"]["val_loss"] for point in records["Vanilla"]])
        floor_fit = fit_saturating_power_law(
            vanilla_compute, vanilla_loss, f"{title} Vanilla")
        for index, (arm, architecture) in enumerate(ARCHITECTURES.items()):
            compute = np.asarray([point["compute"] for point in records[arm]])
            loss = np.asarray([point["raw"]["val_loss"] for point in records[arm]])
            if not (np.all(np.isfinite(compute)) and np.all(compute > 0)
                    and np.all(np.isfinite(loss))):
                raise ValueError(f"Invalid scaling data for {title} {arm}")
            excess = loss - floor_fit.irreducible
            if not np.all(excess > 0):
                raise ValueError(f"Nonpositive floor-subtracted loss for {title} {arm}")

            # Both architectures share the corpus's Vanilla-fitted Huber floor.
            log_compute, log_loss = np.log(compute / C_REF), np.log(excess)
            slope, intercept = np.polyfit(log_compute, log_loss, 1)
            residual = log_loss - (intercept + slope * log_compute)
            r_squared = 1 - np.sum(residual**2) / np.sum((log_loss - log_loss.mean())**2)
            curve_compute = np.geomspace(compute.min(), compute.max(), 200)
            curve_loss = np.exp(intercept) * (curve_compute / C_REF)**slope
            color = ARCHITECTURE_COLORS[architecture]
            for row, (metric, _increasing) in enumerate(METRICS[1:], start=1):
                values = np.asarray([point["raw"][metric] for point in records[arm]])
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Invalid {metric} data for {title} {arm}")
                axes[row, column].plot(
                    compute, values, color=color,
                    marker=ARCHITECTURE_MARKERS[architecture], zorder=3,
                )
            axis.plot(curve_compute, curve_loss, color=color)
            axis.plot(
                compute, excess, linestyle="none", color=color,
                marker=ARCHITECTURE_MARKERS[architecture], zorder=3,
            )
            axis.text(
                0.97, 0.95 - 0.115 * index,
                f"{ARCHITECTURE_LABELS[architecture]}: {slope:.5f}",
                transform=axis.transAxes, ha="right", va="top", fontsize=ANNOTATION_SIZE,
                color=color,
            )
            print(f"| {title} | {ARCHITECTURE_LABELS[architecture]} | "
                  f"{floor_fit.irreducible:.4f} | {slope:.5f} | {r_squared:.5f} |")

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_ylim(0.8, 2.45)
        axis.yaxis.set_major_locator(FixedLocator([0.8, 1.0, 1.2, 1.5, 2.0, 2.4]))
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        axis.yaxis.set_minor_formatter(NullFormatter())

        for row, (metric, increasing) in enumerate(METRICS):
            compute, multiplier = compute_multipliers(records, metric, increasing=increasing)
            axes[row, 2].plot(
                compute, multiplier, color=CORPUS_COLORS[title],
                marker=CORPUS_MARKERS[title], label=title,
            )

    for axis in axes.flat:
        axis.set_xscale("log")

    handles = [
        Line2D([], [], color=ARCHITECTURE_COLORS[architecture],
               marker=ARCHITECTURE_MARKERS[architecture],
               label=ARCHITECTURE_LABELS[architecture])
        for architecture in ARCHITECTURES.values()
    ]
    handles.extend(
        Line2D([], [], color=CORPUS_COLORS[title], marker=CORPUS_MARKERS[title], label=title)
        for title in CORPORA.values()
    )
    add_top_legend(fig, handles, [handle.get_label() for handle in handles], rows=1)
    save_figure(fig, 18, "a", bbox_inches=None)

def main():
    plot_data_comparison()
    plot_corpus_comparison(taskwise.build_taskwise_model("fineweb"),
                           taskwise.build_taskwise_model("fineweb_edu"), vertical=True)

if __name__ == "__main__":
    main()
