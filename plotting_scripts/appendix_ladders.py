from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.optimize import least_squares
from scipy.stats import linregress
from common import arm_color, save_figure
from appendix_style import ARCHITECTURE_LABELS, GRID, add_top_legend, legend_row_count
from appendix_ladders_geometry import MARKER_SIZE, figure_width, apply_style, make_grid

ALL_VARIANT_ARMS = ("van", "van_glr_wd", "van_glr_om", "van_grow", "deep_van", "k1_glr", "k1_van_recipe", "k2", "k2_nci", "k2_fixed_pc", "k2_grow", "dep", "dep_grow")


LADDER_MAJOR_ARMS = ("van", "deep_van", "deep_van_grow", "k1_glr", "k2", "k2_grow", "dep", "dep_grow")


VANILLA_SCALING_ARMS = ("van_constant", "van", "van_glr_om", "van_glr_wd", "van_const_mup_om", "van_mup_om", "van_wdr1")


SHAPE_LADDER_ARMS = ("van", "deep_van", "k1_glr", "deeper_van", "deeper_k1")


K1_ARMS = ("van", "k1_glr", "k1_van_recipe")


CONSTANT_LADDER_ARMS = ("van", "k2", "dep", "k2_grow", "dep_grow")


ARM_LABEL = {
    "van_constant": "Vanilla Constant",
    "van": ARCHITECTURE_LABELS["van"],
    "van_glr_wd": "Vanilla + WD@0.1",
    "van_glr_om": "Vanilla + OM@-0.5",
    "van_const_mup_om": "Vanilla muP OM",
    "van_mup_om": "Vanilla muP OM + GLR",
    "van_wdr1": "Vanilla WDR1",
    "deep_van": "Deep Vanilla",
    "deeper_van": "Deeper Vanilla",
    "deep_van_grow": "Deep Vanilla Grow",
    "van_grow": "Vanilla Grow",
    "k1": ARCHITECTURE_LABELS["k1"],
    "k1_glr": ARCHITECTURE_LABELS["k1"],
    "deeper_k1": "Deeper Operator-1",
    "k1_1c2": "Operator-1 1/C/2",
    "k1_van_recipe": "Operator-1 Vanilla Recipe",
    "k2": ARCHITECTURE_LABELS["k2"],
    "k2_nci": "Loop-2 No-Coda-Inj",
    "k2_fixed_pc": "Loop-2 NCI 2/C/3",
    "k2_grow": ARCHITECTURE_LABELS["k2_grow"],
    "dep": ARCHITECTURE_LABELS["dep"],
    "dep_nonorm": "Untied-2 No-Norm",
    "dep_noinject": "Untied-2 No-Inject",
    "dep_nci": "Untied-2 No-Coda-Inj",
    "dep_grow": ARCHITECTURE_LABELS["dep_grow"],
}


ARM_MARKERS = {"van_constant": "x", "van": "o", "van_glr_wd": "<", "van_glr_om": ">",
               "van_const_mup_om": "v", "van_mup_om": "s", "van_wdr1": "P", "deep_van": "h", "deeper_van": "D", "deep_van_grow": "*", "van_grow": "s", "k1": "s", "k1_glr": "p", "deeper_k1": "X", "k1_1c2": "d", "k1_van_recipe": "H",
               "k2": "^", "k2_nci": "X", "k2_fixed_pc": "*", "k2_grow": "P", "dep": "D", "dep_nonorm": "p", "dep_noinject": "v", "dep_nci": "X", "dep_grow": "v"}


C_REF = 1e18


MARKER_SIZE_MIN = 3.5


MARKER_SIZE_MAX = 7.5


MARKER_PARAMETER_MIN = 50e6


MARKER_PARAMETER_MAX = 2e9


@dataclass(frozen=True)
class Point:
    compute: float
    loss: float
    depth: int
    parameters: int
    effective_depth: float | None = None
    width: int | None = None
    tokens: int | None = None


@dataclass(frozen=True)
class Fit:
    amplitude: float
    alpha: float
    irreducible: float
    r_squared: float
    alpha_standard_error: float | None = None

    def predict(self, compute):
        compute = np.asarray(compute, dtype=float)
        return self.irreducible + self.amplitude * (compute / C_REF) ** (-self.alpha)

    def inverse(self, loss):
        loss = np.asarray(loss, dtype=float)
        return C_REF * ((loss - self.irreducible) / self.amplitude) ** (-1.0 / self.alpha)


def model_marker_size(parameters: int) -> float:
    """Map stored parameter count to marker diameter in points."""
    position = (np.log(float(parameters)) - np.log(MARKER_PARAMETER_MIN)) / (
        np.log(MARKER_PARAMETER_MAX) - np.log(MARKER_PARAMETER_MIN))
    position = float(np.clip(position, 0.0, 1.0))
    return MARKER_SIZE_MIN + position * (MARKER_SIZE_MAX - MARKER_SIZE_MIN)


def plot_sized_markers(axis, x, y, points, *, arm: str, zorder: int = 3,
                       size_scale: float = 1.0,
                       fixed_size: float | None = None, hollow: bool = False) -> None:
    """Plot measured points with size encoding unless a fixed diameter is given."""
    marker_sizes = [
        (fixed_size if fixed_size is not None else size_scale * model_marker_size(point.parameters)) ** 2
        for point in points
    ]
    colors = ({"facecolors": "white", "edgecolors": ARM_COLORS[arm]}
              if hollow else {"color": ARM_COLORS[arm]})
    axis.scatter(
        x,
        y,
        s=marker_sizes,
        marker=ARM_MARKERS[arm],
        **colors,
        linewidths=0.8 if hollow else 0.6,
        zorder=zorder,
    )


def fit_shared_irreducible(data: dict[str, list[Point]], arms: tuple[str, ...]) -> dict[str, Fit]:
    """Joint Huber fit with one E and arm-specific amplitude/exponent."""
    series = {
        arm: (
            np.asarray([point.compute for point in data[arm]], dtype=float) / C_REF,
            np.asarray([point.loss for point in data[arm]], dtype=float),
        )
        for arm in arms
    }
    if any(len(x) < 3 for x, _ in series.values()):
        raise ValueError("each paper-figure arm needs at least three loss measurements")

    def residual(params):
        irreducible = params[-1]
        pieces = []
        for index, arm in enumerate(arms):
            amplitude, alpha = params[2 * index:2 * index + 2]
            x, y = series[arm]
            pieces.append(irreducible + amplitude * x ** (-alpha) - y)
        return np.concatenate(pieces)

    smallest_loss = min(float(y.min()) for _, y in series.values())
    initial = None
    for irreducible in np.linspace(0.0, smallest_loss * 0.995, 600):
        params = []
        valid = True
        for arm in arms:
            x, y = series[arm]
            slope, intercept = np.polyfit(np.log(x), np.log(y - irreducible), 1)
            if slope >= 0:
                valid = False
                break
            params.extend((np.exp(intercept), -slope))
        if not valid:
            continue
        candidate = np.asarray([*params, irreducible])
        squared_error = float(np.sum(residual(candidate) ** 2))
        if initial is None or squared_error < initial[0]:
            initial = squared_error, candidate
    if initial is None:
        raise ValueError("loss series do not admit a decreasing shared-E power-law fit")

    lower = np.zeros(len(initial[1]))
    upper = np.full(len(initial[1]), np.inf)
    upper[-1] = np.nextafter(smallest_loss, 0.0)
    ordinary = least_squares(residual, initial[1], bounds=(lower, upper))
    ordinary_residual = residual(ordinary.x)
    mad = np.median(np.abs(ordinary_residual - np.median(ordinary_residual)))
    huber_scale = max(1.345 * mad / 0.67448975, 1e-6)
    robust = least_squares(
        residual, ordinary.x, bounds=(lower, upper), loss="huber", f_scale=huber_scale)

    irreducible = float(robust.x[-1])
    fits = {}
    for index, arm in enumerate(arms):
        amplitude, alpha = (float(value) for value in robust.x[2 * index:2 * index + 2])
        x, y = series[arm]
        prediction = irreducible + amplitude * x ** (-alpha)
        denominator = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum((prediction - y) ** 2)) / denominator
        fits[arm] = Fit(amplitude, alpha, irreducible, r_squared)
    return fits


def fit_vanilla_baseline_power_laws(
        data: dict[str, list[Point]], arms: tuple[str, ...],
        irreducible: float | None = None,
        shared_base: Point | None = None) -> dict[str, Fit]:
    """Fit or fix Vanilla's E, then regress log(L - E) on the x coordinate."""
    if irreducible is None:
        irreducible = fit_shared_irreducible(data, ("van",))["van"].irreducible
    fits = {}
    for arm in arms:
        compute = np.asarray(
            [point.compute for point in data[arm]], dtype=float) / C_REF
        adjusted_loss = np.asarray(
            [point.loss for point in data[arm]], dtype=float) - irreducible
        if len(compute) < 3:
            raise ValueError("each paper-figure arm needs at least three loss measurements")
        if np.any(adjusted_loss <= 0):
            raise ValueError(
                f"Vanilla fitted E must be below every measured loss for {arm}")
        log_loss = np.log(adjusted_loss)
        regression = linregress(np.log(compute), log_loss)
        slope = float(regression.slope)
        intercept = float(regression.intercept)
        r_squared = float(regression.rvalue ** 2)
        standard_error = float(regression.stderr)
        if shared_base is not None:
            # Regress through the shared base; copied anchors are not replicates.
            x = np.log(compute / (shared_base.compute / C_REF))
            base_log_loss = np.log(shared_base.loss - irreducible)
            y = log_loss - base_log_loss
            slope = float(np.dot(x, y) / np.dot(x, x))
            intercept = float(base_log_loss - slope * np.log(shared_base.compute / C_REF))
            residual_sum = float(np.sum((y - slope * x) ** 2))
            degrees_of_freedom = np.count_nonzero(x) - 1
            standard_error = float(np.sqrt(residual_sum / degrees_of_freedom / np.dot(x, x)))
            r_squared = float(1 - residual_sum / np.sum((log_loss - log_loss.mean()) ** 2))
        if slope >= 0:
            raise ValueError(
                f"{arm} does not admit a decreasing power law above Vanilla's fitted E")
        fits[arm] = Fit(
            float(np.exp(intercept)), float(-slope), irreducible,
            r_squared, standard_error)
    return fits


def interpolate_compute_multiplier(points: list[Point], target_loss: float,
                                   vanilla_compute: float) -> float:
    """Compute C_van/C_arm from a measured loss bracket in log compute.

    Only decreasing adjacent segments are eligible, and no extrapolation is used.
    Targets outside the measured loss range return NaN and are not plotted.
    """
    tolerance = 1e-12 * max(1.0, abs(target_loss))
    candidates = []
    for point in points:
        if abs(point.loss - target_loss) <= tolerance:
            candidates.append(point.compute)
    for point0, point1 in zip(points, points[1:]):
        if point1.loss >= point0.loss:
            continue
        if not point1.loss <= target_loss <= point0.loss:
            continue
        fraction = (target_loss - point0.loss) / (point1.loss - point0.loss)
        log_compute = np.log(point0.compute) + fraction * (
            np.log(point1.compute) - np.log(point0.compute))
        candidates.append(float(np.exp(log_compute)))
    if not candidates:
        return float("nan")
    return vanilla_compute / min(candidates)


def plot_coefficient_panel(axis, data, arms, fits):
    reference_points = [point for point in data["van"] if point.depth == 8]
    if len(reference_points) != 1:
        raise ValueError("expected exactly one Vanilla d8 compute reference")
    reference_compute = reference_points[0].compute
    for arm in arms:
        fit = fits[arm]
        intercept = np.log(fit.amplitude) - fit.alpha * np.log(reference_compute / C_REF)
        axis.plot(intercept, fit.alpha, linestyle="none", color=ARM_COLORS[arm],
                  marker=ARM_MARKERS[arm], markersize=MARKER_SIZE)
    axis.margins(x=0.12, y=0.24)
    axis.invert_xaxis()
    axis.set_ylabel(r"$\beta$")


def plot_figure(data: dict[str, list[Point]], arms: tuple[str, ...], number: int, panel: str,
                *, interpolation_multiplier: bool = False,
                fit_arms: tuple[str, ...] | None = None,
                labels: dict[str, str] | None = None,
                subtract_vanilla_irreducible: bool = False,
                x_label: str = "Compute (FLOPs)",
                scaling_variable: str = "C",
                fixed_irreducible: float | None = None,
                multiplier_baselines: tuple[str, ...] = ("van",),
                coefficient_panel: bool = False,
                shared_base: Point | None = None,
                hollow_arms: tuple[str, ...] = ()) -> None:
    apply_style()
    fit_arms = arms if fit_arms is None else fit_arms
    labels = ARM_LABEL if labels is None else ARM_LABEL | labels
    missing = [arm for arm in dict.fromkeys((*arms, *fit_arms)) if not data[arm]]
    if missing:
        raise ValueError(f"no data for: {', '.join(missing)}")
    fits = (
        fit_vanilla_baseline_power_laws(data, fit_arms, fixed_irreducible, shared_base)
        if subtract_vanilla_irreducible
        else fit_shared_irreducible(data, fit_arms)
    )
    panel_count = 1 + len(multiplier_baselines) + int(coefficient_panel)
    legend_labels = []
    for arm in arms:
        exponent = f"-{fits[arm].alpha:.4f}"
        if panel_count >= 3:
            exponent += f" \\pm {fits[arm].alpha_standard_error:.4f}"
        legend_labels.append(
            f"{labels[arm]} (${scaling_variable}^{{{exponent}}}$)"
            if subtract_vanilla_irreducible else labels[arm])
    width = figure_width(panel_count)
    legend_rows = legend_row_count(legend_labels, width=width)

    fig, axes = make_grid(
        1,
        panel_count,
        titles=(
            "Reducible Loss" if subtract_vanilla_irreducible else "Loss",
            *(f"vs. {labels[baseline]}"
              if len(multiplier_baselines) > 1 else "Compute Multiplier"
              for baseline in multiplier_baselines),
            *(("Scaling coefficients",) if coefficient_panel else ()),
        ),
        shared_xlabel=None if coefficient_panel else x_label,
        column_xlabels=([x_label] * (1 + len(multiplier_baselines)) + [r"$\log B$"]
                        if coefficient_panel else None),
        legend_rows=legend_rows,
    )
    loss_axis = axes[0]
    multiplier_axes = axes[1:1 + len(multiplier_baselines)]
    if coefficient_panel:
        plot_coefficient_panel(axes[-1], data, arms, fits)
    for arm in arms:
        points = data[arm]
        fit = fits[arm]
        measured_compute = np.asarray([point.compute for point in points])
        measured_loss = np.asarray([point.loss for point in points])
        fit_compute = np.geomspace(measured_compute.min(), measured_compute.max(), 300)
        plotted_loss = measured_loss - fit.irreducible if subtract_vanilla_irreducible else measured_loss
        plotted_fit = fit.predict(fit_compute) - fit.irreducible if subtract_vanilla_irreducible else fit.predict(fit_compute)
        loss_axis.plot(
            fit_compute, plotted_fit, color=ARM_COLORS[arm],
            linewidth=1.1 if subtract_vanilla_irreducible else None, zorder=2)
        marker_start = 1 if shared_base is not None else 0
        plot_sized_markers(
            loss_axis, measured_compute[marker_start:], plotted_loss[marker_start:],
            points[marker_start:], arm=arm, zorder=3,
            fixed_size=MARKER_SIZE, hollow=arm in hollow_arms)

    if shared_base is not None:
        loss_axis.plot(shared_base.compute, shared_base.loss - fits["van"].irreducible,
                       marker="o", linestyle="none", color=ARM_COLORS["van"], zorder=6)

    for multiplier_axis, baseline in zip(multiplier_axes, multiplier_baselines):
        baseline_points = [
            point for point in data[baseline]
            if baseline != "van" or 8 <= point.depth <= 20
        ]
        baseline_compute = np.asarray([point.compute for point in baseline_points])
        for arm in arms:
            if interpolation_multiplier:
                multiplier = np.asarray([
                    1.0 if arm == baseline else interpolate_compute_multiplier(
                        data[arm], point.loss, point.compute)
                    for point in baseline_points
                ])
            else:
                baseline_loss = fits[baseline].predict(baseline_compute)
                required_compute = fits[arm].inverse(baseline_loss)
                multiplier = baseline_compute / required_compute
            multiplier_axis.plot(
                baseline_compute, multiplier, color=ARM_COLORS[arm],
                zorder=4 if arm != baseline else 3)
            plot_sized_markers(
                multiplier_axis, baseline_compute[marker_start:], multiplier[marker_start:],
                baseline_points[marker_start:], arm=arm,
                zorder=5 if arm != baseline else 4,
                fixed_size=MARKER_SIZE, hollow=arm in hollow_arms)
        multiplier_axis.set_xscale("log")
        multiplier_axis.set_ylabel("Compute Multiplier")
        multiplier_axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}×"))
        multiplier_axis.axhline(1.0, color=GRID, lw=0.9, zorder=1)
        if shared_base is not None:
            multiplier_axis.plot(shared_base.compute, 1.0, marker="o", linestyle="none",
                                 color=ARM_COLORS["van"], zorder=6)

    loss_axis.set_xscale("log")
    loss_axis.set_yscale("log")
    loss_axis.set_ylabel(
        r"$L - E_{\mathrm{Van}}$" if subtract_vanilla_irreducible else "Loss")
    for formatter in (loss_axis.yaxis.set_major_formatter, loss_axis.yaxis.set_minor_formatter):
        formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    handles = tuple(
        Line2D([], [], color=ARM_COLORS[arm], marker=ARM_MARKERS[arm],
               markersize=MARKER_SIZE, label=legend_label,
               markerfacecolor="white" if arm in hollow_arms else ARM_COLORS[arm],
               markeredgecolor=ARM_COLORS[arm], markeredgewidth=0.8 if arm in hollow_arms else 0.6)
        for arm, legend_label in zip(arms, legend_labels)
    )
    add_top_legend(fig, handles, legend_labels, rows=legend_rows)

    save_figure(fig, number, panel, bbox_inches=None)


    baseline_name = "Vanilla E" if subtract_vanilla_irreducible else "shared E"
    print(f"{baseline_name} = {fits['van'].irreducible:.6f}")
    for arm in arms:
        fit = fits[arm]
        print(
            f"  {labels[arm]:>8}: A={fit.amplitude:.6f} "
            f"alpha={fit.alpha:.6f} R^2={fit.r_squared:.6f}")


def vanilla_scaling_data(data):
    """Use the mean of the two equivalent d8 runs once for every recipe."""
    base_runs = [point for arm in ("van_constant", "van")
                 for point in data[arm] if point.depth == 8]
    if len(base_runs) != 2:
        raise ValueError("expected one constant and one constant + GLR d8 run")
    if any(replace(point, loss=base_runs[0].loss) != base_runs[0] for point in base_runs):
        raise ValueError("the equivalent d8 runs must have identical compute and model metadata")
    shared_base = replace(base_runs[0], loss=float(np.mean([point.loss for point in base_runs])))
    scaling_data = {
        arm: [shared_base, *(point for point in data[arm] if point.depth > 8)]
        for arm in VANILLA_SCALING_ARMS
    }
    return scaling_data, shared_base


def plot_vanilla_scaling(data):
    scaling_data, shared_base = vanilla_scaling_data(data)
    vanilla_scaling_arms = tuple(
        arm for arm in VANILLA_SCALING_ARMS if len(scaling_data[arm]) >= 3)
    plot_figure(
        scaling_data, vanilla_scaling_arms, 16, "a",
        interpolation_multiplier=True, fit_arms=vanilla_scaling_arms,
        subtract_vanilla_irreducible=True, shared_base=shared_base,
        fixed_irreducible=fit_shared_irreducible(data, ("van",))["van"].irreducible,
        labels={
            "van_constant": "Constant",
            "van": "GLR",
            "van_glr_om": "GLR + OM",
            "van_glr_wd": "GLR + WD",
            "van_const_mup_om": "muP OM",
            "van_mup_om": "muP OM + GLR",
            "van_wdr1": "WDR1 + GLR",
        })

# Ablations retain their family color; markers and line styles distinguish recipes.
ARM_COLORS = {arm: arm_color(
    "deep_vanilla_grow" if arm == "deep_van_grow" else
    "deep_vanilla" if arm in {"deep_van", "deeper_van"} else
    "vanilla" if arm.startswith("van") else
    "operator_1" if arm.startswith(("k1", "deeper_k1")) else
    "loop_grow" if arm == "k2_grow" else
    "loop_2" if arm.startswith("k2") else
    "untied_grow" if arm == "dep_grow" else "untied_2") for arm in ARM_LABEL}
# Recipe-only comparisons use distinct tab10 colors, keeping GLR Vanilla gray.
TAB10 = plt.get_cmap("tab10").colors
for arm, index in {"van_constant": 0, "van_glr_wd": 1, "van_glr_om": 2,
                   "van_const_mup_om": 3, "van_mup_om": 4, "van_wdr1": 6}.items():
    ARM_COLORS[arm] = TAB10[index]
DATA = Path(__file__).resolve().parent / "data" / "appendix_ladders"

def load_json(name):
    return json.loads((DATA / f"{name}.json").read_text())

def load_points(name="ladders"):
    return {arm: [Point(**record) for record in records]
            for arm, records in load_json(name).items()}
