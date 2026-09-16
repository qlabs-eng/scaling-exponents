"""Figure 20: taskwise calibration of CORE accuracy on FineWeb-Edu."""
from __future__ import annotations
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.optimize import least_squares
from common import save_figure
from appendix_style import ARCHITECTURE_COLORS, ARCHITECTURE_MARKERS, INK, MUTED, add_top_legend, legend_row_count

from scipy.optimize import brentq
from appendix_style import architecture_label
from appendix_ladders_corpus_style import ANNOTATION_SIZE, apply_style, make_grid
from appendix_ladders import load_json
ARMS = {
    "Vanilla": {
        "depths": tuple(range(6, 22, 2)),
        "color": ARCHITECTURE_COLORS["van"],
        "marker": ARCHITECTURE_MARKERS["van"],
    },
    "Dep Grow": {
        "depths": tuple(range(6, 20, 2)),
        "color": ARCHITECTURE_COLORS["dep_grow"],
        "marker": ARCHITECTURE_MARKERS["dep_grow"],
    },
}


GPT_TARGETS = (
    ("GPT-3 2.7B", 0.329),
    ("GPT-3 6.7B", 0.361),
    ("GPT-3 13B", 0.385),
    ("GPT-3 175B", 0.427),
)


SCALE_COMPUTE = 1e18


TASK_LABELS = {
    "agi_eval_lsat_ar": "LSAT AR",
    "arc_challenge": "ARC Chal.",
    "arc_easy": "ARC Easy",
    "bigbench_dyck_languages": "Dyck",
    "bigbench_operators": "Ops.",
    "bigbench_qa_wikidata": "Wikidata",
    "copa": "COPA",
    "coqa": "CoQA",
    "hellaswag": "HellaSwag",
    "hellaswag_zeroshot": "Hella ZS",
    "jeopardy": "Jeopardy",
    "lambada_openai": "LAMBADA",
    "openbook_qa": "OpenBookQA",
    "piqa": "PIQA",
    "squad": "SQuAD",
    "winograd": "Winograd",
    "winogrande": "WinoGrande",
}


def robust_scale(residual: np.ndarray, minimum: float) -> float:
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    return max(float(1.345 * mad / 0.67448975), minimum)


def power_law(compute, floor, scale, exponent):
    compute = np.asarray(compute, dtype=float)
    return floor + scale * (compute / SCALE_COMPUTE) ** (-exponent)


def fit_power_law(compute, values):
    """Huber fit of E + A(C/1e18)^(-alpha), using multiple starts."""
    compute = np.asarray(compute, dtype=float)
    values = np.asarray(values, dtype=float)

    def residual(parameters):
        return power_law(compute, *parameters) - values

    lower = np.asarray([0.0, 0.0, 0.005])
    upper = np.asarray([float(values.min()) - 1e-8, 50.0, 2.0])
    candidates = []
    for exponent in (0.05, 0.10, 0.20, 0.40, 0.80):
        floor = max(0.0, 0.5 * float(values.min()))
        scale = max(
            0.01,
            (float(values.max()) - floor)
            * (float(compute.min()) / SCALE_COMPUTE) ** exponent,
        )
        fit = least_squares(
            residual,
            np.asarray([floor, scale, exponent]),
            bounds=(lower, upper),
            max_nfev=30_000,
        )
        candidates.append(fit)
    ordinary = min(candidates, key=lambda fit: np.sum(residual(fit.x) ** 2))
    huber = least_squares(
        residual,
        ordinary.x,
        bounds=(lower, upper),
        loss="huber",
        f_scale=robust_scale(residual(ordinary.x), minimum=0.002),
        max_nfev=30_000,
    )
    return huber.x


def task_sigmoid(nll, slope, intercept):
    exponent = np.clip(slope * np.asarray(nll) - intercept, -700, 700)
    return 1.0 / (1.0 + np.exp(exponent))


def fit_task_sigmoid(nll, accuracy):
    """Huber fit of a bounded, monotone accuracy-from-NLL sigmoid."""
    nll = np.asarray(nll, dtype=float)
    accuracy = np.asarray(accuracy, dtype=float)

    def residual(parameters):
        return task_sigmoid(nll, *parameters) - accuracy

    mean_accuracy = float(np.clip(accuracy.mean(), 0.001, 0.999))
    intercept = (
        np.log(mean_accuracy / (1.0 - mean_accuracy)) + np.median(nll))
    ordinary = least_squares(
        residual,
        np.asarray([1.0, intercept]),
        bounds=([0.0, -20.0], [20.0, 20.0]),
        max_nfev=30_000,
    )
    huber = least_squares(
        residual,
        ordinary.x,
        bounds=([0.0, -20.0], [20.0, 20.0]),
        loss="huber",
        f_scale=robust_scale(residual(ordinary.x), minimum=0.005),
        max_nfev=30_000,
    )
    return huber.x


@dataclass(frozen=True)
class TaskwiseCalibration:
    accuracy_fits: dict[str, np.ndarray]
    full_to_filtered_ratio: float
    ratio_rmse: float
    ratio_r2: float
    task_accuracy_rmse: dict[str, float]


@dataclass(frozen=True)
class TaskwiseModel:
    selected_tasks: tuple[str, ...]
    records: dict[str, list[dict]]
    nll_fits: dict[str, dict[str, np.ndarray]]
    accuracy_fits: dict[str, np.ndarray]
    full_to_filtered_ratio: float
    ratio_rmse: float
    ratio_r2: float
    task_accuracy_rmse: dict[str, float]

    def predict_filtered(self, arm: str, compute):
        task_predictions = [
            task_sigmoid(
                power_law(compute, *self.nll_fits[arm][task]),
                *self.accuracy_fits[task],
            )
            for task in self.selected_tasks
        ]
        return np.mean(np.asarray(task_predictions), axis=0)

    def predict_full(self, arm: str, compute):
        return self.full_to_filtered_ratio * self.predict_filtered(arm, compute)

    def compute_for_target(self, arm: str, target: float) -> float:
        root = brentq(
            lambda log_compute: (
                float(self.predict_full(arm, 10**log_compute)) - target),
            15.0,
            40.0,
        )
        return 10**root


def fit_taskwise_calibration(
        selected_tasks: tuple[str, ...],
        record_groups: tuple[list[dict], ...]) -> TaskwiseCalibration:
    """Fit shared task sigmoids and one full/filtered ratio to record groups."""
    calibration_records = [
        record for group in record_groups for record in group
    ]
    filtered = np.asarray([
        record["filtered_accuracy"] for record in calibration_records
    ])
    full = np.asarray([
        record["full_accuracy"] for record in calibration_records
    ])
    full_to_filtered_ratio = float(
        filtered @ full / (filtered @ filtered)
    )
    ratio_prediction = full_to_filtered_ratio * filtered
    ratio_rmse = float(np.sqrt(np.mean(
        (ratio_prediction - full) ** 2)))
    ratio_r2 = float(1.0 - np.sum(
        (ratio_prediction - full) ** 2) / np.sum(
        (full - full.mean()) ** 2))

    accuracy_fits = {}
    task_accuracy_rmse = {}
    for task in selected_tasks:
        pooled_nll = np.asarray([
            record["raw"]["losses"][task]
            for record in calibration_records
        ])
        pooled_accuracy = np.asarray([
            record["raw"]["centered_results"][task]
            for record in calibration_records
        ])
        accuracy_fits[task] = fit_task_sigmoid(
            pooled_nll, pooled_accuracy)
        fitted_accuracy = task_sigmoid(
            pooled_nll, *accuracy_fits[task])
        task_accuracy_rmse[task] = float(np.sqrt(np.mean(
            (fitted_accuracy - pooled_accuracy) ** 2)))

    return TaskwiseCalibration(
        accuracy_fits=accuracy_fits,
        full_to_filtered_ratio=full_to_filtered_ratio,
        ratio_rmse=ratio_rmse,
        ratio_r2=ratio_r2,
        task_accuracy_rmse=task_accuracy_rmse,
    )


def build_taskwise_model(corpus: str = "fineweb") -> TaskwiseModel:
    selected_tasks = tuple(load_json("filter")["selected_tasks"])
    records = load_corpus_records(corpus, selected_tasks)

    nll_fits = {arm: {} for arm in ARMS}
    for task in selected_tasks:
        for arm in ARMS:
            compute = np.asarray([
                record["compute"] for record in records[arm]
            ])
            nll = np.asarray([
                record["raw"]["losses"][task] for record in records[arm]
            ])
            nll_fits[arm][task] = fit_power_law(compute, nll)
    calibration = fit_taskwise_calibration(
        selected_tasks, tuple(records[arm] for arm in ARMS))

    return TaskwiseModel(
        selected_tasks=selected_tasks,
        records=records,
        nll_fits=nll_fits,
        accuracy_fits=calibration.accuracy_fits,
        full_to_filtered_ratio=calibration.full_to_filtered_ratio,
        ratio_rmse=calibration.ratio_rmse,
        ratio_r2=calibration.ratio_r2,
        task_accuracy_rmse=calibration.task_accuracy_rmse,
    )


def plot_calibration(model: TaskwiseModel) -> None:
    labels = (
        "Vanilla",
        architecture_label("Dep Grow"),
        "Shared sigmoid",
        "Full/filtered ratio",
    )
    apply_style()
    legend_rows = legend_row_count(labels)
    titles = [TASK_LABELS[task] for task in model.selected_tasks]
    titles.extend(("Unused", "Unused", "CORE ratio"))
    fig, axes = make_grid(
        4,
        5,
        titles=titles,
        sharey=True,
        shared_xlabel="CORE NLL",
        shared_ylabel="CORE accuracy",
        legend_rows=legend_rows,
        squeeze=False,
    )

    task_axes = axes.flat[:len(model.selected_tasks)]
    for axis in axes.flat[len(model.selected_tasks):-1]:
        axis.set_visible(False)
    for axis, task in zip(task_axes, model.selected_tasks):
        pooled_nll = []
        for arm, attributes in ARMS.items():
            nll = np.asarray([
                record["raw"]["losses"][task]
                for record in model.records[arm]
            ])
            accuracy = np.asarray([
                record["raw"]["centered_results"][task]
                for record in model.records[arm]
            ])
            pooled_nll.extend(nll)
            axis.plot(
                nll,
                accuracy,
                linestyle="none",
                marker=attributes["marker"],
                color=attributes["color"],
                markeredgecolor="white",
                markeredgewidth=0.4,
                zorder=3,
            )
        nll_min = float(np.min(pooled_nll))
        nll_max = float(np.max(pooled_nll))
        nll_span = nll_max - nll_min
        slope, intercept = model.accuracy_fits[task]
        midpoint = intercept / slope if slope > 0 else nll_min
        # Include the fitted midpoint when it lies below the measurements, but
        # keep the displayed NLL domain physically meaningful.
        curve_min = max(
            0.0,
            min(nll_min - 0.75 * nll_span, midpoint - 0.50 * nll_span),
        )
        curve_max = nll_max + 0.50 * nll_span
        curve_nll = np.linspace(curve_min, curve_max, 300)
        axis.plot(
            curve_nll,
            task_sigmoid(curve_nll, *model.accuracy_fits[task]),
            color=INK,
            linestyle="--",
            zorder=1,
        )
        measured_nll = np.linspace(nll_min, nll_max, 200)
        axis.plot(
            measured_nll,
            task_sigmoid(measured_nll, *model.accuracy_fits[task]),
            color=INK,
            zorder=2,
        )
        axis.set_xlim(curve_max, curve_min)
        axis.set_ylim(-0.08, 1.02)
        axis.locator_params(axis="x", nbins=4)
        axis.set_yticks((0.0, 0.25, 0.50, 0.75, 1.0))
        axis.text(
            0.04,
            0.94,
            f"RMSE {model.task_accuracy_rmse[task]:.3f}",
            transform=axis.transAxes,
            color=MUTED,
            va="top",
            fontsize=ANNOTATION_SIZE,
        )

    ratio_axis = axes.flat[-1]
    filtered_minimum = min(
        record["filtered_accuracy"]
        for arm in ARMS for record in model.records[arm]
    )
    filtered_maximum = max(
        record["filtered_accuracy"]
        for arm in ARMS for record in model.records[arm]
    )
    ratio_x = np.linspace(0.0, 1.0, 400)
    before_data = ratio_x <= filtered_minimum
    measured_data = (
        (ratio_x >= filtered_minimum) & (ratio_x <= filtered_maximum))
    after_data = ratio_x >= filtered_maximum
    for domain, linestyle in (
            (before_data, "--"), (measured_data, "-"),
            (after_data, "--")):
        ratio_axis.plot(
            ratio_x[domain],
            model.full_to_filtered_ratio * ratio_x[domain],
            color=RATIO_COLOR, linestyle=linestyle, zorder=2,
        )
    for arm, attributes in ARMS.items():
        ratio_axis.plot(
            [record["filtered_accuracy"] for record in model.records[arm]],
            [record["full_accuracy"] for record in model.records[arm]],
            linestyle="none", marker=attributes["marker"],
            color=RATIO_COLOR, markeredgecolor="white",
            markeredgewidth=0.4, zorder=3,
        )
    ratio_axis.set_xlim(-0.02, 1.02)
    ratio_axis.set_ylim(-0.08, 1.02)
    ratio_axis.set_xticks((0.0, 0.5, 1.0))
    ratio_axis.set_xlabel("Filtered CORE")
    ratio_axis.set_title("CORE ratio", color=RATIO_COLOR)
    ratio_axis.text(
        0.05,
        0.94,
        f"y = {model.full_to_filtered_ratio:.3f}x\n"
        f"RMSE {model.ratio_rmse:.3f}",
        transform=ratio_axis.transAxes,
        color=RATIO_COLOR,
        va="top",
        fontsize=ANNOTATION_SIZE,
    )
    handles = [
        Line2D(
            [], [], linestyle="none", marker=ARMS[arm]["marker"],
            color=ARMS[arm]["color"], markeredgecolor="white",
            markeredgewidth=0.4,
        )
        for arm in ARMS
    ] + [
        Line2D([], [], color=INK),
        Line2D([], [], color=RATIO_COLOR),
    ]
    add_top_legend(fig, handles, labels, rows=legend_rows)
    save_figure(fig, 20, bbox_inches=None)

RATIO_COLOR = plt.get_cmap("tab10")(6)

def load_corpus_records(corpus, selected_tasks):
    """Load the frozen selected records; preserve the original task selection."""
    if tuple(selected_tasks) != tuple(load_json("filter")["selected_tasks"]):
        raise ValueError("task selection differs from the published Vanilla filter")
    if corpus not in {"fineweb", "fineweb_edu"}:
        raise ValueError(f"unknown corpus: {corpus}")
    return load_json(corpus)

def main():
    plot_calibration(build_taskwise_model("fineweb_edu"))

if __name__ == "__main__":
    main()
