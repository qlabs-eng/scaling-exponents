"""Measured appendix inputs, with the paper's original compute conventions."""
from __future__ import annotations
import csv
import math
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data" / "appendix_regimes"

@dataclass(frozen=True)
class Point:
    compute: float
    loss: float
    depth: int
    loops: int

def read_rows(filename):
    with (DATA / filename).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for field in ("compute", "loss"):
            if field in row and not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite {field} in {filename}")
    return rows

def point(row):
    return Point(float(row["compute"]), float(row["loss"]), int(row["depth"]), int(row["loops"]))

def load_regimes():
    series = {}
    for row in read_rows("regimes.csv"):
        series.setdefault(row["regime"], {}).setdefault(int(row["loops"]), []).append(point(row))
    return series

def load_joint_grid():
    # Figure 21 uses FLOPs; Figures 22–24 retain the producers' EFLOP arithmetic.
    return {wd: {k: [Point(p["compute"] * 1e18, p["loss"], p["depth"], k)
                         for p in points] for k, points in series.items()}
            for wd, series in load_loop_grid().items()}


def load_loop_grid():
    grid = {}
    for row in read_rows("weight_decay.csv"):
        p = point(row)
        grid.setdefault(float(row["weight_decay"]), {}).setdefault(p.loops, []).append(vars(p))
    return grid

def load_architectures():
    series = {}
    for row in read_rows("architectures.csv"):
        series.setdefault((row["architecture"], row["family"]), {}).setdefault(int(row["loops"]), []).append(point(row))
    return series

def load_slices():
    slices = {}
    for row in read_rows("hyperparameters.csv"):
        slices.setdefault(row["family"], {}).setdefault(row["knob"], {}).setdefault(int(row["loops"]), []).append(float(row["loss"]))
    return slices
