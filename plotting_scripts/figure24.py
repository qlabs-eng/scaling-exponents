#!/usr/bin/env python3
"""Regenerate paper Figure 24 from bundled measured inputs, offline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from appendix_regimes_style import STYLE
from appendix_regimes_grid import plot_other_architectures
from appendix_regimes_data import load_loop_grid, load_architectures

def main():
    with plt.rc_context(STYLE):
        plot_other_architectures(load_architectures(), load_loop_grid()[0.8], 24)

if __name__ == "__main__":
    main()
