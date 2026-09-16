#!/usr/bin/env python3
"""Regenerate paper Figure 23 from bundled measured inputs, offline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from appendix_regimes_style import STYLE
from appendix_regimes_hyper import plot
from appendix_regimes_data import load_loop_grid, load_slices

def main():
    with plt.rc_context(STYLE):
        plot(load_loop_grid(), load_slices(), 23)

if __name__ == "__main__":
    main()
