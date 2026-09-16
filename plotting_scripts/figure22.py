#!/usr/bin/env python3
"""Regenerate paper Figure 22 from bundled measured inputs, offline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from appendix_regimes_style import STYLE
from appendix_regimes_grid import plot_complete_grid
from appendix_regimes_data import load_loop_grid

def main():
    with plt.rc_context(STYLE):
        plot_complete_grid(load_loop_grid(), 22)

if __name__ == "__main__":
    main()
