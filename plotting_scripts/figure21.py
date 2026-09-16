#!/usr/bin/env python3
"""Regenerate paper Figure 21 from bundled measured inputs, offline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from appendix_regimes_style import STYLE
from appendix_regimes import plot_regimes
from appendix_regimes_data import load_regimes

def main():
    with plt.rc_context(STYLE):
        plot_regimes(load_regimes())

if __name__ == "__main__":
    main()
