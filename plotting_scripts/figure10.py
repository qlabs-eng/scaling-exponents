"""Figure 10: growth surfaces (a–c) and token allocation with ablation (d)."""
from appendix_style import apply_style
from appendix_early_growth import load_rows, load_deepvan_rows, make_arm_figure, make_deepvan_figure
from appendix_early_allocation import main as allocation_figure


def main():
    apply_style()
    rows = load_rows()
    for arm in ("k2", "dep"):
        make_arm_figure(rows, arm)
    make_deepvan_figure(load_deepvan_rows())
    allocation_figure()


if __name__ == "__main__":
    main()
