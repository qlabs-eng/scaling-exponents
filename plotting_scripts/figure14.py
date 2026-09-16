"""Figure 14: Vanilla hyperparameter sensitivity (a), architecture recipe slices (b)."""
from appendix_style import apply_style
from appendix_early_hyper1d import main as one_dimensional
from appendix_early_hyperslices import slice_figure


def main():
    apply_style()
    one_dimensional()
    slice_figure()


if __name__ == "__main__":
    main()
