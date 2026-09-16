# Paper figures

Regenerate all 24 paper figures from bundled data, without network access.
`figure1.py` through `figure24.py` follow the paper's numbering, including the appendix.

## Run

Use Python 3.10+. From the repository root, install the locked dependencies
(including NumPy, Matplotlib, and SciPy) and activate the environment:

```bash
uv sync --frozen
source .venv/bin/activate
```

Figure 1 also needs `pdflatex` with TikZ and `pdftoppm`. On Ubuntu, install
`texlive-latex-base`, `texlive-pictures`, and `poppler-utils`.

From the repository root:

```bash
# All figures, using bundled data
bash plotting_scripts/make_all.sh

# Individual figures
python3 plotting_scripts/figure3.py --csv
python3 plotting_scripts/figure16.py

# Select a Python environment
PLOT_PYTHON=/path/to/venv/bin/python bash plotting_scripts/make_all.sh
```

Outputs go to `plotting_scripts/pdf/` and `plotting_scripts/png/`: **38 pairs**
of vector PDFs and 300-dpi PNGs. Most figures produce `figureN.pdf` and
`figureN.png`; separately exported panels use `figureN/a.pdf`, `b.pdf`, etc.,
with matching PNG paths. Panel letters follow the paper's subfigure order.

## Figure index

| Figure | Contents | Separate panels |
| --- | --- | --- |
| 1 | Architecture schematic | — |
| 2 | Single-epoch and multi-epoch overview | — |
| 3 | FineWeb ladders, compute multipliers, power-law coefficients | — |
| 4 | FineWeb-Edu extrapolation to 7.4B Untied-Grow | — |
| 5 | Data-constrained loop grid and scaling recipes | — |
| 6 | Logit-KL effective depth | — |
| 7 | Recurrence design and prelude/core/coda allocation | — |
| 8 | Fresh-data recurrence optima | — |
| 9 | Tokens per parameter/layer | — |
| 10 | Growth schedules and frontier regret | a–d |
| 11 | Base recipe tuning | — |
| 12 | Tokens-per-parameter curves | — |
| 13 | Growth fits | — |
| 14 | Hyperparameter sensitivity | a–b |
| 15 | Architecture ablations, runtime, constant recipes | a–c |
| 16 | Vanilla scaling, Operator-1, allocation, recurrence, width, optimizers | a–f |
| 17 | Evaluation-time recurrence | — |
| 18 | Cross-corpus transfer and fitted comparison | a–b |
| 19 | Downstream ladders, loss/CORE relation, matched-loss tasks | a–c |
| 20 | Taskwise sigmoid and linear calibration | — |
| 21 | Data-constrained regimes | — |
| 22 | Weight-decay grid | — |
| 23 | Hyperparameter sensitivity across recurrence counts | — |
| 24 | Other data-constrained architectures | — |

## Data

All figures use bundled inputs by default. Figures 2, 3, and 5 can optionally
read your runs when both `--wandb-entity` and `--wandb-project` are supplied;
`--wandb-group` restricts the query to one group. Successful pulls are cached
by source in `data/`; add `--refresh` to fetch again. If W&B is unavailable,
the scripts fall back to bundled measurements. `--csv` forces bundled inputs.
`make_all.sh` forwards these options to Figures 2, 3, and 5; the appendix always
uses bundled inputs.

| Input in `data/` | Contents |
| --- | --- |
| `fineweb_ladders.csv` | Architecture, depth, training FLOPs, final validation loss |
| `loop_grid.csv` | Depth, recurrence, weight decay, FLOPs, final loss; 100M tokens × 10 epochs |
| `fineweb_edu_ladders.json` | FineWeb-Edu ladders and CORE evaluation |
| `effective_depth.json` | Logit-lens effective depths |
| `architecture.tex` | Figure 1's TikZ schematic |

Ladder FLOPs sum the per-token cost over the executed schedule, counting K=2 and
K=4 growth stages separately. Deep Vanilla Grow is included in the data but has
no training script in this release. Refreshing the loop grid replaces cells
available on W&B and keeps bundled values for the rest.

To plot your own ladder runs, use local `<arm>_d<depth>/result.json` files or a
synced W&B group:

```bash
python3 plotting_scripts/figure3.py --runs-dir runs
python3 plotting_scripts/figure3.py --wandb-entity <you> --wandb-project <project> --wandb-group scaling_ladders
```

See the [public run table](../README.md#public-ladder-runs) for published logs.
Appendix input inventories and analysis conventions:
[Figures 7–14](data/appendix_early/README.md),
[Figures 15–20](data/appendix_ladders/README.md),
[Figures 21–24](data/appendix_regimes/README.md).
