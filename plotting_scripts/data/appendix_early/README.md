# Inputs for Figures 7–14

Bundled measurements for offline plotting. Scripts recompute fits, interpolations,
and optima from these inputs. See the [plotting guide](../../README.md) for setup
and commands; Figures 10 and 14 export separate panels.

## Input map

All files below are JSON.

| Figure | Measurements |
| --- | --- |
| 7 | `shell` (25 allocations), `grow_targets` and `grow_lens` (15 each), plus fresh-data, growth, and stored-TPP inputs |
| 8 | `fresh_data`: tied/untied one-epoch ladders with a shared K1 control |
| 9 | `tpl` (204 rows), `tpl_budgets` (7 compute budgets) |
| 10a–c | `growth` (231 cells), `deepvan_growth` (27 cells), `counts` |
| 10d | `token_allocation` (30 growth cells per arm plus controls), `ablation` (25 endpoints) |
| 11 | `tune`: candidates and confirmations; losses determine adoption |
| 12 | `tpp`: 125 iso-compute cells across five architecture families |
| 13 | Figure 10 growth cells at matched budget/model depths 8–10 |
| 14a | `hyper1d`: 169 runs, 212 slice points including shared centers |
| 14b | `hyper_slices`: six architectures' constant/scaled slices and parameter counts |

## Analysis conventions

TPP means tokens per parameter.

- `counts.json` stores the paper model's parameter counts; `tpl_budgets.json`
  stores compute budgets in FLOPs.
- Figure 7's stored-TPP coordinate uses the initial stored parameter count.
- Figure 10d uses `6 * tokens**2 / scheduled_matrix_flops`.
