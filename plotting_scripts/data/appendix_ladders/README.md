# Inputs for Figures 15–20

Bundled measurements for offline plotting. Scripts recompute fits and
interpolations. See the [plotting guide](../../README.md) for setup, commands,
and output paths.

## Input map

All files below are JSON.

| Figure | Contents | Inputs |
| --- | --- | --- |
| 15a | Architecture ablations | `ladders` |
| 15b | Runtime | `timing`, `ladders` |
| 15c | Constant recipes | `constant` |
| 16a–c | Vanilla scaling, Operator-1, stage allocation | `ladders` |
| 16d | Random recurrence | `random` |
| 16e–f | Width and optimizers | `width_optimizer` |
| 17 | Evaluation-time recurrence | `recurrence` |
| 18a–b | Cross-corpus transfer and calibration | `fineweb`, `fineweb_edu`, `filter` |
| 19a–b | Downstream ladders and loss/CORE relation | `downstream` |
| 19c | Matched-loss tasks | `downstream`, `tasks` |
| 20 | Taskwise calibration | `fineweb_edu`, `filter` |

## Data conventions

- `ladders.json`: 181 points across 24 series. Aliases `van`, `k1`, `k2`, and
  `dep` mean Vanilla, Operator-1, Loop-2, and Untied-2.
- `timing.json`: 58 points with `compute` in training hours. The loss floor is
  fitted to Vanilla's FLOP-coordinate ladder.
- `random.json`: 43 completed runs, evaluated at K=4. `width_optimizer.json`:
  57 points; Adam ladders include completed runs, and width-only ladders extend
  to d24.
- `recurrence.json`: seven checkpoints for each of five FineWeb architectures.
  Figure 17 shows the loss difference from the trained recurrence.
- `downstream.json`: 58 revised CORE checkpoints, 22 tasks, three seeds.
  Vanilla and Operator-1 use d06–d20; other architectures use d06–d18.
- Each corpus file has eight Vanilla and seven Untied Grow records, using matched
  FineWeb compute schedules. These original evaluations differ from the revised
  downstream ladder. `filter.json` fixes the 17-task Vanilla filter;
  `tasks.json` supplies task groups and labels.

## Fit conventions

Power-law fits use Huber initialization and fixed-floor log regressions.
Compute multipliers interpolate within improving measured brackets in log
compute; missing overlap remains missing.

| Figure | Convention |
| --- | --- |
| 16a | Average the two equivalent d08 runs into one shared regression anchor |
| 18b | Use cross-corpus calibration and GPT-3 targets for extrapolation |
| 19b | Fit one equal-weight pooled line across checkpoints |
| 19c | Fit tasks on Vanilla; report residual means for all points and the lower half of the validation-loss interval |
| 20 | Fit shared task sigmoids and a no-intercept full/filtered accuracy ratio |
