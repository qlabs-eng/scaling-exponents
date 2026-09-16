# Inputs for Figures 21–24

Four CSVs supply the measured inputs for offline plotting. See the
[plotting guide](../../README.md) for setup and commands. Loss is final validation
loss; figures display compute in FLOPs.

## Input map

| File | Rows | Figures | Compute unit |
| --- | ---: | --- | --- |
| `regimes.csv` | 97 | 21 | FLOPs |
| `weight_decay.csv` | 222 | 21–24 | EFLOP |
| `architectures.csv` | 105 | 24 | EFLOP |
| `hyperparameters.csv` | 150 | 23 | — |

The weight-decay table uses full precision; the main figures use a differently
rounded export. Architecture inputs share K1 baselines between tied and untied
variants and omit the unused tied recurrent ladder. Sensitivity slices retain
repeated center values.

## Analysis conventions

Interpolation is linear in compute. Figures 21 and 24 fit quadratics in log loss
versus log recurrence: filled stars mark interior minima; hollow stars mark the
best measured value when no interior minimum exists.

| Figure | Convention |
| --- | --- |
| 21 | Column 4 selects the best measured K1 weight decay at each depth, then uses it for other recurrence counts |
| 22 | Connect interpolated values without quadratic fitting |
| 23 | Compare sensitivity at fixed token counts, with zoomed weight-decay axes |
| 24 | Use the lower convex envelope in log compute as the dashed tied reference, excluding K6/depth-4 |
