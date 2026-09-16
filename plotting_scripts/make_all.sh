#!/usr/bin/env bash
# Regenerate all 24 figures from bundled inputs, with matching PDF/PNG trees.
# Explicit --wandb-entity/--wandb-project options replace --csv for Figures 2, 3 and 5.
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# == 0 )); then
  set -- --csv
fi
for number in {1..24}; do
  case "$number" in
    2|3|5) "${PLOT_PYTHON:-python3}" "plotting_scripts/figure$number.py" "$@" ;;
    *) "${PLOT_PYTHON:-python3}" "plotting_scripts/figure$number.py" ;;
  esac
done
