#!/usr/bin/env bash
# arXiv Figure 5, third panel: Scale loops at WD=0.8.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export WANDB_MODE="${WANDB_MODE:-offline}"

# Keep global batch size and packing fixed when changing the GPU count.
run() {
    local depth=$1 loops=$2 batch=$3 wd=$4
    local prelude=$((depth / 3))
    local core=$((depth / 3 + (depth % 3 >= 1)))
    local coda=$((depth / 3 + (depth % 3 >= 2)))
    local name
    printf -v name 'loop_d%02d_k%d_wd%s' "$depth" "$loops" "${wd//./p}"
    local -a command=(
        torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" train.py
        --data-dir "${DATA_DIR:-data/fineweb}"
        --runs-dir "${RUNS_DIR:-runs/data_constrained}/scale_loops_fixed_wd"
        --run-name "$name"
        --n_layer "$depth" --n_head "$depth" --n_embd "$((128 * depth))"
        --depth-scale-mode loop --core-repetitions "$loops"
        --prelude-layer-count "$prelude" --core-layer-count "$core" --coda-layer-count "$coda"
        --recurrence-rmsnorm recurrence_only --recurrence-emb-alpha 1.0 --decoder-inject
        --total-batch-size 524288 --device-batch-size "$batch"
        --token-limit-cutoff-devices "$((256 / batch))"
        --train-token-limit 100000000 --val-token-limit 10000000 --num-epochs 10
        --max-train-steps 1900 --lr-schedule-steps 1900 --seed 42
        --matrix-lr 0.04 --embedding-lr 0.03620386719676 --unembedding-lr 0.0032
        --lr_multiplier 1 --adam-betas 0.8 0.98 --adam-eps 1e-10
        --weight-decay "$wd" --embedding-wd-skip-norm 0
        --residual-branch-multiplier 0.5 --output-multiplier 1.0
        --wte-init-std 0.11313708499 --uniform-init-scale 0.353553390593
        --warmup-steps 0 --warmdown-ratio 0.8
        --wandb-project "${WANDB_PROJECT:-loop-paper}"
        --wandb-group "${WANDB_GROUP:-20260916_data_constrained}"
    )
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

# depth  recurrence  device batch  weight decay
run  6  1 32 0.8
run  6  2 32 0.8
run  6  3 32 0.8
run  6  4 32 0.8
run  6  6 32 0.8
run  8  1 32 0.8
run  8  2 32 0.8
run  8  3 32 0.8
run  8  4 32 0.8
run  8  6 32 0.8
run  8  8 16 0.8
run  8 12  8 0.8
run 10  1 32 0.8
run 10  2 32 0.8
run 10  3 32 0.8
run 10  4 16 0.8
run 10  6 16 0.8
run 10  8  8 0.8
run 10 12  8 0.8
run 12  1 32 0.8
run 12  2 16 0.8
run 12  3 16 0.8
run 12  4 16 0.8
run 12  6  8 0.8
run 12  8  8 0.8
run 12 12  8 0.8
run 14  1 16 0.8
run 14  2 16 0.8
run 14  3 16 0.8
run 14  4  8 0.8
run 14  6  8 0.8
run 14  8  4 0.8
run 16  1 16 0.8
run 16  2  8 0.8
run 16  3  8 0.8
run 18  1  8 0.8
