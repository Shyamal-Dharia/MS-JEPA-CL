#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

METHOD="${1:?Usage: $0 jepa|masked A|B}"
DIRECTION="${2:?Usage: $0 jepa|masked A|B}"
EXTRA_SEEDS=(${EXTRA_SEEDS:-4 64 256 512 1024})
EVAL_SEEDS="${EVAL_SEEDS:-0,1,2,3,4}"
DEVICE="${DEVICE:-cuda}"

RUN_TAG="final_run_v1"
JEPA_RESULTS_DIR="results/reports/tables/jepa_fusion/final_run_v1"
MASKED_RESULTS_DIR="results/reports/tables/jepa_fusion/masked_recon_transfer"
mkdir -p "$JEPA_RESULTS_DIR" "$MASKED_RESULTS_DIR"

if [[ "$DIRECTION" == "A" ]]; then
    TRAIN_DS="srpbs,abide1"
    EVAL_DS="abide2"
    # This matches the existing ABIDE-II unseen-center protocol.
    OVERLAP="KKI,NYU,OHSU,SDSU,UCLA_1,USM,Trinity"
elif [[ "$DIRECTION" == "B" ]]; then
    TRAIN_DS="srpbs,abide2"
    EVAL_DS="abide1"
    # Keep this empty to match the current 23-target-site table.
    OVERLAP=""
else
    echo "Unknown direction: ${DIRECTION}. Expected A or B." >&2
    exit 1
fi

run_jepa_seed() {
    local seed="$1"
    local stage_base="${RUN_TAG}_transfer_${TRAIN_DS//,/_}_to_${EVAL_DS}_rotary_curr_replay_nce_seed${seed}"
    local save_dir="checkpoints/jepa_fusion/${RUN_TAG}/${stage_base}"
    local ckpt="${save_dir}/phase2_final.pt"
    local out_gcv="${JEPA_RESULTS_DIR}/${stage_base}_groupedcv"
    local out_sv="${JEPA_RESULTS_DIR}/${stage_base}_subject_vote"

    echo ">>> MS-JEPA ${DIRECTION} seed ${seed} - train"
    if [[ -f "$ckpt" ]]; then
        echo "    Checkpoint exists - skipping"
    else
        python -m src.jepa_fusion.train_jepa \
            --model-family temporal \
            --temporal-variant rotary_conv_net \
            --training-mode curriculum \
            --use-conv-stem \
            --use-network-embeddings \
            --replay-alignment-weight 0.1 \
            --replay-pair-sampling \
            --infonce-weight 0.1 \
            --main-datasets "$TRAIN_DS" \
            --device "$DEVICE" \
            --seed "$seed" \
            --save-dir "$save_dir" \
            --log-file "${save_dir}/train.log"
    fi

    echo ">>> MS-JEPA ${DIRECTION} seed ${seed} - grouped-CV eval"
    if [[ -f "${out_gcv}.json" ]]; then
        echo "    Exists - skipping"
    else
        eval_args=(
            --checkpoint "$ckpt"
            --dataset "$EVAL_DS"
            --device "$DEVICE"
            --seeds "$EVAL_SEEDS"
            --output-json "${out_gcv}.json"
            --output-md "${out_gcv}.md"
        )
        if [[ -n "$OVERLAP" ]]; then
            eval_args+=(--exclude-sites "$OVERLAP")
        fi
        python -m src.jepa_fusion.evaluate_jepa "${eval_args[@]}"
    fi

    echo ">>> MS-JEPA ${DIRECTION} seed ${seed} - subject-vote eval"
    if [[ -f "${out_sv}.json" ]]; then
        echo "    Exists - skipping"
    else
        eval_args=(
            --checkpoint "$ckpt"
            --dataset "$EVAL_DS"
            --device "$DEVICE"
            --seeds "$EVAL_SEEDS"
            --output-json "${out_sv}.json"
            --output-md "${out_sv}.md"
        )
        if [[ -n "$OVERLAP" ]]; then
            eval_args+=(--exclude-sites "$OVERLAP")
        fi
        python -m src.jepa_fusion.evaluate_jepa_subject_vote "${eval_args[@]}"
    fi
}

run_masked_seed() {
    local seed="$1"
    local stage_base="masked_recon_ssl_${TRAIN_DS//,/_}_to_${EVAL_DS}_seed${seed}"
    local save_dir="checkpoints/jepa_fusion/${stage_base}"
    local ckpt="${save_dir}/masked_recon_final.pt"
    local out_gcv="${MASKED_RESULTS_DIR}/${stage_base}_groupedcv"
    local out_sv="${MASKED_RESULTS_DIR}/${stage_base}_subject_vote"

    echo ">>> Masked Recon. SSL ${DIRECTION} seed ${seed} - train"
    if [[ -f "$ckpt" ]]; then
        echo "    Checkpoint exists - skipping"
    else
        python -m src.jepa_fusion.train_masked_recon_ssl \
            --main-datasets "$TRAIN_DS" \
            --temporal-variant rotary_conv_net \
            --device "$DEVICE" \
            --seed "$seed" \
            --save-dir "$save_dir" \
            --log-file "${save_dir}/train.log"
    fi

    echo ">>> Masked Recon. SSL ${DIRECTION} seed ${seed} - grouped-CV eval"
    if [[ -f "${out_gcv}.json" ]]; then
        echo "    Exists - skipping"
    else
        eval_args=(
            --checkpoint "$ckpt"
            --dataset "$EVAL_DS"
            --device "$DEVICE"
            --seeds "$EVAL_SEEDS"
            --output-json "${out_gcv}.json"
            --output-md "${out_gcv}.md"
        )
        if [[ -n "$OVERLAP" ]]; then
            eval_args+=(--exclude-sites "$OVERLAP")
        fi
        python -m src.jepa_fusion.evaluate_jepa "${eval_args[@]}"
    fi

    echo ">>> Masked Recon. SSL ${DIRECTION} seed ${seed} - subject-vote eval"
    if [[ -f "${out_sv}.json" ]]; then
        echo "    Exists - skipping"
    else
        eval_args=(
            --checkpoint "$ckpt"
            --dataset "$EVAL_DS"
            --device "$DEVICE"
            --seeds "$EVAL_SEEDS"
            --output-json "${out_sv}.json"
            --output-md "${out_sv}.md"
        )
        if [[ -n "$OVERLAP" ]]; then
            eval_args+=(--exclude-sites "$OVERLAP")
        fi
        python -m src.jepa_fusion.evaluate_jepa_subject_vote "${eval_args[@]}"
    fi
}

echo "=== Extra zero-shot seeds | method=${METHOD} direction=${DIRECTION} train=${TRAIN_DS} eval=${EVAL_DS} seeds=${EXTRA_SEEDS[*]} ==="
for seed in "${EXTRA_SEEDS[@]}"; do
    case "$METHOD" in
        jepa) run_jepa_seed "$seed" ;;
        masked) run_masked_seed "$seed" ;;
        *) echo "Unknown method: ${METHOD}. Expected jepa or masked." >&2; exit 1 ;;
    esac
    echo ">>> ${METHOD} ${DIRECTION} seed ${seed} - DONE"
    echo ""
done
echo "=== DONE ${METHOD} direction ${DIRECTION} ==="
