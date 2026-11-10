#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DIRECTION="${1:?Usage: $0 A|B}"
SEEDS=(0 8 16 32 128)
EVAL_SEEDS="0,1,2,3,4"
DEVICE="${DEVICE:-cuda}"
RESULTS_DIR="results/reports/tables/jepa_fusion/masked_recon_transfer"
mkdir -p "$RESULTS_DIR"

if [[ "$DIRECTION" == "A" ]]; then
    TRAIN_DS="srpbs,abide1"
    EVAL_DS="abide2"
    OVERLAP="KKI,NYU,OHSU,SDSU,UCLA_1,USM,Trinity"
elif [[ "$DIRECTION" == "B" ]]; then
    TRAIN_DS="srpbs,abide2"
    EVAL_DS="abide1"
    OVERLAP=""
else
    echo "Unknown direction: $DIRECTION" >&2
    exit 1
fi

STAGE_BASE="masked_recon_ssl_${TRAIN_DS//,/_}_to_${EVAL_DS}"

echo "=== Masked Recon. SSL direction ${DIRECTION}: train [${TRAIN_DS}] -> eval [${EVAL_DS}] ==="
if [[ -n "$OVERLAP" ]]; then
    echo "=== Excluding overlapping target centers: ${OVERLAP} ==="
fi

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="checkpoints/jepa_fusion/${STAGE_BASE}_seed${SEED}"
    CKPT="${SAVE_DIR}/masked_recon_final.pt"
    OUT_GCV="${RESULTS_DIR}/${STAGE_BASE}_seed${SEED}_groupedcv"
    OUT_SV="${RESULTS_DIR}/${STAGE_BASE}_seed${SEED}_subject_vote"

    echo ">>> Seed ${SEED} - Training"
    if [[ -f "$CKPT" ]]; then
        echo "    Checkpoint exists - skipping"
    else
        python -m src.jepa_fusion.train_masked_recon_ssl \
            --main-datasets "$TRAIN_DS" \
            --temporal-variant rotary_conv_net \
            --device "$DEVICE" \
            --seed "$SEED" \
            --save-dir "$SAVE_DIR" \
            --log-file "${SAVE_DIR}/train.log"
    fi

    echo ">>> Seed ${SEED} - Grouped-CV eval on ${EVAL_DS}"
    if [[ -f "${OUT_GCV}.json" ]]; then
        echo "    Exists - skipping"
    else
        eval_args=(
            --checkpoint "$CKPT"
            --dataset "$EVAL_DS"
            --device "$DEVICE"
            --seeds "$EVAL_SEEDS"
            --output-json "${OUT_GCV}.json"
            --output-md "${OUT_GCV}.md"
        )
        if [[ -n "$OVERLAP" ]]; then
            eval_args+=(--exclude-sites "$OVERLAP")
        fi
        python -m src.jepa_fusion.evaluate_jepa "${eval_args[@]}"
    fi

    echo ">>> Seed ${SEED} - Subject-vote eval on ${EVAL_DS}"
    if [[ -f "${OUT_SV}.json" ]]; then
        echo "    Exists - skipping"
    else
        eval_args=(
            --checkpoint "$CKPT"
            --dataset "$EVAL_DS"
            --device "$DEVICE"
            --seeds "$EVAL_SEEDS"
            --output-json "${OUT_SV}.json"
            --output-md "${OUT_SV}.md"
        )
        if [[ -n "$OVERLAP" ]]; then
            eval_args+=(--exclude-sites "$OVERLAP")
        fi
        python -m src.jepa_fusion.evaluate_jepa_subject_vote "${eval_args[@]}"
    fi

    echo ">>> Seed ${SEED} - DONE"
    echo ""
done

echo "=== ALL DONE: Masked Recon. SSL direction ${DIRECTION} ==="
