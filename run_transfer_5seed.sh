#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DIRECTION="${1:?Usage: $0 A|B}"
SEEDS=(42 0 1 2 3)
EVAL_SEEDS="0,1,2,3,4"
DEVICE="cuda"

MODEL_ARGS=(
    --model-family temporal
    --temporal-variant rotary_conv_net
    --training-mode curriculum
    --use-conv-stem
    --use-network-embeddings
)

if [[ "$DIRECTION" == "A" ]]; then
    TRAIN_DS="srpbs,abide1"
    EVAL_DS="abide2"
    # ABIDE-II normalized labels for centers also in ABIDE-I
    OVERLAP="KKI,NYU,OHSU,SDSU,UCLA_1,USM,Trinity"
elif [[ "$DIRECTION" == "B" ]]; then
    TRAIN_DS="srpbs,abide2"
    EVAL_DS="abide1"
    # ABIDE-I normalized labels for centers also in ABIDE-II
    OVERLAP="ABIDEI-KKI,ABIDEI-NYU,ABIDEI-OHSU,ABIDEI-SDSU,ABIDEI-UCLA_1,ABIDEI-UCLA_2,ABIDEI-USM,ABIDEI-Trinity"
else
    echo "Unknown direction: $DIRECTION"; exit 1
fi

STAGE_BASE="transfer_${TRAIN_DS//,/_}_to_${EVAL_DS}_rotary_curr"
RESULTS_DIR="results/reports/tables/jepa_fusion/transfer"
mkdir -p "$RESULTS_DIR"

echo "=== Direction $DIRECTION: train [$TRAIN_DS] -> eval [$EVAL_DS] ==="
echo "=== Excluding overlapping centers: $OVERLAP ==="

for SEED in "${SEEDS[@]}"; do
    SAVE_DIR="checkpoints/jepa_fusion/${STAGE_BASE}_seed${SEED}"
    CKPT="${SAVE_DIR}/phase2_final.pt"
    OUT_GCV="${RESULTS_DIR}/${STAGE_BASE}_seed${SEED}_groupedcv"
    OUT_SV="${RESULTS_DIR}/${STAGE_BASE}_seed${SEED}_subject_vote"

    echo ">>> Seed $SEED — Training"
    if [[ -f "$CKPT" ]]; then
        echo "    Checkpoint exists — skipping"
    else
        python -m src.jepa_fusion.train_jepa \
            "${MODEL_ARGS[@]}" \
            --main-datasets "$TRAIN_DS" \
            --device "$DEVICE" \
            --seed "$SEED" \
            --save-dir "$SAVE_DIR" \
            --log-file "${SAVE_DIR}/train.log"
    fi

    echo ">>> Seed $SEED — Grouped-CV eval on $EVAL_DS (unseen sites only)"
    if [[ -f "${OUT_GCV}.json" ]]; then
        echo "    Exists — skipping"
    else
        python -m src.jepa_fusion.evaluate_jepa \
            --checkpoint "$CKPT" \
            --dataset "$EVAL_DS" \
            --device "$DEVICE" \
            --seeds "$EVAL_SEEDS" \
            --exclude-sites "$OVERLAP" \
            --output-json "${OUT_GCV}.json" \
            --output-md "${OUT_GCV}.md"
    fi

    echo ">>> Seed $SEED — Subject-vote LOSO eval on $EVAL_DS (unseen sites only)"
    if [[ -f "${OUT_SV}.json" ]]; then
        echo "    Exists — skipping"
    else
        python -m src.jepa_fusion.evaluate_jepa_subject_vote \
            --checkpoint "$CKPT" \
            --dataset "$EVAL_DS" \
            --device "$DEVICE" \
            --seeds "$EVAL_SEEDS" \
            --exclude-sites "$OVERLAP" \
            --output-json "${OUT_SV}.json" \
            --output-md "${OUT_SV}.md"
    fi

    echo ">>> Seed $SEED — DONE"
    echo ""
done

echo "=== ALL DONE: Direction $DIRECTION ==="
