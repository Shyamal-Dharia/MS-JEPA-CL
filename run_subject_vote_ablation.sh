#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUTDIR="results/reports/tables/jepa_fusion/five_seed_subject_vote"
mkdir -p "$OUTDIR"

declare -A PROFILES
# Pooled ablation
PROFILES[rotary_pooled_adv]="pooled_final.pt"
PROFILES[rotary_pooled_nce]="pooled_final.pt"
PROFILES[rotary_pooled_both]="pooled_final.pt"
# Curriculum ablation
PROFILES[rotary_curr_adv]="phase2_final.pt"
PROFILES[rotary_curr_nce]="phase2_final.pt"
PROFILES[rotary_curr_both]="phase2_final.pt"

for profile in "${!PROFILES[@]}"; do
    ckpt_name="${PROFILES[$profile]}"
    for seed in 42 0 1 2 3; do
        ckpt="checkpoints/jepa_fusion/five_seed_${profile}_seed${seed}/${ckpt_name}"
        out_json="${OUTDIR}/${profile}_seed${seed}_subject_vote.json"
        out_md="${OUTDIR}/${profile}_seed${seed}_subject_vote.md"

        if [ -f "$out_json" ]; then
            echo "SKIP: $out_json already exists"
            continue
        fi

        echo "EVAL: ${profile} seed=${seed}"
        python -m src.jepa_fusion.evaluate_jepa_subject_vote \
            --checkpoint "$ckpt" \
            --dataset all \
            --seeds 0,1,2,3,4 \
            --device cuda \
            --output-json "$out_json" \
            --output-md "$out_md"
    done
done

echo "ALL DONE"
