#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

DEVICE="${DEVICE:-cuda}"
RUN_TAG="${RUN_TAG:-final_run_v1}"
SEEDS_CSV="${SEEDS_CSV:-0,8,16,32,128}"
PRETRAIN_SEEDS=(${PRETRAIN_SEEDS:-0 4 8 16 32 64 128 256 512 1024})

RESULTS_ROOT="results/reports/tables/jepa_fusion/zero_shot_no_exclusion"
mkdir -p "$RESULTS_ROOT"/{msjepa,masked_recon,static_fc}

run_static() {
  local direction="$1"
  local source_ds target_ds stem

  if [[ "$direction" == "A" ]]; then
    source_ds="srpbs,abide1"
    target_ds="abide2"
    stem="srpbs_abide1_to_abide2"
  elif [[ "$direction" == "B" ]]; then
    source_ds="srpbs,abide2"
    target_ds="abide1"
    stem="srpbs_abide2_to_abide1"
  else
    echo "Unknown direction: ${direction}" >&2
    return 1
  fi

  local out_json="${RESULTS_ROOT}/static_fc/static_fc_zero_shot_${stem}_hc_disorder_no_exclusion.json"
  local out_md="${RESULTS_ROOT}/static_fc/static_fc_zero_shot_${stem}_hc_disorder_no_exclusion.md"
  if [[ -f "$out_json" && -f "$out_md" ]]; then
    echo "Static FC ${direction}: exists - skipping"
    return 0
  fi

  python -m src.jepa_fusion.evaluate_static_fc_zero_shot \
    --source-datasets "$source_ds" \
    --target-dataset "$target_ds" \
    --task hc_disorder \
    --seeds "$SEEDS_CSV" \
    --output-json "$out_json" \
    --output-md "$out_md"
}

run_learned_seed() {
  local method="$1"
  local direction="$2"
  local seed="$3"
  local train_ds eval_ds stage_base ckpt out_dir prefix

  if [[ "$direction" == "A" ]]; then
    train_ds="srpbs,abide1"
    eval_ds="abide2"
  elif [[ "$direction" == "B" ]]; then
    train_ds="srpbs,abide2"
    eval_ds="abide1"
  else
    echo "Unknown direction: ${direction}" >&2
    return 1
  fi

  if [[ "$method" == "msjepa" ]]; then
    stage_base="${RUN_TAG}_transfer_${train_ds//,/_}_to_${eval_ds}_rotary_curr_replay_nce_seed${seed}"
    ckpt="checkpoints/jepa_fusion/${RUN_TAG}/${stage_base}/phase2_final.pt"
    out_dir="${RESULTS_ROOT}/msjepa"
    prefix="${stage_base}_no_exclusion"
  elif [[ "$method" == "masked" ]]; then
    stage_base="masked_recon_ssl_${train_ds//,/_}_to_${eval_ds}_seed${seed}"
    ckpt="checkpoints/jepa_fusion/${stage_base}/masked_recon_final.pt"
    out_dir="${RESULTS_ROOT}/masked_recon"
    prefix="${stage_base}_no_exclusion"
  else
    echo "Unknown method: ${method}" >&2
    return 1
  fi

  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    return 1
  fi

  local grouped_json="${out_dir}/${prefix}_groupedcv.json"
  local grouped_md="${out_dir}/${prefix}_groupedcv.md"
  local subject_json="${out_dir}/${prefix}_subject_vote.json"
  local subject_md="${out_dir}/${prefix}_subject_vote.md"

  if [[ -f "$grouped_json" && -f "$grouped_md" ]]; then
    echo "${method} ${direction} seed ${seed}: grouped-CV exists - skipping"
  else
    python -m src.jepa_fusion.evaluate_jepa \
      --checkpoint "$ckpt" \
      --dataset "$eval_ds" \
      --device "$DEVICE" \
      --seeds "$SEEDS_CSV" \
      --output-json "$grouped_json" \
      --output-md "$grouped_md"
  fi

  if [[ -f "$subject_json" && -f "$subject_md" ]]; then
    echo "${method} ${direction} seed ${seed}: subject-vote exists - skipping"
  else
    python -m src.jepa_fusion.evaluate_jepa_subject_vote \
      --checkpoint "$ckpt" \
      --dataset "$eval_ds" \
      --device "$DEVICE" \
      --seeds "$SEEDS_CSV" \
      --output-json "$subject_json" \
      --output-md "$subject_md"
  fi
}

for direction in A B; do
  echo "=== Static FC direction ${direction} | no exclusion ==="
  run_static "$direction"

  for seed in "${PRETRAIN_SEEDS[@]}"; do
    echo "=== MS-JEPA direction ${direction} seed ${seed} | no exclusion ==="
    run_learned_seed msjepa "$direction" "$seed"

    echo "=== Masked Recon. SSL direction ${direction} seed ${seed} | no exclusion ==="
    run_learned_seed masked "$direction" "$seed"
  done
done

echo "=== Table 9 summary and paired sign-flip tests ==="
PRETRAIN_SEEDS_CSV="$(IFS=,; echo "${PRETRAIN_SEEDS[*]}")"
python -m src.jepa_fusion.compare_zero_shot_transfer_stats \
  --results-root "$RESULTS_ROOT" \
  --run-tag "$RUN_TAG" \
  --seeds "$PRETRAIN_SEEDS_CSV"

echo "=== DONE zero-shot no-exclusion inference ==="
