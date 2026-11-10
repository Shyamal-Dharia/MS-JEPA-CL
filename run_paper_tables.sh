#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_TAG="${RUN_TAG:-final_run_v1}"
DEVICE="${DEVICE:-cuda}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

RESULTS_ROOT="results/reports/tables/jepa_fusion/${RUN_TAG}"
CHECKPOINT_ROOT="checkpoints/jepa_fusion/${RUN_TAG}"
LOG_ROOT="logs/${RUN_TAG}"

SEED_LIST=(0 8 16 32 128)
SEEDS_CSV="0,8,16,32,128"

mkdir -p "$RESULTS_ROOT" "$CHECKPOINT_ROOT" "$LOG_ROOT"

STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(
    dataset_summary
    baselines
    pilot_pooled
    curriculum
    aux
    transfer
    lowk
    tables
  )
fi

has_stage() {
  local target="$1"
  local stage
  for stage in "${STAGES[@]}"; do
    if [[ "$stage" == "$target" ]] || [[ "$stage" == "all" ]]; then
      return 0
    fi
  done
  return 1
}

run_cmd() {
  echo
  echo "\$ $*"
  "$@"
}

write_manifest() {
  cat > "${RESULTS_ROOT}/${RUN_TAG}_manifest.txt" <<EOF
run_tag=${RUN_TAG}
root=${ROOT}
device=${DEVICE}
skip_existing=${SKIP_EXISTING}
results_root=${RESULTS_ROOT}
checkpoint_root=${CHECKPOINT_ROOT}
log_root=${LOG_ROOT}

dataset_summary:
  script=generate_table2.py

baselines:
  pretrain_seeds=n/a
  eval_seeds=${SEEDS_CSV}

pilot_pooled:
  models=temporal_basic,st_roi_patch,temporal_conv_net,temporal_rotary_conv_net
  pretrain_seeds=${SEEDS_CSV}
  groupedcv_eval_seeds=${SEEDS_CSV}
  subject_vote_eval_seeds=${SEEDS_CSV}

curriculum:
  models=temporal_rotary_conv_net + replay_alignment
  pretrain_seeds=${SEEDS_CSV}
  groupedcv_eval_seeds=${SEEDS_CSV}
  subject_vote_eval_seeds=${SEEDS_CSV}
  replay_alignment_weight=0.1
  note=curriculum uses replay alignment in both phases

aux:
  profiles=rotary_pooled_adv,rotary_pooled_nce,rotary_pooled_both
  curriculum_profiles=rotary_curr_replay_adv,rotary_curr_replay_nce,rotary_curr_replay_both
  pretrain_seeds=${SEEDS_CSV}
  groupedcv_eval_seeds=${SEEDS_CSV}
  subject_vote_eval_seeds=${SEEDS_CSV}
  curriculum_replay_alignment_weight=0.1

transfer:
  profile=rotary_curr_replay_nce
  pretrain_seeds=${SEEDS_CSV}
  groupedcv_eval_seeds=${SEEDS_CSV}
  subject_vote_eval_seeds=${SEEDS_CSV}
  replay_alignment_weight=0.1
  infonce_weight=0.1
  direction_a=train srpbs,abide1 -> eval abide2
  direction_b=train srpbs,abide2 -> eval abide1

lowk:
  eval_seeds=${SEEDS_CSV}
  k_values=10,20,30,40,50
EOF
}

run_dataset_summary() {
  local out_tex="${RESULTS_ROOT}/${RUN_TAG}_dataset_summary_table.tex"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_tex" ]]; then
    echo "Skipping dataset_summary: ${out_tex} exists"
    return
  fi
  echo
  echo "\$ python generate_table2.py > ${out_tex}"
  python generate_table2.py > "$out_tex"
}

run_baselines() {
  local abide_json="${RESULTS_ROOT}/${RUN_TAG}_abide_hc_asd_mixedsites_baseline_table.json"
  local abide_md="${RESULTS_ROOT}/${RUN_TAG}_abide_hc_asd_mixedsites_baseline_table.md"
  local srpbs_json="${RESULTS_ROOT}/${RUN_TAG}_srpbs_hc_disorder_mixedsites_baseline_table.json"
  local srpbs_md="${RESULTS_ROOT}/${RUN_TAG}_srpbs_hc_disorder_mixedsites_baseline_table.md"

  if [[ "$SKIP_EXISTING" == "1" && -f "$abide_json" && -f "$abide_md" ]]; then
    echo "Skipping baselines abide: ${abide_json} exists"
  else
    run_cmd python -m src.jepa_fusion.evaluate_baselines \
      --dataset abide \
      --seeds "$SEEDS_CSV" \
      --harmonization-covariates age,sex \
      --harmonization-fit-controls-only \
      --output-json "$abide_json" \
      --output-md "$abide_md"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$srpbs_json" && -f "$srpbs_md" ]]; then
    echo "Skipping baselines srpbs: ${srpbs_json} exists"
  else
    run_cmd python -m src.jepa_fusion.evaluate_baselines \
      --dataset srpbs \
      --seeds "$SEEDS_CSV" \
      --harmonization-covariates age,sex \
      --harmonization-fit-controls-only \
      --output-json "$srpbs_json" \
      --output-md "$srpbs_md"
  fi
}

train_and_eval_jepa() {
  local run_name="$1"
  local checkpoint_name="$2"
  local groupedcv_dataset="$3"
  local groupedcv_seeds="$4"
  local subjectvote_dataset="$5"
  local subjectvote_seeds="$6"
  local exclude_sites="$7"
  shift 7
  local train_args=("$@")

  local save_dir="${CHECKPOINT_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}_train.log"
  local checkpoint="${save_dir}/${checkpoint_name}"
  local groupedcv_json="${RESULTS_ROOT}/${run_name}_groupedcv.json"
  local groupedcv_md="${RESULTS_ROOT}/${run_name}_groupedcv.md"
  local subjectvote_json="${RESULTS_ROOT}/${run_name}_subject_vote.json"
  local subjectvote_md="${RESULTS_ROOT}/${run_name}_subject_vote.md"

  mkdir -p "$save_dir"

  if [[ "$SKIP_EXISTING" == "1" && -f "$checkpoint" ]]; then
    echo "Skipping train: ${checkpoint} exists"
  else
    run_cmd python -m src.jepa_fusion.train_jepa \
      "${train_args[@]}" \
      --device "$DEVICE" \
      --seed "${run_name##*_seed}" \
      --save-dir "$save_dir" \
      --log-file "$log_file"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$groupedcv_json" && -f "$groupedcv_md" ]]; then
    echo "Skipping groupedcv eval: ${groupedcv_json} exists"
  else
    local groupedcv_cmd=(
      python -m src.jepa_fusion.evaluate_jepa
      --checkpoint "$checkpoint"
      --dataset "$groupedcv_dataset"
      --device "$DEVICE"
      --seeds "$groupedcv_seeds"
      --output-json "$groupedcv_json"
      --output-md "$groupedcv_md"
    )
    if [[ -n "$exclude_sites" ]]; then
      groupedcv_cmd+=(--exclude-sites "$exclude_sites")
    fi
    run_cmd "${groupedcv_cmd[@]}"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$subjectvote_json" && -f "$subjectvote_md" ]]; then
    echo "Skipping subject-vote eval: ${subjectvote_json} exists"
  else
    local subjectvote_cmd=(
      python -m src.jepa_fusion.evaluate_jepa_subject_vote
      --checkpoint "$checkpoint"
      --dataset "$subjectvote_dataset"
      --device "$DEVICE"
      --seeds "$subjectvote_seeds"
      --output-json "$subjectvote_json"
      --output-md "$subjectvote_md"
    )
    if [[ -n "$exclude_sites" ]]; then
      subjectvote_cmd+=(--exclude-sites "$exclude_sites")
    fi
    run_cmd "${subjectvote_cmd[@]}"
  fi
}

pilot_train_args() {
  local model_key="$1"
  case "$model_key" in
    temporal_basic)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant basic \
        --training-mode pooled \
        --no-conv-stem \
        --no-network-embeddings
      ;;
    st_roi_patch)
      printf '%s\n' \
        --model-family st_roi_patch \
        --training-mode pooled
      ;;
    temporal_conv_net)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant conv_net \
        --training-mode pooled \
        --use-conv-stem \
        --use-network-embeddings
      ;;
    temporal_rotary_conv_net)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode pooled \
        --use-conv-stem \
        --use-network-embeddings
      ;;
    *)
      echo "Unknown pilot model: ${model_key}" >&2
      return 1
      ;;
  esac
}

curriculum_rotary_args() {
  printf '%s\n' \
    --model-family temporal \
    --temporal-variant rotary_conv_net \
    --training-mode curriculum \
    --use-conv-stem \
    --use-network-embeddings \
    --replay-alignment-weight 0.1 \
    --replay-pair-sampling
}

aux_pooled_args() {
  local profile="$1"
  case "$profile" in
    rotary_pooled_adv)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode pooled \
        --use-conv-stem \
        --use-network-embeddings \
        --adversarial-site-weight 0.1
      ;;
    rotary_pooled_nce)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode pooled \
        --use-conv-stem \
        --use-network-embeddings \
        --infonce-weight 0.1
      ;;
    rotary_pooled_both)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode pooled \
        --use-conv-stem \
        --use-network-embeddings \
        --adversarial-site-weight 0.1 \
        --infonce-weight 0.1
      ;;
    *)
      echo "Unknown aux profile: ${profile}" >&2
      return 1
      ;;
  esac
}

transfer_rotary_curr_args() {
  local main_datasets="$1"
  printf '%s\n' \
    --model-family temporal \
    --temporal-variant rotary_conv_net \
    --training-mode curriculum \
    --use-conv-stem \
    --use-network-embeddings \
    --replay-alignment-weight 0.1 \
    --replay-pair-sampling \
    --infonce-weight 0.1 \
    --main-datasets "${main_datasets}"
}

aux_curriculum_args() {
  local profile="$1"
  case "$profile" in
    rotary_curr_replay_adv)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode curriculum \
        --use-conv-stem \
        --use-network-embeddings \
        --replay-alignment-weight 0.1 \
        --replay-pair-sampling \
        --adversarial-site-weight 0.1
      ;;
    rotary_curr_replay_nce)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode curriculum \
        --use-conv-stem \
        --use-network-embeddings \
        --replay-alignment-weight 0.1 \
        --replay-pair-sampling \
        --infonce-weight 0.1
      ;;
    rotary_curr_replay_both)
      printf '%s\n' \
        --model-family temporal \
        --temporal-variant rotary_conv_net \
        --training-mode curriculum \
        --use-conv-stem \
        --use-network-embeddings \
        --replay-alignment-weight 0.1 \
        --replay-pair-sampling \
        --adversarial-site-weight 0.1 \
        --infonce-weight 0.1
      ;;
    *)
      echo "Unknown curriculum aux profile: ${profile}" >&2
      return 1
      ;;
  esac
}

run_pooled_pilot() {
  local model
  local seed
  local run_name
  local -a train_args
  for model in temporal_basic st_roi_patch temporal_conv_net temporal_rotary_conv_net; do
    mapfile -t train_args < <(pilot_train_args "$model")
    for seed in "${SEED_LIST[@]}"; do
      run_name="${RUN_TAG}_pilot_${model}_pooled_seed${seed}"
      train_and_eval_jepa \
        "$run_name" \
        "pooled_final.pt" \
        "all" \
        "$SEEDS_CSV" \
        "all" \
        "$SEEDS_CSV" \
        "" \
        "${train_args[@]}"
    done
  done
}

run_curriculum_rotary() {
  local seed
  local run_name
  local -a train_args
  mapfile -t train_args < <(curriculum_rotary_args)
  for seed in "${SEED_LIST[@]}"; do
    run_name="${RUN_TAG}_rotary_curr_replay_seed${seed}"
    train_and_eval_jepa \
      "$run_name" \
      "phase2_final.pt" \
      "all" \
      "$SEEDS_CSV" \
      "all" \
      "$SEEDS_CSV" \
      "" \
      "${train_args[@]}"
  done
}

run_aux_pooled() {
  local profile
  local seed
  local run_name
  local -a train_args
  for profile in rotary_pooled_adv rotary_pooled_nce rotary_pooled_both; do
    mapfile -t train_args < <(aux_pooled_args "$profile")
    for seed in "${SEED_LIST[@]}"; do
      run_name="${RUN_TAG}_${profile}_seed${seed}"
      train_and_eval_jepa \
        "$run_name" \
        "pooled_final.pt" \
        "all" \
        "$SEEDS_CSV" \
        "all" \
        "$SEEDS_CSV" \
        "" \
        "${train_args[@]}"
    done
  done
}

run_aux_curriculum() {
  local profile
  local seed
  local run_name
  local -a train_args
  for profile in rotary_curr_replay_adv rotary_curr_replay_nce rotary_curr_replay_both; do
    mapfile -t train_args < <(aux_curriculum_args "$profile")
    for seed in "${SEED_LIST[@]}"; do
      run_name="${RUN_TAG}_${profile}_seed${seed}"
      train_and_eval_jepa \
        "$run_name" \
        "phase2_final.pt" \
        "all" \
        "$SEEDS_CSV" \
        "all" \
        "$SEEDS_CSV" \
        "" \
        "${train_args[@]}"
    done
  done
}

run_transfer_direction() {
  local direction="$1"
  local train_ds
  local eval_ds
  local overlap
  local seed
  local run_name
  local -a train_args

  if [[ "$direction" == "A" ]]; then
    train_ds="srpbs,abide1"
    eval_ds="abide2"
    overlap="KKI,NYU,OHSU,SDSU,UCLA_1,USM,Trinity"
  elif [[ "$direction" == "B" ]]; then
    train_ds="srpbs,abide2"
    eval_ds="abide1"
    overlap="ABIDEI-KKI,ABIDEI-NYU,ABIDEI-OHSU,ABIDEI-SDSU,ABIDEI-UCLA_1,ABIDEI-UCLA_2,ABIDEI-USM,ABIDEI-Trinity"
  else
    echo "Unknown transfer direction: ${direction}" >&2
    return 1
  fi

  mapfile -t train_args < <(transfer_rotary_curr_args "$train_ds")
  for seed in "${SEED_LIST[@]}"; do
    run_name="${RUN_TAG}_transfer_${train_ds//,/_}_to_${eval_ds}_rotary_curr_replay_nce_seed${seed}"
    train_and_eval_jepa \
      "$run_name" \
      "phase2_final.pt" \
      "$eval_ds" \
      "$SEEDS_CSV" \
      "$eval_ds" \
      "$SEEDS_CSV" \
      "$overlap" \
      "${train_args[@]}"
  done
}

run_lowk() {
  local srpbs_json="${RESULTS_ROOT}/${RUN_TAG}_srpbs_lowk.json"
  local srpbs_md="${RESULTS_ROOT}/${RUN_TAG}_srpbs_lowk.md"
  local abide_json="${RESULTS_ROOT}/${RUN_TAG}_abide_lowk.json"
  local abide_md="${RESULTS_ROOT}/${RUN_TAG}_abide_lowk.md"

  if [[ "$SKIP_EXISTING" == "1" && -f "$srpbs_json" && -f "$srpbs_md" ]]; then
    echo "Skipping lowk srpbs: ${srpbs_json} exists"
  else
    run_cmd python -m src.jepa_fusion.evaluate_baselines_lowk_newbaseline \
      --dataset srpbs \
      --k-values 10,20,30,40,50 \
      --seeds "$SEEDS_CSV" \
      --output-json "$srpbs_json" \
      --output-md "$srpbs_md"
  fi

  if [[ "$SKIP_EXISTING" == "1" && -f "$abide_json" && -f "$abide_md" ]]; then
    echo "Skipping lowk abide: ${abide_json} exists"
  else
    run_cmd python -m src.jepa_fusion.evaluate_baselines_lowk_newbaseline \
      --dataset abide \
      --k-values 10,20,30,40,50 \
      --seeds "$SEEDS_CSV" \
      --output-json "$abide_json" \
      --output-md "$abide_md"
  fi
  run_cmd python -m src.jepa_fusion.plot_lowk_dual_axis \
    --abide-json "$abide_json" \
    --srpbs-json "$srpbs_json" \
    --abide-baseline-json "${RESULTS_ROOT}/${RUN_TAG}_abide_hc_asd_mixedsites_baseline_table.json" \
    --srpbs-baseline-json "${RESULTS_ROOT}/${RUN_TAG}_srpbs_hc_disorder_mixedsites_baseline_table.json" \
    --outdir "results/figures/jepa_fusion/${RUN_TAG}_lowk"
}

run_tex_tables() {
  run_cmd python -m src.jepa_fusion.build_final_run_tables \
    --run-tag "$RUN_TAG" \
    --results-dir "$RESULTS_ROOT" \
    --seeds "$SEEDS_CSV"
}

write_manifest

if has_stage dataset_summary; then
  run_dataset_summary
fi

if has_stage baselines; then
  run_baselines
fi

if has_stage pilot_pooled; then
  run_pooled_pilot
fi

if has_stage curriculum; then
  run_curriculum_rotary
fi

if has_stage aux; then
  run_aux_pooled
  run_aux_curriculum
fi

if has_stage transfer; then
  run_transfer_direction A
  run_transfer_direction B
fi

if has_stage lowk; then
  run_lowk
fi

if has_stage tables; then
  run_tex_tables
fi

echo
echo "Done. Tagged outputs are under:"
echo "  ${RESULTS_ROOT}"
echo "  ${CHECKPOINT_ROOT}"
echo "  ${LOG_ROOT}"
