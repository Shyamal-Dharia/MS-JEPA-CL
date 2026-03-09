#!/bin/bash

# ==============================================================================
# MS-JEPA RECOVERY PIPELINE (Phase 2 Contrastive & Final Evaluation)
# ==============================================================================

export PYTHONPATH=/data/s.dharia-ra/multi-site-problem/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "======================================================================="
echo "Starting RECOVERY: Phase 2 Contrastive Models -> Final Evaluation"
echo "======================================================================="

# GPU 2: Curriculum Contrastive
echo "[RECOVERY] Launching Phase 2 Contrastive (Curriculum) on GPU 2..."
CUDA_VISIBLE_DEVICES=2 python src/jepa/train_phase2_contrastive.py > results/train_logs/train_phase2_cl_curr.log 2>&1 &
PID_CL_CURR=$!

# GPU 1: No-Curriculum Contrastive
echo "[RECOVERY] Launching Phase 2 Contrastive (No-Curriculum) on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python src/jepa/train_phase2_contrastive.py --no_curriculum > results/train_logs/train_phase2_cl_nocurr.log 2>&1 &
PID_CL_NOCURR=$!

echo "[RECOVERY] Dispatched. Waiting for completion..."

wait $PID_CL_CURR
wait $PID_CL_NOCURR

echo "[RECOVERY] Phase 2 Contrastive Complete! Launching final evaluation..."

# Launch evaluator in background
nohup /data/s.dharia-ra/multi-site-problem/src/jepa/run_all_evals.sh > results/logs/eval_mass_dispatch.log 2>&1 &

echo "======================================================================="
echo "RECOVERY DISPATCHED. Evaluation will start automatically when training ends."
echo "======================================================================="
