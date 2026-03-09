#!/bin/bash

# ==============================================================================
# MS-JEPA FULL MANUSCRIPT PIPELINE
# Automatically trains all baselines, ablations, and the proposed model.
# ==============================================================================

export PYTHONPATH=/data/s.dharia-ra/multi-site-problem/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "======================================================================="
echo "Starting FULL UNIFIED PIPELINE (Baselines -> Phase 1 -> Phase 2 -> Eval)"
echo "======================================================================="

mkdir -p results/train_logs

# ------------------------------------------------------------------------------
# STEP 1: Foundation Models & Baselines (Independent)
# ------------------------------------------------------------------------------

echo "[PIPELINE] Launching Round 1: Foundation Models & Baselines..."

# GPU 0: SimCLR Baseline
CUDA_VISIBLE_DEVICES=0 nohup python src/jepa/baselines/train_simclr.py > results/train_logs/train_simclr.log 2>&1 &
PID_SIMCLR=$!

# GPU 1: VAE
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/baselines/train_vae.py > results/train_logs/train_vae.log 2>&1 &
PID_VAE=$!

# GPU 1 & 2: Phase 1 Foundations
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/train_phase1.py > results/train_logs/train_phase1_curr.log 2>&1 &
PID_P1_CURR=$!

CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/train_phase1.py --no_curriculum > results/train_logs/train_phase1_nocurr.log 2>&1 &
PID_P1_NOCURR=$!

echo "[PIPELINE] Round 1 Dispatched. Waiting for Phase 1 to finish (approx 5-10 mins)..."

# ------------------------------------------------------------------------------
# STEP 2: Wait for Phase 1 Checkpoints
# ------------------------------------------------------------------------------
wait $PID_P1_CURR
wait $PID_P1_NOCURR

echo "[PIPELINE] Phase 1 Checkpoints Saved! Launching Round 2: Phase 2 Ablations..."

# ------------------------------------------------------------------------------
# STEP 3: Phase 2 Ablations (Dependent on Phase 1)
# ------------------------------------------------------------------------------

# GPU 2: Curriculum Variants
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/train_phase2.py > results/train_logs/train_phase2_vanilla_curr.log 2>&1 &
PID_P2_V_CURR=$!
sleep 10
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/train_phase2_contrastive.py > results/train_logs/train_phase2_cl_curr.log 2>&1 &
PID_P2_CL_CURR=$!

# GPU 1: No-Curriculum Variants
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/train_phase2.py --no_curriculum > results/train_logs/train_phase2_vanilla_nocurr.log 2>&1 &
PID_P2_V_NOCURR=$!
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/train_phase2_contrastive.py --no_curriculum > results/train_logs/train_phase2_cl_nocurr.log 2>&1 &
PID_P2_CL_NOCURR=$!

echo "[PIPELINE] Round 2 Dispatched. Everything is now running."
echo "[PIPELINE] Monitor logs in results/train_logs/"

# ------------------------------------------------------------------------------
# STEP 4: Optional Final Evaluation
# ------------------------------------------------------------------------------
# We wait for the longest jobs (SimCLR/VAE or Phase 2 CL)
wait $PID_SIMCLR
wait $PID_VAE
wait $PID_P2_CL_CURR
wait $PID_P2_CL_NOCURR

echo "[PIPELINE] ALL TRAINING COMPLETE! Launching final evaluation suite..."
/data/s.dharia-ra/multi-site-problem/src/jepa/run_all_evals.sh

echo "======================================================================="
echo "MANUSCRIPT PIPELINE FINISHED. Check results/tables/ for publication metrics."
echo "======================================================================="
