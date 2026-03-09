#!/bin/bash

# ==============================================================================
# Multi-GPU Training Launcher (ROUND 1)
# Distributes the independent baseline and Phase 1 training runs across 3 GPUs
# ==============================================================================

export PYTHONPATH=/data/s.dharia-ra/multi-site-problem/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "======================================================================="
echo "Starting Massive Multi-GPU Training Suite (Round 1: First-stage Models)"
echo "======================================================================="

# Ensure log directory exists
mkdir -p results/train_logs

# ------------------------------------------------------------------------------
# We have 3 GPUs (0, 1, 2). 
# We need to run 4 independent models that don't rely on existing checkpoints:
# 1. SimCLR Baseline
# 2. VAE Baseline
# 3. Phase 1 JEPA (With Curriculum - Travelling Only)
# 4. Phase 1 JEPA (No Curriculum - All Subjects)
# ------------------------------------------------------------------------------

# GPU 0: SimCLR Baseline (Heavy)
echo "Launching GPU 0 jobs..."
CUDA_VISIBLE_DEVICES=0 nohup python src/jepa/baselines/train_simclr.py > results/train_logs/train_simclr.log 2>&1 &

# GPU 1: VAE & Phase 1 Vanilla
echo "Launching GPU 1 jobs..."
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/baselines/train_vae.py > results/train_logs/train_vae.log 2>&1 &
# Wait 10 seconds to stagger GPU memory allocation
sleep 10
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/train_phase1.py > results/train_logs/train_phase1_curr.log 2>&1 &

# GPU 2: Phase 1 (No Curriculum)
echo "Launching GPU 2 jobs..."
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/train_phase1.py --no_curriculum > results/train_logs/train_phase1_nocurr.log 2>&1 &

echo "======================================================================="
echo "Round 1 training jobs have been dispatched to the background."
echo "Check progress in results/train_logs/"
echo ""
echo "NOTE: Once these finish, you MUST run Round 2 to train the Phase 2 models:"
echo "CUDA_VISIBLE_DEVICES=0 python src/jepa/train_phase2_contrastive.py"
echo "CUDA_VISIBLE_DEVICES=1 python src/jepa/train_phase2.py"
echo "CUDA_VISIBLE_DEVICES=2 python src/jepa/train_phase2_contrastive.py --no_curriculum"
echo "======================================================================="
