#!/bin/bash

# ==============================================================================
# Multi-GPU Training Launcher (ROUND 2: Phase 2 Ablations)
# Distributes the dependent Phase 2 training runs across 3 GPUs.
# Requires Phase 1 checkpoints to be present in checkpoints/phase1/ and checkpoints/phase1_nocurr/
# ==============================================================================

export PYTHONPATH=/data/s.dharia-ra/multi-site-problem/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "======================================================================="
echo "Starting Massive Multi-GPU Training Suite (Round 2: Phase 2 Models)"
echo "======================================================================="

# Ensure log directory exists
mkdir -p results/train_logs

# ------------------------------------------------------------------------------
# We have 3 GPUs (0, 1, 2). 
# SimCLR is on GPU 0. VAE is on GPU 1.
# GPU 2 is empty. GPU 1 has ~40GB free.
# ------------------------------------------------------------------------------

# GPU 2: Curriculum Variants (The most important ones)
echo "Launching GPU 2 jobs..."
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/train_phase2.py > results/train_logs/train_phase2_vanilla_curr.log 2>&1 &
# Wait for some VRAM allocation
sleep 15
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/train_phase2_contrastive.py > results/train_logs/train_phase2_cl_curr.log 2>&1 &

# GPU 1: No-Curriculum Variants (Baseline Ablations)
echo "Launching GPU 1 jobs..."
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/train_phase2.py --no_curriculum > results/train_logs/train_phase2_vanilla_nocurr.log 2>&1 &
# Wait 
sleep 15
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/train_phase2_contrastive.py --no_curriculum > results/train_logs/train_phase2_cl_nocurr.log 2>&1 &

echo "======================================================================="
echo "Round 2 training jobs have been dispatched to the background."
echo "Check progress in results/train_logs/"
echo "======================================================================="
