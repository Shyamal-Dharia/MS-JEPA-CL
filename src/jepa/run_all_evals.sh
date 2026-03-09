#!/bin/bash

# ==============================================================================
# Multi-GPU Evaluation Launcher
# Distributes the evaluation methods across available GPUs to drastically
# accelerate the complete benchmark run.
# ==============================================================================

export PYTHONPATH=/data/s.dharia-ra/multi-site-problem/src
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "======================================================================="
echo "Starting Massive Multi-GPU Evaluation Suite"
echo "======================================================================="

# Ensure results directory exists
mkdir -p results/logs

# Map specific methods to specific GPUs
# We have 8 methods and 3 GPUs (0, 1, 2). 
# We'll put 3 methods on GPU 0, 3 on GPU 1, and 2 on GPU 2.
# ------------------------------------------------------------------------------

# GPU 0: Foundation Baselines & SimCLR (3 jobs)
echo "Launching GPU 0 jobs..."
CUDA_VISIBLE_DEVICES=0 nohup python src/jepa/evaluate_all.py --method "Raw (Transformer from scratch)" > results/logs/eval_raw.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python src/jepa/evaluate_all.py --method "ComBat (Transformer from scratch)" > results/logs/eval_combat.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python src/jepa/evaluate_all.py --method "SimCLR Baseline" > results/logs/eval_simclr.log 2>&1 &

# GPU 1: VAE & Non-Curriculum Ablations (3 jobs)
echo "Launching GPU 1 jobs..."
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/evaluate_all.py --method "Transformer-VAE" > results/logs/eval_vae.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/evaluate_all.py --method "JEPA Vanilla (No Curriculum)" > results/logs/eval_jepa_vanilla_nocurr.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python src/jepa/evaluate_all.py --method "JEPA + Contrastive (No Curriculum)" > results/logs/eval_jepa_cl_nocurr.log 2>&1 &

# GPU 2: Our Proposed Curriculum Models (2 jobs)
echo "Launching GPU 2 jobs..."
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/evaluate_all.py --method "JEPA Vanilla (With Curriculum)" > results/logs/eval_jepa_vanilla_curr.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python src/jepa/evaluate_all.py --method "JEPA + Contrastive (With Curriculum)" > results/logs/eval_jepa_cl_curr.log 2>&1 &

echo "======================================================================="
echo "All 8 evaluation jobs have been dispatched to the background."
echo "You can monitor their progress individually in the results/logs/ directory."
echo "======================================================================="
