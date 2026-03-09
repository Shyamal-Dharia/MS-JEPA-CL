# src/jepa/run_curriculum.py
"""
End-to-end curriculum learning pipeline.

Runs all 3 phases sequentially:
  Phase 1: Self-supervised JEPA on travelling subjects
  Phase 2: Scale to SRPBS 1600 with 4 harmonization variants
  Phase 3: Comprehensive evaluation (frozen probing + fine-tuning)

Usage:
    cd /data/s.dharia-ra/multi-site-problem
    PYTHONPATH=src nohup python -m jepa.run_curriculum > results/curriculum.log 2>&1 &

Each phase's checkpoint is automatically passed to the next.
If a phase's output checkpoint already exists, it can be skipped with --skip-phase1, etc.
"""

import sys
import os
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("Curriculum")


# ═══════════════════════════════════════════════════════════════
# ★ CENTRAL CONFIGURATION — EDIT ALL HYPERPARAMETERS HERE ★
# ═══════════════════════════════════════════════════════════════

class CurriculumConfig:
    """
    All hyperparameters for the full pipeline in one place.
    Modify these values to change behaviour across ALL phases.
    """

    # ─────────────── Shared model architecture ───────────────
    embed_dim       = 400       # embedding dimension (64 or 128)
    encoder_depth   = 3        # transformer encoder layers
    predictor_depth = 2        # predictor transformer layers
    num_heads       = 4        # attention heads
    mlp_ratio       = 4.0      # MLP hidden dim ratio
    dropout         = 0.2      # dropout rate
    ema_decay       = 0.996    # EMA decay for target encoder

    # ─────────────── Data paths ──────────────────────────────
    travelling_data = "preprocessed_data_fir/roi_timeseries_traveling"
    srpbs_data      = "preprocessed_data_fir/roi_timeseries_1600"

    # ─────────────── Phase 1: Self-supervised JEPA ───────────
    p1_epochs       = 100      # training epochs
    p1_batch_size   = 16       # batch size
    p1_lr           = 1e-4     # learning rate
    p1_mask_ratio   = 0.5      # fraction of tokens to mask
    p1_save_dir     = "checkpoints/phase1"

    # ─────────────── Phase 2: Shared settings ────────────────
    p2_epochs       = 100       # training epochs (all P2 variants)
    p2_batch_main   = 54       # main dataset batch size
    p2_batch_replay = 10       # replay buffer batch size
    p2_lr           = 5e-5     # learning rate
    p2_hc_only      = True     # use only healthy controls

    # ─────── Phase 2a: Vanilla (JEPA only) ────────
    p2a_save_dir    = "checkpoints/phase2"

    # ─────── Phase 2b: Adversarial v2 ─────────────
    p2b_adv_lr       = 1e-3    # adversarial discriminator LR
    p2b_grl_lambda   = 0.1     # gradient reversal strength
    p2b_warmup       = 10      # warmup epochs before adversarial
    p2b_save_dir     = "checkpoints/phase2_adversarial_v2"

    # ─────── Phase 2c: Contrastive (NT-Xent) ─────
    p2c_cl_weight    = 0.5     # contrastive loss weight
    p2c_temperature  = 0.1     # NT-Xent temperature
    p2c_save_dir     = "checkpoints/phase2_contrastive"

    # ─────── Phase 2d: CL + VICReg ───────────────
    p2d_cl_weight    = 0.5     # contrastive loss weight
    p2d_temperature  = 0.1     # NT-Xent temperature
    p2d_vicreg_weight = 1.0    # overall VICReg weight
    p2d_var_weight   = 1.0     # variance term weight
    p2d_cov_weight   = 1.0     # covariance term weight
    p2d_var_target   = 1.0     # target std for variance term
    p2d_save_dir     = "checkpoints/phase2_contrastive_vicreg_w10"

    # ─────────────── Phase 3: Evaluation ─────────────────────
    eval_ft_epochs     = 100    # fine-tuning epochs
    eval_ft_batch      = 32    # fine-tuning batch size
    eval_encoder_lr    = 1e-5  # encoder LR (low, pretrained)
    eval_head_lr       = 1e-3  # head LR (high, random init)
    eval_weight_decay  = 0.01  # weight decay
    eval_head_hidden   = 128   # classification head hidden dim
    eval_head_dropout  = 0.2   # classification head dropout
    eval_cv_folds      = 5     # CV folds
    eval_seeds         = [42, 123, 777]  # seeds for averaging
    eval_save_dir      = "results"

    # ─────────────── Device ──────────────────────────────────
    device = "cuda"  # "cuda" or "cpu"


# Global config instance
CFG = CurriculumConfig()


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def banner(text):
    """Print a clear phase banner."""
    print("\n" + "█" * 70)
    print(f"█  {text}")
    print("█" * 70 + "\n")


def phase_done(label, t0):
    elapsed = time.time() - t0
    logger.info(f"✓ {label} complete ({elapsed/60:.1f} min)")
    return elapsed


def apply_shared_arch(cfg):
    """Apply shared architecture params from CurriculumConfig to a phase config."""
    cfg.embed_dim = CFG.embed_dim
    cfg.encoder_depth = CFG.encoder_depth
    cfg.predictor_depth = CFG.predictor_depth
    cfg.num_heads = CFG.num_heads
    cfg.mlp_ratio = CFG.mlp_ratio
    cfg.dropout = CFG.dropout
    cfg.ema_decay = CFG.ema_decay
    return cfg


# ═══════════════════════════════════════════════════════════════
# Phase 1: Self-supervised JEPA on travelling subjects
# ═══════════════════════════════════════════════════════════════

def run_phase1(skip=False):
    output = f"{CFG.p1_save_dir}/model_A_phase1.pt"

    if skip and os.path.exists(output):
        logger.info(f"Phase 1: SKIPPED (checkpoint exists: {output})")
        return output

    banner("PHASE 1: Self-supervised JEPA on travelling subjects")
    t0 = time.time()

    from jepa.train_phase1 import Config, train_one_model

    cfg = Config()
    cfg = apply_shared_arch(cfg)
    cfg.data_dir = CFG.travelling_data
    cfg.epochs = CFG.p1_epochs
    cfg.batch_size = CFG.p1_batch_size
    cfg.lr = CFG.p1_lr
    cfg.mask_ratio = CFG.p1_mask_ratio
    cfg.save_dir = CFG.p1_save_dir

    model, history = train_one_model("temporal", cfg)

    phase_done("Phase 1", t0)
    return output


# ═══════════════════════════════════════════════════════════════
# Phase 2: Scale up with harmonization variants
# ═══════════════════════════════════════════════════════════════

def run_phase2_vanilla(phase1_ckpt, skip=False):
    output = f"{CFG.p2a_save_dir}/model_A_phase2_final.pt"

    if skip and os.path.exists(output):
        logger.info(f"Phase 2 vanilla: SKIPPED ({output})")
        return output

    banner("PHASE 2a: JEPA vanilla on SRPBS 1600 (HC only)")
    t0 = time.time()

    from jepa.train_phase2 import Phase2Config, train_phase2

    cfg = Phase2Config()
    cfg = apply_shared_arch(cfg)
    cfg.phase1_checkpoint = phase1_ckpt
    cfg.main_data_dir = CFG.srpbs_data
    cfg.replay_data_dir = CFG.travelling_data
    cfg.epochs = CFG.p2_epochs
    cfg.batch_main = CFG.p2_batch_main
    cfg.batch_replay = CFG.p2_batch_replay
    cfg.lr = CFG.p2_lr
    cfg.save_dir = CFG.p2a_save_dir

    train_phase2(cfg)

    phase_done("Phase 2 vanilla", t0)
    return output


def run_phase2_adversarial(phase1_ckpt, skip=False):
    output = f"{CFG.p2b_save_dir}/model_A_phase2advv2_final.pt"

    if skip and os.path.exists(output):
        logger.info(f"Phase 2 adversarial: SKIPPED ({output})")
        return output

    banner("PHASE 2b: JEPA + Adversarial v2 on SRPBS 1600")
    t0 = time.time()

    from jepa.train_p2_advv import Phase2AdvV2Config, train_phase2_adversarial_v2

    cfg = Phase2AdvV2Config()
    cfg = apply_shared_arch(cfg)
    cfg.phase1_checkpoint = phase1_ckpt
    cfg.main_data_dir = CFG.srpbs_data
    cfg.replay_data_dir = CFG.travelling_data
    cfg.epochs = CFG.p2_epochs
    cfg.batch_main = CFG.p2_batch_main
    cfg.batch_replay = CFG.p2_batch_replay
    cfg.lr = CFG.p2_lr
    cfg.adv_lr = CFG.p2b_adv_lr
    cfg.grl_lambda = CFG.p2b_grl_lambda
    cfg.warmup_epochs = CFG.p2b_warmup
    cfg.save_dir = CFG.p2b_save_dir

    train_phase2_adversarial_v2(cfg)

    phase_done("Phase 2 adversarial v2", t0)
    return output


def run_phase2_contrastive(phase1_ckpt, skip=False):
    output = f"{CFG.p2c_save_dir}/model_A_phase2cl_final.pt"

    if skip and os.path.exists(output):
        logger.info(f"Phase 2 contrastive: SKIPPED ({output})")
        return output

    banner("PHASE 2c: JEPA + Contrastive (NT-Xent) on SRPBS 1600")
    t0 = time.time()

    from jepa.train_phase2_contrastive import Phase2ContrastiveConfig, train_phase2_contrastive

    cfg = Phase2ContrastiveConfig()
    cfg = apply_shared_arch(cfg)
    cfg.phase1_checkpoint = phase1_ckpt
    cfg.main_data_dir = CFG.srpbs_data
    cfg.replay_data_dir = CFG.travelling_data
    cfg.epochs = CFG.p2_epochs
    cfg.batch_main = CFG.p2_batch_main
    cfg.batch_replay = CFG.p2_batch_replay
    cfg.lr = CFG.p2_lr
    cfg.contrastive_weight = CFG.p2c_cl_weight
    cfg.temperature = CFG.p2c_temperature
    cfg.save_dir = CFG.p2c_save_dir

    train_phase2_contrastive(cfg)

    phase_done("Phase 2 contrastive", t0)
    return output


def run_phase2_cl_vicreg(phase1_ckpt, skip=False):
    output = f"{CFG.p2d_save_dir}/model_A_phase2cl_vicreg_final.pt"

    if skip and os.path.exists(output):
        logger.info(f"Phase 2 CL+VICReg: SKIPPED ({output})")
        return output

    banner("PHASE 2d: JEPA + Contrastive + VICReg on SRPBS 1600")
    t0 = time.time()

    from jepa.train_phase2_contrastive_vicreg import Phase2CLVICRegConfig, train_phase2_contrastive_vicreg

    cfg = Phase2CLVICRegConfig()
    cfg = apply_shared_arch(cfg)
    cfg.phase1_checkpoint = phase1_ckpt
    cfg.main_data_dir = CFG.srpbs_data
    cfg.replay_data_dir = CFG.travelling_data
    cfg.epochs = CFG.p2_epochs
    cfg.batch_main = CFG.p2_batch_main
    cfg.batch_replay = CFG.p2_batch_replay
    cfg.lr = CFG.p2_lr
    cfg.contrastive_weight = CFG.p2d_cl_weight
    cfg.temperature = CFG.p2d_temperature
    cfg.vicreg_weight = CFG.p2d_vicreg_weight
    cfg.vicreg_var_weight = CFG.p2d_var_weight
    cfg.vicreg_cov_weight = CFG.p2d_cov_weight
    cfg.vicreg_var_target = CFG.p2d_var_target
    cfg.save_dir = CFG.p2d_save_dir

    train_phase2_contrastive_vicreg(cfg)

    phase_done("Phase 2 CL+VICReg", t0)
    return output


# ═══════════════════════════════════════════════════════════════
# Phase 3: Comprehensive evaluation
# ═══════════════════════════════════════════════════════════════

def run_evaluation(skip=False):
    output = f"{CFG.eval_save_dir}/evaluation_results.json"

    if skip and os.path.exists(output):
        logger.info(f"Evaluation: SKIPPED (results exist: {output})")
        return output

    banner("PHASE 3: Comprehensive evaluation (all methods × all tasks)")
    t0 = time.time()

    from jepa.evaluate_all import EvalConfig, evaluate_all

    cfg = EvalConfig()
    # Apply shared architecture
    cfg.embed_dim = CFG.embed_dim
    cfg.encoder_depth = CFG.encoder_depth
    cfg.predictor_depth = CFG.predictor_depth
    cfg.num_heads = CFG.num_heads
    cfg.mlp_ratio = CFG.mlp_ratio
    cfg.dropout = CFG.dropout
    cfg.ema_decay = CFG.ema_decay
    # Apply eval-specific params
    cfg.ft_epochs = CFG.eval_ft_epochs
    cfg.ft_batch_size = CFG.eval_ft_batch
    cfg.ft_encoder_lr = CFG.eval_encoder_lr
    cfg.ft_head_lr = CFG.eval_head_lr
    cfg.ft_weight_decay = CFG.eval_weight_decay
    cfg.head_hidden = CFG.eval_head_hidden
    cfg.head_dropout = CFG.eval_head_dropout
    cfg.n_cv_folds = CFG.eval_cv_folds
    cfg.seeds = CFG.eval_seeds
    cfg.save_dir = CFG.eval_save_dir
    # Update checkpoint paths to match our save dirs
    cfg.methods = [
        ("Raw (no pretrain)",    None,                                                      "raw"),
        ("ComBat",               None,                                                      "combat"),
        ("JEPA vanilla",         f"{CFG.p2a_save_dir}/model_A_phase2_final.pt",             "jepa"),
        ("JEPA + Adversarial",   f"{CFG.p2b_save_dir}/model_A_phase2advv2_final.pt",        "jepa"),
        ("JEPA + Contrastive",   f"{CFG.p2c_save_dir}/model_A_phase2cl_final.pt",           "jepa"),
        ("JEPA + CL + VICReg",   f"{CFG.p2d_save_dir}/model_A_phase2cl_vicreg_final.pt",   "jepa"),
    ]

    evaluate_all(cfg)

    phase_done("Evaluation", t0)
    return output


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end MS-JEPA curriculum learning pipeline"
    )
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip Phase 1 if checkpoint exists")
    parser.add_argument("--skip-phase2", action="store_true",
                        help="Skip all Phase 2 variants if checkpoints exist")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip evaluation if results exist")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip any phase whose output already exists")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run only the evaluation step")
    args = parser.parse_args()

    skip1 = args.skip_phase1 or args.skip_existing
    skip2 = args.skip_phase2 or args.skip_existing
    skip_eval = args.skip_eval or args.skip_existing

    t_total = time.time()

    banner("MS-JEPA CURRICULUM LEARNING PIPELINE")
    logger.info("Configuration:")
    logger.info(f"  embed_dim={CFG.embed_dim}, depth={CFG.encoder_depth}, "
                f"heads={CFG.num_heads}, dropout={CFG.dropout}")
    logger.info(f"  Phase 1: {CFG.p1_epochs} epochs, lr={CFG.p1_lr}")
    logger.info(f"  Phase 2: {CFG.p2_epochs} epochs, lr={CFG.p2_lr}")
    logger.info(f"    Adversarial: grl_lambda={CFG.p2b_grl_lambda}, warmup={CFG.p2b_warmup}")
    logger.info(f"    Contrastive: weight={CFG.p2c_cl_weight}, temp={CFG.p2c_temperature}")
    logger.info(f"    VICReg: weight={CFG.p2d_vicreg_weight}, var={CFG.p2d_var_weight}, cov={CFG.p2d_cov_weight}")
    logger.info(f"  Evaluation: ft_epochs={CFG.eval_ft_epochs}, seeds={CFG.eval_seeds}")

    timings = {}

    if args.eval_only:
        logger.info("\n→ --eval-only: Skipping Phase 1 & 2")
    else:
        # ── Phase 1 ──
        t0 = time.time()
        phase1_ckpt = run_phase1(skip=skip1)
        timings["Phase 1"] = time.time() - t0

        # ── Phase 2 (all variants from Phase 1 checkpoint) ──
        t0 = time.time()
        run_phase2_vanilla(phase1_ckpt, skip=skip2)
        timings["Phase 2a (vanilla)"] = time.time() - t0

        t0 = time.time()
        run_phase2_adversarial(phase1_ckpt, skip=skip2)
        timings["Phase 2b (adversarial)"] = time.time() - t0

        t0 = time.time()
        run_phase2_contrastive(phase1_ckpt, skip=skip2)
        timings["Phase 2c (contrastive)"] = time.time() - t0

        t0 = time.time()
        run_phase2_cl_vicreg(phase1_ckpt, skip=skip2)
        timings["Phase 2d (CL+VICReg)"] = time.time() - t0

    # ── Phase 3: Evaluation ──
    t0 = time.time()
    run_evaluation(skip=skip_eval)
    timings["Evaluation"] = time.time() - t0

    # ── Summary ──
    total = time.time() - t_total
    banner("PIPELINE COMPLETE")
    logger.info("Timing summary:")
    for phase, elapsed in timings.items():
        logger.info(f"  {phase:30s}: {elapsed/60:6.1f} min")
    logger.info(f"  {'TOTAL':30s}: {total/60:6.1f} min")
    logger.info(f"\nResults: {CFG.eval_save_dir}/evaluation_results.json")


if __name__ == "__main__":
    main()
