"""Multi-seed confirmation runner for selected JEPA profiles.

Trains N seeds for each profile and runs both subject-vote and fusion evaluation.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path


CONFIRMATION_PROFILES = {
    "basic_pooled": {
        "model_family": "temporal",
        "temporal_variant": "basic",
        "training_mode": "pooled",
        "extra_args": [],
    },
    "rotary_curr_align": {
        "model_family": "temporal",
        "temporal_variant": "rotary_conv_net",
        "training_mode": "curriculum",
        "extra_args": [
            "--global-loss-weight", "1.0",
            "--replay-alignment-weight", "0.1",
            "--replay-pair-sampling",
        ],
    },
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def run_confirmation_seed(
    profile_name: str,
    profile: dict,
    seed: int,
    base_dir: str,
    device: str,
    dry_run: bool = False,
):
    logger = logging.getLogger("Confirm")
    name = f"confirm_{profile_name}_seed{seed}"
    save_dir = f"{base_dir}/{name}"
    log_file = f"{save_dir}/train.log"

    # Training
    train_cmd = [
        sys.executable, "-m", "src.jepa_fusion.train_jepa",
        "--model-family", profile["model_family"],
        "--temporal-variant", profile["temporal_variant"],
        "--training-mode", profile["training_mode"],
        "--device", device,
        "--save-dir", save_dir,
        "--log-file", log_file,
        "--seed", str(seed),
    ] + profile.get("extra_args", [])

    logger.info("Training %s", name)
    logger.info("Command: %s", " ".join(train_cmd))

    if dry_run:
        logger.info("[DRY RUN] Skipping training")
    else:
        subprocess.run(train_cmd, check=True)

    # Identify checkpoint
    if profile["training_mode"] == "curriculum":
        ckpt = f"{save_dir}/phase2_final.pt"
    else:
        ckpt = f"{save_dir}/pooled_final.pt"

    if not Path(ckpt).exists() and not dry_run:
        logger.warning("Checkpoint not found: %s", ckpt)
        return

    # Subject-vote evaluation
    sv_cmd = [
        sys.executable, "-m", "src.jepa_fusion.evaluate_jepa_subject_vote",
        "--checkpoint", ckpt,
        "--device", device,
    ]
    logger.info("Subject-vote eval: %s", name)
    if not dry_run:
        subprocess.run(sv_cmd, check=True)

    # Fusion evaluation
    fusion_cmd = [
        sys.executable, "-m", "src.jepa_fusion.evaluate_fusion_subject_vote",
        "--checkpoint", ckpt,
        "--device", device,
    ]
    logger.info("Fusion eval: %s", name)
    if not dry_run:
        subprocess.run(fusion_cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-seed JEPA confirmation")
    p.add_argument("--profiles", nargs="+", default=None, choices=sorted(CONFIRMATION_PROFILES))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--base-dir", default="checkpoints/jepa_fusion")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("Confirm")

    profiles = args.profiles or sorted(CONFIRMATION_PROFILES)
    seeds = args.seeds

    logger.info("Running %d profiles x %d seeds", len(profiles), len(seeds))
    for profile_name in profiles:
        profile = CONFIRMATION_PROFILES[profile_name]
        for seed in seeds:
            try:
                run_confirmation_seed(
                    profile_name, profile, seed=seed,
                    base_dir=args.base_dir, device=args.device,
                    dry_run=args.dry_run,
                )
            except subprocess.CalledProcessError as e:
                logger.error("Failed %s seed=%d: %s", profile_name, seed, e)


if __name__ == "__main__":
    main()
