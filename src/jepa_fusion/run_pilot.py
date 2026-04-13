"""One-shot launcher for pilot JEPA training across all model families.

Trains each pilot configuration in sequence and runs subject-vote evaluation.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PILOT_CONFIGS = [
    {
        "name": "pilot_t_basic",
        "model_family": "temporal",
        "temporal_variant": "basic",
        "training_mode": "pooled",
    },
    {
        "name": "pilot_t_basic_curr",
        "model_family": "temporal",
        "temporal_variant": "basic",
        "training_mode": "curriculum",
    },
    {
        "name": "pilot_t_conv_net",
        "model_family": "temporal",
        "temporal_variant": "conv_net",
        "training_mode": "pooled",
    },
    {
        "name": "pilot_t_conv_net_curr",
        "model_family": "temporal",
        "temporal_variant": "conv_net",
        "training_mode": "curriculum",
    },
    {
        "name": "pilot_t_rotary_conv_net",
        "model_family": "temporal",
        "temporal_variant": "rotary_conv_net",
        "training_mode": "pooled",
    },
    {
        "name": "pilot_t_rotary_conv_net_curr",
        "model_family": "temporal",
        "temporal_variant": "rotary_conv_net",
        "training_mode": "curriculum",
    },
]


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def run_pilot_training(config: dict, base_dir: str, device: str, seed: int, dry_run: bool = False):
    logger = logging.getLogger("RunPilot")
    name = config["name"]
    save_dir = f"{base_dir}/{name}"
    log_file = f"{save_dir}/train.log"

    cmd = [
        sys.executable, "-m", "src.jepa_fusion.train_jepa",
        "--model-family", config["model_family"],
        "--temporal-variant", config["temporal_variant"],
        "--training-mode", config["training_mode"],
        "--device", device,
        "--save-dir", save_dir,
        "--log-file", log_file,
        "--seed", str(seed),
    ]

    logger.info("Training %s", name)
    logger.info("Command: %s", " ".join(cmd))

    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return

    subprocess.run(cmd, check=True)

    # Run subject-vote evaluation on the final checkpoint
    if config["training_mode"] == "pooled":
        ckpt = f"{save_dir}/pooled_final.pt"
    else:
        ckpt = f"{save_dir}/phase2_final.pt"

    if Path(ckpt).exists():
        eval_name = f"{name}_{Path(ckpt).stem}"
        eval_cmd = [
            sys.executable, "-m", "src.jepa_fusion.evaluate_jepa_subject_vote",
            "--checkpoint", ckpt,
            "--device", device,
        ]
        logger.info("Evaluating %s", eval_name)
        subprocess.run(eval_cmd, check=True)
    else:
        logger.warning("Checkpoint not found: %s", ckpt)


def parse_args():
    p = argparse.ArgumentParser(description="Run all pilot JEPA training configs")
    p.add_argument("--base-dir", default="checkpoints/jepa_fusion")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--configs", nargs="+", default=None, help="Pilot config names to run (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("RunPilot")

    configs = PILOT_CONFIGS
    if args.configs:
        configs = [c for c in PILOT_CONFIGS if c["name"] in args.configs]

    logger.info("Running %d pilot configurations", len(configs))
    for config in configs:
        try:
            run_pilot_training(
                config,
                base_dir=args.base_dir,
                device=args.device,
                seed=args.seed,
                dry_run=args.dry_run,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Failed on %s: %s", config["name"], e)
            continue


if __name__ == "__main__":
    main()
