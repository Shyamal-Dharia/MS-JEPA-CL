from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


PROFILES = {
    "basic_pooled": {
        "save_prefix": "confirm_basic_pooled",
        "train_args": [
            "--model-family",
            "temporal",
            "--temporal-variant",
            "basic",
            "--training-mode",
            "pooled",
            "--no-conv-stem",
            "--no-network-embeddings",
        ],
        "checkpoint_name": "pooled_final.pt",
    },
    "rotary_curr_align": {
        "save_prefix": "confirm_rotary_curr_align",
        "train_args": [
            "--model-family",
            "temporal",
            "--temporal-variant",
            "rotary_conv_net",
            "--training-mode",
            "curriculum",
            "--use-conv-stem",
            "--use-network-embeddings",
            "--replay-alignment-weight",
            "0.1",
            "--replay-pair-sampling",
        ],
        "checkpoint_name": "phase2_final.pt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate multiple JEPA pretraining seeds for the confirmation "
            "sweep used in the jepa_fusion paper tables."
        )
    )
    parser.add_argument(
        "--profile",
        nargs="+",
        choices=sorted(PROFILES),
        default=["rotary_curr_align", "basic_pooled"],
        help="Which saved JEPA configurations to rerun.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated JEPA pretraining seeds to run.",
    )
    parser.add_argument(
        "--probe-seeds",
        default="0,1,2,3,4",
        help="Comma-separated downstream probe seeds for subject-vote and fusion evaluation.",
    )
    parser.add_argument(
        "--dataset",
        choices=["abide", "abide1", "abide2", "srpbs", "all"],
        default="all",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for the launched subprocesses.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip train/eval steps whose output files already exist.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str]):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    args = parse_args()
    pretrain_seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for profile_name in args.profile:
        profile = PROFILES[profile_name]
        for seed in pretrain_seeds:
            run_name = f"{profile['save_prefix']}_seed{seed}"
            save_dir = ROOT / "checkpoints" / "jepa_fusion" / run_name
            log_file = save_dir / "train.log"
            checkpoint = save_dir / profile["checkpoint_name"]
            subject_vote_json = (
                ROOT
                / "results"
                / "reports"
                / "tables"
                / "jepa_fusion"
                / f"{run_name}_subject_vote.json"
            )
            subject_vote_md = subject_vote_json.with_suffix(".md")
            fusion_json = (
                ROOT
                / "results"
                / "reports"
                / "tables"
                / "jepa_fusion"
                / f"{run_name}_fusion_subject_vote.json"
            )
            fusion_md = fusion_json.with_suffix(".md")

            if not (args.skip_existing and checkpoint.exists()):
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.jepa_fusion.train_jepa",
                        *profile["train_args"],
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--save-dir",
                        str(save_dir),
                        "--log-file",
                        str(log_file),
                    ],
                    env=env,
                )
            else:
                print(f"Skipping train for {run_name}: {checkpoint} exists", flush=True)

            if not (args.skip_existing and subject_vote_json.exists()):
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.jepa_fusion.evaluate_jepa_subject_vote",
                        "--checkpoint",
                        str(checkpoint),
                        "--dataset",
                        args.dataset,
                        "--device",
                        args.device,
                        "--seeds",
                        args.probe_seeds,
                        "--output-json",
                        str(subject_vote_json),
                        "--output-md",
                        str(subject_vote_md),
                    ],
                    env=env,
                )
            else:
                print(f"Skipping subject-vote eval for {run_name}: {subject_vote_json} exists", flush=True)

            if not (args.skip_existing and fusion_json.exists()):
                run_command(
                    [
                        sys.executable,
                        "-m",
                        "src.jepa_fusion.evaluate_fusion_subject_vote",
                        "--checkpoint",
                        str(checkpoint),
                        "--dataset",
                        args.dataset,
                        "--device",
                        args.device,
                        "--seeds",
                        args.probe_seeds,
                        "--output-json",
                        str(fusion_json),
                        "--output-md",
                        str(fusion_md),
                    ],
                    env=env,
                )
            else:
                print(f"Skipping fusion eval for {run_name}: {fusion_json} exists", flush=True)


if __name__ == "__main__":
    main()
