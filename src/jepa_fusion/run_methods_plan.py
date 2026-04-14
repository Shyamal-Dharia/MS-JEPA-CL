from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = ROOT / "results" / "reports" / "tables" / "jepa_fusion"
DEFAULT_CHECKPOINTS_DIR = ROOT / "checkpoints" / "jepa_fusion"
DEFAULT_STAGES = [
    "baseline_tables",
    "pilot_pooled",
    "pilot_curriculum",
    "groupedcv_confirmation",
    "transfer_abide2",
    "transfer_abide1",
]

BASELINE_OUTPUTS = {
    "abide1": "abide1_hc_asd_mixedsites_baseline_table",
    "abide2": "abide2_hc_asd_mixedsites_baseline_table",
    "srpbs": "srpbs_hc_disorder_mixedsites_baseline_table",
}

TRANSFER_PROFILES = {
    "basic_pooled": {
        "train_args": (
            "--model-family",
            "temporal",
            "--temporal-variant",
            "basic",
            "--training-mode",
            "pooled",
            "--no-conv-stem",
            "--no-network-embeddings",
        ),
        "checkpoint_name": "pooled_final.pt",
    },
    "basic_curr": {
        "train_args": (
            "--model-family",
            "temporal",
            "--temporal-variant",
            "basic",
            "--training-mode",
            "curriculum",
            "--no-conv-stem",
            "--no-network-embeddings",
        ),
        "checkpoint_name": "phase2_final.pt",
    },
    "rotary_curr": {
        "train_args": (
            "--model-family",
            "temporal",
            "--temporal-variant",
            "rotary_conv_net",
            "--training-mode",
            "curriculum",
            "--use-conv-stem",
            "--use-network-embeddings",
        ),
        "checkpoint_name": "phase2_final.pt",
    },
    "rotary_curr_align": {
        "train_args": (
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
        ),
        "checkpoint_name": "phase2_final.pt",
    },
    "rotary_curr_align_adv": {
        "train_args": (
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
            "--adversarial-site-weight",
            "0.05",
        ),
        "checkpoint_name": "phase2_final.pt",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the runnable subset of the exact methods plan from plan.docx using the "
            "current jepa_fusion codebase."
        )
    )
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=DEFAULT_STAGES,
        default=list(DEFAULT_STAGES),
        help="One or more plan stages to execute in order.",
    )
    parser.add_argument(
        "--transfer-profile",
        choices=sorted(TRANSFER_PROFILES),
        default="rotary_curr_align",
        help="JEPA configuration used for the cross-dataset transfer stages.",
    )
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--groupedcv-pretrain-seeds", default="42,0,1,2,3")
    parser.add_argument("--groupedcv-eval-seeds", default="42")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--checkpoints-dir", default=str(DEFAULT_CHECKPOINTS_DIR))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], env: dict[str, str], *, dry_run: bool):
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def maybe_add_shared_flags(cmd: list[str], args):
    if args.quick:
        cmd.append("--quick")
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")


def baseline_stage(args, env: dict[str, str]):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    for dataset_key in ("abide1", "abide2", "srpbs"):
        stem = BASELINE_OUTPUTS[dataset_key]
        cmd = [
            sys.executable,
            "-m",
            "src.jepa_fusion.evaluate_baselines",
            "--dataset",
            dataset_key,
            "--seeds",
            args.eval_seeds,
            "--output-json",
            str(results_dir / f"{stem}.json"),
            "--output-md",
            str(results_dir / f"{stem}.md"),
        ]
        run_command(cmd, env, dry_run=args.dry_run)


def pilot_stage(args, env: dict[str, str], *, training_mode: str):
    cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.run_pilot",
        "--training-mode",
        training_mode,
        "--device",
        args.device,
        "--checkpoints-dir",
        args.checkpoints_dir,
        "--results-dir",
        args.results_dir,
        "--train-seed",
        str(args.train_seed),
        "--eval-seeds",
        args.eval_seeds,
    ]
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    maybe_add_shared_flags(cmd, args)
    run_command(cmd, env, dry_run=args.dry_run)


def groupedcv_confirmation_stage(args, env: dict[str, str]):
    cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.run_multiseed_confirmation",
        "--profile",
        "basic_pooled",
        "rotary_curr_align",
        "--seeds",
        args.groupedcv_pretrain_seeds,
        "--probe-seeds",
        args.eval_seeds,
        "--device",
        args.device,
    ]
    if args.cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", args.cuda_visible_devices])
    if args.skip_existing:
        cmd.append("--skip-existing")
    if args.dry_run:
        cmd.append("--dry-run")
    run_command(cmd, env, dry_run=args.dry_run)


def transfer_stage(args, env: dict[str, str], *, train_datasets: str, eval_dataset: str):
    results_dir = Path(args.results_dir)
    checkpoints_dir = Path(args.checkpoints_dir)
    profile = TRANSFER_PROFILES[args.transfer_profile]
    stage_name = f"transfer_{train_datasets.replace(',', '_')}_to_{eval_dataset}_{args.transfer_profile}"
    save_dir = checkpoints_dir / stage_name
    checkpoint = save_dir / profile["checkpoint_name"]
    output_json = results_dir / f"{stage_name}_groupedcv.json"
    output_md = results_dir / f"{stage_name}_groupedcv.md"

    train_cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.train_jepa",
        *profile["train_args"],
        "--main-datasets",
        train_datasets,
        "--device",
        args.device,
        "--seed",
        str(args.train_seed),
        "--save-dir",
        str(save_dir),
        "--log-file",
        str(save_dir / "train.log"),
    ]
    if args.quick:
        train_cmd.append("--quick")

    if args.skip_existing and checkpoint.exists():
        print(f"Skipping transfer train: {checkpoint} exists", flush=True)
    else:
        run_command(train_cmd, env, dry_run=args.dry_run)

    eval_cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.evaluate_jepa",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        eval_dataset,
        "--device",
        args.device,
        "--seeds",
        args.eval_seeds,
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    if args.skip_existing and output_json.exists() and output_md.exists():
        print(f"Skipping transfer eval: {output_json} exists", flush=True)
    else:
        run_command(eval_cmd, env, dry_run=args.dry_run)


def main():
    args = parse_args()
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for stage in args.stage:
        if stage == "baseline_tables":
            baseline_stage(args, env)
        elif stage == "pilot_pooled":
            pilot_stage(args, env, training_mode="pooled")
        elif stage == "pilot_curriculum":
            pilot_stage(args, env, training_mode="curriculum")
        elif stage == "groupedcv_confirmation":
            groupedcv_confirmation_stage(args, env)
        elif stage == "transfer_abide2":
            transfer_stage(args, env, train_datasets="srpbs,abide1", eval_dataset="abide2")
        elif stage == "transfer_abide1":
            transfer_stage(args, env, train_datasets="srpbs,abide2", eval_dataset="abide1")
        else:
            raise ValueError(f"Unsupported stage: {stage}")


if __name__ == "__main__":
    main()
