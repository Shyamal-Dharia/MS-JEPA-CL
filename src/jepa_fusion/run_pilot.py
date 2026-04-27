from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


PILOT_ARCHITECTURES = [
    {
        "base_name": "pilot_t_basic",
        "model_args": [
            "--model-family",
            "temporal",
            "--temporal-variant",
            "basic",
        ],
    },
    {
        "base_name": "pilot_t_conv_net",
        "model_args": [
            "--model-family",
            "temporal",
            "--temporal-variant",
            "conv_net",
        ],
    },
    {
        "base_name": "pilot_t_rotary_conv_net",
        "model_args": [
            "--model-family",
            "temporal",
            "--temporal-variant",
            "rotary_conv_net",
        ],
    },
    {
        "base_name": "pilot_st_roi_patch",
        "model_args": [
            "--model-family",
            "st_roi_patch",
        ],
    },
]


def checkpoint_name(training_mode: str):
    return "pooled_final.pt" if training_mode == "pooled" else "phase2_final.pt"


def selection_stem(training_mode: str):
    return "pilot_selection" if training_mode == "pooled" else "pilot_selection_curriculum"


def build_candidates(training_mode: str):
    if training_mode not in {"pooled", "curriculum"}:
        raise ValueError(f"Unsupported training mode: {training_mode}")
    suffix = "" if training_mode == "pooled" else "_curr"
    candidates = []
    for arch in PILOT_ARCHITECTURES:
        candidates.append(
            {
                "name": f"{arch['base_name']}{suffix}",
                "train_args": [*arch["model_args"], "--training-mode", training_mode],
            }
        )
    return candidates


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(part.strip()) for part in seed_text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the JEPA pilot automatically: train the requested architecture family, "
            "evaluate it with grouped CV, and rank the candidates."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints/jepa_fusion",
        help="Base directory for pilot checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        default="results/reports/tables/jepa_fusion",
        help="Base directory for evaluation and selection outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed through to training and evaluation.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["pooled", "curriculum"],
        default="pooled",
        help="Whether the pilot sweep should use pooled SSL or the two-phase curriculum recipe.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES value for all subprocesses, e.g. `0` or `0,1`.",
    )
    parser.add_argument(
        "--train-seed",
        type=int,
        default=42,
        help="Training seed for all pilot candidates when --train-seeds is not provided.",
    )
    parser.add_argument(
        "--train-seeds",
        default=None,
        help=(
            "Comma-separated JEPA pretraining seeds. Use this for final paper selection, "
            "for example `0,1,2,3,4`."
        ),
    )
    parser.add_argument(
        "--eval-seeds",
        default="0,1,2,3,4",
        help="Comma-separated evaluation seeds passed to evaluate_jepa.py. Pilot default is five seeds.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Grouped-CV folds for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional training batch size override for all pilot candidates.",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional max scans override for evaluation.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the existing quick mode for smoke testing only.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training/evaluation steps whose outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without executing them.",
    )
    return parser.parse_args()


def run_command(cmd: list[str], *, env: dict[str, str], dry_run: bool):
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    checkpoints_dir = Path(args.checkpoints_dir)
    results_dir = Path(args.results_dir)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    candidates = build_candidates(args.training_mode)
    checkpoint_stem = Path(checkpoint_name(args.training_mode)).stem
    train_seeds = parse_seed_list(args.train_seeds) if args.train_seeds is not None else [args.train_seed]
    use_seed_suffix = args.train_seeds is not None or len(train_seeds) > 1

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    eval_jsons: list[str] = []
    for candidate in candidates:
        base_name = candidate["name"]
        for train_seed in train_seeds:
            run_name = f"{base_name}_seed{train_seed}" if use_seed_suffix else base_name
            save_dir = checkpoints_dir / run_name
            save_dir.mkdir(parents=True, exist_ok=True)
            train_log = save_dir / "train.log"
            checkpoint_path = save_dir / checkpoint_name(args.training_mode)

            train_cmd = [
                sys.executable,
                "-m",
                "src.jepa_fusion.train_jepa",
                *candidate["train_args"],
                "--device",
                args.device,
                "--save-dir",
                str(save_dir),
                "--log-file",
                str(train_log),
                "--seed",
                str(train_seed),
            ]
            if args.batch_size is not None:
                train_cmd.extend(["--batch-size", str(args.batch_size)])
            if args.quick:
                train_cmd.append("--quick")

            if args.skip_existing and checkpoint_path.exists():
                print(f"Skipping training for {run_name}: {checkpoint_path} already exists.", flush=True)
            else:
                run_command(train_cmd, env=env, dry_run=args.dry_run)

            eval_json = results_dir / f"{run_name}_{checkpoint_stem}_jepa_eval.json"
            eval_md = results_dir / f"{run_name}_{checkpoint_stem}_jepa_eval.md"
            eval_jsons.append(str(eval_json))
            eval_cmd = [
                sys.executable,
                "-m",
                "src.jepa_fusion.evaluate_jepa",
                "--checkpoint",
                str(checkpoint_path),
                "--device",
                args.device,
                "--seeds",
                args.eval_seeds,
                "--folds",
                str(args.folds),
                "--output-json",
                str(eval_json),
                "--output-md",
                str(eval_md),
            ]
            if args.max_scans is not None:
                eval_cmd.extend(["--max-scans", str(args.max_scans)])

            if args.skip_existing and eval_json.exists() and eval_md.exists():
                print(f"Skipping evaluation for {run_name}: {eval_json} already exists.", flush=True)
            else:
                run_command(eval_cmd, env=env, dry_run=args.dry_run)

            loso_json = results_dir / f"{run_name}_{checkpoint_stem}_subject_vote.json"
            loso_md = results_dir / f"{run_name}_{checkpoint_stem}_subject_vote.md"
            loso_cmd = [
                sys.executable,
                "-m",
                "src.jepa_fusion.evaluate_jepa_subject_vote",
                "--checkpoint",
                str(checkpoint_path),
                "--device",
                args.device,
                "--seeds",
                args.eval_seeds,
                "--output-json",
                str(loso_json),
                "--output-md",
                str(loso_md),
            ]
            if args.max_scans is not None:
                loso_cmd.extend(["--max-scans", str(args.max_scans)])

            if args.skip_existing and loso_json.exists() and loso_md.exists():
                print(f"Skipping LOSO evaluation for {run_name}: {loso_json} already exists.", flush=True)
            else:
                run_command(loso_cmd, env=env, dry_run=args.dry_run)

    selection_json = results_dir / f"{selection_stem(args.training_mode)}.json"
    selection_md = results_dir / f"{selection_stem(args.training_mode)}.md"
    select_cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.select_pilot",
        "--inputs",
        *eval_jsons,
        "--output-json",
        str(selection_json),
        "--output-md",
        str(selection_md),
    ]
    if args.skip_existing and selection_json.exists() and selection_md.exists():
        print(f"Skipping pilot selection: {selection_json} already exists.", flush=True)
    else:
        run_command(select_cmd, env=env, dry_run=args.dry_run)

    if not args.dry_run:
        print(f"Pilot selection markdown: {selection_md}", flush=True)


if __name__ == "__main__":
    main()
