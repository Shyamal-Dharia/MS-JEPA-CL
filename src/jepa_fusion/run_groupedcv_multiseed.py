from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "reports" / "tables" / "jepa_fusion" / "five_seed_groupedcv"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "jepa_fusion"


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    checkpoint_name: str
    train_args: tuple[str, ...]
    known_checkpoints: dict[int, Path]


PROFILES: dict[str, Profile] = {
    "basic_pooled": Profile(
        key="basic_pooled",
        label="JEPA (temporal basic; pooled SSL)",
        checkpoint_name="pooled_final.pt",
        train_args=(
            "--model-family",
            "temporal",
            "--temporal-variant",
            "basic",
            "--training-mode",
            "pooled",
            "--no-conv-stem",
            "--no-network-embeddings",
        ),
        known_checkpoints={
            42: CHECKPOINTS_DIR / "pilot_t_basic" / "pooled_final.pt",
            0: CHECKPOINTS_DIR / "confirm_basic_pooled_seed0" / "pooled_final.pt",
            1: CHECKPOINTS_DIR / "confirm_basic_pooled_seed1" / "pooled_final.pt",
            2: CHECKPOINTS_DIR / "confirm_basic_pooled_seed2" / "pooled_final.pt",
        },
    ),
    "basic_curr": Profile(
        key="basic_curr",
        label="JEPA (temporal basic; curriculum)",
        checkpoint_name="phase2_final.pt",
        train_args=(
            "--model-family",
            "temporal",
            "--temporal-variant",
            "basic",
            "--training-mode",
            "curriculum",
            "--no-conv-stem",
            "--no-network-embeddings",
        ),
        known_checkpoints={
            42: CHECKPOINTS_DIR / "pilot_t_basic_curr" / "phase2_final.pt",
        },
    ),
    "rotary_curr": Profile(
        key="rotary_curr",
        label="JEPA (temporal rotary conv-net; curriculum)",
        checkpoint_name="phase2_final.pt",
        train_args=(
            "--model-family",
            "temporal",
            "--temporal-variant",
            "rotary_conv_net",
            "--training-mode",
            "curriculum",
            "--use-conv-stem",
            "--use-network-embeddings",
        ),
        known_checkpoints={
            42: CHECKPOINTS_DIR / "pilot_t_rotary_conv_net_curr" / "phase2_final.pt",
        },
    ),
    "rotary_curr_align": Profile(
        key="rotary_curr_align",
        label="JEPA (temporal rotary conv-net; curriculum + replay alignment)",
        checkpoint_name="phase2_final.pt",
        train_args=(
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
        known_checkpoints={
            42: CHECKPOINTS_DIR / "rotary_curr_align" / "phase2_final.pt",
            0: CHECKPOINTS_DIR / "confirm_rotary_curr_align_seed0" / "phase2_final.pt",
            1: CHECKPOINTS_DIR / "confirm_rotary_curr_align_seed1" / "phase2_final.pt",
            2: CHECKPOINTS_DIR / "confirm_rotary_curr_align_seed2" / "phase2_final.pt",
        },
    ),
    "rotary_curr_align_adv": Profile(
        key="rotary_curr_align_adv",
        label="JEPA (temporal rotary conv-net; curriculum + replay alignment + domain adversary)",
        checkpoint_name="phase2_final.pt",
        train_args=(
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
        known_checkpoints={
            42: CHECKPOINTS_DIR / "rotary_curr_align_adv" / "phase2_final.pt",
        },
    ),
    # ── Phase 3: 2×4 loss ablation (pooled × curriculum × {adv, nce, both}) ──
    "rotary_pooled_adv": Profile(
        key="rotary_pooled_adv",
        label="JEPA (rotary conv-net; pooled + adversarial)",
        checkpoint_name="pooled_final.pt",
        train_args=(
            "--model-family", "temporal",
            "--temporal-variant", "rotary_conv_net",
            "--training-mode", "pooled",
            "--use-conv-stem", "--use-network-embeddings",
            "--adversarial-site-weight", "0.1",
        ),
        known_checkpoints={},
    ),
    "rotary_pooled_nce": Profile(
        key="rotary_pooled_nce",
        label="JEPA (rotary conv-net; pooled + InfoNCE)",
        checkpoint_name="pooled_final.pt",
        train_args=(
            "--model-family", "temporal",
            "--temporal-variant", "rotary_conv_net",
            "--training-mode", "pooled",
            "--use-conv-stem", "--use-network-embeddings",
            "--infonce-weight", "0.1",
        ),
        known_checkpoints={},
    ),
    "rotary_pooled_both": Profile(
        key="rotary_pooled_both",
        label="JEPA (rotary conv-net; pooled + adversarial + InfoNCE)",
        checkpoint_name="pooled_final.pt",
        train_args=(
            "--model-family", "temporal",
            "--temporal-variant", "rotary_conv_net",
            "--training-mode", "pooled",
            "--use-conv-stem", "--use-network-embeddings",
            "--adversarial-site-weight", "0.1",
            "--infonce-weight", "0.1",
        ),
        known_checkpoints={},
    ),
    "rotary_curr_adv": Profile(
        key="rotary_curr_adv",
        label="JEPA (rotary conv-net; curriculum + adversarial)",
        checkpoint_name="phase2_final.pt",
        train_args=(
            "--model-family", "temporal",
            "--temporal-variant", "rotary_conv_net",
            "--training-mode", "curriculum",
            "--use-conv-stem", "--use-network-embeddings",
            "--adversarial-site-weight", "0.1",
        ),
        known_checkpoints={},
    ),
    "rotary_curr_nce": Profile(
        key="rotary_curr_nce",
        label="JEPA (rotary conv-net; curriculum + InfoNCE)",
        checkpoint_name="phase2_final.pt",
        train_args=(
            "--model-family", "temporal",
            "--temporal-variant", "rotary_conv_net",
            "--training-mode", "curriculum",
            "--use-conv-stem", "--use-network-embeddings",
            "--infonce-weight", "0.1",
        ),
        known_checkpoints={},
    ),
    "rotary_curr_both": Profile(
        key="rotary_curr_both",
        label="JEPA (rotary conv-net; curriculum + adversarial + InfoNCE)",
        checkpoint_name="phase2_final.pt",
        train_args=(
            "--model-family", "temporal",
            "--temporal-variant", "rotary_conv_net",
            "--training-mode", "curriculum",
            "--use-conv-stem", "--use-network-embeddings",
            "--adversarial-site-weight", "0.1",
            "--infonce-weight", "0.1",
        ),
        known_checkpoints={},
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run or aggregate the 5-seed grouped-CV JEPA experiments needed to replace "
            "the single-seed rows in the grouped-CV paper table."
        )
    )
    parser.add_argument(
        "--profile",
        nargs="+",
        choices=sorted(PROFILES),
        default=["basic_pooled"],
        help="Which JEPA grouped-CV profile(s) to process.",
    )
    parser.add_argument(
        "--pretrain-seeds",
        default="42,0,1,2,3",
        help="Comma-separated JEPA pretraining seeds. Seed 42 reuses the original pilot/rescue checkpoints.",
    )
    parser.add_argument(
        "--eval-seeds",
        default="42",
        help="Comma-separated evaluate_jepa probe seeds. Keep 42 to match the original grouped-CV artifacts.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device passed through to train_jepa / evaluate_jepa.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES for all subprocesses.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip train/eval work if the expected outputs already exist.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Only aggregate existing per-seed grouped-CV eval JSON files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def parse_seed_list(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def mean_std(values: list[float]) -> dict[str, float]:
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {"mean": float(mean(values)), "std": float(stdev(values))}


def run_command(cmd: list[str], env: dict[str, str], *, dry_run: bool):
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def managed_checkpoint_dir(profile_key: str, seed: int) -> Path:
    return CHECKPOINTS_DIR / f"five_seed_{profile_key}_seed{seed}"


def eval_output_paths(profile_key: str, seed: int) -> tuple[Path, Path]:
    base = RESULTS_DIR / f"{profile_key}_seed{seed}_groupedcv"
    return base.with_suffix(".json"), base.with_suffix(".md")


def checkpoint_for_seed(profile: Profile, seed: int) -> tuple[Path, bool]:
    if seed in profile.known_checkpoints:
        return profile.known_checkpoints[seed], False
    save_dir = managed_checkpoint_dir(profile.key, seed)
    return save_dir / profile.checkpoint_name, True


def train_missing_checkpoint(profile: Profile, seed: int, env: dict[str, str], *, device: str, dry_run: bool):
    save_dir = managed_checkpoint_dir(profile.key, seed)
    checkpoint = save_dir / profile.checkpoint_name
    if checkpoint.exists():
        return checkpoint
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / "train.log"
    cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.train_jepa",
        *profile.train_args,
        "--device",
        device,
        "--seed",
        str(seed),
        "--save-dir",
        str(save_dir),
        "--log-file",
        str(log_file),
    ]
    run_command(cmd, env, dry_run=dry_run)
    return checkpoint


def evaluate_checkpoint(
    profile_key: str,
    checkpoint: Path,
    eval_seeds: str,
    env: dict[str, str],
    *,
    device: str,
    dry_run: bool,
):
    output_json, output_md = eval_output_paths(profile_key, extract_seed_from_path(profile_key, checkpoint))
    cmd = [
        sys.executable,
        "-m",
        "src.jepa_fusion.evaluate_jepa",
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        "all",
        "--seeds",
        eval_seeds,
        "--device",
        device,
        "--output-json",
        str(output_json),
        "--output-md",
        str(output_md),
    ]
    run_command(cmd, env, dry_run=dry_run)


def extract_seed_from_path(profile_key: str, checkpoint: Path) -> int:
    path_text = checkpoint.as_posix()
    if "pilot_t_basic/pooled_final.pt" in path_text:
        return 42
    if "pilot_t_basic_curr/phase2_final.pt" in path_text:
        return 42
    if "pilot_t_rotary_conv_net_curr/phase2_final.pt" in path_text:
        return 42
    if "rotary_curr_align/phase2_final.pt" in path_text and "confirm_" not in path_text:
        return 42
    if "rotary_curr_align_adv/phase2_final.pt" in path_text:
        return 42
    if "confirm_basic_pooled_seed" in path_text or "confirm_rotary_curr_align_seed" in path_text:
        suffix = checkpoint.parent.name.split("seed")[-1]
        return int(suffix)
    managed_prefix = f"five_seed_{profile_key}_seed"
    if managed_prefix in checkpoint.as_posix():
        suffix = checkpoint.parent.name.split("seed")[-1]
        return int(suffix)
    raise ValueError(f"Could not infer seed from checkpoint path: {checkpoint}")


def aggregate_profile(profile: Profile, seeds: list[int]) -> dict:
    per_seed_payloads = []
    for seed in seeds:
        output_json, _ = eval_output_paths(profile.key, seed)
        if not output_json.exists():
            raise FileNotFoundError(f"Missing grouped-CV eval JSON for {profile.key} seed {seed}: {output_json}")
        with output_json.open() as handle:
            per_seed_payloads.append(json.load(handle))

    dataset_order = per_seed_payloads[0]["order"]
    summary_payload = {
        "profile": profile.key,
        "label": profile.label,
        "pretrain_seeds": list(seeds),
        "eval_seeds": per_seed_payloads[0]["datasets"][dataset_order[0]]["config"]["seeds"],
        "order": list(dataset_order),
        "datasets": {},
    }
    for dataset_key in dataset_order:
        site_vals = [payload["datasets"][dataset_key]["summary"]["site"]["balanced_accuracy"]["mean"] for payload in per_seed_payloads]
        dx_bacc_vals = [
            payload["datasets"][dataset_key]["summary"]["diagnosis"]["balanced_accuracy"]["mean"]
            for payload in per_seed_payloads
        ]
        dx_auc_vals = [
            payload["datasets"][dataset_key]["summary"]["diagnosis"]["roc_auc"]["mean"]
            for payload in per_seed_payloads
        ]
        meta = per_seed_payloads[0]["datasets"][dataset_key]["meta"]
        config = per_seed_payloads[0]["datasets"][dataset_key]["config"]
        summary_payload["datasets"][dataset_key] = {
            "config": config,
            "meta": meta,
            "per_seed": {
                "site_bacc": site_vals,
                "dx_bacc": dx_bacc_vals,
                "dx_auc": dx_auc_vals,
            },
            "summary": {
                "site": {"balanced_accuracy": mean_std(site_vals)},
                "diagnosis": {
                    "balanced_accuracy": mean_std(dx_bacc_vals),
                    "roc_auc": mean_std(dx_auc_vals),
                },
            },
        }
    return summary_payload


def build_markdown(payload: dict) -> str:
    lines = [
        f"# {payload['label']} Five-Seed Grouped-CV Summary",
        "",
        f"Pretraining seeds: `{','.join(str(seed) for seed in payload['pretrain_seeds'])}`",
        f"Grouped-CV probe seeds per checkpoint: `{','.join(str(seed) for seed in payload['eval_seeds'])}`",
        "",
        "| Dataset | Site bAcc | Dx bAcc | Dx ROC-AUC |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        dataset = payload["datasets"][dataset_key]
        lines.append(
            f"| {dataset['config']['dataset_label']} | "
            f"{100.0 * dataset['summary']['site']['balanced_accuracy']['mean']:.1f} ± {100.0 * dataset['summary']['site']['balanced_accuracy']['std']:.1f} | "
            f"{100.0 * dataset['summary']['diagnosis']['balanced_accuracy']['mean']:.1f} ± {100.0 * dataset['summary']['diagnosis']['balanced_accuracy']['std']:.1f} | "
            f"{100.0 * dataset['summary']['diagnosis']['roc_auc']['mean']:.1f} ± {100.0 * dataset['summary']['diagnosis']['roc_auc']['std']:.1f} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_aggregate(profile: Profile, payload: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_DIR / f"{profile.key}_five_seed_summary.json"
    output_md = RESULTS_DIR / f"{profile.key}_five_seed_summary.md"
    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown(payload))
    print(f"Saved summary JSON: {output_json}")
    print(f"Saved summary Markdown: {output_md}")


def main():
    args = parse_args()
    pretrain_seeds = parse_seed_list(args.pretrain_seeds)
    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for profile_key in args.profile:
        profile = PROFILES[profile_key]
        print(f"== Profile: {profile.key} ==", flush=True)
        for seed in pretrain_seeds:
            checkpoint, managed = checkpoint_for_seed(profile, seed)
            output_json, output_md = eval_output_paths(profile.key, seed)
            if not args.aggregate_only and managed:
                if not (args.skip_existing and checkpoint.exists()):
                    checkpoint = train_missing_checkpoint(
                        profile,
                        seed,
                        env,
                        device=args.device,
                        dry_run=args.dry_run,
                    )
                else:
                    print(f"Skipping train for {profile.key} seed {seed}: {checkpoint} exists", flush=True)
            elif not checkpoint.exists():
                raise FileNotFoundError(f"Expected checkpoint for {profile.key} seed {seed} not found: {checkpoint}")

            if args.aggregate_only:
                continue
            if args.skip_existing and output_json.exists() and output_md.exists():
                print(f"Skipping eval for {profile.key} seed {seed}: {output_json} exists", flush=True)
                continue
            evaluate_checkpoint(
                profile.key,
                checkpoint,
                args.eval_seeds,
                env,
                device=args.device,
                dry_run=args.dry_run,
            )

        if args.dry_run:
            continue
        aggregate_payload = aggregate_profile(profile, pretrain_seeds)
        write_aggregate(profile, aggregate_payload)


if __name__ == "__main__":
    main()
