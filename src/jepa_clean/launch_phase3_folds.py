import argparse
import json
import logging
import math
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .evaluate_subject_vote import (
    TASKS,
    add_site_std_to_summary,
    compute_binary_metrics,
    compute_site_level_metrics,
)
from .train_phase3_subject_pooling import build_subject_entries, filter_subject_entries


@dataclass
class FoldLaunchConfig:
    checkpoint_path: str
    task: str
    gpus: list[str]
    data_dirs: list[str] | None
    save_root: str
    log_root: str

    window_seconds: float | None = None
    window_stride_seconds: float | None = None
    target_timepoints: int | None = None
    ft_epochs: int | None = None
    ft_batch_size: int | None = None
    ft_encoder_lr: float | None = None
    ft_head_lr: float | None = None
    contrastive_weight: float | None = None
    contrastive_temperature: float | None = None
    pooling: str | None = None
    attention_hidden: int | None = None
    calibration_mode: str | None = None
    train_windows_per_subject: int | None = None
    head_hidden: int | None = None
    head_dropout: float | None = None
    val_fraction: float | None = None
    early_stop_patience: int | None = None
    max_subjects_per_class: int | None = None
    seed: int | None = None
    deterministic: bool = True
    quick: bool = False


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one Phase 3 task with LOSO folds split across multiple GPUs, then merge results."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--data-dirs", nargs="+", default=None)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--log-root", required=True)

    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=None)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--ft-epochs", type=int, default=None)
    parser.add_argument("--ft-batch-size", type=int, default=None)
    parser.add_argument("--ft-encoder-lr", type=float, default=None)
    parser.add_argument("--ft-head-lr", type=float, default=None)
    parser.add_argument("--contrastive-weight", type=float, default=None)
    parser.add_argument("--contrastive-temperature", type=float, default=None)
    parser.add_argument("--pooling", choices=["mean", "attention"], default=None)
    parser.add_argument("--attention-hidden", type=int, default=None)
    parser.add_argument("--calibration-mode", choices=["none", "temperature"], default=None)
    parser.add_argument("--train-windows-per-subject", type=int, default=None)
    parser.add_argument("--head-hidden", type=int, default=None)
    parser.add_argument("--head-dropout", type=float, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=None)
    parser.add_argument("--max-subjects-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def chunk_sites(sites: list[str], n_chunks: int):
    n_chunks = max(1, n_chunks)
    if not sites:
        return [[] for _ in range(n_chunks)]
    chunk_size = int(math.ceil(len(sites) / n_chunks))
    return [sites[i:i + chunk_size] for i in range(0, len(sites), chunk_size)]


def build_child_command(cfg: FoldLaunchConfig, sites: list[str], save_dir: Path, log_file: Path):
    cmd = [
        sys.executable,
        "-m",
        "jepa_clean.train_phase3_subject_pooling",
        "--checkpoint",
        cfg.checkpoint_path,
        "--device",
        "cuda",
        "--task",
        cfg.task,
        "--held-out-sites",
        *sites,
        "--save-dir",
        str(save_dir),
        "--log-file",
        str(log_file),
    ]

    if cfg.data_dirs is not None:
        cmd += ["--data-dirs", *cfg.data_dirs]
    if cfg.window_seconds is not None:
        cmd += ["--window-seconds", str(cfg.window_seconds)]
    if cfg.window_stride_seconds is not None:
        cmd += ["--window-stride-seconds", str(cfg.window_stride_seconds)]
    if cfg.target_timepoints is not None:
        cmd += ["--target-timepoints", str(cfg.target_timepoints)]
    if cfg.ft_epochs is not None:
        cmd += ["--ft-epochs", str(cfg.ft_epochs)]
    if cfg.ft_batch_size is not None:
        cmd += ["--ft-batch-size", str(cfg.ft_batch_size)]
    if cfg.ft_encoder_lr is not None:
        cmd += ["--ft-encoder-lr", str(cfg.ft_encoder_lr)]
    if cfg.ft_head_lr is not None:
        cmd += ["--ft-head-lr", str(cfg.ft_head_lr)]
    if cfg.contrastive_weight is not None:
        cmd += ["--contrastive-weight", str(cfg.contrastive_weight)]
    if cfg.contrastive_temperature is not None:
        cmd += ["--contrastive-temperature", str(cfg.contrastive_temperature)]
    if cfg.pooling is not None:
        cmd += ["--pooling", cfg.pooling]
    if cfg.attention_hidden is not None:
        cmd += ["--attention-hidden", str(cfg.attention_hidden)]
    if cfg.calibration_mode is not None:
        cmd += ["--calibration-mode", cfg.calibration_mode]
    if cfg.train_windows_per_subject is not None:
        cmd += ["--train-windows-per-subject", str(cfg.train_windows_per_subject)]
    if cfg.head_hidden is not None:
        cmd += ["--head-hidden", str(cfg.head_hidden)]
    if cfg.head_dropout is not None:
        cmd += ["--head-dropout", str(cfg.head_dropout)]
    if cfg.val_fraction is not None:
        cmd += ["--val-fraction", str(cfg.val_fraction)]
    if cfg.early_stop_patience is not None:
        cmd += ["--early-stop-patience", str(cfg.early_stop_patience)]
    if cfg.max_subjects_per_class is not None:
        cmd += ["--max-subjects-per-class", str(cfg.max_subjects_per_class)]
    if cfg.seed is not None:
        cmd += ["--seed", str(cfg.seed)]
    if cfg.deterministic:
        cmd += ["--deterministic"]
    if not cfg.deterministic:
        cmd += ["--non-deterministic"]
    if cfg.quick:
        cmd += ["--quick"]

    return cmd


def merge_chunk_results(task: str, chunk_dirs: list[Path], save_root: Path):
    all_subjects = []
    base_summary = None
    for chunk_dir in chunk_dirs:
        path = chunk_dir / "subject_pooling_results.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing chunk result file: {path}")
        data = json.loads(path.read_text())
        if not isinstance(data, list) or len(data) != 1:
            raise ValueError(f"Unexpected result payload in {path}")
        row = data[0]
        if base_summary is None:
            base_summary = row["summary"]
        all_subjects.extend(row["subjects"])

    if not all_subjects:
        merged = [
            {
                "summary": {
                    "task": TASKS[task][0],
                    "task_key": task,
                    "bacc": float("nan"),
                    "f1_macro": float("nan"),
                    "precision": float("nan"),
                    "recall": float("nan"),
                    "auc": float("nan"),
                    "n_valid_sites": 0,
                    "held_out_sites": [],
                    "n_subjects": 0,
                    "mean_windows_per_subject": float("nan"),
                },
                "subjects": [],
            }
        ]
        add_site_std_to_summary(merged[0]["summary"], [])
    else:
        y_true = np.array([row["y"] for row in all_subjects], dtype=int)
        y_pred = np.array([row["pred"] for row in all_subjects], dtype=int)
        y_prob = np.array([row["prob"] for row in all_subjects], dtype=float)
        summary = compute_binary_metrics(y_true, y_pred, y_prob)
        summary["task"] = TASKS[task][0]
        summary["task_key"] = task
        summary["n_valid_sites"] = len(sorted({row["site"] for row in all_subjects}))
        summary["held_out_sites"] = sorted({row["site"] for row in all_subjects})
        summary["n_subjects"] = len(all_subjects)
        summary["mean_windows_per_subject"] = float(
            np.mean([row["n_windows"] for row in all_subjects])
        )
        if base_summary is not None:
            for key in [
                "pooling_mode",
                "calibration_mode",
                "train_windows_per_subject",
                "contrastive_weight",
                "contrastive_temperature",
            ]:
                if key in base_summary:
                    summary[key] = base_summary[key]
        site_metrics = compute_site_level_metrics(all_subjects)
        add_site_std_to_summary(summary, site_metrics)
        merged = [{"summary": summary, "subjects": all_subjects}]

    out_path = save_root / "subject_pooling_results.json"
    out_path.write_text(json.dumps(merged, indent=2))
    return out_path


def main():
    args = parse_args()
    cfg = FoldLaunchConfig(
        checkpoint_path=args.checkpoint,
        task=args.task,
        gpus=args.gpus,
        data_dirs=args.data_dirs,
        save_root=args.save_root,
        log_root=args.log_root,
        window_seconds=args.window_seconds,
        window_stride_seconds=args.window_stride_seconds,
        target_timepoints=args.target_timepoints,
        ft_epochs=args.ft_epochs,
        ft_batch_size=args.ft_batch_size,
        ft_encoder_lr=args.ft_encoder_lr,
        ft_head_lr=args.ft_head_lr,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        pooling=args.pooling,
        attention_hidden=args.attention_hidden,
        calibration_mode=args.calibration_mode,
        train_windows_per_subject=args.train_windows_per_subject,
        head_hidden=args.head_hidden,
        head_dropout=args.head_dropout,
        val_fraction=args.val_fraction,
        early_stop_patience=args.early_stop_patience,
        max_subjects_per_class=args.max_subjects_per_class,
        seed=args.seed,
        deterministic=not args.non_deterministic,
        quick=args.quick,
    )

    setup_logging()
    logger = logging.getLogger("JEPA-Clean-Phase3-Folds")

    save_root = Path(cfg.save_root)
    log_root = Path(cfg.log_root)
    save_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    (save_root / "launch_manifest.json").write_text(json.dumps(asdict(cfg), indent=2))

    subject_entries = build_subject_entries(
        data_dirs=cfg.data_dirs
        if cfg.data_dirs is not None
        else [
            "preprocessed_data_fir/full/roi_timeseries_1600_full",
            "preprocessed_data_fir/full/roi_timeseries_abide_full",
        ],
        window_seconds=cfg.window_seconds if cfg.window_seconds is not None else 90.0,
        stride_seconds=cfg.window_stride_seconds if cfg.window_stride_seconds is not None else 30.0,
    )
    _, task_entries = filter_subject_entries(subject_entries, cfg.task)
    unique_sites = sorted({entry["site"] for entry in task_entries})
    site_chunks = [chunk for chunk in chunk_sites(unique_sites, len(cfg.gpus)) if chunk]
    logger.info("Task %s split across %d chunks: %s", cfg.task, len(site_chunks), site_chunks)

    base_env = os.environ.copy()
    py_path = base_env.get("PYTHONPATH", "")
    if py_path:
        parts = py_path.split(os.pathsep)
        if "src" not in parts:
            base_env["PYTHONPATH"] = os.pathsep.join(["src", py_path])
    else:
        base_env["PYTHONPATH"] = "src"

    procs = []
    chunk_dirs = []
    try:
        for idx, sites in enumerate(site_chunks):
            gpu_id = cfg.gpus[idx % len(cfg.gpus)]
            chunk_dir = save_root / f"chunk_{idx}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_root / f"chunk_{idx}.log"
            cmd = build_child_command(cfg, sites, chunk_dir, log_file)
            env = dict(base_env)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info("Starting chunk=%d on gpu=%s sites=%s", idx, gpu_id, sites)
            logger.info("Command: %s", " ".join(cmd))
            proc = subprocess.Popen(cmd, env=env)
            procs.append((idx, proc, log_file))
            chunk_dirs.append(chunk_dir)

        failures = []
        for idx, proc, log_file in procs:
            code = proc.wait()
            if code != 0:
                failures.append((idx, code, str(log_file)))

        if failures:
            lines = [
                f"chunk={idx} exit_code={code} log={log_file}"
                for idx, code, log_file in failures
            ]
            raise SystemExit("One or more fold chunks failed:\n" + "\n".join(lines))

    finally:
        for _, proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()

    merged_path = merge_chunk_results(cfg.task, chunk_dirs, save_root)
    logger.info("Merged result saved to %s", merged_path)


if __name__ == "__main__":
    main()
