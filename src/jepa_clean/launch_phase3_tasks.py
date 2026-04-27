import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .evaluate_subject_vote import TASKS


@dataclass
class LaunchConfig:
    checkpoint_path: str = "checkpoints/jepa_clean_curriculum_vicreg_ijepa400_e100/phase2_final.pt"
    tasks: list[str] = field(default_factory=lambda: ["hc_all", "hc_asd", "hc_mdd", "hc_scz"])
    gpus: list[str] = field(default_factory=lambda: ["0"])
    data_dirs: list[str] | None = None

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

    save_root: str = "results/jepa_clean_phase3_parallel"
    log_root: str = "results/logs/jepa_clean_phase3_parallel"
    poll_seconds: float = 10.0


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch Phase 3 task-specific runs in parallel across GPUs and merge results."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--task", nargs="+", default=None, choices=sorted(TASKS))
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--data-dirs", nargs="+", default=None)
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--log-root", default=None)
    parser.add_argument("--poll-seconds", type=float, default=None)

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


def build_task_command(cfg: LaunchConfig, task_key: str, save_dir: Path, log_file: Path):
    cmd = [
        sys.executable,
        "-m",
        "jepa_clean.train_phase3_subject_pooling",
        "--checkpoint",
        cfg.checkpoint_path,
        "--device",
        "cuda",
        "--task",
        task_key,
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


def merge_results(tasks: list[str], save_root: Path):
    merged = []
    for task_key in tasks:
        result_path = save_root / task_key / "subject_pooling_results.json"
        if not result_path.exists():
            raise FileNotFoundError(f"Missing result file: {result_path}")
        rows = json.loads(result_path.read_text())
        if not isinstance(rows, list) or len(rows) != 1:
            raise ValueError(f"Unexpected result payload in {result_path}")
        merged.extend(rows)

    merged_path = save_root / "subject_pooling_results_merged.json"
    merged_path.write_text(json.dumps(merged, indent=2))
    return merged_path


def main():
    args = parse_args()
    cfg = LaunchConfig()
    if args.checkpoint is not None:
        cfg.checkpoint_path = args.checkpoint
    if args.task is not None:
        cfg.tasks = args.task
    if args.gpus is not None:
        cfg.gpus = args.gpus
    if args.data_dirs is not None:
        cfg.data_dirs = args.data_dirs
    if args.save_root is not None:
        cfg.save_root = args.save_root
    if args.log_root is not None:
        cfg.log_root = args.log_root
    if args.poll_seconds is not None:
        cfg.poll_seconds = args.poll_seconds

    if args.window_seconds is not None:
        cfg.window_seconds = args.window_seconds
    if args.window_stride_seconds is not None:
        cfg.window_stride_seconds = args.window_stride_seconds
    if args.target_timepoints is not None:
        cfg.target_timepoints = args.target_timepoints
    if args.ft_epochs is not None:
        cfg.ft_epochs = args.ft_epochs
    if args.ft_batch_size is not None:
        cfg.ft_batch_size = args.ft_batch_size
    if args.ft_encoder_lr is not None:
        cfg.ft_encoder_lr = args.ft_encoder_lr
    if args.ft_head_lr is not None:
        cfg.ft_head_lr = args.ft_head_lr
    if args.contrastive_weight is not None:
        cfg.contrastive_weight = args.contrastive_weight
    if args.contrastive_temperature is not None:
        cfg.contrastive_temperature = args.contrastive_temperature
    if args.pooling is not None:
        cfg.pooling = args.pooling
    if args.attention_hidden is not None:
        cfg.attention_hidden = args.attention_hidden
    if args.calibration_mode is not None:
        cfg.calibration_mode = args.calibration_mode
    if args.train_windows_per_subject is not None:
        cfg.train_windows_per_subject = args.train_windows_per_subject
    if args.head_hidden is not None:
        cfg.head_hidden = args.head_hidden
    if args.head_dropout is not None:
        cfg.head_dropout = args.head_dropout
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.early_stop_patience is not None:
        cfg.early_stop_patience = args.early_stop_patience
    if args.max_subjects_per_class is not None:
        cfg.max_subjects_per_class = args.max_subjects_per_class
    if args.seed is not None:
        cfg.seed = args.seed
    if args.deterministic:
        cfg.deterministic = True
    if args.non_deterministic:
        cfg.deterministic = False
    if args.quick:
        cfg.quick = True

    setup_logging()
    logger = logging.getLogger("JEPA-Clean-Phase3-Launcher")
    if not cfg.gpus:
        raise ValueError("At least one GPU id must be provided via --gpus")

    save_root = Path(cfg.save_root)
    log_root = Path(cfg.log_root)
    save_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    manifest_path = save_root / "launch_manifest.json"
    manifest_path.write_text(json.dumps(asdict(cfg), indent=2))
    logger.info("Launch config saved to %s", manifest_path)

    task_queue = list(cfg.tasks)
    free_gpus = list(cfg.gpus)
    active = []
    completed = []

    base_env = os.environ.copy()
    py_path = base_env.get("PYTHONPATH", "")
    if py_path:
        parts = py_path.split(os.pathsep)
        if "src" not in parts:
            base_env["PYTHONPATH"] = os.pathsep.join(["src", py_path])
    else:
        base_env["PYTHONPATH"] = "src"

    try:
        while task_queue or active:
            while task_queue and free_gpus:
                gpu_id = free_gpus.pop(0)
                task_key = task_queue.pop(0)
                task_save_dir = save_root / task_key
                task_save_dir.mkdir(parents=True, exist_ok=True)
                task_log = log_root / f"{task_key}.log"
                cmd = build_task_command(cfg, task_key, task_save_dir, task_log)

                env = dict(base_env)
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                logger.info("Starting task=%s on gpu=%s", task_key, gpu_id)
                logger.info("Command: %s", " ".join(cmd))
                proc = subprocess.Popen(cmd, env=env)
                active.append(
                    {
                        "task": task_key,
                        "gpu": gpu_id,
                        "proc": proc,
                        "log_file": str(task_log),
                        "save_dir": str(task_save_dir),
                    }
                )

            if not active:
                continue

            time.sleep(cfg.poll_seconds)
            for job in list(active):
                return_code = job["proc"].poll()
                if return_code is None:
                    continue

                active.remove(job)
                free_gpus.append(job["gpu"])
                free_gpus.sort()
                completed.append(
                    {
                        "task": job["task"],
                        "gpu": job["gpu"],
                        "log_file": job["log_file"],
                        "save_dir": job["save_dir"],
                        "return_code": return_code,
                    }
                )

                if return_code != 0:
                    for running in active:
                        running["proc"].terminate()
                    raise SystemExit(
                        f"Task {job['task']} failed with exit code {return_code}. "
                        f"See {job['log_file']}"
                    )

                logger.info("Finished task=%s on gpu=%s", job["task"], job["gpu"])

    finally:
        for job in active:
            if job["proc"].poll() is None:
                job["proc"].terminate()

    merged_path = merge_results(cfg.tasks, save_root)
    summary_path = save_root / "launch_status.json"
    summary_path.write_text(json.dumps(completed, indent=2))
    logger.info("Merged results saved to %s", merged_path)
    logger.info("Task status saved to %s", summary_path)


if __name__ == "__main__":
    main()
