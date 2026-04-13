"""Subject-level sliding-window majority-vote LOSO evaluation for JEPA checkpoints.

Runs per-dataset LOSO evaluation and produces per-dataset summary tables.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..jepa_clean.data import WindowedFMRIDataset
from ..jepa_clean.evaluate_subject_vote import (
    build_model_from_checkpoint,
    extract_window_embeddings,
    filter_task,
    run_subject_vote_loso,
    TASKS,
)
from ..jepa_clean.repro import seed_everything
from .protocol import DATASETS


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )


DATASET_TASKS = {
    "abide1": "hc_asd",
    "abide2": "hc_asd",
    "srpbs": "hc_all",
}


def run_subject_vote(
    checkpoint_path: str,
    device=None,
    probe_seeds=(0, 1, 2, 3, 4),
    classifier="logreg",
):
    logger = logging.getLogger("JEPA-SubjectVote")

    model, ckpt_cfg = build_model_from_checkpoint(checkpoint_path)
    if device is None:
        device = ckpt_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    window_seconds = ckpt_cfg.get("window_seconds", 90.0)
    target_timepoints = ckpt_cfg.get("target_timepoints", 45)

    all_results = {}
    for ds_name, spec in DATASETS.items():
        task_key = DATASET_TASKS[ds_name]
        logger.info("Evaluating %s on %s", task_key, spec.display_name)

        dataset = WindowedFMRIDataset(
            spec.raw_dir,
            window_seconds=window_seconds,
            stride_seconds=30.0,
            target_timepoints=target_timepoints,
            sampling="sliding",
            exclude_unknown_diagnoses=True,
        )

        embeddings, metadata = extract_window_embeddings(
            model, dataset, device=device, batch_size=128, window_feature="mean",
        )
        logger.info("Extracted %d window embeddings for %s", len(embeddings), spec.display_name)

        task_name, task_emb, task_meta, y = filter_task(embeddings, metadata, task_key)

        seed_results = []
        for seed in probe_seeds:
            metrics, _ = run_subject_vote_loso(
                task_emb, task_meta, y,
                seed=seed, classifier=classifier,
            )
            seed_results.append(metrics)

        # Aggregate across probe seeds
        baccs = [r["bacc"] for r in seed_results]
        aucs = [r["auc"] for r in seed_results]
        site_baccs = [r.get("site_bacc_mean", float("nan")) for r in seed_results]
        mean_windows = [r.get("mean_windows_per_subject", float("nan")) for r in seed_results]

        all_results[ds_name] = {
            "display_name": spec.display_name,
            "task": task_name,
            "dx_bacc_mean": float(np.nanmean(baccs)),
            "dx_bacc_std": float(np.nanstd(baccs)),
            "dx_auc_mean": float(np.nanmean(aucs)),
            "dx_auc_std": float(np.nanstd(aucs)),
            "site_bacc_mean": float(np.nanmean(site_baccs)),
            "site_bacc_std": float(np.nanstd(site_baccs)),
            "n_valid_sites": seed_results[0].get("n_valid_sites", 0),
            "mean_windows_per_subject": float(np.nanmean(mean_windows)),
            "seed_results": seed_results,
        }

        logger.info(
            "%s | Dx bAcc=%.1f ± %.1f | Dx AUC=%.3f ± %.3f | Valid sites=%d",
            spec.display_name,
            100 * np.nanmean(baccs),
            100 * np.nanstd(baccs),
            np.nanmean(aucs),
            np.nanstd(aucs),
            seed_results[0].get("n_valid_sites", 0),
        )

    return all_results


def build_markdown(results: dict, checkpoint_path: str) -> str:
    ckpt_name = str(Path(checkpoint_path).relative_to(Path.cwd())) if Path(checkpoint_path).is_absolute() else checkpoint_path
    lines = [
        f"# {Path(checkpoint_path).parent.name}/{Path(checkpoint_path).name} Subject-Vote Evaluation",
        "",
        f"Checkpoint: `{ckpt_name}`",
        "",
        "Evaluation uses sliding windows, subject-level majority vote, and leave-one-site-out diagnosis.",
        "",
        "| Dataset | Dx LOSO bAcc | Dx LOSO ROC-AUC | Held-out Site Mean bAcc | Valid Sites | Mean Windows/Subject |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for ds_name in ["abide1", "abide2", "srpbs"]:
        if ds_name not in results:
            continue
        r = results[ds_name]
        lines.append(
            f"| {r['display_name']} "
            f"| {100 * r['dx_bacc_mean']:.1f} ± {100 * r['dx_bacc_std']:.1f} "
            f"| {100 * r['dx_auc_mean']:.1f} ± {100 * r['dx_auc_std']:.1f} "
            f"| {100 * r['site_bacc_mean']:.1f} ± {100 * r['site_bacc_std']:.1f} "
            f"| {r['n_valid_sites']} "
            f"| {r['mean_windows_per_subject']:.2f} ± 0.00 |"
        )

    return "\n".join(lines) + "\n"


def parse_args():
    p = argparse.ArgumentParser(description="Subject-vote LOSO JEPA evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--classifier", choices=["logreg", "linear_svm", "xgboost"], default="logreg")
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-md", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging()
    seed_everything(42, deterministic=True)

    results = run_subject_vote(
        args.checkpoint,
        device=args.device,
        probe_seeds=tuple(args.seeds),
        classifier=args.classifier,
    )

    logger = logging.getLogger("JEPA-SubjectVote")

    # Auto-name based on checkpoint
    ckpt_path = Path(args.checkpoint)
    ckpt_tag = f"{ckpt_path.parent.name}_{ckpt_path.stem}"
    out_dir = Path("results/reports/tables/jepa_fusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = args.save_json or str(out_dir / f"{ckpt_tag}_subject_vote.json")
    md_path = args.save_md or str(out_dir / f"{ckpt_tag}_subject_vote.md")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", json_path)

    md_content = build_markdown(results, args.checkpoint)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info("Saved Markdown: %s", md_path)


if __name__ == "__main__":
    main()
