import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.covariance import LedoitWolf, OAS
from torch.utils.data import DataLoader

from .data import WindowedFMRIDataset
from .evaluate_fc_vs_raw_loso import run_loso_probe
from .evaluate_subject_vote import TASKS, build_model_from_checkpoint, setup_logging
from .plot_tsne_fc_vs_raw import (
    apply_combat_harmonization,
    apply_covbat_harmonization,
    run_grouped_probe,
    vectorize_fc,
)
from .repro import seed_everything
from .summarize_fc_vs_raw_probe_multiseed import parse_seed_list, summarize_probe_runs


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def build_random_projection_matrix(input_dim: int, output_dim: int, seed: int):
    if output_dim <= 0 or output_dim >= input_dim:
        return None
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal(size=(input_dim, output_dim), dtype=np.float32)
    q, _ = np.linalg.qr(basis)
    return np.asarray(q[:, :output_dim], dtype=np.float32)


def covariance_to_correlation(covariance: np.ndarray):
    covariance = np.asarray(covariance, dtype=np.float32)
    std = np.sqrt(np.clip(np.diag(covariance), a_min=1e-12, a_max=None))
    denom = np.outer(std, std)
    corr = np.divide(covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32, copy=False)


def estimate_fc_matrix(tokens: np.ndarray, estimator: str):
    if estimator == "corr":
        fc = np.corrcoef(tokens, rowvar=False)
        fc = np.asarray(fc, dtype=np.float32)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        fc = 0.5 * (fc + fc.T)
        np.fill_diagonal(fc, 1.0)
        return fc

    if estimator == "cov":
        fc = np.cov(tokens, rowvar=False)
        fc = np.asarray(fc, dtype=np.float32)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        return 0.5 * (fc + fc.T)

    if estimator in {"ledoitwolf_corr", "ledoitwolf_cov"}:
        covariance = LedoitWolf().fit(tokens).covariance_.astype(np.float32, copy=False)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.T)
        return covariance_to_correlation(covariance) if estimator.endswith("_corr") else covariance

    if estimator in {"oas_corr", "oas_cov"}:
        covariance = OAS().fit(tokens).covariance_.astype(np.float32, copy=False)
        covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
        covariance = 0.5 * (covariance + covariance.T)
        return covariance_to_correlation(covariance) if estimator.endswith("_corr") else covariance

    raise ValueError(f"Unsupported fc estimator: {estimator}")


@torch.no_grad()
def extract_subject_latent_fc_features(
    model,
    dataset,
    device,
    batch_size: int = 64,
    triangle: str = "lower",
    fc_estimator: str = "corr",
    projection_dim: int = 0,
    projection_seed: int = 0,
):
    model.eval()
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    by_scan_tokens = defaultdict(list)
    by_scan_meta = {}
    scan_order = []

    for x, labels in loader:
        x = x.to(device)
        sequence = raw_model.encode_sequence(x).detach().cpu().numpy()
        for sample_tokens, label in zip(sequence, labels):
            scan_id = str(label["scan_id"])
            if scan_id not in by_scan_meta:
                by_scan_meta[scan_id] = dict(label)
                scan_order.append(scan_id)
            by_scan_tokens[scan_id].append((int(label.get("window_index", 0)), sample_tokens.astype(np.float32)))

    features = []
    metadata = []
    latent_dim = None
    effective_dim = None
    projection_matrix = None

    for scan_id in scan_order:
        ordered_windows = [
            tokens
            for _, tokens in sorted(by_scan_tokens[scan_id], key=lambda item: item[0])
        ]
        subject_tokens = np.concatenate(ordered_windows, axis=0)
        latent_dim = int(subject_tokens.shape[1])
        if projection_matrix is None:
            projection_matrix = build_random_projection_matrix(
                input_dim=latent_dim,
                output_dim=int(projection_dim),
                seed=int(projection_seed),
            )
        if projection_matrix is not None:
            subject_tokens = subject_tokens @ projection_matrix
        effective_dim = int(subject_tokens.shape[1])

        fc = estimate_fc_matrix(subject_tokens, estimator=fc_estimator)

        feature_row = vectorize_fc(fc, triangle=triangle)
        features.append(feature_row)

        row_meta = dict(by_scan_meta[scan_id])
        row_meta["n_windows"] = len(ordered_windows)
        row_meta["n_latent_timepoints"] = int(subject_tokens.shape[0])
        metadata.append(row_meta)

    if not features:
        raise RuntimeError("No latent FC features were extracted.")

    return np.stack(features, axis=0), metadata, latent_dim, effective_dim


def filter_task(features, metadata, task_key: str):
    task_name, hc_diag, disorder_diag = TASKS[task_key]

    keep_indices = []
    binary_labels = []
    class_labels = []
    for idx, meta in enumerate(metadata):
        diag = int(meta["diag"])
        diag_name = str(meta.get("diag_name", diag))
        if diag < 0:
            continue
        if disorder_diag == "ALL":
            keep_indices.append(idx)
            binary_labels.append(int(diag != hc_diag))
            class_labels.append("HC" if diag == hc_diag else "Disorder")
        elif diag in {hc_diag, disorder_diag}:
            keep_indices.append(idx)
            binary_labels.append(int(diag == disorder_diag))
            class_labels.append("HC" if diag == hc_diag else diag_name)

    if not keep_indices:
        raise ValueError(f"No scans available for task {task_name}")

    keep_indices = np.asarray(keep_indices, dtype=int)
    kept_meta = [metadata[idx] for idx in keep_indices]
    return (
        task_name,
        features[keep_indices],
        kept_meta,
        np.asarray(binary_labels, dtype=int),
        class_labels,
    )


def format_metric(metric: dict):
    return f"{100.0 * metric['mean']:.1f}±{100.0 * metric['std']:.1f}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate subject-level JEPA latent FC features with grouped site probes and LOSO diagnosis probes."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dirs", nargs="+", required=True)
    parser.add_argument("--task", nargs="+", default=["hc_all"], choices=sorted(TASKS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--triangle", choices=["upper", "lower"], default="lower")
    parser.add_argument(
        "--fc-estimator",
        choices=["corr", "cov", "ledoitwolf_corr", "ledoitwolf_cov", "oas_corr", "oas_cov"],
        default="corr",
    )
    parser.add_argument(
        "--projection-dim",
        type=int,
        default=0,
        help="If >0 and < latent_dim, project latent channels to this dimension before FC estimation.",
    )
    parser.add_argument("--projection-seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--include-combat", action="store_true")
    parser.add_argument("--combat-preserve-diagnosis", action="store_true")
    parser.add_argument("--combat-mean-only", action="store_true")
    parser.add_argument("--combat-verbose", action="store_true")
    parser.add_argument("--include-covbat", action="store_true")
    parser.add_argument("--covbat-n-pc", type=int, default=50)
    parser.add_argument("--covbat-pct-var", type=float, default=0.95)
    parser.add_argument("--covbat-verbose", action="store_true")
    parser.add_argument("--keep-unknown-diagnoses", action="store_true")
    parser.add_argument("--save-json", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("JEPA-Clean-LatentFC")

    seeds = parse_seed_list(args.seeds)
    seed_everything(seeds[0], deterministic=True)

    model, ckpt_cfg = build_model_from_checkpoint(args.checkpoint, override_target_timepoints=args.target_timepoints)
    device = args.device or ckpt_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    window_seconds = args.window_seconds or ckpt_cfg.get("window_seconds", 90.0)
    target_timepoints = args.target_timepoints or ckpt_cfg.get("target_timepoints", 45)
    dataset = WindowedFMRIDataset(
        args.data_dirs,
        window_seconds=window_seconds,
        stride_seconds=args.window_stride_seconds,
        target_timepoints=target_timepoints,
        sampling="sliding",
        exclude_unknown_diagnoses=not args.keep_unknown_diagnoses,
    )
    logger.info(
        "Sliding dataset: %d windows from %d scans | window=%.1fs stride=%.1fs target_tp=%d",
        len(dataset),
        len(dataset.records),
        window_seconds,
        args.window_stride_seconds,
        target_timepoints,
    )

    latent_fc_features, metadata, latent_dim, effective_dim = extract_subject_latent_fc_features(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        triangle=args.triangle,
        fc_estimator=args.fc_estimator,
        projection_dim=args.projection_dim,
        projection_seed=args.projection_seed,
    )
    logger.info(
        "Latent FC features: %s | latent_dim=%d effective_dim=%d estimator=%s triangle=%s",
        latent_fc_features.shape,
        latent_dim,
        effective_dim,
        args.fc_estimator,
        args.triangle,
    )

    sites = [meta["site"] for meta in metadata]
    subjects = [meta["subject_id"] for meta in metadata]
    diagnosis_names = [str(meta.get("diag_name", meta["diag"])) for meta in metadata]

    feature_sets = {"latent_fc": latent_fc_features}
    if args.include_combat:
        combat_covariates = {"diagnosis": diagnosis_names} if args.combat_preserve_diagnosis else None
        logger.info("Running ComBat on latent FC features")
        feature_sets["latent_fc_combat"] = apply_combat_harmonization(
            latent_fc_features,
            sites,
            categorical_covariates=combat_covariates,
            mean_only=args.combat_mean_only,
            verbose=args.combat_verbose,
        )
    if args.include_covbat:
        logger.info("Running CovBat on latent FC features")
        feature_sets["latent_fc_covbat"] = apply_covbat_harmonization(
            latent_fc_features,
            sites,
            n_pc=args.covbat_n_pc,
            pct_var=args.covbat_pct_var,
            verbose=args.covbat_verbose,
        )

    site_runs = {name: [] for name in feature_sets}
    diagnosis_runs = {task_key: {name: [] for name in feature_sets} for task_key in args.task}

    for seed in seeds:
        for method_name, features in feature_sets.items():
            site_runs[method_name].append(
                run_grouped_probe(
                    features,
                    sites,
                    groups=subjects,
                    seed=seed,
                    folds=args.folds,
                    task_name="site",
                )
            )

            for task_key in args.task:
                task_name, task_features, task_meta, _, task_labels = filter_task(features, metadata, task_key)
                task_sites = [meta["site"] for meta in task_meta]
                diagnosis_runs[task_key][method_name].append(
                    run_loso_probe(task_features, task_labels, task_sites, seed=seed)
                )

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "data_dirs": args.data_dirs,
            "task": args.task,
            "window_seconds": float(window_seconds),
            "window_stride_seconds": float(args.window_stride_seconds),
            "target_timepoints": int(target_timepoints),
            "triangle": args.triangle,
            "fc_estimator": args.fc_estimator,
            "projection_dim": int(args.projection_dim),
            "projection_seed": int(args.projection_seed),
            "folds": int(args.folds),
            "seeds": seeds,
            "combat": {
                "include": bool(args.include_combat),
                "preserve_diagnosis": bool(args.combat_preserve_diagnosis),
                "mean_only": bool(args.combat_mean_only),
            },
            "covbat": {
                "include": bool(args.include_covbat),
                "n_pc": int(args.covbat_n_pc),
                "pct_var": float(args.covbat_pct_var),
            },
        },
        "meta": {
            "n_scans": int(len(metadata)),
            "n_subjects": int(len(set(subjects))),
            "n_sites": int(len(set(sites))),
            "latent_dim": int(latent_dim),
            "effective_dim": int(effective_dim),
            "feature_dim": int(latent_fc_features.shape[1]),
            "mean_windows_per_scan": float(np.mean([meta["n_windows"] for meta in metadata])),
            "mean_latent_timepoints_per_scan": float(np.mean([meta["n_latent_timepoints"] for meta in metadata])),
        },
        "per_seed": {
            "site": site_runs,
            "diagnosis": diagnosis_runs,
        },
        "summary": {
            "site": {
                method_name: summarize_probe_runs(runs)
                for method_name, runs in site_runs.items()
            },
            "diagnosis": {
                task_key: {
                    method_name: summarize_probe_runs(
                        runs,
                        metric_sources={
                            "accuracy": "mean_fold_accuracy",
                            "balanced_accuracy": "mean_fold_balanced_accuracy",
                            "pooled_accuracy": "pooled_accuracy",
                            "pooled_balanced_accuracy": "pooled_balanced_accuracy",
                        },
                        passthrough_keys=[
                            "n_classes",
                            "classes",
                            "n_samples_total",
                            "n_samples_valid",
                            "n_valid_sites",
                            "n_skipped_sites",
                            "valid_sites",
                            "cv",
                        ],
                    )
                    for method_name, runs in task_runs.items()
                }
                for task_key, task_runs in diagnosis_runs.items()
            },
        },
    }

    for method_name, summary in payload["summary"]["site"].items():
        logger.info(
            "Site probe | %s | bAcc=%s",
            method_name,
            format_metric(summary["balanced_accuracy"]),
        )
    for task_key in args.task:
        task_name = TASKS[task_key][0]
        for method_name, summary in payload["summary"]["diagnosis"][task_key].items():
            logger.info(
                "Diagnosis LOSO | %s | %s | bAcc=%s pooled_bAcc=%s",
                task_name,
                method_name,
                format_metric(summary["balanced_accuracy"]),
                format_metric(summary["pooled_balanced_accuracy"]),
            )

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(payload, indent=2))
        logger.info("Saved results: %s", save_path)


if __name__ == "__main__":
    main()
