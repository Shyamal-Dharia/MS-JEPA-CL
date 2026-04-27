from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

try:
    from .evaluate_jepa_fourier_patch import collate_fn
    from .model_fourier_patch import build_model_from_config
    from .protocol import center_site_label, dataset_order_for_choice, get_dataset_preset
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.evaluate_jepa_fourier_patch import collate_fn
    from src.jepa_fusion.model_fourier_patch import build_model_from_config
    from src.jepa_fusion.protocol import center_site_label, dataset_order_for_choice, get_dataset_preset
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.pooling import pool_sequence_features
    from src.jepa_clean.repro import seed_everything
else:
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.pooling import pool_sequence_features
    from ..jepa_clean.repro import seed_everything


@dataclass(frozen=True)
class EvalRecord:
    index: int
    subject: str
    site: str
    age: float


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fourier-patch JEPA checkpoints for age regression.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    return parser.parse_args()


def parse_seed_list(seed_text: str):
    seeds = []
    for item in seed_text.split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    if not seeds:
        raise ValueError("At least one seed is required.")
    return seeds


def mean_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    }


def load_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility: "
            f"missing={sorted(incompatible.missing_keys)} "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return model, checkpoint, cfg


@torch.no_grad()
def extract_embeddings(
    model,
    dataset,
    device,
    *,
    batch_size: int,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
):
    model.eval()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    embeddings = []
    metadata = []
    for x, labels in loader:
        x = x.to(device)
        sequence = model.encode_sequence(x)
        pooled = pool_sequence_features(
            sequence,
            mode=window_feature,
            covariance_dim=covariance_dim,
            covariance_seed=covariance_seed,
        )
        embeddings.append(pooled.cpu().numpy())
        metadata.extend(labels)
    return np.concatenate(embeddings, axis=0), metadata


def filter_labeled_age(records: list[EvalRecord]):
    kept_records = []
    ages = []
    for record in records:
        age = float(record.age)
        if not np.isfinite(age) or age < 0:
            continue
        kept_records.append(record)
        ages.append(age)
    if not kept_records:
        raise ValueError("No labeled records with age information were found.")
    return kept_records, np.asarray(ages, dtype=np.float64)


def _effective_group_folds(groups: np.ndarray, requested_folds: int):
    return min(requested_folds, len(np.unique(groups)))


def _make_regressor(feature_dim: int, train_count: int, seed: int, pca_components: int, ridge_alpha: float):
    n_components = min(int(pca_components), feature_dim, max(1, train_count - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        Ridge(alpha=float(ridge_alpha), random_state=seed),
    )


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def run_grouped_cv(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
    pca_components: int,
    ridge_alpha: float,
):
    effective_folds = _effective_group_folds(groups, folds)
    if effective_folds < 2:
        raise RuntimeError(f"Need at least 2 effective folds, got {effective_folds}.")

    cv = GroupKFold(n_splits=effective_folds)
    predictions = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    sample_indices = np.arange(len(y), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(sample_indices, y, groups=groups), start=1):
        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        reg = _make_regressor(
            feature_dim=x_train.shape[1],
            train_count=len(train_idx),
            seed=seed,
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
        )
        reg.fit(x_train, y_train)
        fold_pred = reg.predict(x_test)
        predictions[test_idx] = fold_pred

        fold_metrics = compute_regression_metrics(y_test, fold_pred)
        fold_metrics.update(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        fold_rows.append(fold_metrics)

    metrics = compute_regression_metrics(y, predictions)
    metrics.update(
        {
            "folds": int(effective_folds),
            "fold_rows": fold_rows,
            "n_samples": int(len(y)),
        }
    )
    return metrics


def summarize_runs(runs: list[dict]):
    return {
        "r2": mean_std([run["r2"] for run in runs]),
        "mae": mean_std([run["mae"] for run in runs]),
        "rmse": mean_std([run["rmse"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "folds": int(runs[0]["folds"]),
    }


def compute_site_level_metrics(subject_rows: list[dict]):
    grouped = {}
    for row in subject_rows:
        grouped.setdefault(row["site"], []).append(row)

    site_metrics = []
    for site in sorted(grouped):
        rows = grouped[site]
        if len(rows) < 2:
            continue
        y_true = np.asarray([row["y"] for row in rows], dtype=np.float64)
        y_pred = np.asarray([row["pred"] for row in rows], dtype=np.float64)
        metrics = compute_regression_metrics(y_true, y_pred)
        metrics["site"] = site
        metrics["n_subjects"] = len(rows)
        metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in rows]))
        site_metrics.append(metrics)
    return site_metrics


def add_site_std_to_summary(summary: dict, site_metrics: list[dict]):
    summary["site_metrics"] = site_metrics
    for key in ("r2", "mae", "rmse"):
        summary[f"site_{key}_mean"] = float("nan")
        summary[f"site_{key}_std"] = float("nan")

    if not site_metrics:
        return summary

    for key in ("r2", "mae", "rmse"):
        vals = np.asarray([m[key] for m in site_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[f"site_{key}_mean"] = float(np.mean(vals))
        summary[f"site_{key}_std"] = float(np.std(vals, ddof=0))
    return summary


def aggregate_subject_predictions(metadata: list[dict], preds: np.ndarray):
    grouped = {}
    for meta, pred in zip(metadata, preds):
        subject_id = str(meta["subject_id"])
        entry = grouped.setdefault(
            subject_id,
            {"preds": [], "site": None, "y": None, "n_windows": 0},
        )
        entry["preds"].append(float(pred))
        entry["site"] = str(meta["site"])
        entry["y"] = float(meta["y"])
        entry["n_windows"] += 1

    subject_rows = []
    for subject_id, entry in grouped.items():
        subject_rows.append(
            {
                "subject_id": subject_id,
                "site": entry["site"],
                "y": entry["y"],
                "pred": float(np.mean(entry["preds"])),
                "n_windows": entry["n_windows"],
            }
        )
    return subject_rows


def run_subject_vote_loso(
    embeddings: np.ndarray,
    metadata: list[dict],
    y: np.ndarray,
    *,
    seed: int,
    pca_components: int,
    ridge_alpha: float,
):
    sites = np.asarray([str(meta["site"]) for meta in metadata], dtype=object)
    unique_sites = np.unique(sites)
    aggregated = []
    valid_sites = []

    for held_out in unique_sites:
        train_mask = sites != held_out
        test_mask = sites == held_out

        x_train = embeddings[train_mask]
        x_test = embeddings[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        test_metadata = [metadata[idx] for idx in np.flatnonzero(test_mask)]

        if len(y_train) < 2 or len(y_test) < 2:
            continue

        reg = _make_regressor(
            feature_dim=x_train.shape[1],
            train_count=len(y_train),
            seed=seed,
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
        )
        reg.fit(x_train, y_train)
        test_preds = reg.predict(x_test)
        subject_rows = aggregate_subject_predictions(test_metadata, test_preds)
        if len(subject_rows) < 2:
            continue
        aggregated.extend(subject_rows)
        valid_sites.append(str(held_out))

    if not aggregated:
        return {
            "r2": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "n_valid_sites": 0,
            "n_subjects": 0,
            "mean_windows_per_subject": float("nan"),
            "site_r2_mean": float("nan"),
            "site_r2_std": float("nan"),
            "site_mae_mean": float("nan"),
            "site_mae_std": float("nan"),
            "site_rmse_mean": float("nan"),
            "site_rmse_std": float("nan"),
            "site_metrics": [],
        }, []

    y_true = np.asarray([row["y"] for row in aggregated], dtype=np.float64)
    y_pred = np.asarray([row["pred"] for row in aggregated], dtype=np.float64)
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics["n_valid_sites"] = len(set(valid_sites))
    metrics["n_subjects"] = len(aggregated)
    metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in aggregated]))
    site_metrics = compute_site_level_metrics(aggregated)
    add_site_std_to_summary(metrics, site_metrics)
    return metrics, aggregated


def summarize_subject_vote_runs(runs: list[dict]):
    return {
        "r2": mean_std([run["r2"] for run in runs]),
        "mae": mean_std([run["mae"] for run in runs]),
        "rmse": mean_std([run["rmse"] for run in runs]),
        "site_r2_mean": mean_std([run["site_r2_mean"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_valid_sites": int(runs[0]["n_valid_sites"]),
        "n_subjects": int(runs[0]["n_subjects"]),
        "mean_windows_per_subject": mean_std([run["mean_windows_per_subject"] for run in runs]),
    }


def format_metric(metric: dict, decimals: int = 3):
    return f"{metric['mean']:.{decimals}f} ± {metric['std']:.{decimals}f}"


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}_fourier_patch_age_eval.json", base_dir / f"{stem}_fourier_patch_age_eval.md"


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} Fourier-Patch JEPA Age Evaluation",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "## Centered Window Grouped-CV",
        "",
        "| Dataset | Age R2 | Age MAE | Age RMSE | N Scans |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["centered"]["summary"]
        config = payload["datasets"][dataset_key]["config"]
        lines.append(
            f"| {config['dataset_label']} | "
            f"{format_metric(summary['r2'])} | "
            f"{format_metric(summary['mae'])} | "
            f"{format_metric(summary['rmse'])} | "
            f"{summary['n_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Subject Vote LOSO",
            "",
            "| Dataset | Age LOSO R2 | Age LOSO MAE | Age LOSO RMSE | Held-out Site Mean R2 | Valid Sites | Mean Windows/Subject |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["subject_vote"]["summary"]
        config = payload["datasets"][dataset_key]["config"]
        lines.append(
            f"| {config['dataset_label']} | "
            f"{format_metric(summary['r2'])} | "
            f"{format_metric(summary['mae'])} | "
            f"{format_metric(summary['rmse'])} | "
            f"{format_metric(summary['site_r2_mean'])} | "
            f"{summary['n_valid_sites']} | "
            f"{summary['mean_windows_per_subject']['mean']:.2f} ± {summary['mean_windows_per_subject']['std']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_dataset(
    *,
    model,
    checkpoint_cfg: dict,
    dataset_key: str,
    batch_size: int,
    folds: int,
    seeds: list[int],
    max_scans: int | None,
    device: str,
    window_feature: str,
    pca_components: int,
    covariance_dim: int,
    covariance_seed: int,
    window_seconds: float | None,
    window_stride_seconds: float,
    target_timepoints: int | None,
    ridge_alpha: float,
):
    preset = get_dataset_preset(dataset_key)
    resolved_window_seconds = float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0))
    resolved_target_timepoints = int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45))

    centered_dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=resolved_window_seconds,
        stride_seconds=float(checkpoint_cfg.get("window_stride_seconds", 30.0)),
        target_timepoints=resolved_target_timepoints,
        sampling="center",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )
    centered_embeddings, centered_metadata = extract_embeddings(
        model,
        centered_dataset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )
    centered_records = [
        EvalRecord(
            index=idx,
            subject=str(meta["subject_id"]),
            site=center_site_label(str(meta.get("site_raw", meta["site"]))),
            age=float(meta.get("age", -1)),
        )
        for idx, meta in enumerate(centered_metadata)
    ]
    centered_records, centered_y = filter_labeled_age(centered_records)
    centered_keep = np.asarray([record.index for record in centered_records], dtype=np.int64)
    centered_features = centered_embeddings[centered_keep]
    centered_groups = np.asarray([record.site for record in centered_records], dtype=object)

    centered_runs = []
    for seed in seeds:
        centered_runs.append(
            run_grouped_cv(
                centered_features,
                centered_y,
                centered_groups,
                seed=seed,
                folds=folds,
                pca_components=pca_components,
                ridge_alpha=ridge_alpha,
            )
        )

    sliding_dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=resolved_window_seconds,
        stride_seconds=window_stride_seconds,
        target_timepoints=resolved_target_timepoints,
        sampling="sliding",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )
    sliding_embeddings, sliding_metadata = extract_embeddings(
        model,
        sliding_dataset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )

    sliding_keep_indices = []
    sliding_y = []
    sliding_eval_metadata = []
    for idx, meta in enumerate(sliding_metadata):
        age = float(meta.get("age", -1))
        if not np.isfinite(age) or age < 0:
            continue
        sliding_keep_indices.append(idx)
        sliding_y.append(age)
        copied = dict(meta)
        copied["site"] = center_site_label(str(meta.get("site_raw", meta["site"])))
        copied["y"] = age
        sliding_eval_metadata.append(copied)
    if not sliding_keep_indices:
        raise ValueError(f"No sliding-window age labels available for dataset {dataset_key}.")

    sliding_features = sliding_embeddings[np.asarray(sliding_keep_indices, dtype=np.int64)]
    sliding_y = np.asarray(sliding_y, dtype=np.float64)

    subject_runs = []
    for seed in seeds:
        metrics, _ = run_subject_vote_loso(
            sliding_features,
            sliding_eval_metadata,
            sliding_y,
            seed=seed,
            pca_components=pca_components,
            ridge_alpha=ridge_alpha,
        )
        subject_runs.append(metrics)

    return {
        "config": {
            "dataset_key": dataset_key,
            "dataset_label": preset.label,
            "window_seconds": resolved_window_seconds,
            "window_stride_seconds": float(window_stride_seconds),
            "target_timepoints": resolved_target_timepoints,
            "window_feature": window_feature,
            "pca_components": int(pca_components),
            "ridge_alpha": float(ridge_alpha),
        },
        "centered": {
            "runs": centered_runs,
            "summary": summarize_runs(centered_runs),
        },
        "subject_vote": {
            "runs": subject_runs,
            "summary": summarize_subject_vote_runs(subject_runs),
        },
    }


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    seed_everything(int(seeds[0]), deterministic=True)

    model, _checkpoint, checkpoint_cfg = load_checkpoint(args.checkpoint)
    device = args.device or checkpoint_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = model.to(device)

    dataset_order = dataset_order_for_choice(args.dataset)
    datasets = {}
    for dataset_key in dataset_order:
        datasets[dataset_key] = evaluate_dataset(
            model=model,
            checkpoint_cfg=checkpoint_cfg,
            dataset_key=dataset_key,
            batch_size=args.batch_size,
            folds=args.folds,
            seeds=seeds,
            max_scans=args.max_scans,
            device=device,
            window_feature=args.window_feature,
            pca_components=args.pca_components,
            covariance_dim=args.covariance_dim,
            covariance_seed=args.covariance_seed,
            window_seconds=args.window_seconds,
            window_stride_seconds=args.window_stride_seconds,
            target_timepoints=args.target_timepoints,
            ridge_alpha=args.ridge_alpha,
        )

    payload = {
        "checkpoint": {
            "path": args.checkpoint,
            "label": f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).stem}",
        },
        "config": {
            "dataset_choice": args.dataset,
            "seeds": seeds,
            "batch_size": int(args.batch_size),
            "folds": int(args.folds),
            "window_feature": args.window_feature,
            "pca_components": int(args.pca_components),
            "ridge_alpha": float(args.ridge_alpha),
        },
        "order": dataset_order,
        "datasets": datasets,
    }

    output_json = Path(args.output_json) if args.output_json else None
    output_md = Path(args.output_md) if args.output_md else None
    if output_json is None or output_md is None:
        default_json, default_md = default_output_paths(args.checkpoint)
        if output_json is None:
            output_json = default_json
        if output_md is None:
            output_md = default_md

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown(payload))

    print(json.dumps(payload["config"], indent=2))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
