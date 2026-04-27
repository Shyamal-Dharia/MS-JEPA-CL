from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

try:
    from .model import build_model_from_config
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        center_site_label,
        class_counts,
        dataset_order_for_choice,
        filter_records_for_task,
        get_dataset_preset,
    )
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.model import build_model_from_config
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        center_site_label,
        class_counts,
        dataset_order_for_choice,
        filter_records_for_task,
        get_dataset_preset,
    )
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.pooling import pool_sequence_features
    from src.jepa_clean.repro import seed_everything
else:
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.pooling import pool_sequence_features
    from ..jepa_clean.repro import seed_everything


@dataclass(frozen=True)
class EvalWindowRecord:
    index: int
    subject: str
    site: str
    diagnosis: str


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate jepa_fusion checkpoints with sliding-window subject-vote LOSO."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--task", choices=["default", "hc_asd", "hc_disorder"], default="default")
    parser.add_argument("--all-sites", action="store_true")
    parser.add_argument("--exclude-sites", default=None, help="Comma-separated center site labels to exclude from eval")
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--classifier", choices=["logreg"], default="logreg")
    parser.add_argument("--threshold-mode", choices=["fixed", "calibrated"], default="fixed")
    parser.add_argument("--threshold-val-fraction", type=float, default=0.2)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--healthy-control-label", default=HEALTHY_CONTROL_LABEL)
    parser.add_argument("--autism-label", default=AUTISM_LABEL)
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


def format_pct(metric: dict):
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def load_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model_from_config(cfg)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    allowed_missing = {
        "global_predictor.0.weight",
        "global_predictor.0.bias",
        "global_predictor.1.weight",
        "global_predictor.1.bias",
        "global_predictor.1.running_mean",
        "global_predictor.1.running_var",
        "global_predictor.1.num_batches_tracked",
        "global_predictor.3.weight",
        "global_predictor.3.bias",
    }
    missing = set(incompatible.missing_keys)
    if missing - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected checkpoint incompatibility: "
            f"missing={sorted(missing)} unexpected={sorted(incompatible.unexpected_keys)}"
        )
    return model, checkpoint, cfg


@torch.no_grad()
def extract_window_embeddings(
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
    loader = DataLoader(
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


def attach_labels(metadata, y):
    labeled = []
    for meta, label in zip(metadata, y):
        copied = dict(meta)
        copied["y"] = int(label)
        labeled.append(copied)
    return labeled


def get_default_threshold(classifier: str):
    if classifier != "logreg":
        raise ValueError(f"Unsupported classifier: {classifier}")
    return 0.5


def threshold_scores(scores, classifier: str, threshold: float):
    if classifier != "logreg":
        raise ValueError(f"Unsupported classifier: {classifier}")
    scores = np.asarray(scores, dtype=float)
    return (scores >= threshold).astype(int)


def get_threshold_grid(scores, classifier: str):
    if classifier != "logreg":
        raise ValueError(f"Unsupported classifier: {classifier}")
    default_threshold = get_default_threshold(classifier)
    grid = np.linspace(0.01, 0.99, num=99, dtype=float)
    if not np.any(np.isclose(grid, default_threshold)):
        grid = np.unique(np.concatenate([grid, np.array([default_threshold], dtype=float)]))
    return grid


def aggregate_subject_predictions(metadata, scores, classifier: str, threshold: float):
    grouped = {}
    for meta, score in zip(metadata, scores):
        pred = int(threshold_scores(np.array([score]), classifier, threshold)[0])
        subject_id = str(meta["subject_id"])
        entry = grouped.setdefault(
            subject_id,
            {"scores": [], "preds": [], "site": None, "y": None, "n_windows": 0},
        )
        entry["scores"].append(float(score))
        entry["preds"].append(int(pred))
        entry["site"] = str(meta["site"])
        entry["y"] = int(meta["y"])
        entry["n_windows"] += 1

    subject_rows = []
    for subject_id, entry in grouped.items():
        votes = np.asarray(entry["preds"], dtype=int)
        score_mean = float(np.mean(entry["scores"]))
        vote_sum = int(votes.sum())
        if vote_sum > len(votes) / 2:
            subject_pred = 1
        elif vote_sum < len(votes) / 2:
            subject_pred = 0
        else:
            subject_pred = int(score_mean >= threshold)

        subject_rows.append(
            {
                "subject_id": subject_id,
                "site": entry["site"],
                "y": entry["y"],
                "pred": subject_pred,
                "prob": score_mean,
                "n_windows": entry["n_windows"],
            }
        )
    return subject_rows


def split_train_val_by_subject(metadata, y, val_fraction: float, seed: int):
    if val_fraction <= 0.0:
        return np.arange(len(metadata), dtype=int), np.array([], dtype=int)

    by_subject = {}
    subject_label = {}
    for idx, (meta, label) in enumerate(zip(metadata, y)):
        subject_id = str(meta["subject_id"])
        by_subject.setdefault(subject_id, []).append(idx)
        label = int(label)
        if subject_id in subject_label and subject_label[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent labels")
        subject_label[subject_id] = label

    subject_ids = np.asarray(sorted(by_subject), dtype=object)
    subject_y = np.asarray([subject_label[subject_id] for subject_id in subject_ids], dtype=int)
    if len(subject_ids) < 2 or len(np.unique(subject_y)) < 2:
        return np.arange(len(metadata), dtype=int), np.array([], dtype=int)

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_subject_idx, val_subject_idx = next(splitter.split(np.zeros(len(subject_ids)), subject_y))
    except ValueError:
        return np.arange(len(metadata), dtype=int), np.array([], dtype=int)

    train_subjects = set(subject_ids[train_subject_idx].tolist())
    val_subjects = set(subject_ids[val_subject_idx].tolist())
    train_indices = np.asarray(
        [idx for idx, meta in enumerate(metadata) if str(meta["subject_id"]) in train_subjects],
        dtype=int,
    )
    val_indices = np.asarray(
        [idx for idx, meta in enumerate(metadata) if str(meta["subject_id"]) in val_subjects],
        dtype=int,
    )
    return train_indices, val_indices


def find_optimal_subject_threshold(metadata, y, scores, classifier: str):
    if len(metadata) == 0 or len(np.unique(y)) < 2:
        return get_default_threshold(classifier)

    best_threshold = get_default_threshold(classifier)
    best_bacc = -1.0
    for threshold in get_threshold_grid(scores, classifier):
        subject_rows = aggregate_subject_predictions(metadata, scores, classifier=classifier, threshold=threshold)
        y_true = np.asarray([row["y"] for row in subject_rows], dtype=int)
        y_pred = np.asarray([row["pred"] for row in subject_rows], dtype=int)
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            continue
        bacc = float(balanced_accuracy_score(y_true, y_pred))
        if bacc > best_bacc:
            best_bacc = bacc
            best_threshold = float(threshold)
    return best_threshold


def compute_binary_metrics(y_true, y_pred, y_prob):
    metrics = {
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = float("nan")
    return metrics


def compute_site_level_metrics(subject_rows):
    grouped = {}
    for row in subject_rows:
        grouped.setdefault(row["site"], []).append(row)

    site_metrics = []
    for site in sorted(grouped):
        rows = grouped[site]
        y_true = np.asarray([row["y"] for row in rows], dtype=int)
        y_pred = np.asarray([row["pred"] for row in rows], dtype=int)
        y_prob = np.asarray([row["prob"] for row in rows], dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        metrics["site"] = site
        metrics["n_subjects"] = len(rows)
        metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in rows]))
        site_metrics.append(metrics)
    return site_metrics


def add_site_std_to_summary(summary: dict, site_metrics: list[dict]):
    summary["site_metrics"] = site_metrics
    summary["site_bacc_mean"] = float("nan")
    summary["site_bacc_std"] = float("nan")
    summary["site_auc_mean"] = float("nan")
    summary["site_auc_std"] = float("nan")

    if not site_metrics:
        return summary

    for key, out_mean, out_std in [
        ("bacc", "site_bacc_mean", "site_bacc_std"),
        ("auc", "site_auc_mean", "site_auc_std"),
    ]:
        vals = np.asarray([m[key] for m in site_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[out_mean] = float(np.mean(vals))
        summary[out_std] = float(np.std(vals, ddof=0))
    return summary


def build_probe_classifier(seed: int, y_train):
    n_comp = min(50, len(y_train))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp, random_state=seed),
        LogisticRegression(
            C=0.1,
            max_iter=2000,
            random_state=seed,
            class_weight="balanced",
        ),
    )


def get_probe_scores(clf, x_test):
    return clf.predict_proba(x_test)[:, 1]


def run_subject_vote_loso(
    embeddings,
    metadata,
    y,
    *,
    seed: int,
    classifier: str,
    threshold_mode: str,
    val_fraction: float,
):
    sites = np.asarray([str(meta["site"]) for meta in metadata], dtype=object)
    unique_sites = np.unique(sites)
    aggregated = []
    held_out_sites = []
    fold_thresholds = []
    calibrated_sites = []

    for held_out in unique_sites:
        train_mask = sites != held_out
        test_mask = sites == held_out

        train_embeddings = embeddings[train_mask]
        test_embeddings = embeddings[test_mask]
        train_metadata = [metadata[idx] for idx in np.flatnonzero(train_mask)]
        test_metadata_raw = [metadata[idx] for idx in np.flatnonzero(test_mask)]
        y_train = y[train_mask]
        y_test = y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        threshold = get_default_threshold(classifier)
        if threshold_mode == "calibrated":
            train_inner_idx, val_idx = split_train_val_by_subject(
                train_metadata,
                y_train,
                val_fraction=val_fraction,
                seed=seed,
            )
            if (
                len(val_idx) > 0
                and len(np.unique(y_train[train_inner_idx])) >= 2
                and len(np.unique(y_train[val_idx])) >= 2
            ):
                clf = build_probe_classifier(seed, y_train[train_inner_idx])
                clf.fit(train_embeddings[train_inner_idx], y_train[train_inner_idx])
                val_scores = get_probe_scores(clf, train_embeddings[val_idx])
                val_metadata = attach_labels([train_metadata[idx] for idx in val_idx], y_train[val_idx])
                threshold = find_optimal_subject_threshold(
                    val_metadata,
                    y_train[val_idx],
                    val_scores,
                    classifier=classifier,
                )
                clf = build_probe_classifier(seed, y_train)
                clf.fit(train_embeddings, y_train)
                calibrated_sites.append(str(held_out))
            else:
                clf = build_probe_classifier(seed, y_train)
                clf.fit(train_embeddings, y_train)
        else:
            clf = build_probe_classifier(seed, y_train)
            clf.fit(train_embeddings, y_train)

        test_scores = get_probe_scores(clf, test_embeddings)
        test_metadata = attach_labels(test_metadata_raw, y_test)
        subject_rows = aggregate_subject_predictions(
            test_metadata,
            test_scores,
            classifier=classifier,
            threshold=threshold,
        )
        aggregated.extend(subject_rows)
        held_out_sites.append(str(held_out))
        fold_thresholds.append(float(threshold))

    if not aggregated:
        return {
            "bacc": float("nan"),
            "auc": float("nan"),
            "f1_macro": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "n_valid_sites": 0,
            "n_subjects": 0,
            "mean_windows_per_subject": float("nan"),
            "threshold_mode": threshold_mode,
            "val_fraction": float(val_fraction),
            "mean_threshold": float("nan"),
            "n_calibrated_sites": 0,
            "site_bacc_mean": float("nan"),
            "site_bacc_std": float("nan"),
            "site_auc_mean": float("nan"),
            "site_auc_std": float("nan"),
            "site_metrics": [],
        }, []

    y_true = np.asarray([row["y"] for row in aggregated], dtype=int)
    y_pred = np.asarray([row["pred"] for row in aggregated], dtype=int)
    y_prob = np.asarray([row["prob"] for row in aggregated], dtype=float)
    metrics = compute_binary_metrics(y_true, y_pred, y_prob)
    metrics["n_valid_sites"] = len(set(held_out_sites))
    metrics["n_subjects"] = len(aggregated)
    metrics["mean_windows_per_subject"] = float(np.mean([row["n_windows"] for row in aggregated]))
    metrics["threshold_mode"] = threshold_mode
    metrics["val_fraction"] = float(val_fraction)
    metrics["mean_threshold"] = float(np.mean(fold_thresholds)) if fold_thresholds else float("nan")
    metrics["n_calibrated_sites"] = len(calibrated_sites)
    site_metrics = compute_site_level_metrics(aggregated)
    add_site_std_to_summary(metrics, site_metrics)
    return metrics, aggregated


def summarize_runs(runs: list[dict]):
    return {
        "bacc": mean_std([run["bacc"] for run in runs]),
        "auc": mean_std([run["auc"] for run in runs]),
        "site_bacc_mean": mean_std([run["site_bacc_mean"] for run in runs]),
        "site_auc_mean": mean_std([run["site_auc_mean"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_valid_sites": int(runs[0]["n_valid_sites"]),
        "n_subjects": int(runs[0]["n_subjects"]),
        "mean_windows_per_subject": mean_std([run["mean_windows_per_subject"] for run in runs]),
    }


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}_subject_vote.json", base_dir / f"{stem}_subject_vote.md"


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} Subject-Vote Evaluation",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "Evaluation uses sliding windows, subject-level majority vote, and leave-one-center-out diagnosis.",
        "",
        "| Dataset | Dx LOSO bAcc | Dx LOSO ROC-AUC | Held-out Site Mean bAcc | Valid Sites | Mean Windows/Subject |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["summary"]
        config = payload["datasets"][dataset_key]["config"]
        lines.append(
            f"| {config['dataset_label']} | "
            f"{format_pct(summary['bacc'])} | "
            f"{format_pct(summary['auc'])} | "
            f"{format_pct(summary['site_bacc_mean'])} | "
            f"{summary['n_valid_sites']} | "
            f"{summary['mean_windows_per_subject']['mean']:.2f} ± {summary['mean_windows_per_subject']['std']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def evaluate_dataset(
    *,
    model,
    checkpoint_cfg: dict,
    dataset_key: str,
    task_arg: str,
    all_sites: bool,
    batch_size: int,
    seeds: list[int],
    max_scans: int | None,
    device: str,
    window_feature: str,
    covariance_dim: int,
    covariance_seed: int,
    window_seconds: float | None,
    window_stride_seconds: float,
    target_timepoints: int | None,
    classifier: str,
    threshold_mode: str,
    threshold_val_fraction: float,
    healthy_control_label: str,
    autism_label: str,
    exclude_sites: set[str] | None = None,
):
    preset = get_dataset_preset(dataset_key)
    task_key = preset.default_task if task_arg == "default" else task_arg
    mixed_sites_only = False if all_sites else preset.default_mixed_sites_only

    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(window_stride_seconds),
        target_timepoints=int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45)),
        sampling="sliding",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )
    embeddings, metadata = extract_window_embeddings(
        model,
        dataset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )

    eval_records = [
        EvalWindowRecord(
            index=idx,
            subject=str(meta["subject_id"]),
            site=center_site_label(str(meta.get("site_raw", meta["site"]))),
            diagnosis=str(meta["diag_name"]),
        )
        for idx, meta in enumerate(metadata)
    ]
    task_result = filter_records_for_task(
        eval_records,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=healthy_control_label,
        autism_label=autism_label,
    )
    # Optionally exclude sites (for transfer experiments)
    if exclude_sites:
        keep_mask = [record.site not in exclude_sites for record in task_result.records]
        task_result_records = [r for r, k in zip(task_result.records, keep_mask) if k]
        task_result_y = task_result.y[keep_mask]
        n_excluded = sum(1 for k in keep_mask if not k)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} records from sites: {sorted(exclude_sites)}")
    else:
        task_result_records = list(task_result.records)
        task_result_y = task_result.y

    keep_indices = np.asarray([record.index for record in task_result_records], dtype=np.int64)
    task_embeddings = embeddings[keep_indices]
    task_y = task_result_y
    task_metadata = [
        {
            "subject_id": task_result_records[idx].subject,
            "site": task_result_records[idx].site,
        }
        for idx in range(len(task_result_records))
    ]

    diagnosis_runs = []
    run_rows = []
    for seed in seeds:
        metrics, subject_rows = run_subject_vote_loso(
            task_embeddings,
            task_metadata,
            task_y,
            seed=seed,
            classifier=classifier,
            threshold_mode=threshold_mode,
            val_fraction=threshold_val_fraction,
        )
        diagnosis_runs.append(metrics)
        run_rows.append(
            {
                "seed": int(seed),
                "summary": metrics,
                "subjects": subject_rows,
            }
        )

    return {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "raw_dir": preset.raw_dir,
            "task_key": task_result.task_key,
            "task_label": task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "window_feature": window_feature,
            "covariance_dim": int(covariance_dim),
            "covariance_seed": int(covariance_seed),
            "classifier": classifier,
            "threshold_mode": threshold_mode,
            "threshold_val_fraction": float(threshold_val_fraction),
            "window_stride_seconds": float(window_stride_seconds),
        },
        "meta": {
            "n_windows_total": int(len(metadata)),
            "n_windows_post_task_filter": int(len(task_result.records)),
            "removed_unknown": int(task_result.removed_unknown),
            "removed_task_mismatch": int(task_result.removed_task_mismatch),
            "removed_single_class_sites": int(task_result.removed_single_class_sites),
            "diagnosis_class_counts": class_counts(task_y, task_result.class_names),
        },
        "runs": run_rows,
        "summary": summarize_runs(diagnosis_runs),
    }


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    seed_everything(seeds[0], deterministic=True)
    model, checkpoint, cfg = load_checkpoint(args.checkpoint)
    device = args.device or cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset_order = dataset_order_for_choice(args.dataset)
    payload = {
        "checkpoint": {
            "path": str(Path(args.checkpoint)),
            "label": f"{Path(args.checkpoint).parent.name}/{Path(args.checkpoint).stem}",
            "config": cfg,
        },
        "order": list(dataset_order),
        "datasets": {},
    }

    for dataset_key in dataset_order:
        payload["datasets"][dataset_key] = evaluate_dataset(
            model=model,
            checkpoint_cfg=cfg,
            dataset_key=dataset_key,
            task_arg=args.task,
            all_sites=args.all_sites,
            batch_size=args.batch_size,
            seeds=seeds,
            max_scans=args.max_scans,
            device=device,
            window_feature=args.window_feature,
            covariance_dim=args.covariance_dim,
            covariance_seed=args.covariance_seed,
            window_seconds=args.window_seconds,
            window_stride_seconds=args.window_stride_seconds,
            target_timepoints=args.target_timepoints,
            classifier=args.classifier,
            threshold_mode=args.threshold_mode,
            threshold_val_fraction=args.threshold_val_fraction,
            healthy_control_label=args.healthy_control_label,
            autism_label=args.autism_label,
            exclude_sites=set(args.exclude_sites.split(",")) if args.exclude_sites else None,
        )

    output_json, output_md = default_output_paths(args.checkpoint)
    if args.output_json is not None:
        output_json = Path(args.output_json)
    if args.output_md is not None:
        output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(payload, handle, indent=2)
    output_md.write_text(build_markdown(payload))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
