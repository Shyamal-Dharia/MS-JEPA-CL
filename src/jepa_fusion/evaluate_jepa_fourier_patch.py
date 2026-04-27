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
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

try:
    from .model_fourier_patch import build_model_from_config
    from .protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
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
    from src.jepa_fusion.model_fourier_patch import build_model_from_config
    from src.jepa_fusion.protocol import (
        AUTISM_LABEL,
        HEALTHY_CONTROL_LABEL,
        batch_site_label,
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
class EvalRecord:
    index: int
    subject: str
    site: str
    diagnosis: str


@dataclass(frozen=True)
class EvalWindowRecord:
    index: int
    subject: str
    site: str
    diagnosis: str


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fourier-patch JEPA checkpoints with grouped CV.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--task", choices=["default", "hc_asd", "hc_disorder"], default="hc_disorder")
    parser.add_argument("--all-sites", action="store_true")
    parser.add_argument("--exclude-sites", default=None, help="Comma-separated center site labels to exclude from eval")
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--healthy-control-label", default=HEALTHY_CONTROL_LABEL)
    parser.add_argument("--autism-label", default=AUTISM_LABEL)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--classifier", choices=["logreg"], default="logreg")
    parser.add_argument("--threshold-mode", choices=["fixed", "calibrated"], default="fixed")
    parser.add_argument("--threshold-val-fraction", type=float, default=0.2)
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


def _make_linear_probe(feature_dim: int, train_count: int, seed: int, pca_components: int):
    n_components = min(int(pca_components), feature_dim, max(1, train_count - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(
            max_iter=10000,
            random_state=seed,
            class_weight="balanced",
        ),
    )


def _effective_group_folds(y: np.ndarray, groups: np.ndarray, requested_folds: int):
    per_class_group_counts = []
    for cls in np.unique(y):
        cls_groups = np.unique(groups[y == cls])
        per_class_group_counts.append(len(cls_groups))
    return min(requested_folds, min(per_class_group_counts))


def _balanced_accuracy_present(y_true: np.ndarray, y_pred: np.ndarray):
    recalls = []
    for cls in np.unique(y_true):
        cls_mask = y_true == cls
        recalls.append(float((y_pred[cls_mask] == cls).mean()))
    return float(np.mean(recalls))


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    if scores.ndim == 1:
        return float(roc_auc_score(y_true, scores))
    return float(
        roc_auc_score(
            y_true,
            scores,
            labels=np.arange(scores.shape[1]),
            multi_class="ovr",
            average="macro",
        )
    )


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


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


@torch.no_grad()
def extract_scan_embeddings(
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
    return extract_scan_embeddings(
        model,
        dataset,
        device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )


def run_grouped_cv(
    features: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    groups: np.ndarray,
    *,
    seed: int,
    folds: int,
    pca_components: int,
):
    if len(np.unique(y)) < 2:
        raise RuntimeError(
            f"Need at least 2 classes for grouped CV, got classes={np.unique(y).tolist()} "
            f"with names={class_names}."
        )
    effective_folds = _effective_group_folds(y, groups, folds)
    if effective_folds < 2:
        raise RuntimeError(f"Need at least 2 effective folds, got {effective_folds}.")

    cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.empty_like(y)
    n_classes = len(class_names)
    oof_scores = np.full((len(y), n_classes), 0.0, dtype=np.float64)
    fold_rows = []

    sample_indices = np.arange(len(y), dtype=np.int64)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(sample_indices, y, groups=groups), start=1):
        x_train = features[train_idx]
        x_test = features[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        clf = _make_linear_probe(
            feature_dim=x_train.shape[1],
            train_count=len(train_idx),
            seed=seed,
            pca_components=pca_components,
        )
        clf.fit(x_train, y_train)
        fold_pred = clf.predict(x_test)
        preds[test_idx] = fold_pred

        clf_classes = clf.named_steps["logisticregression"].classes_
        fold_scores = clf.predict_proba(x_test)
        aligned_scores = np.zeros((len(test_idx), n_classes), dtype=np.float64)
        aligned_scores[:, clf_classes] = fold_scores
        oof_scores[test_idx] = aligned_scores

        fold_row = {
            "fold": int(fold_idx),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "accuracy": float(accuracy_score(y_test, fold_pred)),
            "balanced_accuracy": _balanced_accuracy_present(y_test, fold_pred),
        }
        if n_classes == 2:
            fold_row["roc_auc"] = _safe_roc_auc(y_test, aligned_scores[:, 1])
        else:
            fold_row["roc_auc"] = _safe_roc_auc(y_test, aligned_scores)
        fold_rows.append(fold_row)

    result = {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": _balanced_accuracy_present(y, preds),
        "folds": int(effective_folds),
        "fold_rows": fold_rows,
        "n_samples": int(len(y)),
        "classes": list(class_names),
    }
    if n_classes == 2:
        result["roc_auc"] = _safe_roc_auc(y, oof_scores[:, 1])
    else:
        result["roc_auc"] = _safe_roc_auc(y, oof_scores)
    return result


def summarize_runs(runs: list[dict]):
    return {
        "accuracy": mean_std([run["accuracy"] for run in runs]),
        "balanced_accuracy": mean_std([run["balanced_accuracy"] for run in runs]),
        "roc_auc": mean_std([run["roc_auc"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "folds": int(runs[0]["folds"]),
        "classes": list(runs[0]["classes"]),
    }


def format_pct(metric: dict):
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} Fourier-Patch JEPA Evaluation",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "## Centered Window Grouped-CV",
        "",
        "| Dataset | Site Probe bAcc | Site Probe ROC-AUC | Diagnosis bAcc | Diagnosis ROC-AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["centered"]["summary"]
        config = payload["datasets"][dataset_key]["config"]
        lines.append(
            f"| {config['dataset_label']} | "
            f"{format_pct(summary['site']['balanced_accuracy'])} | "
            f"{format_pct(summary['site']['roc_auc'])} | "
            f"{format_pct(summary['diagnosis']['balanced_accuracy'])} | "
            f"{format_pct(summary['diagnosis']['roc_auc'])} |"
        )
    lines.extend(
        [
            "",
            "## Subject Vote LOSO",
            "",
            "| Dataset | Dx LOSO bAcc | Dx LOSO ROC-AUC | Held-out Site Mean bAcc | Valid Sites | Mean Windows/Subject |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["subject_vote"]["summary"]
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


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}_fourier_patch_eval.json", base_dir / f"{stem}_fourier_patch_eval.md"


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


def build_probe_classifier(seed: int, y_train, pca_components: int):
    n_comp = min(int(pca_components), len(y_train))
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
    pca_components: int,
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
                clf = build_probe_classifier(seed, y_train[train_inner_idx], pca_components=pca_components)
                clf.fit(train_embeddings[train_inner_idx], y_train[train_inner_idx])
                val_scores = get_probe_scores(clf, train_embeddings[val_idx])
                val_metadata = attach_labels([train_metadata[idx] for idx in val_idx], y_train[val_idx])
                threshold = find_optimal_subject_threshold(
                    val_metadata,
                    y_train[val_idx],
                    val_scores,
                    classifier=classifier,
                )
                clf = build_probe_classifier(seed, y_train, pca_components=pca_components)
                clf.fit(train_embeddings, y_train)
                calibrated_sites.append(str(held_out))
            else:
                clf = build_probe_classifier(seed, y_train, pca_components=pca_components)
                clf.fit(train_embeddings, y_train)
        else:
            clf = build_probe_classifier(seed, y_train, pca_components=pca_components)
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


def summarize_subject_vote_runs(runs: list[dict]):
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


def evaluate_dataset(
    *,
    model,
    checkpoint_cfg: dict,
    dataset_key: str,
    task_arg: str,
    all_sites: bool,
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

    centered_dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(checkpoint_cfg.get("window_stride_seconds", 30.0)),
        target_timepoints=int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45)),
        sampling="center",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )
    embeddings, metadata = extract_scan_embeddings(
        model,
        centered_dataset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )

    eval_records = [
        EvalRecord(
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

    if exclude_sites:
        keep_mask = [record.site not in exclude_sites for record in task_result.records]
        task_result_records = [r for r, k in zip(task_result.records, keep_mask) if k]
        task_result_y = task_result.y[keep_mask]
    else:
        task_result_records = list(task_result.records)
        task_result_y = task_result.y

    keep_indices = np.asarray([record.index for record in task_result_records], dtype=np.int64)
    task_embeddings = embeddings[keep_indices]
    task_y = task_result_y
    subjects = np.asarray([task_result_records[idx].subject for idx in range(len(task_result_records))], dtype=object)
    batch_sites = np.asarray(
        [
            batch_site_label(
                str(metadata[idx].get("site_raw", metadata[idx]["site"])),
                source_dataset=str(metadata[idx].get("dataset", "")),
            )
            for idx in keep_indices.tolist()
        ],
        dtype=object,
    )

    site_encoder = LabelEncoder()
    y_site = site_encoder.fit_transform(batch_sites)
    site_class_names = [str(name) for name in site_encoder.classes_.tolist()]

    site_runs = []
    diagnosis_runs = []
    for seed in seeds:
        site_runs.append(
            run_grouped_cv(
                task_embeddings,
                y_site,
                site_class_names,
                subjects,
                seed=seed,
                folds=folds,
                pca_components=pca_components,
            )
        )
        diagnosis_runs.append(
            run_grouped_cv(
                task_embeddings,
                task_y,
                task_result.class_names,
                subjects,
                seed=seed,
                folds=folds,
                pca_components=pca_components,
            )
        )

    centered_payload = {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "raw_dir": preset.raw_dir,
            "task_key": task_result.task_key,
            "task_label": task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "folds": int(folds),
            "seeds": list(seeds),
            "window_feature": window_feature,
            "pca_components": int(pca_components),
            "covariance_dim": int(covariance_dim),
            "covariance_seed": int(covariance_seed),
        },
        "meta": {
            "n_records_total": int(len(centered_dataset.records)),
            "n_records_post_task_filter": int(len(task_result.records)),
            "removed_unknown": int(task_result.removed_unknown),
            "removed_task_mismatch": int(task_result.removed_task_mismatch),
            "removed_single_class_sites": int(task_result.removed_single_class_sites),
            "n_sites": int(len(site_class_names)),
            "site_classes": list(site_class_names),
            "diagnosis_class_counts": class_counts(task_y, task_result.class_names),
            "excluded_sites": sorted(exclude_sites) if exclude_sites else [],
        },
        "runs": {
            "site": site_runs,
            "diagnosis": diagnosis_runs,
        },
        "summary": {
            "site": summarize_runs(site_runs),
            "diagnosis": summarize_runs(diagnosis_runs),
        },
    }

    sliding_dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(window_seconds or checkpoint_cfg.get("window_seconds", 90.0)),
        stride_seconds=float(window_stride_seconds),
        target_timepoints=int(target_timepoints or checkpoint_cfg.get("target_timepoints", 45)),
        sampling="sliding",
        max_scans=max_scans,
        exclude_unknown_diagnoses=False,
    )
    window_embeddings, window_metadata = extract_window_embeddings(
        model,
        sliding_dataset,
        device=device,
        batch_size=batch_size,
        window_feature=window_feature,
        covariance_dim=covariance_dim,
        covariance_seed=covariance_seed,
    )
    window_eval_records = [
        EvalWindowRecord(
            index=idx,
            subject=str(meta["subject_id"]),
            site=center_site_label(str(meta.get("site_raw", meta["site"]))),
            diagnosis=str(meta["diag_name"]),
        )
        for idx, meta in enumerate(window_metadata)
    ]
    window_task_result = filter_records_for_task(
        window_eval_records,
        task_key,
        mixed_sites_only=mixed_sites_only,
        healthy_control_label=healthy_control_label,
        autism_label=autism_label,
    )
    if exclude_sites:
        keep_mask = [record.site not in exclude_sites for record in window_task_result.records]
        window_task_records = [r for r, k in zip(window_task_result.records, keep_mask) if k]
        window_task_y = window_task_result.y[keep_mask]
    else:
        window_task_records = list(window_task_result.records)
        window_task_y = window_task_result.y

    keep_indices = np.asarray([record.index for record in window_task_records], dtype=np.int64)
    task_window_embeddings = window_embeddings[keep_indices]
    task_window_metadata = [
        {
            "subject_id": window_task_records[idx].subject,
            "site": window_task_records[idx].site,
        }
        for idx in range(len(window_task_records))
    ]
    diagnosis_vote_runs = []
    vote_run_rows = []
    for seed in seeds:
        metrics, subject_rows = run_subject_vote_loso(
            task_window_embeddings,
            task_window_metadata,
            window_task_y,
            seed=seed,
            classifier=classifier,
            threshold_mode=threshold_mode,
            val_fraction=threshold_val_fraction,
            pca_components=pca_components,
        )
        diagnosis_vote_runs.append(metrics)
        vote_run_rows.append(
            {
                "seed": int(seed),
                "summary": metrics,
                "subjects": subject_rows,
            }
        )

    subject_vote_payload = {
        "config": {
            "dataset": preset.key,
            "dataset_label": preset.label,
            "raw_dir": preset.raw_dir,
            "task_key": window_task_result.task_key,
            "task_label": window_task_result.task_label,
            "mixed_sites_only": bool(mixed_sites_only),
            "window_feature": window_feature,
            "pca_components": int(pca_components),
            "covariance_dim": int(covariance_dim),
            "covariance_seed": int(covariance_seed),
            "classifier": classifier,
            "threshold_mode": threshold_mode,
            "threshold_val_fraction": float(threshold_val_fraction),
            "window_stride_seconds": float(window_stride_seconds),
        },
        "meta": {
            "n_windows_total": int(len(window_metadata)),
            "n_windows_post_task_filter": int(len(window_task_result.records)),
            "removed_unknown": int(window_task_result.removed_unknown),
            "removed_task_mismatch": int(window_task_result.removed_task_mismatch),
            "removed_single_class_sites": int(window_task_result.removed_single_class_sites),
            "diagnosis_class_counts": class_counts(window_task_y, window_task_result.class_names),
        },
        "runs": vote_run_rows,
        "summary": summarize_subject_vote_runs(diagnosis_vote_runs),
    }

    return {
        "config": centered_payload["config"],
        "centered": centered_payload,
        "subject_vote": subject_vote_payload,
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
