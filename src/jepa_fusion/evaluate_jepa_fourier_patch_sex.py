from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
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


SEX_TO_LABEL = {"M": 0, "F": 1}
CLASS_NAMES = ["M", "F"]


@dataclass(frozen=True)
class EvalRecord:
    index: int
    subject: str
    site: str
    sex: str


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Fourier-patch JEPA checkpoints for sex prediction.")
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


def filter_labeled_sex(records: list[EvalRecord]):
    kept_records = []
    y = []
    for record in records:
        sex = str(record.sex).strip().upper()
        if sex not in SEX_TO_LABEL:
            continue
        kept_records.append(record)
        y.append(SEX_TO_LABEL[sex])
    if not kept_records:
        raise ValueError("No labeled records with sex information were found.")
    y = np.asarray(y, dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Sex evaluation needs both classes present.")
    return kept_records, y


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
    oof_scores = np.full((len(y), len(class_names)), 0.0, dtype=np.float64)
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
        aligned_scores = np.zeros((len(test_idx), len(class_names)), dtype=np.float64)
        aligned_scores[:, clf_classes] = fold_scores
        oof_scores[test_idx] = aligned_scores

        fold_rows.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, fold_pred)),
                "balanced_accuracy": _balanced_accuracy_present(y_test, fold_pred),
                "roc_auc": _safe_roc_auc(y_test, aligned_scores[:, 1]),
            }
        )

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": _balanced_accuracy_present(y, preds),
        "roc_auc": _safe_roc_auc(y, oof_scores[:, 1]),
        "folds": int(effective_folds),
        "fold_rows": fold_rows,
        "n_samples": int(len(y)),
        "classes": list(class_names),
    }


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


def attach_labels(metadata, y):
    labeled = []
    for meta, label in zip(metadata, y):
        copied = dict(meta)
        copied["y"] = int(label)
        labeled.append(copied)
    return labeled


def get_default_threshold():
    return 0.5


def threshold_scores(scores, threshold: float):
    scores = np.asarray(scores, dtype=float)
    return (scores >= threshold).astype(int)


def get_threshold_grid():
    grid = np.linspace(0.01, 0.99, num=99, dtype=float)
    if not np.any(np.isclose(grid, 0.5)):
        grid = np.unique(np.concatenate([grid, np.array([0.5], dtype=float)]))
    return grid


def aggregate_subject_predictions(metadata, scores, threshold: float):
    grouped = {}
    for meta, score in zip(metadata, scores):
        pred = int(threshold_scores(np.array([score]), threshold)[0])
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


def find_optimal_subject_threshold(metadata, y, scores):
    if len(metadata) == 0 or len(np.unique(y)) < 2:
        return get_default_threshold()

    best_threshold = get_default_threshold()
    best_bacc = -1.0
    for threshold in get_threshold_grid():
        subject_rows = aggregate_subject_predictions(metadata, scores, threshold=threshold)
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


def run_subject_vote_loso(
    embeddings,
    metadata,
    y,
    *,
    seed: int,
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

        threshold = get_default_threshold()
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
                val_scores = clf.predict_proba(train_embeddings[val_idx])[:, 1]
                val_metadata = attach_labels([train_metadata[idx] for idx in val_idx], y_train[val_idx])
                threshold = find_optimal_subject_threshold(val_metadata, y_train[val_idx], val_scores)
                clf = build_probe_classifier(seed, y_train, pca_components=pca_components)
                clf.fit(train_embeddings, y_train)
                calibrated_sites.append(str(held_out))
            else:
                clf = build_probe_classifier(seed, y_train, pca_components=pca_components)
                clf.fit(train_embeddings, y_train)
        else:
            clf = build_probe_classifier(seed, y_train, pca_components=pca_components)
            clf.fit(train_embeddings, y_train)

        test_scores = clf.predict_proba(test_embeddings)[:, 1]
        test_metadata = attach_labels(test_metadata_raw, y_test)
        subject_rows = aggregate_subject_predictions(test_metadata, test_scores, threshold=threshold)
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


def format_pct(metric: dict):
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def default_output_paths(checkpoint_path: str):
    checkpoint = Path(checkpoint_path)
    stem = f"{checkpoint.parent.name}_{checkpoint.stem}"
    base_dir = Path("results/reports/tables/jepa_fusion")
    return base_dir / f"{stem}_fourier_patch_sex_eval.json", base_dir / f"{stem}_fourier_patch_sex_eval.md"


def build_markdown(payload: dict):
    lines = [
        f"# {payload['checkpoint']['label']} Fourier-Patch JEPA Sex Evaluation",
        "",
        f"Checkpoint: `{payload['checkpoint']['path']}`",
        "",
        "## Centered Window Grouped-CV",
        "",
        "| Dataset | Sex Probe bAcc | Sex Probe ROC-AUC | N Scans |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        summary = payload["datasets"][dataset_key]["centered"]["summary"]
        config = payload["datasets"][dataset_key]["config"]
        lines.append(
            f"| {config['dataset_label']} | "
            f"{format_pct(summary['balanced_accuracy'])} | "
            f"{format_pct(summary['roc_auc'])} | "
            f"{summary['n_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Subject Vote LOSO",
            "",
            "| Dataset | Sex LOSO bAcc | Sex LOSO ROC-AUC | Held-out Site Mean bAcc | Valid Sites | Mean Windows/Subject |",
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
    threshold_mode: str,
    threshold_val_fraction: float,
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
            sex=str(meta.get("sex", "")),
        )
        for idx, meta in enumerate(centered_metadata)
    ]
    centered_records, centered_y = filter_labeled_sex(centered_records)
    centered_keep = np.asarray([record.index for record in centered_records], dtype=np.int64)
    centered_features = centered_embeddings[centered_keep]
    centered_groups = np.asarray([record.site for record in centered_records], dtype=object)

    centered_runs = []
    for seed in seeds:
        centered_runs.append(
            run_grouped_cv(
                centered_features,
                centered_y,
                CLASS_NAMES,
                centered_groups,
                seed=seed,
                folds=folds,
                pca_components=pca_components,
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
        sex = str(meta.get("sex", "")).strip().upper()
        if sex not in SEX_TO_LABEL:
            continue
        sliding_keep_indices.append(idx)
        sliding_y.append(SEX_TO_LABEL[sex])
        copied = dict(meta)
        copied["site"] = center_site_label(str(meta.get("site_raw", meta["site"])))
        sliding_eval_metadata.append(copied)
    if not sliding_keep_indices:
        raise ValueError(f"No sliding-window sex labels available for dataset {dataset_key}.")

    sliding_features = sliding_embeddings[np.asarray(sliding_keep_indices, dtype=np.int64)]
    sliding_y = np.asarray(sliding_y, dtype=np.int64)
    if len(np.unique(sliding_y)) < 2:
        raise ValueError(f"Sliding-window sex labels for dataset {dataset_key} contain fewer than two classes.")

    subject_runs = []
    for seed in seeds:
        metrics, _ = run_subject_vote_loso(
            sliding_features,
            sliding_eval_metadata,
            sliding_y,
            seed=seed,
            threshold_mode=threshold_mode,
            val_fraction=threshold_val_fraction,
            pca_components=pca_components,
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
            "sex_positive_label": "F",
            "sex_negative_label": "M",
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
            threshold_mode=args.threshold_mode,
            threshold_val_fraction=args.threshold_val_fraction,
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
            "threshold_mode": args.threshold_mode,
            "threshold_val_fraction": float(args.threshold_val_fraction),
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
