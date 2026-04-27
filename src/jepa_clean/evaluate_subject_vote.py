import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from .data import WindowedFMRIDataset
from .model import JEPA
from .pooling import pool_sequence_features
from .repro import seed_everything


TASKS = {
    "hc_all": ("HC vs ALL", 0, "ALL"),
    "hc_asd": ("HC vs ASD", 0, 1),
    "hc_mdd": ("HC vs MDD", 0, 2),
    "hc_scz": ("HC vs SCZ", 0, 4),
    "hc_bipolar": ("HC vs Bipolar", 0, 7),
    "hc_pain": ("HC vs Pain", 0, 5),
    "hc_stroke": ("HC vs Stroke", 0, 6),
    "hc_dysthymia": ("HC vs Dysthymia", 0, 8),
    "hc_others": ("HC vs Others", 0, 99),
}


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)


def collate_fn(batch):
    tensors, labels = zip(*batch)
    return torch.stack(tensors, dim=0), list(labels)


def load_checkpoint_config(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    return checkpoint, cfg


def load_checkpoint_weights(
    model: JEPA,
    checkpoint: dict,
    cfg: dict,
    override_target_timepoints: int | None = None,
):
    state_dict = checkpoint["model_state_dict"]
    checkpoint_target_timepoints = int(cfg.get("target_timepoints", 45))

    if override_target_timepoints is None or int(override_target_timepoints) == checkpoint_target_timepoints:
        model.load_state_dict(state_dict, strict=True)
        return

    pos_embed_mode = cfg.get("pos_embed_mode", "learned")
    if pos_embed_mode == "sincos":
        # Regenerate deterministic sin-cos buffers at the requested sequence length.
        pos_embed_keys = {
            "token_embed.pos_embed",
            "target_token_embed.pos_embed",
            "predictor.pos_embed",
        }
        filtered_state_dict = {k: v for k, v in state_dict.items() if k not in pos_embed_keys}
        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        missing_keys = set(incompatible.missing_keys)
        if missing_keys != pos_embed_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected checkpoint incompatibility while overriding target_timepoints: "
                f"missing={sorted(missing_keys)} unexpected={sorted(incompatible.unexpected_keys)}"
            )
        return

    if pos_embed_mode == "rotary":
        model.load_state_dict(state_dict, strict=True)
        return

    raise ValueError(
        "Overriding target_timepoints is only supported for sincos/rotary checkpoints. "
        f"Checkpoint uses pos_embed_mode={pos_embed_mode!r}."
    )


def build_model_from_checkpoint(checkpoint_path: str, override_target_timepoints: int | None = None):
    checkpoint, cfg = load_checkpoint_config(checkpoint_path)
    target_timepoints = override_target_timepoints or cfg.get("target_timepoints", 45)
    model = JEPA(
        input_dim=400,
        max_seq_len=target_timepoints,
        embed_dim=cfg.get("embed_dim", 128),
        encoder_depth=cfg.get("encoder_depth", 3),
        predictor_depth=cfg.get("predictor_depth", 2),
        num_heads=cfg.get("num_heads", 4),
        pos_embed_mode=cfg.get("pos_embed_mode", "learned"),
        use_conv_stem=cfg.get("use_conv_stem", False),
        conv_kernel_size=cfg.get("conv_kernel_size", 3),
        use_roi_embeddings=cfg.get("use_roi_embeddings", False),
        use_network_embeddings=cfg.get("use_network_embeddings", False),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        dropout=cfg.get("dropout", 0.1),
        ema_decay=cfg.get("ema_decay", 0.996),
    )
    load_checkpoint_weights(
        model,
        checkpoint,
        cfg,
        override_target_timepoints=override_target_timepoints,
    )
    return model, cfg


@torch.no_grad()
def extract_window_embeddings(
    model,
    dataset,
    device,
    batch_size: int = 64,
    window_feature: str = "mean",
    covariance_dim: int = 32,
    covariance_seed: int = 0,
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


def filter_task(embeddings, metadata, task_key: str):
    task_name, hc_diag, disorder_diag = TASKS[task_key]

    keep_indices = []
    y = []
    for idx, meta in enumerate(metadata):
        diag = int(meta["diag"])
        if diag < 0:
            continue
        if disorder_diag == "ALL":
            keep_indices.append(idx)
            y.append(int(diag != hc_diag))
        elif diag in {hc_diag, disorder_diag}:
            keep_indices.append(idx)
            y.append(int(diag == disorder_diag))

    if not keep_indices:
        raise ValueError(f"No windows available for task {task_name}")

    keep_indices = np.array(keep_indices, dtype=int)
    kept_meta = [metadata[i] for i in keep_indices]
    return task_name, embeddings[keep_indices], kept_meta, np.array(y, dtype=int)


def maybe_subsample_subjects(embeddings, metadata, y, max_subjects_per_class: int | None, seed: int):
    if max_subjects_per_class is None:
        return embeddings, metadata, y

    rng = np.random.default_rng(seed)
    by_subject = {}
    for idx, meta in enumerate(metadata):
        by_subject.setdefault(meta["subject_id"], []).append(idx)

    subject_label = {}
    for subject_id, indices in by_subject.items():
        labels = {int(y[i]) for i in indices}
        if len(labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels")
        subject_label[subject_id] = next(iter(labels))

    selected_subjects = []
    for cls in sorted(set(subject_label.values())):
        cls_subjects = [subject_id for subject_id, label in subject_label.items() if label == cls]
        if len(cls_subjects) > max_subjects_per_class:
            cls_subjects = rng.choice(cls_subjects, size=max_subjects_per_class, replace=False).tolist()
        selected_subjects.extend(cls_subjects)

    selected_subjects = set(selected_subjects)
    keep = np.array([meta["subject_id"] in selected_subjects for meta in metadata], dtype=bool)
    kept_meta = [meta for meta, flag in zip(metadata, keep) if flag]
    return embeddings[keep], kept_meta, y[keep]


def attach_labels(metadata, y):
    labeled = []
    for meta, label in zip(metadata, y):
        copied = dict(meta)
        copied["y"] = int(label)
        labeled.append(copied)
    return labeled


def get_default_threshold(classifier: str):
    return 0.0 if classifier == "linear_svm" else 0.5


def threshold_scores(scores, classifier: str, threshold: float):
    scores = np.asarray(scores, dtype=float)
    return (scores >= threshold).astype(int)


def get_threshold_grid(scores, classifier: str):
    default_threshold = get_default_threshold(classifier)
    if classifier in {"logreg", "xgboost"}:
        grid = np.linspace(0.01, 0.99, num=99, dtype=float)
    else:
        scores = np.asarray(scores, dtype=float)
        if len(scores) == 0 or np.allclose(scores, scores[0]):
            grid = np.array([default_threshold], dtype=float)
        else:
            grid = np.quantile(scores, np.linspace(0.01, 0.99, num=99))
            grid = np.unique(grid.astype(float))

    if not np.any(np.isclose(grid, default_threshold)):
        grid = np.unique(np.concatenate([grid, np.array([default_threshold], dtype=float)]))
    return grid


def aggregate_subject_predictions(metadata, scores, classifier: str, threshold: float):
    grouped = defaultdict(lambda: {"scores": [], "preds": [], "site": None, "y": None, "n_windows": 0})
    for meta, score in zip(metadata, scores):
        pred = int(threshold_scores(np.array([score]), classifier, threshold)[0])
        entry = grouped[meta["subject_id"]]
        entry["scores"].append(float(score))
        entry["preds"].append(int(pred))
        entry["site"] = meta["site"]
        entry["y"] = int(meta["y"])
        entry["n_windows"] += 1

    subject_rows = []
    for subject_id, entry in grouped.items():
        votes = np.array(entry["preds"], dtype=int)
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
        subject_id = meta["subject_id"]
        by_subject.setdefault(subject_id, []).append(idx)
        label = int(label)
        if subject_id in subject_label and subject_label[subject_id] != label:
            raise ValueError(f"Subject {subject_id} has inconsistent labels")
        subject_label[subject_id] = label

    subject_ids = np.array(sorted(by_subject), dtype=object)
    subject_y = np.array([subject_label[subject_id] for subject_id in subject_ids], dtype=int)

    if len(subject_ids) < 2 or len(np.unique(subject_y)) < 2:
        return np.arange(len(metadata), dtype=int), np.array([], dtype=int)

    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_subject_idx, val_subject_idx = next(splitter.split(np.zeros(len(subject_ids)), subject_y))
    except ValueError:
        return np.arange(len(metadata), dtype=int), np.array([], dtype=int)

    train_subjects = set(subject_ids[train_subject_idx].tolist())
    val_subjects = set(subject_ids[val_subject_idx].tolist())

    train_indices = np.array(
        [idx for idx, meta in enumerate(metadata) if meta["subject_id"] in train_subjects],
        dtype=int,
    )
    val_indices = np.array(
        [idx for idx, meta in enumerate(metadata) if meta["subject_id"] in val_subjects],
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
        y_true = np.array([row["y"] for row in subject_rows], dtype=int)
        y_pred = np.array([row["pred"] for row in subject_rows], dtype=int)
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
    grouped = defaultdict(list)
    for row in subject_rows:
        grouped[row["site"]].append(row)

    site_metrics = []
    for site in sorted(grouped):
        rows = grouped[site]
        y_true = np.array([row["y"] for row in rows], dtype=int)
        y_pred = np.array([row["pred"] for row in rows], dtype=int)
        y_prob = np.array([row["prob"] for row in rows], dtype=float)
        metrics = compute_binary_metrics(y_true, y_pred, y_prob)
        metrics["site"] = site
        metrics["n_subjects"] = len(rows)
        metrics["mean_windows_per_subject"] = float(
            np.mean([row["n_windows"] for row in rows])
        )
        site_metrics.append(metrics)
    return site_metrics


def add_site_std_to_summary(summary: dict, site_metrics: list[dict]):
    summary["site_metrics"] = site_metrics
    summary["site_bacc_mean"] = float("nan")
    summary["site_bacc_std"] = float("nan")
    summary["site_auc_mean"] = float("nan")
    summary["site_auc_std"] = float("nan")
    summary["site_f1_macro_mean"] = float("nan")
    summary["site_f1_macro_std"] = float("nan")
    summary["site_precision_mean"] = float("nan")
    summary["site_precision_std"] = float("nan")
    summary["site_recall_mean"] = float("nan")
    summary["site_recall_std"] = float("nan")

    if not site_metrics:
        return summary

    for key, out_mean, out_std in [
        ("bacc", "site_bacc_mean", "site_bacc_std"),
        ("auc", "site_auc_mean", "site_auc_std"),
        ("f1_macro", "site_f1_macro_mean", "site_f1_macro_std"),
        ("precision", "site_precision_mean", "site_precision_std"),
        ("recall", "site_recall_mean", "site_recall_std"),
    ]:
        vals = np.array([m[key] for m in site_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        summary[out_mean] = float(np.mean(vals))
        summary[out_std] = float(np.std(vals, ddof=0))
    return summary


def build_probe_classifier(classifier: str, seed: int, y_train=None):
    if classifier == "logreg":
        n_comp = 50 if y_train is None else min(50, len(y_train))
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
    if classifier == "linear_svm":
        return make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=0.1,
                class_weight="balanced",
                random_state=seed,
                max_iter=5000,
            ),
        )
    if classifier == "xgboost":
        scale_pos_weight = 1.0
        if y_train is not None:
            y_train = np.asarray(y_train, dtype=int)
            pos = int((y_train == 1).sum())
            neg = int((y_train == 0).sum())
            if pos > 0:
                scale_pos_weight = float(neg / pos)
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            n_jobs=8,
            tree_method="hist",
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def get_probe_scores_and_preds(clf, x_test, classifier: str, threshold: float | None = None):
    threshold = get_default_threshold(classifier) if threshold is None else float(threshold)

    if classifier == "logreg":
        scores = clf.predict_proba(x_test)[:, 1]
        preds = threshold_scores(scores, classifier, threshold)
        return scores, preds

    if classifier == "linear_svm":
        scores = clf.decision_function(x_test)
        preds = threshold_scores(scores, classifier, threshold)
        return scores, preds

    if classifier == "xgboost":
        scores = clf.predict_proba(x_test)[:, 1]
        preds = threshold_scores(scores, classifier, threshold)
        return scores, preds

    raise ValueError(f"Unsupported classifier: {classifier}")


def run_subject_vote_loso(
    embeddings,
    metadata,
    y,
    seed: int = 42,
    classifier: str = "logreg",
    threshold_mode: str = "fixed",
    val_fraction: float = 0.2,
):
    sites = np.array([meta["site"] for meta in metadata])

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
                clf = build_probe_classifier(classifier, seed, y_train=y_train[train_inner_idx])
                clf.fit(train_embeddings[train_inner_idx], y_train[train_inner_idx])

                val_scores, _ = get_probe_scores_and_preds(
                    clf,
                    train_embeddings[val_idx],
                    classifier,
                    threshold=get_default_threshold(classifier),
                )
                val_metadata = attach_labels([train_metadata[idx] for idx in val_idx], y_train[val_idx])
                threshold = find_optimal_subject_threshold(
                    val_metadata,
                    y_train[val_idx],
                    val_scores,
                    classifier=classifier,
                )
                clf = build_probe_classifier(classifier, seed, y_train=y_train)
                clf.fit(train_embeddings, y_train)
                calibrated_sites.append(str(held_out))
            else:
                clf = build_probe_classifier(classifier, seed, y_train=y_train)
                clf.fit(train_embeddings, y_train)
        else:
            clf = build_probe_classifier(classifier, seed, y_train=y_train)
            clf.fit(train_embeddings, y_train)

        test_scores, _ = get_probe_scores_and_preds(clf, test_embeddings, classifier, threshold=threshold)
        test_metadata = attach_labels(test_metadata_raw, y_test)
        subject_rows = aggregate_subject_predictions(
            test_metadata,
            test_scores,
            classifier=classifier,
            threshold=threshold,
        )
        aggregated.extend(subject_rows)
        held_out_sites.append(held_out)
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
        }, []

    y_true = np.array([row["y"] for row in aggregated], dtype=int)
    y_pred = np.array([row["pred"] for row in aggregated], dtype=int)
    y_prob = np.array([row["prob"] for row in aggregated], dtype=float)
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


def parse_args():
    parser = argparse.ArgumentParser(description="Subject-level sliding-window majority-vote evaluation for clean JEPA checkpoints")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        default=[
            "preprocessed_data_fir/full/roi_timeseries_1600_full",
            "preprocessed_data_fir/full/roi_timeseries_abide_full",
        ],
    )
    parser.add_argument("--task", nargs="+", default=["hc_all"], choices=sorted(TASKS))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--window-stride-seconds", type=float, default=30.0)
    parser.add_argument("--target-timepoints", type=int, default=None)
    parser.add_argument("--max-subjects-per-class", type=int, default=None)
    parser.add_argument("--classifier", choices=["logreg", "linear_svm", "xgboost"], default="logreg")
    parser.add_argument("--threshold-mode", choices=["fixed", "calibrated"], default="fixed")
    parser.add_argument("--threshold-val-fraction", type=float, default=0.2)
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--keep-unknown-diagnoses", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("JEPA-Clean-SubjectVote")
    seed_everything(args.seed, deterministic=True)

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

    embeddings, metadata = extract_window_embeddings(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        window_feature=args.window_feature,
        covariance_dim=args.covariance_dim,
        covariance_seed=args.covariance_seed,
    )
    logger.info(
        "Extracted features: %s | window_feature=%s covariance_dim=%d",
        embeddings.shape,
        args.window_feature,
        args.covariance_dim,
    )

    results = []
    for task_key in args.task:
        task_name, task_embeddings, task_metadata, y = filter_task(embeddings, metadata, task_key)
        task_embeddings, task_metadata, y = maybe_subsample_subjects(
            task_embeddings,
            task_metadata,
            y,
            max_subjects_per_class=args.max_subjects_per_class,
            seed=args.seed,
        )

        subject_ids = sorted({meta["subject_id"] for meta in task_metadata})
        logger.info(
            "Task %s | windows=%d subjects=%d class_counts=%s",
            task_name,
            len(task_metadata),
            len(subject_ids),
            {int(cls): int((y == cls).sum()) for cls in np.unique(y)},
        )

        metrics, subject_rows = run_subject_vote_loso(
            task_embeddings,
            task_metadata,
            y,
            seed=args.seed,
            classifier=args.classifier,
            threshold_mode=args.threshold_mode,
            val_fraction=args.threshold_val_fraction,
        )
        result = {
            "task": task_name,
            "task_key": task_key,
            "classifier": args.classifier,
            "window_feature": args.window_feature,
            "covariance_dim": int(args.covariance_dim),
            "covariance_seed": int(args.covariance_seed),
            **metrics,
        }
        results.append({"summary": result, "subjects": subject_rows})
        logger.info(
            "%s | bAcc=%.1f%% AUC=%.3f F1=%.3f precision=%.3f recall=%.3f | valid_sites=%d subjects=%d mean_windows=%.2f",
            task_name,
            100.0 * result["bacc"],
            result["auc"],
            result["f1_macro"],
            result["precision"],
            result["recall"],
            result["n_valid_sites"],
            result["n_subjects"],
            result["mean_windows_per_subject"],
        )
        logger.info(
            "%s | threshold_mode=%s mean_threshold=%.3f calibrated_sites=%d",
            task_name,
            result["threshold_mode"],
            result["mean_threshold"],
            result["n_calibrated_sites"],
        )

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved results: %s", save_path)


if __name__ == "__main__":
    main()
