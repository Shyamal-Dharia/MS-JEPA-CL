from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .dataset_utils import matched_records_for_preset
    from .evaluate_baselines import (
        mean_std,
        transductive_transform_features,
    )
    from .evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from .protocol import dataset_order_for_choice, get_dataset_preset
    from ..jepa_clean.data import WindowedFMRIDataset
    from ..jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation
    from ..jepa_clean.repro import seed_everything
except ImportError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.jepa_fusion.dataset_utils import matched_records_for_preset
    from src.jepa_fusion.evaluate_baselines import (
        mean_std,
        transductive_transform_features,
    )
    from src.jepa_fusion.evaluate_jepa import extract_scan_embeddings, load_checkpoint
    from src.jepa_fusion.protocol import dataset_order_for_choice, get_dataset_preset
    from src.jepa_clean.data import WindowedFMRIDataset
    from src.jepa_clean.plot_tsne_fc_vs_raw import build_fc_features, get_roi_permutation
    from src.jepa_clean.repro import seed_everything


SEX_CLASS_NAMES = ["M", "F"]
MALE_VALUES = {"m", "male", "1", "1.0", "b"}
FEMALE_VALUES = {"f", "female", "2", "2.0", "g", "0", "0.0"}
METHOD_LABELS = {
    "fc": "Static FC",
    "fc_combat": "Static FC + ComBat",
    "fc_covbat": "Static FC + CovBat",
    "jepa": "JEPA embeddings",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate sex classification and age regression from static FC variants "
            "and optional JEPA checkpoint embeddings."
        )
    )
    parser.add_argument("--checkpoint", default=None, help="Optional JEPA checkpoint to evaluate.")
    parser.add_argument("--dataset", choices=["abide", "abide1", "abide2", "srpbs", "all"], default="all")
    parser.add_argument("--methods", default="fc,fc_combat,fc_covbat,jepa")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-scans", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--window-feature", choices=["mean", "mean_std", "mean_cov"], default="mean")
    parser.add_argument(
        "--jepa-sampling",
        choices=["center", "sliding"],
        default="sliding",
        help="JEPA window extraction. Default averages embeddings across all sliding windows per subject.",
    )
    parser.add_argument("--covariance-dim", type=int, default=32)
    parser.add_argument("--covariance-seed", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--triangle", choices=["upper", "lower"], default="upper")
    parser.add_argument("--reorder", choices=["none", "hemi", "hemi_network"], default="hemi_network")
    parser.add_argument("--covbat-n-pc", type=int, default=50)
    parser.add_argument("--covbat-pct-var", type=float, default=0.95)
    parser.add_argument("--covbat-max-components", type=int, default=256)
    parser.add_argument(
        "--scan-level",
        action="store_true",
        help="Evaluate scan/window rows directly instead of averaging features to one row per subject.",
    )
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


def parse_methods(method_text: str, checkpoint: str | None):
    methods = [item.strip() for item in method_text.split(",") if item.strip()]
    invalid = sorted(set(methods) - set(METHOD_LABELS))
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Valid methods: {sorted(METHOD_LABELS)}")
    if "jepa" in methods and checkpoint is None:
        print("Skipping method `jepa` because --checkpoint was not provided.")
        methods = [method for method in methods if method != "jepa"]
    if not methods:
        raise ValueError("No evaluation methods remain after parsing --methods.")
    return methods


def sex_to_label(value):
    text = str(value).strip().lower()
    if text in MALE_VALUES:
        return 0
    if text in FEMALE_VALUES:
        return 1
    return None


def filter_sex(records, features: np.ndarray):
    keep = []
    y = []
    for idx, record in enumerate(records):
        label = sex_to_label(getattr(record, "sex", None))
        if label is None:
            continue
        keep.append(idx)
        y.append(label)
    if len(set(y)) < 2:
        raise ValueError("Sex prediction requires both M and F labels.")
    return features[np.asarray(keep, dtype=np.int64)], np.asarray(y, dtype=np.int64), [records[idx] for idx in keep]


def filter_age(records, features: np.ndarray):
    keep = []
    y = []
    for idx, record in enumerate(records):
        age = getattr(record, "age", None)
        if age is None:
            continue
        age = float(age)
        if not np.isfinite(age) or age <= 0:
            continue
        keep.append(idx)
        y.append(age)
    if not keep:
        raise ValueError("Age prediction requires at least one finite positive age label.")
    return features[np.asarray(keep, dtype=np.int64)], np.asarray(y, dtype=np.float64), [records[idx] for idx in keep]


def get_subject_groups(records):
    return np.asarray([str(getattr(record, "subject_id", getattr(record, "subject", ""))) for record in records], dtype=object)


def copy_record_with_count(record, n_items: int):
    class Record:
        pass

    copied = Record()
    for key, value in vars(record).items():
        setattr(copied, key, value)
    copied.n_items = int(n_items)
    return copied


def aggregate_features_by_subject(features: np.ndarray, records: list):
    grouped = {}
    order = []
    for idx, record in enumerate(records):
        subject_id = str(getattr(record, "subject_id", getattr(record, "subject", idx)))
        if subject_id not in grouped:
            grouped[subject_id] = {"indices": [], "record": record}
            order.append(subject_id)
        grouped[subject_id]["indices"].append(idx)

    aggregated_features = []
    aggregated_records = []
    for subject_id in order:
        indices = np.asarray(grouped[subject_id]["indices"], dtype=np.int64)
        aggregated_features.append(features[indices].mean(axis=0))
        aggregated_records.append(copy_record_with_count(grouped[subject_id]["record"], len(indices)))

    return np.stack(aggregated_features, axis=0), aggregated_records


def _make_classifier(feature_dim: int, train_count: int, seed: int, pca_components: int):
    n_components = min(int(pca_components), feature_dim, max(1, train_count - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        LogisticRegression(max_iter=10000, random_state=seed, class_weight="balanced"),
    )


def _make_regressor(feature_dim: int, train_count: int, seed: int, pca_components: int, ridge_alpha: float):
    n_components = min(int(pca_components), feature_dim, max(1, train_count - 1))
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=seed),
        Ridge(alpha=float(ridge_alpha), random_state=seed),
    )


def effective_stratified_group_folds(y: np.ndarray, groups: np.ndarray, requested_folds: int):
    per_class_group_counts = []
    for cls in np.unique(y):
        per_class_group_counts.append(len(np.unique(groups[y == cls])))
    return min(int(requested_folds), min(per_class_group_counts))


def effective_group_folds(groups: np.ndarray, requested_folds: int):
    return min(int(requested_folds), len(np.unique(groups)))


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def run_sex_cv(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    records: list,
    *,
    seed: int,
    folds: int,
    pca_components: int,
):
    effective_folds = effective_stratified_group_folds(y, groups, folds)
    if effective_folds < 2:
        raise RuntimeError(f"Need at least 2 effective folds for sex prediction, got {effective_folds}.")

    cv = StratifiedGroupKFold(n_splits=effective_folds, shuffle=True, random_state=seed)
    preds = np.full(len(y), -1, dtype=np.int64)
    scores = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    fold_rows = []
    indices = np.arange(len(y), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(indices, y, groups), start=1):
        clf = _make_classifier(features.shape[1], len(train_idx), seed, pca_components)
        clf.fit(features[train_idx], y[train_idx])
        fold_pred = clf.predict(features[test_idx])
        preds[test_idx] = fold_pred
        fold_ids[test_idx] = fold_idx
        probabilities = clf.predict_proba(features[test_idx])
        classes = clf.named_steps["logisticregression"].classes_
        if 1 in classes:
            class_pos = int(np.flatnonzero(classes == 1)[0])
            fold_scores = probabilities[:, class_pos]
        else:
            fold_scores = np.zeros(len(test_idx), dtype=np.float64)
        scores[test_idx] = fold_scores
        fold_rows.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], fold_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], fold_pred)),
                "roc_auc": safe_roc_auc(y[test_idx], fold_scores),
            }
        )

    prediction_rows = []
    for idx, record in enumerate(records):
        prediction_rows.append(
            {
                "subject_id": str(getattr(record, "subject_id", getattr(record, "subject", ""))),
                "subject": str(getattr(record, "subject", "")),
                "site": str(getattr(record, "site", "")),
                "site_batch": str(getattr(record, "site_batch", "")),
                "n_aggregated_items": int(getattr(record, "n_items", 1)),
                "fold": int(fold_ids[idx]),
                "y_true": SEX_CLASS_NAMES[int(y[idx])],
                "y_pred": SEX_CLASS_NAMES[int(preds[idx])],
                "female_probability": float(scores[idx]),
            }
        )

    return {
        "accuracy": float(accuracy_score(y, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "roc_auc": safe_roc_auc(y, scores),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "folds": int(effective_folds),
        "fold_rows": fold_rows,
        "predictions": prediction_rows,
        "n_samples": int(len(y)),
        "classes": list(SEX_CLASS_NAMES),
    }


def compute_age_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
    }


def run_age_cv(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    records: list,
    *,
    seed: int,
    folds: int,
    pca_components: int,
    ridge_alpha: float,
):
    effective_folds = effective_group_folds(groups, folds)
    if effective_folds < 2:
        raise RuntimeError(f"Need at least 2 effective folds for age prediction, got {effective_folds}.")

    cv = GroupKFold(n_splits=effective_folds)
    preds = np.full(len(y), np.nan, dtype=np.float64)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    fold_rows = []
    indices = np.arange(len(y), dtype=np.int64)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(indices, y, groups), start=1):
        reg = _make_regressor(features.shape[1], len(train_idx), seed, pca_components, ridge_alpha)
        reg.fit(features[train_idx], y[train_idx])
        fold_pred = reg.predict(features[test_idx])
        preds[test_idx] = fold_pred
        fold_ids[test_idx] = fold_idx
        fold_metrics = compute_age_metrics(y[test_idx], fold_pred)
        fold_metrics.update(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
        fold_rows.append(fold_metrics)

    prediction_rows = []
    for idx, record in enumerate(records):
        prediction_rows.append(
            {
                "subject_id": str(getattr(record, "subject_id", getattr(record, "subject", ""))),
                "subject": str(getattr(record, "subject", "")),
                "site": str(getattr(record, "site", "")),
                "site_batch": str(getattr(record, "site_batch", "")),
                "n_aggregated_items": int(getattr(record, "n_items", 1)),
                "fold": int(fold_ids[idx]),
                "y_true": float(y[idx]),
                "y_pred": float(preds[idx]),
                "absolute_error": float(abs(y[idx] - preds[idx])),
            }
        )

    metrics = compute_age_metrics(y, preds)
    metrics.update(
        {
            "folds": int(effective_folds),
            "fold_rows": fold_rows,
            "predictions": prediction_rows,
            "n_samples": int(len(y)),
        }
    )
    return metrics


def summarize_sex_runs(runs: list[dict]):
    return {
        "accuracy": mean_std([run["accuracy"] for run in runs]),
        "balanced_accuracy": mean_std([run["balanced_accuracy"] for run in runs]),
        "roc_auc": mean_std([run["roc_auc"] for run in runs]),
        "f1": mean_std([run["f1"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "folds": int(runs[0]["folds"]),
        "classes": list(runs[0]["classes"]),
    }


def summarize_age_runs(runs: list[dict]):
    return {
        "r2": mean_std([run["r2"] for run in runs]),
        "mae": mean_std([run["mae"] for run in runs]),
        "rmse": mean_std([run["rmse"] for run in runs]),
        "n_runs": int(len(runs)),
        "n_samples": int(runs[0]["n_samples"]),
        "folds": int(runs[0]["folds"]),
    }


def records_from_jepa_metadata(metadata: list[dict]):
    class Record:
        pass

    records = []
    for meta in metadata:
        record = Record()
        record.subject = str(meta["subject"])
        record.subject_id = str(meta["subject_id"])
        record.site = str(meta.get("site_raw", meta.get("site", "")))
        record.site_batch = str(meta.get("site", meta.get("site_raw", "")))
        record.source_dataset = str(meta.get("dataset", ""))
        record.sex = meta.get("sex")
        record.age = meta.get("age")
        record.window_index = meta.get("window_index")
        record.n_windows = meta.get("n_windows")
        records.append(record)
    return records


def build_fc_method_features(records, args, methods: list[str], seeds: list[int]):
    permutation, _ = get_roi_permutation(args.reorder)
    fc_features = build_fc_features(records, permutation=permutation, triangle=args.triangle)
    batch_sites = np.asarray([str(record.site_batch) for record in records], dtype=object)

    features = {}
    if "fc" in methods:
        features["fc"] = fc_features
    if "fc_combat" in methods:
        features["fc_combat"] = transductive_transform_features(
            "fc_combat",
            fc_features,
            batch_sites,
            seed=seeds[0],
            covbat_n_pc=args.covbat_n_pc,
            covbat_pct_var=args.covbat_pct_var,
            covbat_max_components=args.covbat_max_components,
        )
    if "fc_covbat" in methods:
        features["fc_covbat"] = transductive_transform_features(
            "fc_covbat",
            fc_features,
            batch_sites,
            seed=seeds[0],
            covbat_n_pc=args.covbat_n_pc,
            covbat_pct_var=args.covbat_pct_var,
            covbat_max_components=args.covbat_max_components,
        )
    return features


def build_jepa_features(model, cfg, preset, args, device):
    dataset = WindowedFMRIDataset(
        preset.raw_dir,
        window_seconds=float(cfg.get("window_seconds", 90.0)),
        stride_seconds=float(cfg.get("window_stride_seconds", 30.0)),
        target_timepoints=int(cfg.get("target_timepoints", 45)),
        sampling=args.jepa_sampling,
        max_scans=args.max_scans,
        exclude_unknown_diagnoses=False,
    )
    embeddings, metadata = extract_scan_embeddings(
        model,
        dataset,
        device=device,
        batch_size=args.batch_size,
        window_feature=args.window_feature,
        covariance_dim=args.covariance_dim,
        covariance_seed=args.covariance_seed,
    )
    return embeddings, records_from_jepa_metadata(metadata)


def evaluate_method(features: np.ndarray, records: list, seeds: list[int], args):
    if not args.scan_level:
        features, records = aggregate_features_by_subject(features, records)

    sex_features, sex_y, sex_records = filter_sex(records, features)
    sex_groups = get_subject_groups(sex_records)
    age_features, age_y, age_records = filter_age(records, features)
    age_groups = get_subject_groups(age_records)

    sex_runs = [
        run_sex_cv(
            sex_features,
            sex_y,
            sex_groups,
            sex_records,
            seed=seed,
            folds=args.folds,
            pca_components=args.pca_components,
        )
        for seed in seeds
    ]
    age_runs = [
        run_age_cv(
            age_features,
            age_y,
            age_groups,
            age_records,
            seed=seed,
            folds=args.folds,
            pca_components=args.pca_components,
            ridge_alpha=args.ridge_alpha,
        )
        for seed in seeds
    ]
    return {
        "runs": {
            "sex": sex_runs,
            "age": age_runs,
        },
        "summary": {
            "sex": summarize_sex_runs(sex_runs),
            "age": summarize_age_runs(age_runs),
        },
        "meta": {
            "n_records": int(len(records)),
            "n_subjects": int(len(set(get_subject_groups(records).tolist()))),
            "subject_level": not bool(args.scan_level),
            "n_sex_labeled": int(len(sex_records)),
            "n_age_labeled": int(len(age_records)),
        },
    }


def format_pct(metric: dict):
    return f"{100.0 * metric['mean']:.1f} ± {100.0 * metric['std']:.1f}"


def format_value(metric: dict, digits: int = 2):
    return f"{metric['mean']:.{digits}f} ± {metric['std']:.{digits}f}"


def build_markdown(payload: dict):
    lines = [
        "# Demographic Prediction",
        "",
        (
            "Sex prediction uses subject-grouped stratified CV. "
            "Age prediction uses subject-grouped CV. "
            "By default, features are averaged to one row per subject; JEPA uses mean pooled sliding-window embeddings. "
            "ComBat/CovBat are applied to static FC using site labels only before the probes."
        ),
        "",
        "| Dataset | Method | Sex bAcc | Sex ROC-AUC | Age MAE | Age RMSE | Age R2 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset_key in payload["order"]:
        dataset_payload = payload["datasets"][dataset_key]
        dataset_label = dataset_payload["config"]["dataset_label"]
        for method_key in payload["methods"]:
            if method_key not in dataset_payload["methods"]:
                continue
            summary = dataset_payload["methods"][method_key]["summary"]
            lines.append(
                f"| {dataset_label} | {METHOD_LABELS[method_key]} | "
                f"{format_pct(summary['sex']['balanced_accuracy'])} | "
                f"{format_pct(summary['sex']['roc_auc'])} | "
                f"{format_value(summary['age']['mae'])} | "
                f"{format_value(summary['age']['rmse'])} | "
                f"{format_value(summary['age']['r2'], digits=3)} |"
            )
    return "\n".join(lines) + "\n"


def default_output_paths(dataset_choice: str, checkpoint: str | None):
    base_dir = Path("results/reports/tables/jepa_fusion")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        stem = f"{checkpoint_path.parent.name}_{checkpoint_path.stem}_{dataset_choice}_demographics"
    else:
        stem = f"{dataset_choice}_fc_demographics"
    return base_dir / f"{stem}.json", base_dir / f"{stem}.md"


def main():
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    methods = parse_methods(args.methods, args.checkpoint)
    seed_everything(seeds[0], deterministic=True)

    model = None
    cfg = {}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if "jepa" in methods:
        model, checkpoint, cfg = load_checkpoint(args.checkpoint)
        device = args.device or cfg.get("device") or device
        model = model.to(device)

    dataset_order = dataset_order_for_choice(args.dataset)
    payload = {
        "config": {
            "dataset": args.dataset,
            "checkpoint": None if args.checkpoint is None else str(Path(args.checkpoint)),
            "folds": int(args.folds),
            "seeds": list(seeds),
            "pca_components": int(args.pca_components),
            "ridge_alpha": float(args.ridge_alpha),
            "feature_level": "scan_or_window" if args.scan_level else "subject_mean",
            "harmonization": {
                "mode": "transductive_site_only",
                "covariates": [],
                "covbat_n_pc": int(args.covbat_n_pc),
                "covbat_pct_var": float(args.covbat_pct_var),
                "covbat_max_components": int(args.covbat_max_components),
            },
            "jepa": {
                "device": device,
                "window_feature": args.window_feature,
                "sampling": args.jepa_sampling,
                "aggregation": "scan_or_window" if args.scan_level else "subject_mean",
                "covariance_dim": int(args.covariance_dim),
                "covariance_seed": int(args.covariance_seed),
                "batch_size": int(args.batch_size),
                "max_scans": None if args.max_scans is None else int(args.max_scans),
            },
        },
        "methods": list(methods),
        "order": list(dataset_order),
        "datasets": {},
    }

    for dataset_key in dataset_order:
        preset = get_dataset_preset(dataset_key)
        dataset_payload = {
            "config": {
                "dataset": preset.key,
                "dataset_label": preset.label,
                "fc_dir": preset.fc_dir,
                "raw_dir": preset.raw_dir,
            },
            "methods": {},
        }

        fc_methods = [method for method in methods if method.startswith("fc")]
        if fc_methods:
            fc_records = matched_records_for_preset(preset)
            if not fc_records:
                raise FileNotFoundError(
                    f"No matched FC/raw records found for dataset {preset.label} "
                    f"under {preset.fc_dir!r} and {preset.raw_dir!r}."
                )
            fc_feature_map = build_fc_method_features(fc_records, args, fc_methods, seeds)
            for method_key, features in fc_feature_map.items():
                dataset_payload["methods"][method_key] = evaluate_method(features, fc_records, seeds, args)

        if "jepa" in methods:
            jepa_features, jepa_records = build_jepa_features(model, cfg, preset, args, device)
            dataset_payload["methods"]["jepa"] = evaluate_method(jepa_features, jepa_records, seeds, args)

        payload["datasets"][dataset_key] = dataset_payload

    output_json, output_md = default_output_paths(args.dataset, args.checkpoint)
    if args.output_json is not None:
        output_json = Path(args.output_json)
    if args.output_md is not None:
        output_md = Path(args.output_md)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2))
    output_md.write_text(build_markdown(payload))
    print(build_markdown(payload))
    print(f"Saved JSON: {output_json}")
    print(f"Saved Markdown: {output_md}")


if __name__ == "__main__":
    main()
