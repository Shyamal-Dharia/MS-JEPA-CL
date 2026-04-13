"""Shared evaluation protocol for the jepa_fusion branch.

Provides:
- dataset definitions (FC dirs, ROI dirs, tasks)
- mixed-site-only filtering
- fold-wise ComBat / CovBat wrappers
- subject-grouped CV splits
- unified probe pipeline
"""

import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


@dataclass
class DatasetSpec:
    name: str
    display_name: str
    fc_dir: str
    raw_dir: str
    task: str          # e.g. "hc_asd", "hc_disorder"
    hc_diag: int
    disorder_diag: object  # int or "ALL"


DATASETS = {
    "abide1": DatasetSpec(
        name="abide1",
        display_name="ABIDE-I",
        fc_dir="preprocessed_data_fir/FC_abide_full",
        raw_dir="preprocessed_data_fir/full/roi_timeseries_abide_full",
        task="hc_asd",
        hc_diag=0,
        disorder_diag=1,
    ),
    "abide2": DatasetSpec(
        name="abide2",
        display_name="ABIDE-II",
        fc_dir="preprocessed_data_fir/FC_abide2_full",
        raw_dir="preprocessed_data_fir/full/roi_timeseries_abide2_full",
        task="hc_asd",
        hc_diag=0,
        disorder_diag=1,
    ),
    "srpbs": DatasetSpec(
        name="srpbs",
        display_name="SRPBS-1600",
        fc_dir="preprocessed_data_fir/FC_1600",
        raw_dir="preprocessed_data_fir/full/roi_timeseries_1600_full",
        task="hc_disorder",
        hc_diag=0,
        disorder_diag="ALL",
    ),
}


# ---------------------------------------------------------------------------
# Scan / FC record loading
# ---------------------------------------------------------------------------


ABIDE_SITE_NORMALIZATION = {
    "ABIDEII-KKI_1": "KKI",
    "ABIDEII-NYU_1": "NYU",
    "ABIDEII-OHSU_1": "OHSU",
    "ABIDEII-SDSU_1": "SDSU",
    "ABIDEII-TCD_1": "Trinity",
    "ABIDEII-UCLA_1": "UCLA_1",
    "ABIDEII-USM_1": "USM",
}


def normalize_site(site: str) -> str:
    return ABIDE_SITE_NORMALIZATION.get(site, site)


@dataclass
class FCRecord:
    fc_path: str
    meta_path: str
    subject: str
    site: str
    site_raw: str
    dataset: str
    diag: int
    diag_name: str
    scan_id: str


@dataclass
class RawRecord:
    ts_path: str
    meta_path: str
    subject: str
    site: str
    site_raw: str
    dataset: str
    diag: int
    diag_name: str
    tr: float
    n_timepoints: int
    scan_id: str


def _parse_bids_entities(stem: str) -> dict:
    entities = {}
    for part in stem.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        entities[key] = value
    return entities


def load_fc_records(fc_dir: str, exclude_unknown: bool = True) -> list[FCRecord]:
    """Load FC matrices and their metadata."""
    records = []
    for fc_path in sorted(glob.glob(os.path.join(fc_dir, "*_fc.npy"))):
        meta_path = fc_path.replace("_fc.npy", "_fc_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        diag = int(meta.get("diag", meta.get("dx", -1)))
        if exclude_unknown and diag < 0:
            continue
        stem = os.path.basename(fc_path).replace("_fc.npy", "")
        entities = _parse_bids_entities(stem)
        subject = meta.get("subject", f"sub-{entities.get('sub', stem)}")
        site_raw = meta.get("site", "")
        records.append(FCRecord(
            fc_path=fc_path,
            meta_path=meta_path,
            subject=subject,
            site=normalize_site(site_raw),
            site_raw=site_raw,
            dataset=meta.get("dataset", "unknown"),
            diag=diag,
            diag_name=meta.get("diag_name", meta.get("dx_name", "unknown")),
            scan_id=meta.get("scan_id", stem),
        ))
    return records


def load_raw_records(raw_dir: str, exclude_unknown: bool = True) -> list[RawRecord]:
    """Load ROI time-series metadata."""
    records = []
    for ts_path in sorted(glob.glob(os.path.join(raw_dir, "*_roi_ts.npy"))):
        meta_path = ts_path.replace("_roi_ts.npy", "_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        diag = int(meta.get("diag", meta.get("dx", -1)))
        if exclude_unknown and diag < 0:
            continue
        stem = os.path.basename(ts_path).replace("_roi_ts.npy", "")
        entities = _parse_bids_entities(stem)
        subject = meta.get("subject", f"sub-{entities.get('sub', stem)}")
        site_raw = meta.get("site", "")
        records.append(RawRecord(
            ts_path=ts_path,
            meta_path=meta_path,
            subject=subject,
            site=normalize_site(site_raw),
            site_raw=site_raw,
            dataset=meta.get("dataset", "unknown"),
            diag=diag,
            diag_name=meta.get("diag_name", meta.get("dx_name", "unknown")),
            tr=float(meta.get("tr", 2.0) or 2.0),
            n_timepoints=int(meta.get("n_timepoints", 0)),
            scan_id=meta.get("scan_id", stem),
        ))
    return records


# ---------------------------------------------------------------------------
# Task filtering
# ---------------------------------------------------------------------------


def make_subject_id(record) -> str:
    return f"{record.dataset}:{record.subject}"


def filter_task_records(records, hc_diag: int, disorder_diag):
    """Keep records relevant to a binary diagnosis task."""
    kept = []
    labels = []
    for rec in records:
        d = rec.diag
        if d < 0:
            continue
        if disorder_diag == "ALL":
            kept.append(rec)
            labels.append(int(d != hc_diag))
        elif d in {hc_diag, disorder_diag}:
            kept.append(rec)
            labels.append(int(d == disorder_diag))
    return kept, np.array(labels, dtype=int)


def filter_mixed_sites(records, labels):
    """Remove single-class sites — keep only sites with both classes."""
    site_classes = defaultdict(set)
    for rec, lab in zip(records, labels):
        site_classes[rec.site].add(int(lab))
    mixed_sites = {s for s, classes in site_classes.items() if len(classes) >= 2}
    kept = []
    kept_labels = []
    removed = defaultdict(int)
    for rec, lab in zip(records, labels):
        if rec.site in mixed_sites:
            kept.append(rec)
            kept_labels.append(lab)
        else:
            removed[rec.site] += 1
    if removed:
        logger.info("Removed single-class sites: %s", dict(removed))
    return kept, np.array(kept_labels, dtype=int)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def load_fc_features(records: list[FCRecord], triangle: str = "upper") -> np.ndarray:
    """Load FC matrices and extract upper-triangle vectors."""
    features = []
    for rec in records:
        fc = np.load(rec.fc_path).astype(np.float32)
        if triangle == "upper":
            idx = np.triu_indices_from(fc, k=1)
            features.append(fc[idx])
        else:
            features.append(fc.flatten())
    return np.stack(features, axis=0)


def load_raw_features(records: list[RawRecord], target_timepoints: int = 45) -> np.ndarray:
    """Load ROI time-series, center-window, resample, and flatten."""
    from ..jepa_clean.data import resample_time_window

    features = []
    for rec in records:
        ts = np.load(rec.ts_path).astype(np.float32)
        # Center window
        native_window = max(4, int(round(90.0 / max(rec.tr, 1e-6))))
        native_window = min(native_window, ts.shape[0])
        if ts.shape[0] > native_window:
            start = (ts.shape[0] - native_window) // 2
            ts = ts[start : start + native_window]
        window = resample_time_window(ts, target_timepoints)
        features.append(window.flatten())
    return np.stack(features, axis=0)


# ---------------------------------------------------------------------------
# ComBat / CovBat
# ---------------------------------------------------------------------------


def apply_combat_foldwise(X_train, X_test, sites_train, sites_test):
    """Fit ComBat on training data, apply to both train and test."""
    try:
        from neuroCombat import neuroCombat
    except ImportError:
        from combat import combat as neuroCombat

    import pandas as pd

    sites_all = np.concatenate([sites_train, sites_test])
    X_all = np.concatenate([X_train, X_test], axis=0)
    n_train = X_train.shape[0]

    # neuroCombat expects features as rows
    data = pd.DataFrame(X_all.T)
    covars = pd.DataFrame({"batch": sites_all})

    try:
        result = neuroCombat(dat=data, covars=covars, batch_col="batch")
        harmonized = result["data"].values.T if isinstance(result, dict) else result.T
    except Exception:
        # Fall back to no harmonization if ComBat fails
        logger.warning("ComBat failed, returning unharmonized features")
        return X_train.copy(), X_test.copy()

    return harmonized[:n_train], harmonized[n_train:]


def apply_covbat_foldwise(
    X_train, X_test, sites_train, sites_test,
    n_pc=50, pct_var=0.95, max_components=256,
):
    """Approximate CovBat: ComBat + PCA-based covariance matching."""
    # Step 1: ComBat
    X_train_c, X_test_c = apply_combat_foldwise(X_train, X_test, sites_train, sites_test)

    # Step 2: PC-score batch matching on leading components
    n_comp = min(n_pc, X_train_c.shape[0] - 1, X_train_c.shape[1])
    if n_comp < 2:
        return X_train_c, X_test_c

    pca = PCA(n_components=n_comp, random_state=0)
    scores_train = pca.fit_transform(X_train_c)

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_use = int(np.searchsorted(cum_var, pct_var) + 1)
    n_use = min(n_use, max_components, n_comp)

    # Compute per-site scaling factors on train set
    unique_sites = np.unique(sites_train)
    global_std = scores_train[:, :n_use].std(axis=0, keepdims=True).clip(min=1e-8)
    site_scales = {}
    for site in unique_sites:
        mask = sites_train == site
        if mask.sum() < 2:
            site_scales[site] = np.ones((1, n_use))
        else:
            site_std = scores_train[mask, :n_use].std(axis=0, keepdims=True).clip(min=1e-8)
            site_scales[site] = global_std / site_std

    # Apply to train
    for site in unique_sites:
        mask = sites_train == site
        scores_train[mask, :n_use] *= site_scales[site]

    X_train_out = pca.inverse_transform(scores_train)

    # Apply to test
    scores_test = pca.transform(X_test_c)
    for site in np.unique(sites_test):
        mask = sites_test == site
        if site in site_scales:
            scores_test[mask, :n_use] *= site_scales[site]
        # else leave unscaled (unseen site)
    X_test_out = pca.inverse_transform(scores_test)

    return X_train_out.astype(np.float32), X_test_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Grouped cross-validation
# ---------------------------------------------------------------------------


def grouped_cv_splits(records, labels, n_folds=5, seed=0):
    """Subject-grouped stratified K-fold splits."""
    subjects = np.array([make_subject_id(rec) for rec in records])
    le = LabelEncoder()
    subject_ids_encoded = le.fit_transform(subjects)

    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(records)), labels, groups=subject_ids_encoded))


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------


def build_probe(n_train: int, seed: int = 0):
    """Shared StandardScaler -> PCA -> LogisticRegression probe."""
    n_components = min(50, n_train - 1)
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=max(1, n_components), random_state=seed),
        LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            random_state=seed,
        ),
    )


def evaluate_probe_cv(
    features, labels, records,
    n_folds=5, seeds=(0, 1, 2, 3, 4),
    harmonize=None,
):
    """Run grouped CV with optional fold-wise harmonization.

    harmonize: None, "combat", or "covbat"
    Returns dict of {metric_name: (mean, std)}.
    """
    sites = np.array([rec.site for rec in records])

    all_dx_bacc = []
    all_dx_auc = []
    all_site_bacc = []

    for seed in seeds:
        splits = grouped_cv_splits(records, labels, n_folds=n_folds, seed=seed)
        fold_dx_bacc = []
        fold_dx_auc = []
        fold_site_bacc = []

        for train_idx, test_idx in splits:
            X_train = features[train_idx].copy()
            X_test = features[test_idx].copy()
            y_train = labels[train_idx]
            y_test = labels[test_idx]
            sites_train = sites[train_idx]
            sites_test = sites[test_idx]

            # Fold-wise harmonization
            if harmonize == "combat":
                X_train, X_test = apply_combat_foldwise(X_train, X_test, sites_train, sites_test)
            elif harmonize == "covbat":
                X_train, X_test = apply_covbat_foldwise(X_train, X_test, sites_train, sites_test)

            # Diagnosis probe
            if len(np.unique(y_train)) >= 2:
                clf = build_probe(len(y_train), seed=seed)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None
                fold_dx_bacc.append(balanced_accuracy_score(y_test, y_pred))
                try:
                    fold_dx_auc.append(roc_auc_score(y_test, y_prob))
                except ValueError:
                    fold_dx_auc.append(float("nan"))

            # Site probe
            site_enc = LabelEncoder()
            y_site_train = site_enc.fit_transform(sites_train)
            y_site_test = site_enc.transform(sites_test)
            if len(np.unique(y_site_train)) >= 2:
                site_clf = build_probe(len(y_site_train), seed=seed)
                site_clf.fit(X_train, y_site_train)
                site_pred = site_clf.predict(X_test)
                fold_site_bacc.append(balanced_accuracy_score(y_site_test, site_pred))

        all_dx_bacc.append(np.mean(fold_dx_bacc) if fold_dx_bacc else float("nan"))
        all_dx_auc.append(np.mean(fold_dx_auc) if fold_dx_auc else float("nan"))
        all_site_bacc.append(np.mean(fold_site_bacc) if fold_site_bacc else float("nan"))

    return {
        "dx_bacc": (np.nanmean(all_dx_bacc), np.nanstd(all_dx_bacc)),
        "dx_auc": (np.nanmean(all_dx_auc), np.nanstd(all_dx_auc)),
        "site_bacc": (np.nanmean(all_site_bacc), np.nanstd(all_site_bacc)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def format_metric(mean_std_pair, scale=100.0):
    mean, std = mean_std_pair
    return f"{mean * scale:.1f} ± {std * scale:.1f}"


def save_results(results: dict, json_path: str, md_path: str, title: str):
    """Save results as both JSON and a summary markdown table."""
    path_json = Path(json_path)
    path_md = Path(md_path)
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_md.parent.mkdir(parents=True, exist_ok=True)

    with open(path_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", path_json)

    with open(path_md, "w") as f:
        f.write(f"# {title}\n\n")
        if "task_display" in results:
            f.write(f"Task: `{results['task_display']}`\n\n")
        if "table" in results:
            f.write(results["table"])
            f.write("\n")
    logger.info("Saved Markdown: %s", path_md)
